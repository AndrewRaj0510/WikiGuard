"""The public API. Everything a browser can do goes through here, and roles are checked here."""

import hashlib
import hmac
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from psycopg.types.json import Jsonb
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from wikiguard_core.chain import ChainError, ServingChain
from wikiguard_core.features import build_features
from wikiguard_core.predictor import AllModelsFailed
from wikiguard_core.schemas import EditEvent

from wikiguard_api import models, queries
from wikiguard_api.access import USER, Access, Principal, audit
from wikiguard_api.cache import TtlCache
from wikiguard_api.explain import Explainer, Provider
from wikiguard_api.foundations import (
    PROBLEM_TYPE,
    Database,
    Problem,
    RateLimiter,
    Settings,
    problem_body,
)

logger = logging.getLogger("wikiguard_api")

SESSION_COOKIE = "wikiguard_session"
TAB_HEADER = "x-tab-id"
CSRF_HEADER = "x-csrf-token"
REQUEST_ID_HEADER = "x-request-id"
EDGE_SECRET_HEADER = "x-edge-secret"  # noqa: S105 - a header name
# Set by the Worker (edge/worker.js), and believed only with the secret above. Not
# cf-connecting-ip: Cloudflare sets that one itself on a Worker's requests out, to its own
# address, and Render sits behind Cloudflare too, so it can't carry the visitor (DEC-086).
EDGE_CLIENT_IP_HEADER = "x-wikiguard-client-ip"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{8,64}$")
# /health must stay reachable from Render's own checks, which don't come through the Worker.
EDGE_EXEMPT_PATHS = {"/health"}
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 50
PREDICT_CACHE_SIZE = 512
READY_CHECK_SECONDS = 5.0
MODEL_RETRY_SECONDS = 60.0
# (limit, window in seconds)
LIMIT_READ = (120, 60)
LIMIT_GUEST_SIGN_IN = (5, 60)
LIMIT_USER_SIGN_IN = (10, 60)
LIMIT_EXPLAIN = (10, 60)
LIMIT_PREDICT_GUEST = (10, 60)
LIMIT_PREDICT_USER = (60, 60)
LIMIT_WRITE = (60, 60)
RETRY_MODEL_SECONDS = 30
SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    # This API returns JSON and never a page, so nothing here should ever run or load.
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
    "cross-origin-resource-policy": "same-site",
}
HSTS_HEADER = "max-age=31536000; includeSubDomains"


@dataclass
class ApiMetrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.requests = Counter(
            "api_requests", "Requests", ["method", "route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "api_request_seconds", "Request time", ["route"], registry=self.registry
        )
        self.cache = Counter(
            "api_cache_events", "Cache use", ["cache", "event"], registry=self.registry
        )
        self.llm = Counter(
            "api_llm_calls", "LLM calls", ["provider", "outcome"], registry=self.registry
        )
        self.predict_busy = Counter(
            "api_predict_rejected", "Refused by the concurrency cap", registry=self.registry
        )
        self.predict_answers = Counter(
            "api_predict_answers", "Answers", ["model", "fallback"], registry=self.registry
        )
        self.model_ready = Gauge("api_model_ready", "1 once loaded", registry=self.registry)
        self.reload_failures = Gauge(
            "api_reload_failures", "Failed alias reloads", registry=self.registry
        )

    def cache_event(self, cache: str, event: str) -> None:
        self.cache.labels(cache, event).inc()

    def llm_call(self, provider: str, outcome: str) -> None:
        self.llm.labels(provider, outcome).inc()


class PredictCache:
    """Repeating the same text shouldn't repeat inference. Bounded, as anyone can vary the text."""

    def __init__(self, size: int = PREDICT_CACHE_SIZE) -> None:
        self.size = size
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


@dataclass
class Context:
    settings: Settings
    db: Database
    access: Access
    limiter: RateLimiter
    metrics: ApiMetrics
    chain: ServingChain | None
    explainer: Explainer | None
    queue_cache: TtlCache
    feed_cache: TtlCache
    models_cache: TtlCache
    health_cache: TtlCache
    predict_cache: PredictCache
    predict_slots: threading.BoundedSemaphore
    stop: threading.Event = field(default_factory=threading.Event)
    model_error: str | None = None
    _ready_checked_at: float = 0.0
    _database_up: bool = False

    def database_up(self) -> bool:
        """Cached, because /ready polling shouldn't count against Supabase's egress."""
        now = time.monotonic()
        if now - self._ready_checked_at >= READY_CHECK_SECONDS or not self._database_up:
            self._database_up = self.db.check()
            self._ready_checked_at = now
        return self._database_up

    def clear_model_caches(self) -> None:
        self.models_cache.clear()
        self.health_cache.clear()
        self.predict_cache.clear()


def get_context(request: Request) -> Context:
    context: Context = request.app.state.context
    return context


Ctx = Annotated[Context, Depends(get_context)]


def client_ip(request: Request, settings: Settings) -> str:
    """Trust a forwarded address only from the Worker, proven by its secret.

    X-Forwarded-For is never read: anyone can send it, so it would make every limit by IP optional.
    """
    secret = settings.edge_shared_secret
    if secret is not None:
        sent = request.headers.get(EDGE_SECRET_HEADER, "")
        if hmac.compare_digest(sent, secret.get_secret_value()):
            forwarded = request.headers.get(EDGE_CLIENT_IP_HEADER)
            if forwarded:
                return forwarded.strip()[:64]
    return request.client.host if request.client else "unknown"


def request_id_of(request: Request) -> str | None:
    value: str | None = getattr(request.state, "request_id", None)
    return value


def limit_by_ip(request: Request, context: Context, name: str, limit: tuple[int, int]) -> None:
    context.limiter.require(f"{name}:{client_ip(request, context.settings)}", *limit)


def seated(request: Request, context: Ctx) -> Principal:
    """Every protected read goes through here: a session, the seat, and a limit by IP."""
    limit_by_ip(request, context, "read", LIMIT_READ)
    principal = context.access.seated(
        request.cookies.get(SESSION_COOKIE), request.headers.get(TAB_HEADER)
    )
    request.state.role = principal.role
    return principal


Seated = Annotated[Principal, Depends(seated)]


def check_origin(request: Request, context: Context) -> None:
    """Browsers send Origin on writes. A site not on the list can't act with someone's cookie."""
    origin = request.headers.get("origin")
    if origin is not None and origin not in allowed_origins(context.settings):
        raise Problem(403, "bad_origin", "This request came from another site")


def writer(request: Request, context: Ctx, principal: Seated) -> Principal:
    check_origin(request, context)
    sent = request.headers.get(CSRF_HEADER, "")
    if not hmac.compare_digest(sent, principal.csrf_token):
        raise Problem(403, "bad_csrf", "The request's security token is missing or wrong")
    limit_by_ip(request, context, "write", LIMIT_WRITE)
    return principal


Writer = Annotated[Principal, Depends(writer)]


def _deny_if_guest(request: Request, context: Context, principal: Principal) -> None:
    if principal.role != USER:
        with context.db.tx() as tx:
            audit(tx, principal.role, "auth.denied", request.url.path, request_id_of(request))
        raise Problem(403, "forbidden", "This action needs the `user` account")


def owner(request: Request, context: Ctx, principal: Writer) -> Principal:
    _deny_if_guest(request, context, principal)
    return principal


def owner_read(request: Request, context: Ctx, principal: Seated) -> Principal:
    _deny_if_guest(request, context, principal)
    return principal


Owner = Annotated[Principal, Depends(owner)]
OwnerRead = Annotated[Principal, Depends(owner_read)]


def writable(context: Ctx) -> None:
    if context.settings.read_only_mode:
        raise Problem(503, "read_only", "WikiGuard is read-only right now", "Reading still works.")


Writable = Depends(writable)


def allowed_origins(settings: Settings) -> list[str]:
    return [item.strip() for item in settings.cors_allowed_origins.split(",") if item.strip()]


def etag_json(request: Request, payload: Any, status: int = 200) -> Response:
    """A poll that finds nothing new gets 304 and no body, which saves Supabase and bandwidth."""
    body = json.dumps(jsonable_encoder(payload), separators=(",", ":"), sort_keys=True).encode()
    tag = f'W/"{hashlib.sha256(body).hexdigest()[:32]}"'
    # private: this depends on being signed in, so no shared cache may keep it.
    headers = {"etag": tag, "cache-control": "private, no-cache"}
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers=headers)
    return Response(body, status, headers=headers, media_type="application/json")


def set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=settings.environment != "dev",
        samesite="lax",
        path="/",
    )


def session_info(context: Context, principal: Principal) -> dict[str, Any]:
    left = None
    if principal.role != USER:
        left = max(context.settings.guest_explain_limit - principal.fresh_explains, 0)
    return {
        "role": principal.role,
        "csrf_token": principal.csrf_token,
        "expires_at": principal.expires_at,
        "fresh_explains_left": left,
    }


class RequestMiddleware:
    """Runs around every request: ID, edge check, body limit, security headers, metrics, and logs.

    Written as plain ASGI so it also catches unexpected errors, and can answer them in the same
    format with the same headers.
    """

    def __init__(self, app: ASGIApp, settings: Settings, metrics: ApiMetrics) -> None:
        self.app = app
        self.settings = settings
        self.metrics = metrics
        self.secure = settings.environment != "dev"

    def _reject(self, status: int, code: str, title: str, request_id: str) -> Response:
        return Response(
            json.dumps(problem_body(status, code, title, request_id)),
            status,
            media_type=PROBLEM_TYPE,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        headers = {key.decode("latin-1"): value for key, value in scope["headers"]}
        sent_id = headers.get(REQUEST_ID_HEADER, b"").decode("latin-1")
        # Only a safe shape is accepted, so a client can't write whatever it likes into the logs.
        request_id = sent_id if REQUEST_ID_PATTERN.match(sent_id) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status_seen = {"status": 500, "started": False}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_seen["status"] = message["status"]
                status_seen["started"] = True
                extra = dict(SECURITY_HEADERS) | {REQUEST_ID_HEADER: request_id}
                if self.secure:
                    extra["strict-transport-security"] = HSTS_HEADER
                current = list(message.get("headers", []))
                names = {key.decode("latin-1").lower() for key, _ in current}
                if "cache-control" not in names:
                    extra["cache-control"] = "no-store"
                current += [(key.encode(), value.encode()) for key, value in extra.items()]
                message["headers"] = current
            await send(message)

        limit = self.settings.max_body_bytes
        received = {"bytes": 0}

        async def limited_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                received["bytes"] += len(message.get("body", b""))
                # Chunked bodies declare no size, so they're counted as they arrive.
                if received["bytes"] > limit:
                    raise Problem(413, "body_too_large", "That request is too large")
            return message

        response: Response | None = None
        secret = self.settings.edge_shared_secret
        if (
            secret is not None
            and scope["path"] not in EDGE_EXEMPT_PATHS
            and not hmac.compare_digest(
                headers.get(EDGE_SECRET_HEADER, b"").decode("latin-1"), secret.get_secret_value()
            )
        ):
            response = self._reject(403, "direct_access", "Use the site's address", request_id)
        length = headers.get("content-length", b"0").decode("latin-1")
        if response is None and length.isdigit() and int(length) > limit:
            response = self._reject(413, "body_too_large", "That request is too large", request_id)
        try:
            if response is not None:
                await response(scope, receive, send_with_headers)
            else:
                await self.app(scope, limited_receive, send_with_headers)
        except Exception as error:  # noqa: BLE001 - the last stop: never show the client details
            logger.error(
                "Unhandled error: %s",
                type(error).__name__,
                extra={"request_id": request_id, "route": scope["path"]},
            )
            if not status_seen["started"]:
                reply = self._reject(500, "internal_error", "Something went wrong", request_id)
                await reply(scope, receive, send_with_headers)
        route = scope.get("route")
        template = getattr(route, "path", "unmatched")
        seconds = time.perf_counter() - started
        status = int(status_seen["status"])
        self.metrics.requests.labels(scope["method"], template, str(status)).inc()
        self.metrics.latency.labels(template).observe(seconds)
        path_params = scope.get("path_params") or {}
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": scope["method"],
                "route": template,
                "status": status,
                "duration_ms": round(seconds * 1000, 1),
                "rev_id": path_params.get("rev_id"),
                "role": scope["state"].get("role"),
            },
        )


def install_error_handlers(app: FastAPI) -> None:
    def respond(request: Request, status: int, body: dict[str, Any], retry: int | None) -> Response:
        headers = {"retry-after": str(retry)} if retry else None
        return Response(json.dumps(body), status, headers=headers, media_type=PROBLEM_TYPE)

    @app.exception_handler(Problem)
    async def on_problem(request: Request, error: Problem) -> Response:
        body = problem_body(
            error.status, error.code, error.title, request_id_of(request), error.detail, error.extra
        )
        return respond(request, error.status, body, error.retry_after)

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, error: RequestValidationError) -> Response:
        # Where and why, but never the value: pydantic's errors can quote the input.
        fields = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        body = problem_body(
            422,
            "invalid_request",
            "The request isn't valid",
            request_id_of(request),
            extra={"errors": fields},
        )
        return respond(request, 422, body, None)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, error: StarletteHTTPException) -> Response:
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "http_error")
        title = {404: "Not found", 405: "Method not allowed"}.get(error.status_code, "Bad request")
        body = problem_body(error.status_code, code, title, request_id_of(request))
        return respond(request, error.status_code, body, None)


def load_models(context: Context) -> None:
    """On a background thread: the API must answer /health and serve stored data while this runs."""
    chain = context.chain
    if chain is None:
        return
    while not context.stop.is_set():
        try:
            chain.load()
            context.metrics.model_ready.set(1)
            context.model_error = None
            logger.info("Models loaded")
            break
        except (ChainError, OSError, ValueError) as error:
            # A FORCE_MODEL that isn't loaded lands here too, and /ready stays false until fixed.
            context.model_error = type(error).__name__
            logger.error("Models failed to load: %s", type(error).__name__)
            context.stop.wait(MODEL_RETRY_SECONDS)
    seconds = context.settings.model_poll_seconds
    if seconds and not context.stop.is_set():
        chain.poll(seconds, context.stop)


def document_problem_format(app: FastAPI) -> None:
    """Make the published schema describe the errors this API really sends.

    FastAPI documents every 422 as its own default error shape. The handlers above replace that
    with the problem format, so the schema was wrong, and the frontend's types and the API fuzzer
    are both generated from it. Schemathesis found the mismatch (Phase 6).
    """
    build = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = build()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        problem = models.ProblemDetails.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        components |= problem.pop("$defs", {})
        components["ProblemDetails"] = problem
        for unused in ("HTTPValidationError", "ValidationError"):
            components.pop(unused, None)
        for operations in schema["paths"].values():
            for operation in operations.values():
                if "422" in operation.get("responses", {}):
                    operation["responses"]["422"] = {
                        "description": "The request isn't valid",
                        "content": {
                            PROBLEM_TYPE: {
                                "schema": {"$ref": "#/components/schemas/ProblemDetails"}
                            }
                        },
                    }
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def create_app(
    settings: Settings,
    chain: ServingChain | None = None,
    providers: list[Provider] | None = None,
    explain_daily_cap: int = 0,
    on_statement: Any = None,
) -> FastAPI:
    metrics = ApiMetrics()
    db = Database(
        settings.database_url.get_secret_value(),
        settings.db_statement_timeout_ms,
        settings.db_pool_size,
        on_statement,
    )
    context = Context(
        settings=settings,
        db=db,
        access=Access(settings, db),
        limiter=RateLimiter(),
        metrics=metrics,
        chain=chain,
        explainer=(
            Explainer(db, providers, explain_daily_cap, settings.guest_explain_limit, metrics)
            if providers
            else None
        ),
        queue_cache=TtlCache("queue", 2.0, metrics=metrics),
        feed_cache=TtlCache("feed", 2.0, metrics=metrics),
        models_cache=TtlCache("models", 300.0, metrics=metrics),
        health_cache=TtlCache("health", 60.0, metrics=metrics),
        predict_cache=PredictCache(),
        predict_slots=threading.BoundedSemaphore(settings.predict_max_concurrency),
    )
    if chain is not None:
        chain._on_change = context.clear_model_caches  # noqa: SLF001 - wiring at startup only

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        db.open()
        loader = threading.Thread(target=load_models, args=(context,), daemon=True)
        loader.start()
        yield
        # Uvicorn has already finished in-flight requests by this point (SIGTERM drains).
        context.stop.set()
        db.close()

    app = FastAPI(
        title="WikiGuard API",
        version="1.0.0",
        lifespan=lifespan,
        openapi_url="/v1/openapi.json",
        docs_url=None,
        redoc_url=None,
    )
    app.state.context = context
    install_error_handlers(app)
    document_problem_format(app)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RequestMiddleware, settings=settings, metrics=metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(settings),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["content-type", "x-csrf-token", "x-tab-id", "x-request-id"],
        expose_headers=["x-request-id", "etag", "retry-after"],
        max_age=600,
    )

    # --- Operations ---

    @app.get("/health", include_in_schema=False)
    def health(context: Ctx) -> dict[str, str]:
        # The commit lets the deploy pipeline wait for the new build before it tests anything.
        return {"status": "ok", "commit": context.settings.git_commit or "unknown"}

    @app.get("/ready", include_in_schema=False)
    def ready(response: Response, context: Ctx) -> dict[str, Any]:
        """False until the models load. The frontend shows "waking up" until this turns true."""
        needs_model = context.settings.predict_enabled and context.chain is not None
        model_ok = context.chain.ready if needs_model and context.chain else True
        database_ok = context.database_up()
        is_ready = model_ok and database_ok
        response.status_code = 200 if is_ready else 503
        return {"ready": is_ready, "model_loaded": model_ok, "database": database_ok}

    @app.get("/metrics", include_in_schema=False)
    def prometheus(request: Request, context: Ctx) -> Response:
        expected = f"Bearer {context.settings.metrics_bearer_token.get_secret_value()}"
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise Problem(401, "not_signed_in", "A token is required")
        if context.chain is not None:
            context.metrics.reload_failures.set(context.chain.reload_failures)
        return Response(generate_latest(context.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # --- Sign-in ---

    @app.post("/v1/auth/guest", response_model=models.SessionInfo, tags=["auth"])
    def sign_in_guest(
        body: models.GuestSignIn, request: Request, response: Response, context: Ctx
    ) -> dict[str, Any]:
        check_origin(request, context)
        limit_by_ip(request, context, "guest", LIMIT_GUEST_SIGN_IN)
        ip = client_ip(request, context.settings)
        token, _ = context.access.sign_in_guest(ip, body.tab_id, request_id_of(request))
        set_session_cookie(response, token, context.settings)
        return session_info(context, context.access.session(token))

    @app.post("/v1/auth/login", response_model=models.SessionInfo, tags=["auth"])
    def sign_in_user(
        body: models.UserSignIn, request: Request, response: Response, context: Ctx
    ) -> dict[str, Any]:
        check_origin(request, context)
        limit_by_ip(request, context, "login", LIMIT_USER_SIGN_IN)
        token, _ = context.access.sign_in_user(
            body.password,
            client_ip(request, context.settings),
            body.tab_id,
            body.take_over,
            request_id_of(request),
        )
        set_session_cookie(response, token, context.settings)
        return session_info(context, context.access.session(token))

    @app.post("/v1/auth/logout", status_code=204, tags=["auth"])
    def sign_out(request: Request, response: Response, context: Ctx) -> None:
        check_origin(request, context)
        principal = context.access.session(request.cookies.get(SESSION_COOKIE))
        if not hmac.compare_digest(request.headers.get(CSRF_HEADER, ""), principal.csrf_token):
            raise Problem(403, "bad_csrf", "The request's security token is missing or wrong")
        context.access.sign_out(principal, request_id_of(request))
        response.delete_cookie(SESSION_COOKIE, path="/")

    @app.get("/v1/auth/me", response_model=models.SessionInfo, tags=["auth"])
    def who_am_i(request: Request, context: Ctx) -> dict[str, Any]:
        """No seat needed: after a reload, the page needs its token before it can claim the seat."""
        limit_by_ip(request, context, "read", LIMIT_READ)
        principal = context.access.session(request.cookies.get(SESSION_COOKIE))
        return session_info(context, principal)

    # --- The gate ---

    @app.get("/v1/gate/status", response_model=models.GateStatus, tags=["gate"])
    def gate_status(request: Request, context: Ctx) -> dict[str, Any]:
        limit_by_ip(request, context, "read", LIMIT_READ)
        return context.access.status(request.cookies.get(SESSION_COOKIE))

    def signed_in_writer(request: Request, context: Context) -> Principal:
        check_origin(request, context)
        principal = context.access.session(request.cookies.get(SESSION_COOKIE))
        if not hmac.compare_digest(request.headers.get(CSRF_HEADER, ""), principal.csrf_token):
            raise Problem(403, "bad_csrf", "The request's security token is missing or wrong")
        return principal

    @app.post("/v1/gate/claim", status_code=204, tags=["gate"])
    def claim_seat(body: models.ClaimSeat, request: Request, context: Ctx) -> None:
        principal = signed_in_writer(request, context)
        context.access.claim(principal, body.tab_id, body.take_over, request_id_of(request))

    @app.post("/v1/gate/heartbeat", response_model=models.HeartbeatReply, tags=["gate"])
    def heartbeat(body: models.Heartbeat, request: Request, context: Ctx) -> dict[str, int]:
        principal = signed_in_writer(request, context)
        tab_id = request.headers.get(TAB_HEADER)
        if not tab_id:
            raise Problem(400, "tab_required", "This request needs a tab ID")
        left = context.access.heartbeat(principal, tab_id, body.active)
        return {
            "guest_seconds_left": left,
            "lease_seconds": context.settings.gate_lease_seconds,
        }

    @app.post("/v1/gate/release", status_code=204, tags=["gate"])
    def release_seat(body: models.ReleaseSeat, request: Request, context: Ctx) -> None:
        check_origin(request, context)
        principal = context.access.session(request.cookies.get(SESSION_COOKIE))
        if not hmac.compare_digest(body.csrf_token, principal.csrf_token):
            raise Problem(403, "bad_csrf", "The request's security token is missing or wrong")
        context.access.release(principal, body.tab_id)

    # --- Reading ---

    data_mode = settings.data_mode
    snapshot = data_mode == "snapshot"

    @app.get("/v1/queue", response_model=models.EditPage, tags=["edits"])
    def review_queue(
        request: Request,
        context: Ctx,
        _: Seated,
        editor: Literal["all", "logged_out", "registered"] = "all",
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        cursor: Annotated[str | None, Query(max_length=256)] = None,
        include_reviewed: bool = False,
    ) -> Response:
        def load() -> dict[str, Any]:
            with context.db.tx() as tx:
                return queries.review_queue(
                    tx, editor, limit, cursor, include_reviewed, snapshot=snapshot
                )

        page, stale = context.queue_cache.get((editor, limit, cursor, include_reviewed), load)
        return etag_json(request, page | {"stale": stale, "data_mode": data_mode})

    @app.get("/v1/feed", response_model=models.EditPage, tags=["edits"])
    def live_feed(
        request: Request,
        context: Ctx,
        _: Seated,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        cursor: Annotated[str | None, Query(max_length=256)] = None,
    ) -> Response:
        def load() -> dict[str, Any]:
            with context.db.tx() as tx:
                return queries.live_feed(tx, limit, cursor, snapshot=snapshot)

        page, stale = context.feed_cache.get((limit, cursor), load)
        return etag_json(request, page | {"stale": stale, "data_mode": data_mode})

    @app.get("/v1/edits/{rev_id}", response_model=models.EditDetail, tags=["edits"])
    def edit_detail(rev_id: int, context: Ctx, _: Seated) -> dict[str, Any]:
        with context.db.tx() as tx:
            row = queries.edit_detail(tx, rev_id)
        if row is None:
            raise Problem(404, "edit_not_found", "No such edit")
        return row

    @app.get("/v1/models", response_model=models.ModelList, tags=["models"])
    def model_leaderboard(request: Request, context: Ctx, _: Seated) -> Response:
        def load() -> list[dict[str, Any]]:
            with context.db.tx() as tx:
                return queries.model_list(tx)

        items, stale = context.models_cache.get("all", load)
        return etag_json(request, {"items": items, "stale": stale})

    @app.get("/v1/health/summary", response_model=models.HealthSummary, tags=["models"])
    def health_summary(request: Request, context: Ctx, _: Seated) -> Response:
        def load() -> dict[str, Any]:
            with context.db.tx() as tx:
                return queries.health_summary(tx, snapshot=snapshot)

        summary, stale = context.health_cache.get("summary", load)
        chain = context.chain
        loaded = chain is not None and chain.ready
        statuses = chain.current.predictor.status() if chain is not None and loaded else []
        return etag_json(
            request,
            summary
            | {
                "stale": stale,
                "models_loaded": loaded,
                "chain": [
                    {
                        "alias": item.alias,
                        "model_name": item.model_name,
                        "model_version": item.model_version,
                        "breaker": item.breaker.value,
                    }
                    for item in statuses
                ],
            },
        )

    # --- Prediction ---

    def run_model(context: Context, body: models.PredictRequest) -> dict[str, Any]:
        chain = context.chain
        if chain is None or not chain.ready:
            raise Problem(
                503,
                "model_loading",
                "The model is still loading",
                retry_after=RETRY_MODEL_SECONDS,
            )
        version = chain.current.refs[0].version
        content = body.model_dump_json(exclude={"rev_id"})
        key = hashlib.sha256(f"{version}:{content}".encode()).hexdigest()
        known = context.predict_cache.get(key)
        if known is not None:
            return known | {"source": "cache"}
        # A full set of slots answers at once: queueing would only slow everyone and hold memory.
        if not context.predict_slots.acquire(blocking=False):
            context.metrics.predict_busy.inc()
            raise Problem(503, "predict_busy", "Too many predictions at once", retry_after=2)
        try:
            edit = EditEvent(
                rev_id=0,
                parent_rev_id=None,
                title="",
                timestamp=datetime.now(UTC),
                is_minor=body.is_minor,
                is_anonymous=body.is_anonymous,
                user_hash=None,
                comment=body.comment,
                old_length=body.old_length,
                new_length=body.new_length,
                added_text=body.added_text,
                removed_text=body.removed_text,
                user_edit_count=body.user_edit_count,
            )
            prediction = chain.current.predictor.score([build_features(edit).as_dict()])
        except AllModelsFailed as error:
            raise Problem(
                503, "models_unavailable", "No model could answer", retry_after=10
            ) from error
        finally:
            context.predict_slots.release()
        context.metrics.predict_answers.labels(
            prediction.model_name, str(prediction.used_fallback).lower()
        ).inc()
        result = {
            "score": float(prediction.scores[0]),
            "threshold": prediction.threshold,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "used_fallback": prediction.used_fallback,
            "forced": prediction.forced,
            "latency_ms": prediction.latency_ms,
        }
        # A forced or fallback answer isn't kept, so recovery shows up straight away.
        if not prediction.used_fallback and not prediction.forced:
            context.predict_cache.put(key, result)
        return result | {"source": "model"}

    @app.post("/v1/predict", response_model=models.PredictReply, tags=["models"])
    def predict(
        body: models.PredictRequest, request: Request, context: Ctx, principal: Writer
    ) -> dict[str, Any]:
        if not context.settings.predict_enabled:
            raise Problem(503, "feature_disabled", "Prediction is switched off")
        limit = LIMIT_PREDICT_USER if principal.role == USER else LIMIT_PREDICT_GUEST
        limit_by_ip(request, context, "predict", limit)
        if body.rev_id is not None:
            with context.db.tx() as tx:
                stored = queries.stored_score(tx, body.rev_id)
            if stored is None:
                raise Problem(404, "edit_not_found", "No stored score for that edit")
            return stored | {"source": "stored"}
        return run_model(context, body)

    # --- Explain ---

    @app.post("/v1/explain/{rev_id}", response_model=models.ExplainReply, tags=["edits"])
    def explain(rev_id: int, request: Request, context: Ctx, principal: Writer) -> dict[str, Any]:
        if not context.settings.explain_enabled or context.explainer is None:
            raise Problem(503, "feature_disabled", "Explain is switched off")
        limit_by_ip(request, context, "explain", LIMIT_EXPLAIN)
        answer = context.explainer.explain(
            rev_id,
            principal.session_id,
            principal.role != USER,
            context.settings.read_only_mode,
            request_id_of(request),
        )
        return {
            "rev_id": rev_id,
            "explanation": answer.explanation,
            "available": answer.available,
            "cached": answer.cached,
            "reason": answer.reason,
            "llm_model": answer.llm_model,
        }

    # --- Reviewing ---

    @app.put(
        "/v1/edits/{rev_id}/review",
        response_model=models.ReviewReply,
        tags=["edits"],
        dependencies=[Writable],
    )
    def review_edit(
        rev_id: int, body: models.ReviewRequest, request: Request, context: Ctx, principal: Owner
    ) -> dict[str, Any]:
        """PUT, because repeating it gives the same result: one review per edit (idempotent)."""
        with context.db.tx() as tx:
            if tx.one("SELECT 1 AS found FROM edits WHERE rev_id = %s", (rev_id,)) is None:
                raise Problem(404, "edit_not_found", "No such edit")
            row = tx.one(
                """
                INSERT INTO reviews (rev_id, decision, note) VALUES (%s, %s, %s)
                ON CONFLICT (rev_id) DO UPDATE SET
                    decision = EXCLUDED.decision, note = EXCLUDED.note, updated_at = now()
                RETURNING (xmax <> 0) AS updated
                """,
                (rev_id, body.decision, body.note),
            )
            audit(
                tx,
                principal.role,
                "review.saved",
                str(rev_id),
                request_id_of(request),
                {"decision": body.decision},
            )
        context.queue_cache.clear()
        context.feed_cache.clear()
        return {
            "rev_id": rev_id,
            "decision": body.decision,
            "updated": bool(row and row["updated"]),
        }

    # --- Administration. Hidden from the public schema, and `user` only. ---

    def add_request(
        context: Context,
        request: Request,
        kind: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render holds no cluster credentials, so asking means adding a row that the cluster
        polls for. A request already waiting is returned instead of adding a duplicate."""
        with context.db.tx() as tx:
            existing = tx.one(
                f"SELECT {queries.REQUEST_COLUMNS} FROM retrain_requests "
                "WHERE kind = %s AND status IN ('pending', 'running') ORDER BY id DESC LIMIT 1",
                (kind,),
            )
            if existing is not None:
                return existing
            row = tx.one(
                f"""
                INSERT INTO retrain_requests (kind, reason, requested_by, details)
                VALUES (%s, %s, %s, %s) RETURNING {queries.REQUEST_COLUMNS}
                """,
                (kind, reason, USER, Jsonb(details or {})),
            )
            audit(
                tx,
                USER,
                f"admin.{kind}",
                str(row["id"]) if row else None,
                request_id_of(request),
                {"reason": reason},
            )
        assert row is not None  # noqa: S101 - RETURNING on an INSERT
        return row

    @app.post(
        "/v1/admin/retrain",
        include_in_schema=False,
        status_code=202,
        response_model=models.RequestInfo,
        dependencies=[Writable],
    )
    def admin_retrain(
        body: models.AdminRequest, request: Request, context: Ctx, _: Owner
    ) -> dict[str, Any]:
        return add_request(context, request, "retrain", body.reason)

    @app.post(
        "/v1/admin/promote",
        include_in_schema=False,
        status_code=202,
        response_model=models.RequestInfo,
        dependencies=[Writable],
    )
    def admin_promote(
        body: models.PromoteRequest, request: Request, context: Ctx, _: Owner
    ) -> dict[str, Any]:
        return add_request(
            context, request, "promote", body.reason, {"model_version": body.model_version}
        )

    @app.post(
        "/v1/admin/rollback",
        include_in_schema=False,
        status_code=202,
        response_model=models.RequestInfo,
        dependencies=[Writable],
    )
    def admin_rollback(
        body: models.AdminRequest, request: Request, context: Ctx, _: Owner
    ) -> dict[str, Any]:
        return add_request(context, request, "rollback", body.reason)

    @app.get("/v1/admin/requests", include_in_schema=False, response_model=models.RequestList)
    def admin_requests(context: Ctx, _: OwnerRead) -> dict[str, Any]:
        with context.db.tx() as tx:
            rows = tx.all(
                f"SELECT {queries.REQUEST_COLUMNS} FROM retrain_requests ORDER BY id DESC LIMIT 50"
            )
        return {"items": rows}

    @app.get("/v1/admin/audit-log", include_in_schema=False, response_model=models.AuditPage)
    def admin_audit(
        context: Ctx,
        _: OwnerRead,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        cursor: Annotated[str | None, Query(max_length=256)] = None,
    ) -> dict[str, Any]:
        with context.db.tx() as tx:
            return queries.audit_page(tx, limit, cursor)

    return app
