"""Request and response shapes: the public contract, and the frontend's types come from it.

No model has a field for a username, IP, or user_hash, so none can be returned by accident.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Limits are generous for a real edit, and stop anything built to be expensive.
MAX_PREDICT_TEXT = 20_000
MAX_PREDICT_COMMENT = 1000
TabId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")]
Decision = Literal["vandalism", "good_faith_mistake", "ok", "skip"]


class Closed(BaseModel):
    """Unknown fields are refused, so a typo can't silently change what a request means."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GuestSignIn(Closed):
    tab_id: TabId


class UserSignIn(Closed):
    password: str = Field(min_length=1, max_length=256)
    tab_id: TabId
    take_over: bool = False


class SessionInfo(Closed):
    role: Literal["guest", "user"]
    csrf_token: str
    expires_at: datetime
    fresh_explains_left: int | None = None


class GateStatus(Closed):
    enabled: bool
    state: Literal["free", "in_use"]
    retry_after_seconds: int | None
    mine: bool


class ClaimSeat(Closed):
    tab_id: TabId
    take_over: bool = False


class Heartbeat(Closed):
    # True only if a person did something since the last one. It keeps a guest's idle timer alive.
    active: bool = False


class HeartbeatReply(Closed):
    guest_seconds_left: int
    lease_seconds: int


class ReleaseSeat(Closed):
    """Sent by sendBeacon on pagehide, which can't set headers, so the token travels in the body."""

    tab_id: TabId
    csrf_token: str = Field(max_length=128)


class EditSummary(Closed):
    """A row in the feed or queue. Never carries diff text, so list views stay small."""

    rev_id: int
    title: str
    edited_at: datetime
    is_minor: bool
    is_anonymous: bool
    comment: str | None
    bytes_delta: int | None
    score: float | None
    model_name: str | None
    model_version: str | None
    threshold: float | None
    used_fallback: bool | None
    load_shed: bool | None
    decision: Decision | None
    content_hidden: bool
    wikipedia_url: str


class EditPage(Closed):
    items: list[EditSummary]
    next_cursor: str | None
    stale: bool
    # "snapshot" when the edits are a fixed, already-scored dataset, not the live stream (DEC-091).
    data_mode: Literal["live", "snapshot"]


class ScoreInfo(Closed):
    model_name: str
    model_version: str
    alias: str | None
    score: float
    threshold: float | None
    used_fallback: bool
    load_shed: bool
    forced: bool
    latency_ms: float | None
    scored_at: datetime


class EditDetail(EditSummary):
    parent_rev_id: int | None
    added_text: str | None
    removed_text: str | None
    comment_hidden: bool
    user_hidden: bool
    user_edit_count: int | None
    scores: list[ScoreInfo]
    label: int | None
    label_excluded_reason: str | None
    has_explanation: bool


class ReviewRequest(Closed):
    decision: Decision
    note: str | None = Field(default=None, max_length=500)


class ReviewReply(Closed):
    rev_id: int
    decision: Decision
    updated: bool


class ModelEntry(Closed):
    model_name: str
    model_version: str
    alias: str | None
    feature_version: int
    snapshot_hash: str
    metrics: dict[str, Any]
    registered_at: datetime


class ModelList(Closed):
    items: list[ModelEntry]
    stale: bool


class BreakerInfo(Closed):
    alias: str
    model_name: str
    model_version: str
    breaker: str


class HealthSummary(Closed):
    stale: bool
    # In "snapshot" mode the counts cover the whole dataset, which runs from data_from to data_to,
    # not the last 24 hours (DEC-091).
    data_mode: Literal["live", "snapshot"]
    data_from: datetime | None
    data_to: datetime | None
    scored_last_24h: int
    fallback_rate_24h: float | None
    load_shed_rate_24h: float | None
    last_scored_at: datetime | None
    unscored_edits: int
    models_loaded: bool
    chain: list[BreakerInfo]
    live_metrics: list[dict[str, Any]]
    drift_reports: list[dict[str, Any]]
    retrain_runs: list[dict[str, Any]]


class PredictRequest(Closed):
    """Either a known edit, which reads its stored score, or raw text to score now."""

    rev_id: int | None = None
    added_text: str = Field(default="", max_length=MAX_PREDICT_TEXT)
    removed_text: str = Field(default="", max_length=MAX_PREDICT_TEXT)
    comment: str = Field(default="", max_length=MAX_PREDICT_COMMENT)
    is_anonymous: bool = False
    is_minor: bool = False
    old_length: int = Field(default=0, ge=0, le=100_000_000)
    new_length: int = Field(default=0, ge=0, le=100_000_000)
    user_edit_count: int | None = Field(default=None, ge=0)


class PredictReply(Closed):
    score: float
    threshold: float | None
    model_name: str
    model_version: str
    used_fallback: bool
    forced: bool
    latency_ms: float | None
    source: Literal["stored", "cache", "model"]


class ExplainReply(Closed):
    rev_id: int
    explanation: str
    available: bool
    cached: bool
    reason: str | None
    llm_model: str | None


class AdminRequest(Closed):
    reason: str = Field(min_length=3, max_length=500)


class PromoteRequest(AdminRequest):
    model_version: str = Field(min_length=1, max_length=64)


class RequestInfo(Closed):
    id: int
    kind: str
    status: str
    reason: str
    requested_by: str
    requested_at: datetime
    picked_up_at: datetime | None
    finished_at: datetime | None
    details: dict[str, Any]


class RequestList(Closed):
    items: list[RequestInfo]


class AuditEntry(Closed):
    id: int
    at: datetime
    actor: str
    action: str
    target: str | None
    request_id: str | None
    details: dict[str, Any]


class AuditPage(Closed):
    items: list[AuditEntry]
    next_cursor: str | None


class FieldError(BaseModel):
    field: str
    message: str


class ProblemDetails(BaseModel):
    """Every error the API sends, in one format (RFC 9457), as application/problem+json."""

    type: str
    title: str
    status: int
    code: str
    request_id: str | None
    detail: str | None = None
    # Only on a 422: which fields were wrong and why, never the values that were sent.
    errors: list[FieldError] | None = None
