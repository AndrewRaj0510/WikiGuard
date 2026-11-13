"""Error tracking with Sentry's free tier: off unless SENTRY_DSN is set.

What reaches Sentry is kept to the error itself, because diffs, comments, and titles are written by
vandals and can hold personal data, and an exception message can quote a row:

- No request bodies, headers, cookies, or query strings, and no local variables.
- No logged messages as breadcrumbs: logs are already scrubbed, but nothing needs them there.
- Exception messages pass through the same IP and temporary-account redaction as the logs, and are
  cut short. Postgres quotes a failing row in its DETAIL line, so that line is dropped.
- No performance tracing: the free plan's quota is for errors.

The SDK is imported only when a DSN is set, so a service without one never loads it.
"""

import re
from typing import Any

from pydantic import SecretStr

from wikiguard_core.identity import redact_identifiers

MAX_MESSAGE = 300
ROW_DETAIL = re.compile(r"\n?DETAIL:.*", re.DOTALL)


def scrub_message(text: str) -> str:
    return redact_identifiers(ROW_DETAIL.sub("", text))[:MAX_MESSAGE]


def scrub_event(event: dict[str, Any], _hint: Any = None) -> dict[str, Any]:
    """Sentry's before_send: keep the stack and the error's type, drop what could carry data."""
    request = event.get("request")
    if isinstance(request, dict):
        event["request"] = {
            key: request[key] for key in ("method", "url") if isinstance(request.get(key), str)
        }
        if "url" in event["request"]:
            event["request"]["url"] = event["request"]["url"].split("?", 1)[0]
    for key in ("extra", "user", "breadcrumbs"):
        event.pop(key, None)
    for exception in (event.get("exception") or {}).get("values") or []:
        if isinstance(exception.get("value"), str):
            exception["value"] = scrub_message(exception["value"])
        for frame in (exception.get("stacktrace") or {}).get("frames") or []:
            frame.pop("vars", None)
    if isinstance(event.get("message"), str):
        event["message"] = scrub_message(event["message"])
    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        logentry.pop("params", None)
        if isinstance(logentry.get("message"), str):
            logentry["message"] = scrub_message(logentry["message"])
    return event


def init_error_tracking(
    dsn: SecretStr | None, service: str, environment: str, release: str | None = None
) -> bool:
    """Start Sentry for this process. Returns whether it's on."""
    if dsn is None or not dsn.get_secret_value():
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn.get_secret_value(),
        environment=environment,
        release=release or None,
        server_name=service,
        # What integrations may collect. scrub_event below is the last word either way.
        data_collection={
            "user_info": False,
            "http_bodies": [],
            "database_query_data": False,
            "queues": False,
            "stack_frame_variables": False,
        },
        max_request_body_size="never",
        max_breadcrumbs=0,
        traces_sample_rate=0.0,
        # Sentry types the event as a TypedDict. It is the same dict at run time.
        before_send=scrub_event,  # type: ignore[arg-type]
    )
    sentry_sdk.set_tag("service", service)
    return True
