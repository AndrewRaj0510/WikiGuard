"""A Schemathesis check: the API may refuse on purpose, and must never crash.

Schemathesis's own check fails on any 5xx. This API answers 503 on purpose in a few cases: a
feature is switched off, `/predict` is at its concurrency cap, the database is down. Each of
those is a decision, written in the problem format with a code a client can read. A crash is
different: a 500, or any 5xx the API didn't write itself. Found when the first scan over HTTP
failed on "Explain is switched off" (F-138).

Used by the in-process fuzz test (services/api/tests/test_schema_fuzz.py), and by the scan of
staging through SCHEMATHESIS_HOOKS=fuzz_checks (.github/workflows/security-scan.yml).
"""

from typing import Any

import schemathesis

PROBLEM_TYPE = "application/problem+json"
DELIBERATE_STATUS = 503


def is_deliberate_refusal(status: int, content_type: str, body: Any) -> bool:
    """A 503 the API wrote itself: the problem format, with a code."""
    return (
        status == DELIBERATE_STATUS
        and content_type.startswith(PROBLEM_TYPE)
        and isinstance(body, dict)
        and isinstance(body.get("code"), str)
        and body.get("status") == DELIBERATE_STATUS
    )


@schemathesis.check
def never_crashes(ctx: Any, response: Any, case: Any) -> None:
    if response.status_code < 500:
        return
    content_type = response.headers.get("content-type", "")
    if isinstance(content_type, list):
        content_type = content_type[0] if content_type else ""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - a body that isn't JSON is simply not a problem document
        body = None
    if not is_deliberate_refusal(response.status_code, content_type, body):
        raise AssertionError(f"The server failed with {response.status_code}")
