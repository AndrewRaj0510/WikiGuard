"""The fuzzing check must tell a deliberate refusal from a crash, in both directions."""

from typing import Any

import pytest
from fuzz_checks import PROBLEM_TYPE, is_deliberate_refusal, never_crashes


class Reply:
    def __init__(self, status: int, body: Any, content_type: str = PROBLEM_TYPE) -> None:
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


SWITCHED_OFF = {"status": 503, "code": "feature_disabled", "title": "Explain is switched off"}


def test_a_refusal_the_api_wrote_is_not_a_crash() -> None:
    never_crashes(None, Reply(503, SWITCHED_OFF), None)


@pytest.mark.parametrize("status", [200, 204, 401, 404, 422, 429])
def test_ordinary_answers_pass(status: int) -> None:
    never_crashes(None, Reply(status, {}), None)


@pytest.mark.parametrize(
    "reply",
    [
        Reply(500, {"status": 500, "code": "internal_error"}),
        Reply(502, {"status": 502, "code": "bad_gateway"}),
        # A 503 from a proxy or a crashed worker: not the API's own format.
        Reply(503, "Service Unavailable", "text/html"),
        Reply(503, ValueError("not JSON")),
        Reply(503, {"status": 503}),
        Reply(503, SWITCHED_OFF, "application/json"),
    ],
)
def test_a_crash_fails(reply: Reply) -> None:
    with pytest.raises(AssertionError, match="The server failed"):
        never_crashes(None, reply, None)


def test_the_status_inside_the_body_must_agree() -> None:
    assert not is_deliberate_refusal(503, PROBLEM_TYPE, {"status": 500, "code": "x"})
