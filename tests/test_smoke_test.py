"""The deploy smoke test, against a pretend API. The real app is checked in services/api/tests."""

from typing import Any

import httpx
import pytest
from smoke_test import SmokeFailure, check_origin_closed, run, same_commit, wait_until

GOOD_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "x-request-id": "request-1234",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
}
PROBLEM = {"content-type": "application/problem+json"}


def api(**changes: Any) -> httpx.Client:
    """A healthy deployed API, with any route swapped out by path."""
    routes: dict[str, httpx.Response] = {
        "/health": httpx.Response(200, json={"status": "ok", "commit": "0123abcdef"}),
        "/ready": httpx.Response(200, json={"ready": True}),
        "/v1/queue": httpx.Response(401, json={"code": "not_signed_in"}, headers=PROBLEM),
        "/metrics": httpx.Response(401, json={"code": "not_signed_in"}, headers=PROBLEM),
        "/v1/openapi.json": httpx.Response(200, json={"paths": {"/v1/queue": {}}}),
        "/v1/no-such-route": httpx.Response(404, json={"code": "not_found"}, headers=GOOD_HEADERS),
    }
    routes |= {path.replace("__", "/"): response for path, response in changes.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        return routes[request.url.path]

    return httpx.Client(base_url="https://api.test", transport=httpx.MockTransport(handler))


def no_sleep(_: float) -> None:
    return None


def test_a_healthy_deploy_passes() -> None:
    done = run(api(), None, "0123abc", 0, no_sleep)
    assert len(done) == 3


def test_an_old_build_still_serving_fails_after_the_wait() -> None:
    with pytest.raises(SmokeFailure, match="still serving commit 0123abcdef"):
        run(api(), None, "ffff999", 0, no_sleep)


def test_a_build_that_names_no_commit_is_not_taken_for_the_new_one() -> None:
    old = httpx.Response(200, json={"status": "ok", "commit": "unknown"})
    with pytest.raises(SmokeFailure, match="still serving commit unknown"):
        run(api(__health=old), None, "0123abc", 0, no_sleep)


@pytest.mark.parametrize(
    "served,wanted,same",
    [
        ("0123abcdef", "0123abc", True),
        ("0123abc", "0123abcdef", True),
        ("0123abcdef", "0123abd", False),
        # Too short to mean anything: an empty or one-letter value must never match.
        ("", "0123abc", False),
        ("0", "0123abc", False),
    ],
)
def test_commits_match_by_their_shared_prefix(served: str, wanted: str, same: bool) -> None:
    assert same_commit(served, wanted) is same


def test_the_wait_retries_until_the_new_build_arrives() -> None:
    answers = iter(["old build", "old build", None])
    naps: list[float] = []
    wait_until(lambda: next(answers), 600, naps.append, clock=lambda: 0.0)
    assert len(naps) == 2


def test_an_api_that_never_becomes_ready_fails() -> None:
    asleep = httpx.Response(503, json={"ready": False})
    with pytest.raises(SmokeFailure, match="/ready returned 503"):
        run(api(__ready=asleep), None, None, 0, no_sleep)


@pytest.mark.parametrize(
    "path,response,message",
    [
        ("__v1__queue", httpx.Response(200, json={"items": []}), "without a session"),
        ("__metrics", httpx.Response(200, text="# metrics"), "without a token"),
        (
            "__v1__openapi.json",
            httpx.Response(200, json={"paths": {"/v1/queue": {}, "/v1/admin/retrain": {}}}),
            "admin routes",
        ),
        (
            "__v1__no-such-route",
            httpx.Response(404, text='Traceback: File "app.py"', headers=GOOD_HEADERS),
            "leaks internals",
        ),
        (
            "__v1__no-such-route",
            httpx.Response(404, json={}, headers=GOOD_HEADERS | {"x-frame-options": "ALLOW"}),
            "x-frame-options",
        ),
    ],
)
def test_each_lost_control_fails_the_deploy(
    path: str, response: httpx.Response, message: str
) -> None:
    with pytest.raises(SmokeFailure, match=message):
        run(api(**{path: response}), None, None, 0, no_sleep)


def test_an_origin_that_answers_directly_fails() -> None:
    with pytest.raises(SmokeFailure, match="direct request"):
        check_origin_closed(api())


def test_an_origin_that_refuses_direct_requests_passes() -> None:
    refused = httpx.Response(403, json={"code": "direct_access"})
    done = run(api(), api(__ready=refused), None, 0, no_sleep)
    assert "origin" in done[-1]
