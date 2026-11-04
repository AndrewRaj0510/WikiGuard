"""Wikimedia is faked with a local transport. The responses come from recorded fixtures."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from wikiguard_core.labels import LABEL_WAIT, decide_label
from wikiguard_core.wikimedia import (
    MAX_ATTEMPTS,
    DiffStatus,
    WikimediaClient,
    WikimediaError,
    backoff_seconds,
    chunks,
)

UA = "WikiGuardTest/0 (https://example.org)"
Handler = Callable[[httpx.Request], httpx.Response]
LoadCase = Callable[[str], dict[str, Any]]


def client_for(handler: Handler, sleeps: list[float] | None = None) -> WikimediaClient:
    record = sleeps if sleeps is not None else []
    return WikimediaClient(UA, transport=httpx.MockTransport(handler), sleep=record.append)


def ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body)


def test_every_request_identifies_itself_and_sets_maxlag() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"query": {}})

    client_for(handler).query(action="query")

    assert seen[0].headers["User-Agent"] == UA
    assert seen[0].url.params["maxlag"] == "5"


def test_retry_after_is_honoured_on_a_rate_limit() -> None:
    sleeps: list[float] = []
    answers = [httpx.Response(429, headers={"Retry-After": "7"}), ok({"query": {}})]

    assert client_for(lambda _: answers.pop(0), sleeps).query(action="query") == {"query": {}}
    assert sleeps == [7.0]


def test_maxlag_waits_then_retries() -> None:
    sleeps: list[float] = []
    lagged = httpx.Response(200, headers={"Retry-After": "5"}, json={"error": {"code": "maxlag"}})
    answers = [lagged, ok({"query": {"done": True}})]

    result = client_for(lambda _: answers.pop(0), sleeps).query(action="query")

    assert result["query"]["done"] is True
    assert sleeps == [5.0]


def test_network_failure_retries_and_then_gives_up() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("no route")

    with pytest.raises(WikimediaError, match="ConnectError"):
        client_for(handler).query(action="query")
    assert len(calls) == MAX_ATTEMPTS


def test_client_error_does_not_retry() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(403)

    with pytest.raises(WikimediaError):
        client_for(handler).query(action="query")
    assert len(calls) == 1


def test_backoff_grows_stays_capped_and_uses_jitter() -> None:
    assert backoff_seconds(2, rng=lambda: 1.0) == 4.0
    assert backoff_seconds(30, rng=lambda: 1.0) == 60.0
    assert backoff_seconds(2, rng=lambda: 0.5) == 2.0


def test_compare_returns_the_diff(load_case: LoadCase) -> None:
    response = load_case("compare_multi_section")

    result = client_for(lambda _: ok(response)).compare(1, 2)

    assert result.status is DiffStatus.OK
    assert "diff-lineno" in result.html


@pytest.mark.parametrize(
    ("case", "status"),
    [("compare_hidden", DiffStatus.HIDDEN), ("compare_nonexistent", DiffStatus.MISSING)],
)
def test_compare_errors_inside_http_200(load_case: LoadCase, case: str, status: DiffStatus) -> None:
    response = load_case(case)

    result = client_for(lambda _: ok(response)).compare(1, 2)

    assert result.status is status
    assert result.html == ""


def test_revision_states_reads_tags_from_a_real_batch(load_case: LoadCase) -> None:
    response = load_case("revisions_tag_batch")
    ids = [r["revid"] for p in response["query"]["pages"] for r in p["revisions"]]

    states = client_for(lambda _: ok(response)).revision_states(ids)

    assert set(states) == set(ids)
    assert any("mw-reverted" in state.tags for state in states.values())
    assert not any(state.missing for state in states.values())


def test_revision_states_reads_hidden_flags(load_case: LoadCase) -> None:
    response = load_case("revisions_hidden")
    ids = [r["revid"] for p in response["query"]["pages"] for r in p["revisions"]]

    states = client_for(lambda _: ok(response)).revision_states(ids)

    assert any(state.content_hidden for state in states.values())
    assert any(state.comment_hidden for state in states.values())
    assert any(state.user_hidden for state in states.values())
    assert any(state.has_hidden_content for state in states.values())


def test_revision_states_marks_unknown_ids_missing(load_case: LoadCase) -> None:
    response = load_case("revisions_nonexistent")
    rev_id = int(next(iter(response["query"]["badrevids"])))

    states = client_for(lambda _: ok(response)).revision_states([rev_id])

    assert states[rev_id].missing
    assert states[rev_id].has_hidden_content


def test_revision_states_answers_for_ids_the_api_left_out() -> None:
    states = client_for(lambda _: ok({"query": {"pages": []}})).revision_states([9])

    assert states[9].missing


def test_revision_states_sends_at_most_50_ids_at_once_and_follows_continue() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.url.params["revids"].split("|")))
        return ok({"query": {"pages": []}})

    client_for(handler).revision_states(range(120))

    assert sizes == [50, 50, 20]


def test_recent_changes_follows_continue_and_stops_at_the_limit() -> None:
    pages: list[dict[str, Any]] = [
        {"continue": {"rccontinue": "x"}, "query": {"recentchanges": [{"revid": 1}, {"revid": 2}]}},
        {"query": {"recentchanges": [{"revid": 3}]}},
    ]

    rows = client_for(lambda _: ok(pages.pop(0))).recent_changes(
        datetime(2026, 11, 1, tzinfo=UTC), 3
    )

    assert [row["revid"] for row in rows] == [1, 2, 3]


def test_edits_before_counts_and_respects_the_cap(load_case: LoadCase) -> None:
    response = load_case("usercontribs_registered")
    moment = datetime(2026, 11, 2, tzinfo=UTC)

    client = client_for(lambda _: ok({"query": response["query"]}))

    assert client.edits_before("A", moment, cap=500) == len(response["query"]["usercontribs"])
    assert client.edits_before("A", moment, cap=2) <= 2


def test_user_edit_counts_reads_real_shape(load_case: LoadCase) -> None:
    response = load_case("users_temporary")
    name = response["query"]["users"][0]["name"]

    assert client_for(lambda _: ok(response)).user_edit_counts([name]) == {name: 1}


def test_chunks_splits_evenly_and_keeps_the_rest() -> None:
    assert list(chunks(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_label_rule() -> None:
    edited = datetime(2026, 11, 1, tzinfo=UTC)

    assert decide_label(edited, {"mw-reverted"}, edited) == 1
    assert decide_label(edited, {"mobile edit"}, edited) is None
    assert decide_label(edited, set(), edited + LABEL_WAIT) == 0
