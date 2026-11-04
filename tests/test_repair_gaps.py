import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from repair_gaps import repair
from wikiguard_core.kafka import EDITS_DLQ, EDITS_RAW
from wikiguard_core.schemas import EditEvent
from wikiguard_core.wikimedia import WikimediaClient

FIXTURES = Path(__file__).parent / "fixtures" / "wikimedia"
KEY = "x-test-only-x-test-only-x-test-only-x-test-only"
SINCE = datetime(2026, 11, 4, 10, tzinfo=UTC)
UNTIL = datetime(2026, 11, 4, 12, tzinfo=UTC)


def recorded(case: str) -> dict[str, Any]:
    body: dict[str, Any] = json.loads((FIXTURES / f"{case}.json").read_text("utf-8"))["response"]
    return body


def row(rev_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "revid": rev_id,
        "old_revid": rev_id - 1,
        "title": "Moon",
        "user": "Real Name",
        "timestamp": "2026-11-04T11:00:00Z",
        "comment": "fix",
        "oldlen": 1,
        "newlen": 2,
        **extra,
    }


class Publisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, bytes]] = []
        self.flushes = 0

    def publish(self, topic: str, key: str, value: bytes) -> None:
        self.messages.append((topic, key, value))

    def flush(self, timeout: float = 30.0) -> None:
        self.flushes += 1

    def keys(self, topic: str) -> list[str]:
        return [key for t, key, _ in self.messages if t == topic]


class Api:
    def __init__(self, rows: list[dict[str, Any]], compare: str = "compare_multi_section") -> None:
        self.rows = rows
        self.compare = recorded(compare)
        self.diff_requests = 0
        self.list_params: dict[str, str] = {}
        self.fail_for: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("action") == "compare":
            self.diff_requests += 1
            if params["torev"] in self.fail_for:
                return httpx.Response(403)
            return httpx.Response(200, json=self.compare)
        self.list_params = dict(params)
        return httpx.Response(200, json={"query": {"recentchanges": self.rows}})


def client(api: Api) -> WikimediaClient:
    return WikimediaClient("t", transport=httpx.MockTransport(api), sleep=lambda _: None)


def test_refills_the_window_with_full_edits_and_no_edit_count() -> None:
    api = Api([row(11), row(12)])
    publisher = Publisher()

    results = repair(client(api), publisher, SINCE, UNTIL, KEY)

    assert results["published"] == 2
    assert api.list_params["rcstart"] == "2026-11-04T12:00:00Z"
    assert api.list_params["rcend"] == "2026-11-04T10:00:00Z"
    edit = EditEvent.model_validate_json(publisher.messages[0][2])
    assert edit.added_text
    assert edit.user_edit_count is None
    assert b"Real Name" not in publisher.messages[0][2]
    assert publisher.flushes == 1


def test_edits_already_stored_are_skipped_without_a_request() -> None:
    api = Api([row(11), row(12)])
    publisher = Publisher()

    results = repair(client(api), publisher, SINCE, UNTIL, KEY, known=lambda _: {11})

    assert publisher.keys(EDITS_RAW) == ["12"]
    assert results["already_present"] == 1
    assert api.diff_requests == 1


def test_running_twice_publishes_the_same_keys() -> None:
    publisher = Publisher()
    for _ in range(2):
        repair(client(Api([row(11)])), publisher, SINCE, UNTIL, KEY)

    # Consumers upsert on rev_id, so the repeat changes nothing downstream.
    assert publisher.keys(EDITS_RAW) == ["11", "11"]


def test_deleted_revisions_are_counted_and_not_published() -> None:
    publisher = Publisher()

    results = repair(client(Api([row(11)], "compare_nonexistent")), publisher, SINCE, UNTIL, KEY)

    assert results["missing"] == 1
    assert publisher.messages == []


def test_hidden_rows_are_published_without_a_diff_request() -> None:
    api = Api([row(11, sha1hidden=True)])
    publisher = Publisher()

    repair(client(api), publisher, SINCE, UNTIL, KEY)

    assert api.diff_requests == 0
    assert EditEvent.model_validate_json(publisher.messages[0][2]).content_hidden


def test_one_failed_diff_does_not_stop_the_repair() -> None:
    api = Api([row(11), row(12)])
    api.fail_for = {"11"}
    publisher = Publisher()

    results = repair(client(api), publisher, SINCE, UNTIL, KEY)

    assert results["api_error"] == 1
    assert publisher.keys(EDITS_DLQ) == ["11"]
    assert publisher.keys(EDITS_RAW) == ["12"]
