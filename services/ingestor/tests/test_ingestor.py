"""The ingestor on a recorded stream, with Wikimedia and Redpanda faked."""

import json
import threading
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from ingestor import (
    EditCountCache,
    Ingestor,
    Metrics,
    Position,
    parse_sse,
    run_forever,
)
from wikiguard_core.kafka import EDITS_DLQ, EDITS_RAW, PublishError
from wikiguard_core.ops import serve_ops
from wikiguard_core.schemas import EditEvent
from wikiguard_core.wikimedia import WikimediaClient

FIXTURES = Path(__file__).parents[3] / "tests" / "fixtures" / "wikimedia"
HASH_KEY = "x-test-only-x-test-only-x-test-only-x-test-only"


def recorded(case: str) -> dict[str, Any]:
    response: dict[str, Any] = json.loads((FIXTURES / f"{case}.json").read_text("utf-8"))[
        "response"
    ]
    return response


def stream_rows() -> list[dict[str, Any]]:
    lines = (FIXTURES / "stream_events.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def in_scope_events() -> list[dict[str, Any]]:
    return [row["event"] for row in stream_rows() if row["category"].startswith("in_scope")]


def as_sse(rows: Iterable[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        lines += [f"id: {row['sse_id']}", f"data: {json.dumps(row['event'])}", ""]
    return lines


class FakePublisher:
    def __init__(self) -> None:
        self.pending: list[tuple[str, str, bytes]] = []
        self.confirmed: list[tuple[str, str, bytes]] = []
        self.fail_flush = False

    def publish(self, topic: str, key: str, value: bytes) -> None:
        self.pending.append((topic, key, value))

    def flush(self, timeout: float = 30.0) -> None:
        if self.fail_flush:
            raise PublishError("down")
        self.confirmed += self.pending
        self.pending = []

    def on(self, topic: str) -> list[dict[str, Any]]:
        return [json.loads(value) for t, _, value in self.confirmed if t == topic]


class FakeApi:
    def __init__(self, compare: str = "compare_multi_section") -> None:
        self.compare = recorded(compare)
        self.calls: dict[str, int] = {}
        self.fail_compare = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        kind = params.get("action") if params.get("action") == "compare" else params.get("list")
        self.calls[str(kind)] = self.calls.get(str(kind), 0) + 1
        if kind == "compare":
            if self.fail_compare:
                return httpx.Response(403)
            return httpx.Response(200, json=self.compare)
        name = params["ususers"]
        return httpx.Response(
            200, json={"query": {"users": [{"name": name, "editcount": 41, "userid": 1}]}}
        )


def make(
    tmp_path: Path, api: FakeApi | None = None, publisher: FakePublisher | None = None
) -> tuple[Ingestor, FakeApi, FakePublisher]:
    api = api or FakeApi()
    publisher = publisher or FakePublisher()
    client = WikimediaClient("t", transport=httpx.MockTransport(api), sleep=lambda _: None)
    position = Position(tmp_path / "position.json")
    return Ingestor(client, publisher, position, HASH_KEY, Metrics()), api, publisher


def test_sse_parser_handles_comments_ids_and_bad_data() -> None:
    lines = [":ok", "", "event: message", "id: a", 'data: {"x": 1}', "", "id: b", "data: {bad", ""]

    assert list(parse_sse(lines)) == [("a", {"x": 1}), ("b", None)]


def test_recorded_stream_publishes_only_in_scope_edits(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)
    rows = stream_rows()

    ingestor.consume(parse_sse(as_sse(rows)))

    published = publisher.on(EDITS_RAW)
    assert len(published) == len(in_scope_events())
    assert publisher.on(EDITS_DLQ) == []
    assert {p["rev_id"] for p in published} == {e["revision"]["new"] for e in in_scope_events()}
    for message in published:
        edit = EditEvent.model_validate(message)
        assert edit.added_text.split("\n") == ["2nd", "2nd", "1st"]
        assert edit.user_edit_count == 41


def test_temporary_accounts_are_flagged_and_no_username_is_published(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)
    rows = [r for r in stream_rows() if r["category"] == "in_scope_temporary"]

    ingestor.consume(parse_sse(as_sse(rows)))

    assert all(message["is_anonymous"] for message in publisher.on(EDITS_RAW))
    text = json.dumps(publisher.on(EDITS_RAW))
    for row in rows:
        assert row["event"]["user"] not in text


def test_repeat_editor_uses_the_cache_and_counts_up(tmp_path: Path) -> None:
    ingestor, api, publisher = make(tmp_path)
    event = in_scope_events()[0]

    ingestor.handle("1", event)
    ingestor.handle("2", event)

    assert api.calls["users"] == 1
    assert [m["user_edit_count"] for m in publisher.on(EDITS_RAW)] == [41, 42]


def test_a_failed_edit_count_keeps_the_edit_with_the_count_missing(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("action") == "compare":
            return httpx.Response(200, json=recorded("compare_multi_section"))
        return httpx.Response(403)

    publisher = FakePublisher()
    client = WikimediaClient("t", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    ingestor = Ingestor(client, publisher, Position(tmp_path / "p.json"), HASH_KEY, Metrics())

    ingestor.handle("1", in_scope_events()[0])

    assert publisher.on(EDITS_RAW)[0]["user_edit_count"] is None
    assert publisher.on(EDITS_DLQ) == []


def test_cache_evicts_the_oldest_editor() -> None:
    cache = EditCountCache(size=2)
    cache.put("a", 1)
    cache.put("b", 1)
    cache.get("a")
    cache.put("c", 1)

    assert cache.get("b") is None
    assert cache.get("a") == 3


def test_hidden_diff_is_published_flagged_with_no_text(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path, FakeApi("compare_hidden"))

    ingestor.handle("1", in_scope_events()[0])

    message = publisher.on(EDITS_RAW)[0]
    assert message["content_hidden"] is True
    assert message["added_text"] == ""


def test_missing_revision_goes_to_the_dead_letter_topic(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path, FakeApi("compare_nonexistent"))

    ingestor.handle("1", in_scope_events()[0])

    assert publisher.on(EDITS_RAW) == []
    assert publisher.on(EDITS_DLQ)[0]["reason"] == "missing"


def test_api_failure_goes_to_the_dead_letter_topic_and_the_stream_continues(
    tmp_path: Path,
) -> None:
    api = FakeApi()
    api.fail_compare = True
    ingestor, _, publisher = make(tmp_path, api)
    event = in_scope_events()[0]

    ingestor.handle("1", event)

    letter = publisher.on(EDITS_DLQ)[0]
    assert letter["reason"] == "api_error"
    assert letter["rev_id"] == event["revision"]["new"]

    api.fail_compare = False
    ingestor.handle("2", event)
    assert len(publisher.on(EDITS_RAW)) == 1


@pytest.mark.parametrize(
    "event",
    [
        {"wiki": "enwiki", "type": "edit", "namespace": 0, "bot": False},
        {
            "wiki": "enwiki",
            "type": "edit",
            "namespace": 0,
            "bot": False,
            "user": 5,
            "title": None,
            "timestamp": "soon",
            "revision": "x",
        },
    ],
)
def test_poison_messages_go_to_the_dead_letter_topic(tmp_path: Path, event: dict[str, Any]) -> None:
    ingestor, _, publisher = make(tmp_path)

    ingestor.handle("1", event)

    assert publisher.on(EDITS_RAW) == []
    assert publisher.on(EDITS_DLQ)[0]["reason"] == "malformed"


def test_unreadable_stream_data_goes_to_the_dead_letter_topic(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)

    ingestor.handle("1", None)

    assert publisher.on(EDITS_DLQ)[0]["reason"] == "malformed"


def test_dead_letters_hold_no_username_or_comment(tmp_path: Path) -> None:
    api = FakeApi()
    api.fail_compare = True
    ingestor, _, publisher = make(tmp_path, api)
    event = {**in_scope_events()[0], "user": "Real Name", "comment": "private words"}

    ingestor.handle("1", event)

    text = json.dumps(publisher.on(EDITS_DLQ))
    assert "Real Name" not in text
    assert "private words" not in text


def test_position_moves_only_after_redpanda_confirms(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)
    event = in_scope_events()[0]
    ingestor.handle("first", event)
    assert ingestor.position.event_id == "first"

    publisher.fail_flush = True
    with pytest.raises(PublishError):
        ingestor.handle("second", event)

    assert ingestor.position.event_id == "first"


def test_restart_resumes_from_the_saved_position(tmp_path: Path) -> None:
    ingestor, _, _ = make(tmp_path)
    ingestor.position.update("saved-id", force=True)

    assert Position(tmp_path / "position.json").event_id == "saved-id"


def test_a_broken_position_file_starts_from_now(tmp_path: Path) -> None:
    path = tmp_path / "position.json"
    path.write_text("{broken", encoding="utf-8")

    assert Position(path).event_id is None


def test_large_jump_between_events_is_reported(tmp_path: Path) -> None:
    ingestor, _, _ = make(tmp_path)
    event = in_scope_events()[0]

    ingestor.handle("1", {**event, "timestamp": 1000})
    ingestor.handle("2", {**event, "timestamp": 2000})

    gap = ingestor.metrics.registry.get_sample_value("ingestor_last_gap_seconds")
    assert gap == 1000


def test_reconnects_with_the_last_position_after_failures(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)
    rows = [r for r in stream_rows() if r["category"] == "in_scope_registered"][:2]
    stop = threading.Event()
    opened_with: list[str | None] = []
    waits: list[float] = []

    def open_stream(last_id: str | None) -> Iterator[str]:
        opened_with.append(last_id)
        if len(opened_with) == 1:
            yield from as_sse(rows[:1])
            raise httpx.ReadTimeout("dropped")
        if len(opened_with) == 2:
            raise httpx.ConnectError("refused")
        yield from as_sse(rows[1:])
        stop.set()

    def wait(seconds: float) -> bool:
        waits.append(seconds)
        return False

    run_forever(ingestor, open_stream, stop, wait)

    assert opened_with == [None, rows[0]["sse_id"], rows[0]["sse_id"]]
    assert len(waits) == 2
    assert [m["rev_id"] for m in publisher.on(EDITS_RAW)] == [
        rows[0]["event"]["revision"]["new"],
        rows[1]["event"]["revision"]["new"],
    ]
    assert ingestor.metrics.registry.get_sample_value("ingestor_reconnects_total") == 2


def test_redpanda_outage_retries_the_same_events(tmp_path: Path) -> None:
    ingestor, _, publisher = make(tmp_path)
    rows = [r for r in stream_rows() if r["category"] == "in_scope_registered"][:1]
    stop = threading.Event()
    opens = []

    def open_stream(last_id: str | None) -> Iterator[str]:
        opens.append(last_id)
        publisher.fail_flush = len(opens) == 1
        yield from as_sse(rows)
        if len(opens) == 2:
            stop.set()

    run_forever(ingestor, open_stream, stop, lambda _: False)

    assert opens == [None, None]
    assert rows[0]["event"]["revision"]["new"] in {m["rev_id"] for m in publisher.on(EDITS_RAW)}


def test_stop_saves_the_position(tmp_path: Path) -> None:
    ingestor, _, _ = make(tmp_path)
    stop = threading.Event()
    rows = [r for r in stream_rows() if r["category"] == "in_scope_registered"][:1]

    def open_stream(_: str | None) -> Iterator[str]:
        yield from as_sse(rows)
        stop.set()

    run_forever(ingestor, open_stream, stop, lambda _: False)

    saved = json.loads((tmp_path / "position.json").read_text("utf-8"))
    assert saved["event_id"] == rows[0]["sse_id"]


def test_health_and_metrics_endpoints(tmp_path: Path) -> None:
    ingestor, _, _ = make(tmp_path)
    ingestor.handle("1", in_scope_events()[0])
    server = serve_ops(0, ingestor.metrics.registry, ingestor.is_healthy, host="127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/health") as response:  # noqa: S310
            assert response.status == 200
        with urllib.request.urlopen(f"{base}/metrics") as response:  # noqa: S310
            body = response.read().decode()
    finally:
        server.shutdown()

    assert 'ingestor_events_total{kind="in_scope"} 1.0' in body
    assert "ingestor_published_total 1.0" in body


def test_health_fails_when_events_stop(tmp_path: Path) -> None:
    ingestor, _, _ = make(tmp_path)

    assert ingestor.is_healthy()
    ingestor.last_event_at -= 10_000
    assert not ingestor.is_healthy()
