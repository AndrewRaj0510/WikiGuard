"""Parts of the API with no database, so they run in every CI job."""

import json
import logging
import threading

import pytest
from wikiguard_api.app import PredictCache
from wikiguard_api.cache import SingleFlight, TtlCache
from wikiguard_api.foundations import JsonFormatter, Problem, RateLimiter, problem_body
from wikiguard_api.queries import decode_cursor, encode_cursor, with_link


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# --- Rate limits ---


def test_the_limit_allows_up_to_the_count_then_refuses() -> None:
    limiter = RateLimiter(Clock())
    assert [limiter.hit("a", 2, 60) for _ in range(3)][:2] == [None, None]
    assert limiter.hit("a", 2, 60) is not None


def test_keys_are_separate() -> None:
    limiter = RateLimiter(Clock())
    limiter.hit("a", 1, 60)
    assert limiter.hit("b", 1, 60) is None


def test_the_window_resets() -> None:
    clock = Clock()
    limiter = RateLimiter(clock)
    limiter.hit("a", 1, 60)
    assert limiter.hit("a", 1, 60) is not None
    clock.now += 60
    assert limiter.hit("a", 1, 60) is None


def test_the_wait_is_within_the_window() -> None:
    clock = Clock()
    limiter = RateLimiter(clock)
    limiter.hit("a", 1, 60)
    clock.now += 50
    assert 1 <= (limiter.hit("a", 1, 60) or 0) <= 11


def test_require_raises_a_429_with_retry_after() -> None:
    limiter = RateLimiter(Clock())
    limiter.require("a", 1, 60)
    with pytest.raises(Problem) as caught:
        limiter.require("a", 1, 60)
    assert caught.value.status == 429
    assert caught.value.retry_after


def test_the_table_cant_grow_without_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wikiguard_api.foundations.RATE_TABLE_LIMIT", 100)
    limiter = RateLimiter(Clock())
    for index in range(500):
        limiter.hit(f"ip-{index}", 5, 60)
    assert len(limiter._windows) == 100


# --- The cache ---


def cache(clock: Clock | None = None, **options: float) -> TtlCache:
    return TtlCache("test", 10.0, clock=clock or Clock(), rand=lambda: 0.0, **options)


def test_a_value_is_served_until_it_expires() -> None:
    clock = Clock()
    store = cache(clock)
    calls: list[int] = []
    for _ in range(3):
        store.get("k", lambda: calls.append(1) or "value")
    assert len(calls) == 1
    clock.now += 11
    store.get("k", lambda: calls.append(1) or "value")
    assert len(calls) == 2


def test_jitter_only_lengthens_the_lifetime() -> None:
    clock = Clock()
    store = TtlCache("test", 10.0, clock=clock, rand=lambda: 1.0)
    store.get("k", lambda: "value")
    assert store._entries["k"].expires_at == pytest.approx(clock.now + 11)


def test_stale_is_served_when_reloading_fails() -> None:
    clock = Clock()
    store = cache(clock)
    store.get("k", lambda: "old")
    clock.now += 11

    def fail() -> str:
        raise RuntimeError("database down")

    assert store.get("k", fail) == ("old", True)


def test_nothing_kept_means_the_error_comes_through() -> None:
    def fail() -> str:
        raise RuntimeError("database down")

    with pytest.raises(RuntimeError):
        cache().get("k", fail)


def test_concurrent_misses_share_one_load() -> None:
    store = cache()
    calls: list[int] = []
    release = threading.Event()
    results: list[tuple[str, bool]] = []

    def load() -> str:
        calls.append(1)
        release.wait(5)
        return "value"

    threads = [
        threading.Thread(target=lambda: results.append(store.get("k", load))) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    release.wait(0.2)
    release.set()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert results == [("value", False)] * 8


def test_clear_removes_values() -> None:
    store = cache()
    store.get("k", lambda: "old")
    store.clear()
    assert store.get("k", lambda: "new") == ("new", False)


def test_a_load_that_started_before_a_clear_isnt_kept() -> None:
    """Otherwise an invalidation on alias change could be undone by a slow query finishing."""
    store = cache()

    def load() -> str:
        store.clear()
        return "out of date"

    store.get("k", load)
    assert "k" not in store._entries


def test_the_cache_is_bounded() -> None:
    store = TtlCache("test", 10.0, max_entries=3, clock=Clock(), rand=lambda: 0.0)
    for index in range(10):
        store.get(index, lambda: "value")
    assert len(store._entries) == 3


def test_hits_and_misses_are_reported() -> None:
    events: list[tuple[str, str]] = []

    class Metrics:
        def cache_event(self, name: str, event: str) -> None:
            events.append((name, event))

    store = TtlCache("queue", 10.0, metrics=Metrics(), clock=Clock(), rand=lambda: 0.0)
    store.get("k", lambda: "value")
    store.get("k", lambda: "value")
    assert events == [("queue", "miss"), ("queue", "hit")]


def test_single_flight_shares_a_failure_and_then_recovers() -> None:
    flight: SingleFlight[str] = SingleFlight()

    def fail() -> str:
        raise ValueError("no")

    with pytest.raises(ValueError):
        flight.do("k", fail)
    assert flight.do("k", lambda: "yes") == ("yes", True)


def test_predict_cache_drops_the_oldest() -> None:
    store = PredictCache(size=2)
    for key in ("a", "b", "c"):
        store.put(key, {"key": key})
    assert store.get("a") is None
    assert store.get("c") == {"key": "c"}


# --- Formats ---


def test_problem_body_is_rfc_9457_shaped() -> None:
    body = problem_body(404, "x", "Not here", "req", "Details", {"extra": 1})
    assert body == {
        "type": "/problems/x",
        "title": "Not here",
        "status": 404,
        "code": "x",
        "request_id": "req",
        "detail": "Details",
        "extra": 1,
    }


def test_json_logs_keep_only_named_fields() -> None:
    record = logging.LogRecord("api", logging.INFO, "f", 1, "request", None, None)
    record.request_id = "req"
    record.rev_id = 5
    record.body = "SECRET DIFF"
    line = json.loads(JsonFormatter().format(record))
    assert line["request_id"] == "req"
    assert line["rev_id"] == 5
    assert "body" not in line
    assert "SECRET" not in json.dumps(line)


def test_json_logs_keep_only_the_errors_type() -> None:
    try:
        raise ValueError("SECRET value")
    except ValueError:
        import sys

        record = logging.LogRecord("api", logging.ERROR, "f", 1, "boom", None, sys.exc_info())
    line = JsonFormatter().format(record)
    assert "ValueError" in line
    assert "SECRET" not in line


def test_a_cursor_round_trips() -> None:
    assert decode_cursor(encode_cursor([0.5, 7]), (float, int)) == [0.5, 7]


def test_links_are_built_from_the_integer() -> None:
    assert with_link({"rev_id": 42})["wikipedia_url"].endswith("diff=42")
    with pytest.raises(ValueError):
        with_link({"rev_id": "1&evil=1"})
