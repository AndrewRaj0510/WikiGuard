"""The scorer's rules. Database tests connect as the real wikiguard_scorer role."""

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest
from confluent_kafka import KafkaException
from scorer import (
    REASON_DATABASE,
    REASON_SCORING,
    REASON_UNREADABLE,
    ScoringWorker,
)
from wikiguard_core.artifacts import write_checksums
from wikiguard_core.chain import ModelCache, ModelRef, ServingChain
from wikiguard_core.features import FEATURE_VERSION
from wikiguard_core.kafka import EDITS_DLQ, EDITS_SCORED, PublishError
from wikiguard_core.schemas import ScoreRecord

NOW = datetime.now(UTC)


def edit_json(rev_id: int = 1, **overrides: Any) -> bytes:
    body = {
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "title": "Moon",
        "timestamp": NOW.isoformat(),
        "is_minor": False,
        "is_anonymous": True,
        "user_hash": "hash",
        "comment": "a comment",
        "old_length": 100,
        "new_length": 120,
        "added_text": "added words",
        "removed_text": "removed",
        "user_edit_count": 3,
    }
    return json.dumps(body | overrides).encode()


class FakeMessage:
    def __init__(self, value: bytes | None, key: bytes | None = None, error: Any = None) -> None:
        self._value, self._key, self._error = value, key, error

    def error(self) -> Any:
        return self._error

    def key(self) -> bytes | None:
        return self._key

    def value(self) -> bytes | None:
        return self._value


class FakePosition:
    def __init__(self, offset: int) -> None:
        self.offset = offset


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage] | None = None) -> None:
        self.messages = list(messages or [])
        self.commits = 0
        self.lag = 0
        self.fail_commit = False
        self.committed_after: list[int] = []

    def poll(self, timeout: float) -> FakeMessage | None:
        return self.messages.pop(0) if self.messages else None

    def commit(self, asynchronous: bool = False) -> None:
        if self.fail_commit:
            raise KafkaException("rebalanced")
        self.commits += 1

    def assignment(self) -> list[Any]:
        return [object()]

    def position(self, partitions: list[Any]) -> list[Any]:
        return [FakePosition(0)]

    def get_watermark_offsets(self, partition: Any, timeout: float) -> tuple[int, int]:
        return (0, self.lag)


class FakePublisher:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, bytes]] = []
        self.fail_flushes = 0
        self.flushed = 0

    def publish(self, topic: str, key: str, value: bytes) -> None:
        self.sent.append((topic, key, value))

    def flush(self, timeout: float = 30.0) -> None:
        if self.fail_flushes:
            self.fail_flushes -= 1
            raise PublishError("redpanda down")
        self.flushed += 1

    def on(self, topic: str) -> list[dict[str, Any]]:
        return [json.loads(value) for name, _, value in self.sent if name == topic]


class FakeModel:
    threshold: float | None = 0.7

    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0
        self.fail = False
        self.batches: list[int] = []

    def probabilities(self, rows: Any) -> Any:
        self.calls += 1
        self.batches.append(len(rows))
        if self.fail:
            raise RuntimeError("broken")
        if any(row.get("comment") == "poison" for row in rows):
            raise KeyError("hour_of_day")
        return np.full(len(rows), self.score)


class FakeSource:
    def __init__(self, tmp_path: Path, names: list[str]) -> None:
        self.tmp_path = tmp_path
        self.refs = []
        for index, name in enumerate(names):
            digest = self._artifact(name)
            alias = "champion" if index == 0 else f"fallback-{index}"
            self.refs.append(ModelRef(alias, name, str(index + 1), digest))

    def _artifact(self, name: str) -> str:
        folder = self.tmp_path / "store" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"feature_version": FEATURE_VERSION}))
        (folder / "model.onnx").write_bytes(b"x")
        return write_checksums(folder)

    def resolve(self) -> list[ModelRef]:
        return self.refs

    def fetch(self, ref: ModelRef, target: Path) -> None:
        for path in (self.tmp_path / "store" / ref.model_name).iterdir():
            (target / path.name).write_bytes(path.read_bytes())


class Harness:
    def __init__(self, tmp_path: Path, url: str | None = None) -> None:
        self.models = {"roberta": FakeModel(0.8), "bilstm": FakeModel(0.5), "mlp": FakeModel(0.2)}
        self.chain = ServingChain(
            FakeSource(tmp_path, list(self.models)),
            ModelCache(tmp_path / "cache"),
            5000,
            loader=lambda folder, _: self.models[folder.name.split("-")[0]],
        )
        self.chain.load()
        for model in self.models.values():
            model.calls = 0
            model.batches = []
        self.consumer = FakeConsumer()
        self.publisher = FakePublisher()
        self.url = url
        self.sleeps: list[float] = []
        self.now = 1000.0
        self.worker = ScoringWorker(
            self.consumer,
            self.publisher,
            self.chain,
            self.connect,
            clock=lambda: self.now,
            sleep=self.sleep_for,
        )

    def connect(self) -> psycopg.Connection[Any]:
        assert self.url is not None
        return psycopg.connect(self.url, prepare_threshold=None, connect_timeout=5)

    def sleep_for(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.url = self.restore_url or self.url

    restore_url: str | None = None

    def feed(self, *values: bytes) -> None:
        self.consumer.messages = [
            FakeMessage(value, str(index).encode()) for index, value in enumerate(values)
        ]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def live(tmp_path: Path, db: psycopg.Connection[Any], scorer_url: str) -> Harness:
    db.rollback()
    return Harness(tmp_path, scorer_url)


def rows(db: psycopg.Connection[Any], query: str, *params: Any) -> list[tuple[Any, ...]]:
    result = db.execute(query, params).fetchall()
    db.commit()
    return result


# --- Reading and batching ---


def test_collect_stops_at_the_batch_size(harness: Harness) -> None:
    harness.feed(*[edit_json(i) for i in range(40)])
    harness.worker.batch_size = 32
    assert len(harness.worker.collect()) == 32
    assert len(harness.consumer.messages) == 8


def test_collect_returns_what_it_has_when_the_topic_runs_dry(harness: Harness) -> None:
    harness.feed(edit_json(1), edit_json(2))
    assert len(harness.worker.collect()) == 2


def test_collect_stops_when_the_window_ends(harness: Harness) -> None:
    harness.feed(*[edit_json(i) for i in range(10)])
    harness.worker.window_seconds = 0
    assert len(harness.worker.collect()) == 1


def test_an_unreadable_message_goes_to_the_dlq_and_the_rest_continue(harness: Harness) -> None:
    harness.feed(b"not json", edit_json(5))
    items = harness.worker._parse(harness.worker.collect())
    assert [item.edit.rev_id for item in items] == [5]
    dead = harness.publisher.on(EDITS_DLQ)
    assert [entry["reason"] for entry in dead] == [REASON_UNREADABLE]


def test_a_dlq_entry_never_holds_edit_text(harness: Harness) -> None:
    harness.consumer.messages = [
        FakeMessage(edit_json(5, is_minor="SECRET vandal text", added_text=123), b"5")
    ]
    harness.worker._parse(harness.worker.collect())
    body = harness.publisher.sent[0][2].decode()
    assert "SECRET" not in body
    assert json.loads(body)["rev_id"] == 5


def test_consumer_errors_are_skipped(harness: Harness) -> None:
    harness.consumer.messages = [FakeMessage(None, error="broker unavailable")]
    assert harness.worker._parse(harness.worker.collect()) == []


# --- Scoring ---


def parsed(harness: Harness, *values: bytes) -> list[Any]:
    harness.feed(*values)
    return harness.worker._parse(harness.worker.collect())


def test_the_champion_scores_a_batch(harness: Harness) -> None:
    items = harness.worker.score(parsed(harness, edit_json(1), edit_json(2)))
    assert [item.score["model_name"] for item in items] == ["roberta", "roberta"]
    assert items[0].score["used_fallback"] is False
    assert items[0].score["threshold"] == 0.7
    assert harness.models["bilstm"].calls == 0


def test_a_failing_champion_falls_back_and_flags_it(harness: Harness) -> None:
    harness.models["roberta"].fail = True
    items = harness.worker.score(parsed(harness, edit_json(1)))
    assert items[0].score["model_name"] == "bilstm"
    assert items[0].score["used_fallback"] is True


def test_a_poison_edit_goes_to_the_dlq_and_the_batch_still_scores(harness: Harness) -> None:
    items = harness.worker.score(
        parsed(harness, edit_json(1), edit_json(2, comment="poison"), edit_json(3))
    )
    assert [item.edit.rev_id for item in items] == [1, 3]
    assert all(item.score is not None for item in items)
    dead = harness.publisher.on(EDITS_DLQ)
    assert [(entry["reason"], entry["rev_id"]) for entry in dead] == [(REASON_SCORING, 2)]


def test_a_poison_edit_opens_no_breakers(harness: Harness) -> None:
    for _ in range(5):
        harness.worker.score(parsed(harness, edit_json(2, comment="poison")))
    assert all(
        status.breaker.value == "closed" for status in harness.chain.current.predictor.status()
    )


def test_when_every_model_is_broken_edits_are_kept_unscored(harness: Harness) -> None:
    for model in harness.models.values():
        model.fail = True
    items = harness.worker.score(parsed(harness, edit_json(1), edit_json(2)))
    assert [item.edit.rev_id for item in items] == [1, 2]
    assert all(item.score is None for item in items)
    assert harness.publisher.on(EDITS_DLQ) == []


# --- Load shedding ---


def test_shedding_turns_on_past_the_limit_and_uses_a_fast_model(harness: Harness) -> None:
    harness.consumer.lag = 501
    harness.now += 10
    harness.worker._update_shedding()
    assert harness.worker.shedding is True
    items = harness.worker.score(parsed(harness, edit_json(1)))
    assert items[0].score["model_name"] == "bilstm"
    assert items[0].score["load_shed"] is True
    assert harness.models["roberta"].calls == 0


def test_shedding_needs_the_backlog_to_halve_before_turning_off(harness: Harness) -> None:
    harness.consumer.lag = 600
    harness.now += 10
    harness.worker._update_shedding()
    harness.consumer.lag = 300
    harness.now += 10
    harness.worker._update_shedding()
    assert harness.worker.shedding is True
    harness.consumer.lag = 250
    harness.now += 10
    harness.worker._update_shedding()
    assert harness.worker.shedding is False


def test_lag_isnt_checked_on_every_batch(harness: Harness) -> None:
    harness.now += 10
    harness.worker._update_shedding()
    harness.consumer.lag = 10_000
    harness.now += 1
    harness.worker._update_shedding()
    assert harness.worker.shedding is False


# --- Writing, against real Postgres as the real role ---


@pytest.mark.integration
def test_write_then_publish_then_commit(live: Harness, db: psycopg.Connection[Any]) -> None:
    live.feed(edit_json(1), edit_json(2))
    live.worker.run_once()
    assert rows(db, "SELECT rev_id, latest_score, latest_model_version FROM edits ORDER BY 1") == [
        (1, pytest.approx(0.8), "1"),
        (2, pytest.approx(0.8), "1"),
    ]
    stored = rows(
        db,
        "SELECT model_name, alias, threshold, latency_ms, used_fallback, load_shed, forced "
        "FROM scores WHERE rev_id = 1",
    )
    assert stored[0][:3] == ("roberta", "champion", pytest.approx(0.7))
    assert stored[0][3] >= 0
    assert stored[0][4:] == (False, False, False)
    assert [entry["rev_id"] for entry in live.publisher.on(EDITS_SCORED)] == [1, 2]
    ScoreRecord(**live.publisher.on(EDITS_SCORED)[0])
    assert live.consumer.commits == 1


@pytest.mark.integration
@pytest.mark.parametrize("tiny", [1e-50, 1e-300, 5e-324])
def test_a_vanishingly_small_score_is_stored_not_refused(
    live: Harness, db: psycopg.Connection[Any], tiny: float
) -> None:
    live.models["roberta"].score = tiny
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT score FROM scores") == [(0.0,)]
    assert live.publisher.on(EDITS_DLQ) == []


@pytest.mark.integration
def test_a_replay_changes_nothing(live: Harness, db: psycopg.Connection[Any]) -> None:
    live.feed(edit_json(1))
    live.worker.run_once()
    first = rows(db, "SELECT score, scored_at FROM scores")
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT score, scored_at FROM scores") == first
    assert rows(db, "SELECT count(*) FROM edits") == [(1,)]


@pytest.mark.integration
def test_a_replay_cant_bring_back_purged_text(live: Harness, db: psycopg.Connection[Any]) -> None:
    live.feed(edit_json(1))
    live.worker.run_once()
    db.execute("UPDATE edits SET added_text = NULL, content_hidden = true WHERE rev_id = 1")
    db.commit()
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT added_text, content_hidden FROM edits") == [(None, True)]


@pytest.mark.integration
def test_a_new_model_version_becomes_the_latest_score(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    live.feed(edit_json(1))
    live.worker.run_once()
    live.models["roberta"].fail = True
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT count(*) FROM scores") == [(2,)]
    assert rows(db, "SELECT latest_score, latest_model_version FROM edits") == [
        (pytest.approx(0.5), "2")
    ]


@pytest.mark.integration
def test_unscored_edits_are_stored_without_a_score(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    for model in live.models.values():
        model.fail = True
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT latest_score FROM edits") == [(None,)]
    assert rows(db, "SELECT count(*) FROM scores") == [(0,)]
    assert live.publisher.on(EDITS_SCORED) == []
    assert live.consumer.commits == 1


@pytest.mark.integration
def test_rescoring_picks_up_stored_edits_when_models_recover(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    for model in live.models.values():
        model.fail = True
    old = (NOW - timedelta(minutes=10)).isoformat()
    live.feed(edit_json(1, timestamp=old))
    live.worker.run_once()
    for model in live.models.values():
        model.fail = False
    commits = live.consumer.commits
    live.now += 120
    live.worker.rescore_due()
    assert rows(db, "SELECT latest_score, latest_model_version FROM edits") == [
        (pytest.approx(0.8), "1")
    ]
    assert [entry["rev_id"] for entry in live.publisher.on(EDITS_SCORED)] == [1]
    assert live.consumer.commits == commits  # Rescoring has no offset to move.


@pytest.mark.integration
def test_rescoring_leaves_edits_with_no_text_and_very_new_ones(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    for model in live.models.values():
        model.fail = True
    old = (NOW - timedelta(minutes=10)).isoformat()
    live.feed(edit_json(1), edit_json(2, timestamp=old))
    live.worker.run_once()
    db.execute("UPDATE edits SET added_text = NULL WHERE rev_id = 2")
    db.commit()
    for model in live.models.values():
        model.fail = False
    live.now += 120
    live.worker.rescore_due()
    assert rows(db, "SELECT count(*) FROM edits WHERE latest_score IS NOT NULL") == [(0,)]


@pytest.mark.integration
def test_a_row_postgres_refuses_goes_to_the_dlq_and_the_rest_are_stored(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    """JSON can carry a NUL character, and Postgres text can't hold one."""
    live.feed(edit_json(1), edit_json(2, comment="bad\u0000byte"), edit_json(3))
    live.worker.run_once()
    assert rows(db, "SELECT rev_id FROM edits ORDER BY 1") == [(1,), (3,)]
    dead = live.publisher.on(EDITS_DLQ)
    assert [(entry["reason"], entry["rev_id"]) for entry in dead] == [(REASON_DATABASE, 2)]
    assert live.consumer.commits == 1


@pytest.mark.integration
def test_database_down_retries_and_commits_only_after_success(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    good_url = live.url
    live.restore_url = good_url
    live.url = "postgresql://nobody:wrong@127.0.0.1:1/none"
    live.feed(edit_json(1))
    live.worker.run_once()
    assert live.sleeps == [1.0]
    assert rows(db, "SELECT count(*) FROM edits") == [(1,)]
    assert live.consumer.commits == 1


@pytest.mark.integration
def test_nothing_is_committed_while_redpanda_refuses(
    live: Harness, db: psycopg.Connection[Any]
) -> None:
    live.publisher.fail_flushes = 2
    live.feed(edit_json(1))
    live.worker.run_once()
    assert live.sleeps == [1.0, 2.0]
    assert live.consumer.commits == 1
    assert live.publisher.flushed == 1
    assert rows(db, "SELECT count(*) FROM scores") == [(1,)]


def test_stop_ends_the_retry_loop_without_committing(harness: Harness) -> None:
    harness.url = "postgresql://nobody:wrong@127.0.0.1:1/none"
    stop = harness.worker.stop

    def stop_after_sleep(_: float) -> None:
        stop.set()

    harness.worker.sleep = stop_after_sleep
    harness.feed(edit_json(1))
    harness.worker.run_once()
    assert harness.consumer.commits == 0


@pytest.mark.integration
def test_a_failed_commit_only_replays(live: Harness, db: psycopg.Connection[Any]) -> None:
    live.consumer.fail_commit = True
    live.feed(edit_json(1))
    live.worker.run_once()
    assert rows(db, "SELECT count(*) FROM edits") == [(1,)]


# --- Health ---


def test_health_follows_the_chain_and_outages(harness: Harness) -> None:
    assert harness.worker.is_healthy() is True
    harness.worker._outage_started = harness.now
    harness.now += 299
    assert harness.worker.is_healthy() is True
    harness.now += 2
    assert harness.worker.is_healthy() is False


def test_metrics_report_breakers_and_scores(harness: Harness) -> None:
    harness.worker.score(parsed(harness, edit_json(1)))
    harness.worker._export_chain_state()
    registry = harness.worker.metrics.registry
    assert (
        registry.get_sample_value(
            "scorer_edits_scored_total",
            {"model": "roberta", "fallback": "false", "load_shed": "false"},
        )
        == 1
    )
    assert registry.get_sample_value("scorer_breaker_open", {"model": "roberta"}) == 0


def test_the_stop_event_is_a_real_event(harness: Harness) -> None:
    assert isinstance(harness.worker.stop, threading.Event)
