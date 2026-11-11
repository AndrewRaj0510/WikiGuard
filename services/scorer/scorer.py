"""Consume edits.raw, score each edit with the fallback chain, and write the result to Postgres.

The order is what makes a crash harmless: write to Postgres, publish to edits.scored, then commit
the Kafka offset. A crash replays some edits, and the inserts ignore what's already there. They also
never overwrite an edit, so a replay can't bring back text the purge job removed.

What happens to an edit when something goes wrong (section 14 of the plan):

- Unreadable, or Postgres refuses it: edits.dlq, so it can't block the edits behind it.
- Every model fails: if a known-good edit still scores, this edit is the problem and goes to the
  DLQ. If that fails too, the models are the problem, so the edit is stored unscored and rescored
  later.
- Postgres or Redpanda down: nothing is committed, and the batch is retried until it works.
- Too far behind: switch to the fast models and flag those scores, until the backlog halves.
"""

import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

import psycopg
from confluent_kafka import Consumer, KafkaException
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from pydantic import ValidationError
from wikiguard_core.chain import (
    SMOKE_EDIT,
    ServingChain,
    ShadowSlot,
    chain_from_settings,
    shadow_from_settings,
)
from wikiguard_core.config import (
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    KafkaSettings,
    ScorerSettings,
    ServingSettings,
    WorkerSettings,
    load_settings,
)
from wikiguard_core.features import build_features
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.kafka import (
    EDITS_DLQ,
    EDITS_RAW,
    EDITS_SCORED,
    KafkaPublisher,
    Publisher,
    PublishError,
    ensure_topics,
)
from wikiguard_core.ops import serve_ops
from wikiguard_core.predictor import AllModelsFailed, Prediction, Predictor
from wikiguard_core.schemas import EditEvent, ScoreRecord

logger = logging.getLogger("scorer")

CONSUMER_GROUP = "scorer"
POLL_SECONDS = 1.0
LAG_CHECK_SECONDS = 5.0
WATERMARK_TIMEOUT_SECONDS = 2.0
MAX_RETRY_SECONDS = 30.0
HEALTHY_OUTAGE_SECONDS = 300.0
# A rebalance would only replay edits, so this just has to outlast an ordinary database outage.
MAX_POLL_INTERVAL_MS = 60 * 60 * 1000
RESCORE_LIMIT = 32
# Newer edits may still be in a batch that's being retried.
RESCORE_MIN_AGE = timedelta(minutes=2)
# Past this, the retention job is about to remove the text.
RESCORE_MAX_AGE = timedelta(hours=47)
SCORE_DECIMALS = 7
REASON_UNREADABLE = "unreadable"
REASON_FEATURES = "features_failed"
REASON_SCORING = "scoring_failed"
REASON_DATABASE = "database_refused"

INSERT_EDIT = """
    INSERT INTO edits (
        rev_id, parent_rev_id, schema_version, title, edited_at, is_minor, is_anonymous, user_hash,
        comment, old_length, new_length, added_text, removed_text, user_edit_count, user_hidden,
        comment_hidden, content_hidden
    ) VALUES (
        %(rev_id)s, %(parent_rev_id)s, %(schema_version)s, %(title)s, %(timestamp)s, %(is_minor)s,
        %(is_anonymous)s, %(user_hash)s, %(comment)s, %(old_length)s, %(new_length)s,
        %(added_text)s, %(removed_text)s, %(user_edit_count)s, %(user_hidden)s, %(comment_hidden)s,
        %(content_hidden)s
    ) ON CONFLICT (rev_id) DO NOTHING
"""
INSERT_SCORE = """
    INSERT INTO scores (
        rev_id, model_name, model_version, score, used_fallback, scored_at, alias, threshold,
        latency_ms, load_shed, forced
    ) VALUES (
        %(rev_id)s, %(model_name)s, %(model_version)s, %(score)s, %(used_fallback)s, %(scored_at)s,
        %(alias)s, %(threshold)s, %(latency_ms)s, %(load_shed)s, %(forced)s
    ) ON CONFLICT (rev_id, model_version) DO NOTHING
"""
# Only moves forward, so replaying an old edit can't replace a newer model's score.
UPDATE_LATEST = """
    UPDATE edits SET
        latest_score = %(score)s,
        latest_model_version = %(model_version)s,
        latest_scored_at = %(scored_at)s
    WHERE rev_id = %(rev_id)s
      AND (latest_scored_at IS NULL OR latest_scored_at <= %(scored_at)s)
"""
# A candidate's live scores during its shadow period. Their own table, so they never reach users.
INSERT_SHADOW = """
    INSERT INTO shadow_scores (
        rev_id, model_name, model_version, score, error, latency_ms, served_score, scored_at
    ) VALUES (
        %(rev_id)s, %(model_name)s, %(model_version)s, %(score)s, %(error)s, %(latency_ms)s,
        %(served_score)s, %(scored_at)s
    ) ON CONFLICT (rev_id, model_version) DO NOTHING
"""
SELECT_UNSCORED = """
    SELECT rev_id, parent_rev_id, schema_version, title, edited_at, is_minor, is_anonymous,
           user_hash, comment, old_length, new_length, added_text, removed_text, user_edit_count,
           user_hidden, comment_hidden, content_hidden
    FROM edits
    WHERE latest_score IS NULL AND edited_at > %s AND edited_at < %s AND added_text IS NOT NULL
    ORDER BY edited_at DESC
    LIMIT %s
"""


class Settings(
    CommonSettings,
    DatabaseSettings,
    KafkaSettings,
    ServingSettings,
    ScorerSettings,
    WorkerSettings,
):
    pass


class Message(Protocol):
    def error(self) -> Any: ...
    def key(self) -> bytes | None: ...
    def value(self) -> bytes | None: ...


class MessageConsumer(Protocol):
    def poll(self, timeout: float) -> Message | None: ...
    def commit(self, asynchronous: bool = False) -> Any: ...
    def assignment(self) -> list[Any]: ...
    def position(self, partitions: list[Any]) -> list[Any]: ...
    def get_watermark_offsets(self, partition: Any, timeout: float) -> tuple[int, int] | None: ...


@dataclass
class Item:
    edit: EditEvent
    features: dict[str, Any]
    # Stays None when no model could score it, so the edit is stored unscored.
    score: dict[str, Any] | None = None


@dataclass
class Metrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.scored = Counter(
            "scorer_edits_scored",
            "Edits scored",
            ["model", "fallback", "load_shed"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "scorer_batch_latency_seconds",
            "Scoring time per batch",
            ["model"],
            registry=self.registry,
        )
        self.delay = Histogram(
            "scorer_edit_to_score_seconds",
            "Time from the edit to its stored score",
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 3600),
            registry=self.registry,
        )
        self.dead = Counter(
            "scorer_dlq", "Edits sent to the DLQ", ["reason"], registry=self.registry
        )
        self.unscored = Counter(
            "scorer_stored_unscored", "Edits stored with no score", registry=self.registry
        )
        self.rescored = Counter(
            "scorer_rescored", "Stored edits scored later", registry=self.registry
        )
        self.lag = Gauge("scorer_consumer_lag", "Edits behind the stream", registry=self.registry)
        self.shedding = Gauge(
            "scorer_load_shedding", "1 while using only fast models", registry=self.registry
        )
        self.retries = Counter(
            "scorer_delivery_retries", "Failed writes that will be retried", registry=self.registry
        )
        self.breaker = Gauge(
            "scorer_breaker_open",
            "1 while a model's breaker isn't closed",
            ["model"],
            registry=self.registry,
        )
        self.reload_failures = Gauge(
            "scorer_reload_failures", "Failed alias reloads", registry=self.registry
        )
        self.shadow = Counter(
            "scorer_shadow_scored",
            "Edits the shadow candidate scored",
            ["outcome"],
            registry=self.registry,
        )
        self.shadow_loaded = Gauge(
            "scorer_shadow_loaded", "1 while a candidate is in shadow", registry=self.registry
        )


def edit_from_row(row: Sequence[Any]) -> EditEvent:
    return EditEvent(
        rev_id=row[0],
        parent_rev_id=row[1],
        schema_version=row[2],
        title=row[3],
        timestamp=row[4],
        is_minor=row[5],
        is_anonymous=row[6],
        user_hash=row[7],
        comment=row[8],
        old_length=row[9],
        new_length=row[10],
        added_text=row[11] or "",
        removed_text=row[12] or "",
        user_edit_count=row[13],
        user_hidden=row[14],
        comment_hidden=row[15],
        content_hidden=row[16],
    )


def score_columns(item: Item, prediction: Prediction, value: float, shed: bool) -> dict[str, Any]:
    return {
        "rev_id": item.edit.rev_id,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        # Postgres `real` refuses values too small to hold, so a score of 1e-50 would be an error
        # (F-104). A `real` holds about 7 digits anyway, so rounding loses nothing.
        "score": round(value, SCORE_DECIMALS),
        "used_fallback": prediction.used_fallback,
        "scored_at": datetime.now(UTC),
        "alias": prediction.alias,
        "threshold": prediction.threshold,
        "latency_ms": prediction.latency_ms / max(prediction.scores.size, 1),
        "load_shed": shed,
        "forced": prediction.forced,
    }


@dataclass
class ScoringWorker:
    consumer: MessageConsumer
    publisher: Publisher
    chain: ServingChain
    connect: Callable[[], psycopg.Connection[Any]]
    batch_size: int = 32
    window_seconds: float = 0.05
    shed_lag: int = 500
    rescore_seconds: float = 60.0
    stop: threading.Event = field(default_factory=threading.Event)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    metrics: Metrics = field(default_factory=Metrics)
    shedding: bool = False
    shadow: ShadowSlot | None = None
    _connection: psycopg.Connection[Any] | None = None
    _last_lag_check: float = 0.0
    _last_rescore: float = 0.0
    _outage_started: float | None = None

    def __post_init__(self) -> None:
        self._last_rescore = self.clock()

    # --- Reading ---

    def collect(self) -> list[Message]:
        """Up to batch_size messages. Waits briefly for more once the first has arrived."""
        first = self.consumer.poll(POLL_SECONDS)
        if first is None:
            return []
        messages = [first]
        deadline = self.clock() + self.window_seconds
        while len(messages) < self.batch_size and (remaining := deadline - self.clock()) > 0:
            message = self.consumer.poll(remaining)
            if message is None:
                break
            messages.append(message)
        return messages

    def _dead_letter(self, reason: str, rev_id: int | None, detail: str = "") -> None:
        """Enough to find the edit again, and no text or user: the DLQ outlives purges."""
        body = {"reason": reason, "detail": detail[:200], "rev_id": rev_id}
        self.publisher.publish(EDITS_DLQ, str(rev_id), json.dumps(body).encode())
        self.metrics.dead.labels(reason).inc()
        logger.warning("Sent to the DLQ: rev_id=%s reason=%s", rev_id, reason)

    def _parse(self, messages: Sequence[Message]) -> list[Item]:
        items: list[Item] = []
        for message in messages:
            if message.error():
                logger.warning("Consumer error: %s", message.error())
                continue
            value = message.value()
            if value is None:
                continue
            try:
                edit = EditEvent.model_validate_json(value)
            except ValidationError as error:
                # Field names only: pydantic's messages can quote the input.
                fields = sorted({str(item["loc"][0]) for item in error.errors() if item["loc"]})
                key = message.key()
                rev_id = int(key) if key and key.isdigit() else None
                self._dead_letter(REASON_UNREADABLE, rev_id, ",".join(fields))
                continue
            try:
                items.append(Item(edit, build_features(edit).as_dict()))
            except Exception as error:  # noqa: BLE001 - one edit must not stop the stream
                self._dead_letter(REASON_FEATURES, edit.rev_id, type(error).__name__)
        return items

    # --- Scoring ---

    def _update_shedding(self) -> None:
        if self.clock() - self._last_lag_check < LAG_CHECK_SECONDS:
            return
        self._last_lag_check = self.clock()
        try:
            lag = 0
            assigned = self.consumer.assignment()
            for position in self.consumer.position(assigned) if assigned else []:
                marks = self.consumer.get_watermark_offsets(position, WATERMARK_TIMEOUT_SECONDS)
                if marks is not None and position.offset >= 0:
                    lag += max(marks[1] - position.offset, 0)
        except KafkaException:
            return
        self.metrics.lag.set(lag)
        # Different limits for on and off, so a backlog near the line doesn't flip it constantly.
        if lag > self.shed_lag:
            self.shedding = True
        elif lag <= self.shed_lag // 2:
            self.shedding = False
        self.metrics.shedding.set(int(self.shedding))

    def _models_work(self, predictor: Predictor) -> bool:
        try:
            predictor.score([build_features(SMOKE_EDIT).as_dict()])
        except AllModelsFailed:
            return False
        return True

    def _apply(self, items: Sequence[Item], prediction: Prediction, shed: bool) -> None:
        for item, value in zip(items, prediction.scores.tolist(), strict=True):
            item.score = score_columns(item, prediction, value, shed)
        labels = (prediction.model_name, str(prediction.used_fallback).lower(), str(shed).lower())
        self.metrics.scored.labels(*labels).inc(len(items))
        self.metrics.latency.labels(prediction.model_name).observe(prediction.latency_ms / 1000)

    def score(self, items: list[Item]) -> list[Item]:
        """Fill in scores. Returns the items to store, which leaves out any poison ones."""
        if not items:
            return items
        current = self.chain.current
        shed = self.shedding
        predictor = current.fast if shed else current.predictor
        try:
            self._apply(items, predictor.score([item.features for item in items]), shed)
            return items
        except AllModelsFailed:
            pass
        # Find which edit it was. Most of this batch will score fine alone.
        kept: list[Item] = []
        for item in items:
            try:
                self._apply([item], predictor.score([item.features]), shed)
            except AllModelsFailed as failure:
                if self._models_work(predictor):
                    self._dead_letter(REASON_SCORING, item.edit.rev_id, str(failure))
                    continue
                self.metrics.unscored.inc()
            kept.append(item)
        return kept

    # --- Writing ---

    def _db(self) -> psycopg.Connection[Any]:
        if self._connection is None or self._connection.closed:
            self._connection = self.connect()
        return self._connection

    def _write(self, connection: psycopg.Connection[Any], items: Sequence[Item]) -> None:
        with connection.transaction(), connection.cursor() as cursor:
            cursor.executemany(INSERT_EDIT, [item.edit.model_dump() for item in items])
            scored = [item.score for item in items if item.score is not None]
            if scored:
                cursor.executemany(INSERT_SCORE, scored)
                cursor.executemany(UPDATE_LATEST, scored)

    def _write_each(self, connection: psycopg.Connection[Any], items: list[Item]) -> list[Item]:
        """Postgres refused something in the batch, such as a NUL byte. Keep all the others."""
        kept: list[Item] = []
        for item in items:
            try:
                self._write(connection, [item])
                kept.append(item)
            except (psycopg.DataError, psycopg.IntegrityError) as error:
                self._dead_letter(REASON_DATABASE, item.edit.rev_id, type(error).__name__)
        return kept

    def _publish_scores(self, items: Sequence[Item]) -> None:
        for item in items:
            if item.score is None:
                continue
            record = ScoreRecord(
                **{
                    key: item.score[key]
                    for key in (
                        "rev_id",
                        "model_name",
                        "model_version",
                        "score",
                        "used_fallback",
                        "scored_at",
                        "threshold",
                        "alias",
                    )
                }
            )
            self.publisher.publish(
                EDITS_SCORED, str(record.rev_id), record.model_dump_json().encode()
            )

    def _observe_delay(self, items: Sequence[Item]) -> None:
        now = datetime.now(UTC)
        for item in items:
            if item.score is not None:
                self.metrics.delay.observe((now - item.edit.timestamp).total_seconds())

    def deliver(self, items: list[Item], commit: bool = True) -> None:
        """Retries until Postgres and Redpanda accept it. The offset moves only after both have."""
        delay = 1.0
        while not self.stop.is_set():
            try:
                connection = self._db()
                try:
                    self._write(connection, items)
                except (psycopg.DataError, psycopg.IntegrityError):
                    items = self._write_each(connection, items)
                self._publish_scores(items)
                self.publisher.flush()
            except (psycopg.OperationalError, psycopg.InterfaceError, PublishError) as error:
                self.metrics.retries.inc()
                if self._outage_started is None:
                    self._outage_started = self.clock()
                logger.warning("Write failed, retrying: %s", type(error).__name__)
                if self._connection is not None:
                    self._connection.close()
                self.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_SECONDS)
                continue
            self._outage_started = None
            self._observe_delay(items)
            if commit:
                try:
                    self.consumer.commit(asynchronous=False)
                except KafkaException as error:
                    # Only replays these edits, and the writes above ignore what's already stored.
                    logger.warning("Commit failed: %s", error)
            return

    # --- Shadow ---

    def shadow_rows(self, items: Sequence[Item]) -> list[dict[str, Any]]:
        """The candidate scores each edit alone, like serving does, and failures are kept too.

        Skipped while shedding load: the shadow must never slow down what users see.
        """
        loaded = self.shadow.current if self.shadow is not None else None
        if loaded is None or self.shedding:
            return []
        ref, predictor = loaded
        rows = []
        for item in items:
            if item.score is None:
                continue
            row: dict[str, Any] = {
                "rev_id": item.edit.rev_id,
                "model_name": ref.model_name,
                "model_version": ref.version,
                "score": None,
                "error": None,
                "latency_ms": None,
                "served_score": item.score["score"],
                "scored_at": datetime.now(UTC),
            }
            try:
                prediction = predictor.score([item.features])
            except AllModelsFailed as failure:
                attempts = failure.attempts
                row["error"] = str(attempts[-1].outcome) if attempts else "failed"
                self.metrics.shadow.labels("error").inc()
            else:
                row["score"] = round(float(prediction.scores[0]), SCORE_DECIMALS)
                row["latency_ms"] = prediction.latency_ms
                self.metrics.shadow.labels("scored").inc()
            rows.append(row)
        return rows

    def write_shadow(self, items: Sequence[Item]) -> None:
        """Best effort, after the real write: a lost shadow score costs nothing but a sample."""
        rows = self.shadow_rows(items)
        if not rows:
            return
        try:
            connection = self._db()
            with connection.transaction(), connection.cursor() as cursor:
                cursor.executemany(INSERT_SHADOW, rows)
        except psycopg.Error as error:
            self.metrics.shadow.labels("not_stored").inc(len(rows))
            logger.warning("Shadow scores not stored: %s", type(error).__name__)

    # --- Rescoring ---

    def rescore_due(self) -> None:
        if self.clock() - self._last_rescore < self.rescore_seconds:
            return
        self._last_rescore = self.clock()
        now = datetime.now(UTC)
        try:
            connection = self._db()
            rows = connection.execute(
                SELECT_UNSCORED, (now - RESCORE_MAX_AGE, now - RESCORE_MIN_AGE, RESCORE_LIMIT)
            ).fetchall()
            connection.commit()
        except (psycopg.OperationalError, psycopg.InterfaceError) as error:
            logger.warning("Rescore query failed: %s", type(error).__name__)
            if self._connection is not None:
                self._connection.close()
            return
        if not rows:
            return
        edits = [edit_from_row(row) for row in rows]
        items = [Item(edit, build_features(edit).as_dict()) for edit in edits]
        kept = self.score(items)
        scored = [item for item in kept if item.score is not None]
        if scored:
            self.deliver(scored, commit=False)
            self.metrics.rescored.inc(len(scored))

    # --- The loop ---

    def run_once(self) -> None:
        self._update_shedding()
        messages = self.collect()
        if messages:
            items = self.score(self._parse(messages))
            self.deliver(items)
            self.write_shadow(items)
        self.rescore_due()
        self._export_chain_state()

    def _export_chain_state(self) -> None:
        for status in self.chain.current.predictor.status():
            self.metrics.breaker.labels(status.model_name).set(
                int(status.breaker.value != "closed")
            )
        self.metrics.reload_failures.set(self.chain.reload_failures)
        self.metrics.shadow_loaded.set(
            int(self.shadow is not None and self.shadow.current is not None)
        )

    def is_healthy(self) -> bool:
        if not self.chain.ready:
            return False
        started = self._outage_started
        return started is None or self.clock() - started < HEALTHY_OUTAGE_SECONDS


def main() -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2

    ensure_topics(settings.kafka_bootstrap_servers)
    chain = chain_from_settings(settings)
    url = settings.database_url.get_secret_value()

    def connect() -> psycopg.Connection[Any]:
        # prepare_threshold=None because Supabase's transaction pooler can't hold prepared
        # statements across connections.
        return psycopg.connect(url, prepare_threshold=None, connect_timeout=10)

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": CONSUMER_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": MAX_POLL_INTERVAL_MS,
        }
    )
    worker = ScoringWorker(
        cast(MessageConsumer, consumer),
        KafkaPublisher(settings.kafka_bootstrap_servers),
        chain,
        connect,
        settings.scorer_batch_size,
        settings.scorer_batch_window_ms / 1000,
        settings.scorer_shed_lag,
        settings.scorer_rescore_seconds,
    )
    # The cluster's scorer runs the shadow. Render's API never does: it has no room (DEC-050).
    worker.shadow = shadow_from_settings(settings)
    serve_ops(settings.ops_port, worker.metrics.registry, worker.is_healthy)

    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: worker.stop.set())
    # Models first: consuming with nothing to score with would only fill the unscored list.
    chain.load()
    worker.shadow.refresh()
    if settings.model_poll_seconds:
        chain.poll(settings.model_poll_seconds, worker.stop)
        worker.shadow.poll(settings.model_poll_seconds, worker.stop)
    consumer.subscribe([EDITS_RAW])
    while not worker.stop.is_set():
        worker.run_once()
    consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
