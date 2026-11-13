"""Write edits, scores, and labels from their topics to the parquet archive.

The offset is committed only after a file is written, so a crash writes some rows twice but loses
none, and readers remove the duplicates. This reads the topics and never reads from Supabase,
because Supabase reads count against its monthly egress (section 2 of the plan).

    python archiver.py            run the consumer
    python archiver.py --cleanup  delete anything past the retention window, as a daily CronJob
    python archiver.py --cleanup --mirror
                                  the same, then copy new and changed files to the bucket (DEC-083)
    python archiver.py --restore-mirror
                                  after a rebuild: copy what the working copy lacks back from the
                                  bucket
"""

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from confluent_kafka import Consumer
from prometheus_client import CollectorRegistry, Counter, Gauge
from pydantic import BaseModel, ValidationError
from wikiguard_core.archive import (
    EDITS,
    LABELS,
    SCORES,
    Archive,
    Root,
    delete_older_than,
    mirror,
    open_archive,
    open_bucket,
    restore_from_mirror,
    size_in_bytes,
    write_rows,
)
from wikiguard_core.config import (
    AlertSettings,
    ArchiveSettings,
    CommonSettings,
    ConfigError,
    KafkaSettings,
    WorkerSettings,
    load_settings,
)
from wikiguard_core.errors import init_error_tracking
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.kafka import EDITS_RAW, EDITS_SCORED, LABELS_FINAL
from wikiguard_core.ops import serve_ops
from wikiguard_core.schemas import EditEvent, LabelRecord, ScoreRecord

logger = logging.getLogger("archiver")

TOPIC_DATASETS: dict[str, tuple[str, type[BaseModel]]] = {
    EDITS_RAW: (EDITS, EditEvent),
    EDITS_SCORED: (SCORES, ScoreRecord),
    LABELS_FINAL: (LABELS, LabelRecord),
}
FLUSH_ROWS = 1000
FLUSH_SECONDS = 300.0
POLL_SECONDS = 1.0
CONSUMER_GROUP = "archiver"
HEALTHY_WITHIN_SECONDS = 900.0
MEASURE_EVERY_SECONDS = 600.0


class Settings(CommonSettings, KafkaSettings, ArchiveSettings, WorkerSettings, AlertSettings):
    pass


class Message(Protocol):
    def error(self) -> Any: ...
    def topic(self) -> str | None: ...
    def value(self) -> bytes | None: ...


class MessageConsumer(Protocol):
    def poll(self, timeout: float) -> Message | None: ...
    def commit(self, asynchronous: bool = False) -> Any: ...


@dataclass
class Archiver:
    consumer: MessageConsumer
    root: Root
    flush_rows: int = FLUSH_ROWS
    flush_seconds: float = FLUSH_SECONDS
    clock: Callable[[], float] = time.monotonic
    buffers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_flush: float = 0.0
    last_write: float = 0.0
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.last_flush = self.clock()
        self.last_write = self.clock()
        self.rows_written = Counter(
            "archiver_rows_written", "Rows written", ["dataset"], registry=self.registry
        )
        self.poison = Counter(
            "archiver_poison_messages", "Unreadable messages skipped", registry=self.registry
        )
        self.buffered = Gauge("archiver_buffered_rows", "Rows waiting", registry=self.registry)
        # For the free-tier dashboard: the mirror has to fit Oracle's 20 GB. Measured every
        # few minutes, not on each scrape, because it walks the whole tree.
        self.archive_bytes = Gauge(
            "archiver_archive_bytes", "Parquet bytes in the archive", registry=self.registry
        )
        self.last_measured = float("-inf")

    def add(self, topic: str, value: bytes) -> None:
        dataset, model = TOPIC_DATASETS[topic]
        try:
            row = model.model_validate_json(value).model_dump()
        except ValidationError:
            # Skip it. Stopping here would block every later message behind this one.
            self.poison.inc()
            logger.warning("Skipped an unreadable message on %s", topic)
            return
        self.buffers.setdefault(dataset, []).append(row)
        self.buffered.set(sum(len(rows) for rows in self.buffers.values()))

    def flush_if_due(self) -> None:
        count = sum(len(rows) for rows in self.buffers.values())
        if count and (
            count >= self.flush_rows or self.clock() - self.last_flush >= self.flush_seconds
        ):
            self.flush()
        elif not count:
            self.last_flush = self.clock()

    def flush(self) -> None:
        """Write first and commit second, so the commit can never cover unwritten rows."""
        for dataset, rows in self.buffers.items():
            if rows:
                write_rows(self.root, dataset, rows)
                self.rows_written.labels(dataset).inc(len(rows))
        self.consumer.commit(asynchronous=False)
        self.buffers = {}
        self.buffered.set(0)
        self.last_flush = self.clock()
        self.last_write = self.clock()

    def poll_once(self) -> None:
        message = self.consumer.poll(POLL_SECONDS)
        if message is not None and message.error():
            logger.warning("Consumer error: %s", message.error())
        elif message is not None:
            topic, value = message.topic(), message.value()
            if topic is not None and value is not None:
                self.add(topic, value)
        self.flush_if_due()
        if self.clock() - self.last_measured >= MEASURE_EVERY_SECONDS:
            self.last_measured = self.clock()
            self.archive_bytes.set(size_in_bytes(self.root))

    def is_healthy(self) -> bool:
        return self.clock() - self.last_write < HEALTHY_WITHIN_SECONDS


def clean_up(root: Root, retention_days: int, now: datetime | None = None) -> int:
    """Delete folders past the window. Settings refuse a window under 3 days, so edits still
    waiting for their 48-hour label are never touched."""
    now = now or datetime.now(UTC)
    return delete_older_than(root, (now - timedelta(days=retention_days)).date())


def maintain(settings: Settings, root: Archive, cleanup: bool, copy: bool, restore: bool) -> int:
    """The daily CronJob's work, and the rebuild step. Returns the exit code."""
    if cleanup:
        logger.info("Removed %d date folders", clean_up(root, settings.archive_retention_days))
    if not (copy or restore):
        return 0
    bucket = open_bucket(settings, settings.archive_mirror_prefix)
    if bucket is None or not root.local:
        logger.error("The mirror needs ARCHIVE_STORE=local and ARCHIVE_S3_BUCKET")
        return 2
    if restore:
        logger.info("Restored %d files from the mirror", restore_from_mirror(bucket, root))
    if copy:
        today = datetime.now(UTC).date()
        result = mirror(root, bucket, settings.archive_retention_days, today)
        logger.info("Mirror: %s", result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--restore-mirror", action="store_true")
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    init_error_tracking(settings.sentry_dsn, "archiver", settings.environment)

    root = open_archive(settings)
    if args.cleanup or args.mirror or args.restore_mirror:
        return maintain(settings, root, args.cleanup, args.mirror, args.restore_mirror)

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": CONSUMER_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe(list(TOPIC_DATASETS))
    archiver = Archiver(consumer, root)
    serve_ops(settings.ops_port, archiver.registry, archiver.is_healthy)

    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    while not stop.is_set():
        archiver.poll_once()
    archiver.flush()
    consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
