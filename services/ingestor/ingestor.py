"""Read live edits from Wikimedia, add their diffs, and publish them to Redpanda.

The stream position is saved only after a message is confirmed, so a crash replays a few edits
but never loses one. A replay is harmless, because consumers key on rev_id. A bad edit goes to
edits.dlq, so one poison message can't stop the stream.
"""

import json
import logging
import signal
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge
from pydantic import ValidationError
from wikiguard_core.config import (
    AlertSettings,
    CommonSettings,
    ConfigError,
    KafkaSettings,
    PrivacySettings,
    WikimediaSettings,
    WorkerSettings,
    load_settings,
)
from wikiguard_core.errors import init_error_tracking
from wikiguard_core.identity import RedactingFilter, hash_user, redact_identifiers
from wikiguard_core.ingest import REASON_API_ERROR, REASON_MALFORMED, attach_diff
from wikiguard_core.kafka import (
    EDITS_DLQ,
    EDITS_RAW,
    KafkaPublisher,
    Publisher,
    PublishError,
    ensure_topics,
)
from wikiguard_core.ops import serve_ops
from wikiguard_core.schemas import EditEvent, edit_from_stream, is_in_scope_stream_event
from wikiguard_core.wikimedia import WikimediaClient, WikimediaError, backoff_seconds

logger = logging.getLogger("ingestor")

STREAM_READ_TIMEOUT_SECONDS = 60.0
SAVE_POSITION_EVERY_SECONDS = 5.0
USER_COUNT_CACHE_SIZE = 50_000
HEALTHY_WITHIN_SECONDS = 300.0
# A jump this big between stream events means edits may have been missed during an outage.
GAP_WARNING_SECONDS = 120.0
RESET_BACKOFF_AFTER_SECONDS = 60.0


class Settings(
    CommonSettings,
    KafkaSettings,
    WikimediaSettings,
    PrivacySettings,
    WorkerSettings,
    AlertSettings,
):
    pass


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
    """Yield (event id, event) from server-sent event lines. A bad event comes back as None."""
    event_id = None
    data: list[str] = []
    for line in lines:
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line.strip() and data:
            try:
                parsed = json.loads("\n".join(data))
            except json.JSONDecodeError:
                parsed = None
            yield event_id, parsed if isinstance(parsed, dict) else None
            data = []


class EditCountCache:
    """Recent editors' counts, to avoid an API call for every edit by the same person.

    Keyed by hash, so no raw username stays in memory. A cached count goes up by one per edit,
    which keeps it right for as long as the editor stays in the cache.
    """

    def __init__(self, size: int = USER_COUNT_CACHE_SIZE) -> None:
        self.size = size
        self.counts: OrderedDict[str, int] = OrderedDict()

    def get(self, user_hash: str) -> int | None:
        count = self.counts.get(user_hash)
        if count is None:
            return None
        self.counts[user_hash] = count + 1
        self.counts.move_to_end(user_hash)
        return count + 1

    def put(self, user_hash: str, count: int) -> None:
        self.counts[user_hash] = count
        self.counts.move_to_end(user_hash)
        while len(self.counts) > self.size:
            self.counts.popitem(last=False)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.events = Counter(
            "ingestor_events", "Stream events seen", ["kind"], registry=self.registry
        )
        self.published = Counter(
            "ingestor_published", "Edits published to edits.raw", registry=self.registry
        )
        self.dead_letters = Counter(
            "ingestor_dead_letters",
            "Messages sent to edits.dlq",
            ["reason"],
            registry=self.registry,
        )
        self.reconnects = Counter(
            "ingestor_reconnects", "Stream reconnects", registry=self.registry
        )
        self.lag = Gauge(
            "ingestor_lag_seconds", "Age of the newest handled edit", registry=self.registry
        )
        self.gap = Gauge(
            "ingestor_last_gap_seconds", "Largest jump between events", registry=self.registry
        )


class Position:
    """The last confirmed stream position, saved so a restart resumes from it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.event_id: str | None = None
        self.saved_at = 0.0
        if path.exists():
            try:
                self.event_id = json.loads(path.read_text(encoding="utf-8")).get("event_id")
            except json.JSONDecodeError:
                logger.warning("Stream position file is unreadable, so starting from now")

    def update(self, event_id: str | None, force: bool = False) -> None:
        if event_id:
            self.event_id = event_id
        if self.event_id and (
            force or time.monotonic() - self.saved_at >= SAVE_POSITION_EVERY_SECONDS
        ):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"event_id": self.event_id}), encoding="utf-8")
            temporary.replace(self.path)
            self.saved_at = time.monotonic()


class Ingestor:
    def __init__(
        self,
        client: WikimediaClient,
        publisher: Publisher,
        position: Position,
        hash_key: str,
        metrics: Metrics | None = None,
    ) -> None:
        self.client = client
        self.publisher = publisher
        self.position = position
        self.hash_key = hash_key
        self.metrics = metrics or Metrics()
        self.counts = EditCountCache()
        self.last_event_at = time.monotonic()
        self.last_event_time: float | None = None
        self.should_stop: Callable[[], bool] = lambda: False

    def edit_count(self, name: str) -> int | None:
        user_hash = hash_user(name, self.hash_key)
        cached = self.counts.get(user_hash)
        if cached is not None:
            return cached
        try:
            count = self.client.user_edit_counts([name]).get(name)
        except WikimediaError:
            # The count is one feature. Losing the whole edit over it would be worse.
            self.metrics.events.labels("edit_count_failed").inc()
            return None
        if count is not None:
            self.counts.put(user_hash, count)
        return count

    def dead_letter(self, reason: str, event: dict[str, Any] | None, detail: str = "") -> None:
        """Keep enough to find the edit again, and no username or edit text."""
        event = event or {}
        revision = event.get("revision") if isinstance(event.get("revision"), dict) else {}
        body = {
            "reason": reason,
            "detail": redact_identifiers(detail)[:200],
            "rev_id": revision.get("new"),
            "parent_rev_id": revision.get("old"),
            "timestamp": event.get("timestamp"),
        }
        self.publisher.publish(EDITS_DLQ, str(revision.get("new")), json.dumps(body).encode())
        self.metrics.dead_letters.labels(reason).inc()

    def _note_timing(self, event: dict[str, Any]) -> None:
        moment = event.get("timestamp")
        if not isinstance(moment, int | float):
            return
        # Measured from the newest time seen, not the last event's: the stream sometimes carries
        # an event stamped years back, and the next ordinary one would look like a jump of years.
        if self.last_event_time is not None:
            jump = moment - self.last_event_time
            if jump > GAP_WARNING_SECONDS:
                self.metrics.gap.set(jump)
                logger.warning("Stream jumped %d seconds; consider gap repair", jump)
        self.last_event_time = max(moment, self.last_event_time or moment)
        self.metrics.lag.set(max(0.0, time.time() - self.last_event_time))

    def build_edit(self, event: dict[str, Any]) -> EditEvent | None:
        """A full edit, or None after sending the reason to the dead-letter topic."""
        try:
            edit = edit_from_stream(event, self.hash_key)
        except (KeyError, TypeError, ValueError, ValidationError, AttributeError) as error:
            self.dead_letter(REASON_MALFORMED, event, type(error).__name__)
            return None
        try:
            outcome = attach_diff(self.client, edit)
            if outcome.edit is None:
                self.dead_letter(outcome.failure_reason or REASON_API_ERROR, event)
                return None
            return outcome.edit.with_edit_count(self.edit_count(event["user"]))
        except WikimediaError as error:
            self.dead_letter(REASON_API_ERROR, event, str(error))
            return None

    def handle(self, event_id: str | None, event: dict[str, Any] | None) -> None:
        self.last_event_at = time.monotonic()
        if event is None:
            self.metrics.events.labels("unreadable").inc()
            self.dead_letter(REASON_MALFORMED, None, "not JSON")
            self.publisher.flush()
            self.position.update(event_id)
            return
        self._note_timing(event)
        if not is_in_scope_stream_event(event):
            self.metrics.events.labels("out_of_scope").inc()
            self.position.update(event_id)
            return
        self.metrics.events.labels("in_scope").inc()
        edit = self.build_edit(event)
        if edit is not None:
            self.publisher.publish(EDITS_RAW, str(edit.rev_id), edit.model_dump_json().encode())
            self.metrics.published.inc()
        # Only move the position once Redpanda has confirmed everything before it.
        self.publisher.flush()
        self.position.update(event_id)

    def is_healthy(self) -> bool:
        return time.monotonic() - self.last_event_at < HEALTHY_WITHIN_SECONDS

    def consume(self, events: Iterable[tuple[str | None, dict[str, Any] | None]]) -> None:
        for event_id, event in events:
            if self.should_stop():
                return
            self.handle(event_id, event)


def stream_lines(settings: WikimediaSettings, last_event_id: str | None) -> Iterator[str]:
    headers = {"User-Agent": settings.wikimedia_user_agent, "Accept": "text/event-stream"}
    if last_event_id:
        headers["Last-Event-ID"] = last_event_id
    timeout = httpx.Timeout(STREAM_READ_TIMEOUT_SECONDS, connect=30.0)
    with httpx.stream("GET", settings.wikimedia_stream_url, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        yield from r.iter_lines()


def run_forever(
    ingestor: Ingestor,
    open_stream: Callable[[str | None], Iterable[str]],
    stop: threading.Event,
    wait: Callable[[float], bool] | None = None,
) -> None:
    """Reconnect with backoff and jitter whenever the stream or Redpanda fails."""
    pause = wait or stop.wait
    ingestor.should_stop = stop.is_set
    attempt = 0
    while not stop.is_set():
        started = time.monotonic()
        try:
            ingestor.consume(parse_sse(open_stream(ingestor.position.event_id)))
        except (httpx.HTTPError, PublishError) as error:
            logger.warning("Stream interrupted: %s", type(error).__name__)
        if stop.is_set():
            break
        ingestor.metrics.reconnects.inc()
        # A connection that ran well earns a fresh start, while repeated quick failures slow down.
        attempt = 0 if time.monotonic() - started > RESET_BACKOFF_AFTER_SECONDS else attempt + 1
        pause(backoff_seconds(attempt))
    ingestor.publisher.flush()
    ingestor.position.update(None, force=True)


def main() -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    init_error_tracking(settings.sentry_dsn, "ingestor", settings.environment)

    ensure_topics(settings.kafka_bootstrap_servers)
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    ingestor = Ingestor(
        client,
        KafkaPublisher(settings.kafka_bootstrap_servers),
        Position(Path(settings.ingest_state_path)),
        settings.user_hash_key.get_secret_value(),
    )
    serve_ops(settings.ops_port, ingestor.metrics.registry, ingestor.is_healthy)

    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    run_forever(ingestor, lambda event_id: stream_lines(settings, event_id), stop)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
