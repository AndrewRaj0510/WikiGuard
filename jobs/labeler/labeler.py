"""Hourly: give edits that reached 48 hours their final label, from the mw-reverted tag.

Each edit is looked up once, when it turns 48 hours old, rather than every hour until then. A tag
doesn't go away once set, so a later look sees everything an earlier one would have. It takes
about 80 requests an hour instead of about 4,000 (DEC-037).

- Reruns are safe, because a label that exists is never changed.
- After downtime, the backlog is worked through oldest first. Those labels may include a revert
  after the 48-hour mark, so they're flagged decided_late for the evaluation to handle.
- Hidden and deleted revisions get a reason instead of a label, so they're counted.
"""

import logging
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from wikiguard_core.config import (
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    KafkaSettings,
    WikimediaSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.kafka import LABELS_FINAL, KafkaPublisher, Publisher, ensure_topics
from wikiguard_core.labels import LABEL_WAIT, decide_label
from wikiguard_core.schemas import LabelRecord
from wikiguard_core.wikimedia import BATCH_SIZE, RevisionState, WikimediaClient, chunks

logger = logging.getLogger("labeler")

# Hourly runs reach an edit within about an hour of the mark, so this allows for one missed run.
LATE_AFTER = LABEL_WAIT + timedelta(hours=2)
# One run's limit, so a long catch-up can't monopolize the API. The next run continues.
MAX_EDITS_PER_RUN = 50_000
EXCLUDED_HIDDEN = "hidden"
EXCLUDED_DELETED = "deleted"


class Settings(CommonSettings, DatabaseSettings, KafkaSettings, WikimediaSettings):
    pass


def decide(rev_id: int, edited_at: datetime, state: RevisionState, now: datetime) -> LabelRecord:
    """Hidden content can't be fairly labeled, even when a revert tag is present."""
    if state.missing:
        return LabelRecord(
            rev_id=rev_id, label=None, excluded_reason=EXCLUDED_DELETED, decided_at=now
        )
    if state.content_hidden:
        return LabelRecord(
            rev_id=rev_id, label=None, excluded_reason=EXCLUDED_HIDDEN, decided_at=now
        )
    return LabelRecord(
        rev_id=rev_id,
        label=decide_label(edited_at, state.tags, now),
        decided_at=now,
        decided_late=now - edited_at > LATE_AFTER,
    )


def due_edits(
    connection: psycopg.Connection[Any], now: datetime, limit: int
) -> Sequence[tuple[int, datetime]]:
    rows: list[tuple[int, datetime]] = connection.execute(
        """
        SELECT e.rev_id, e.edited_at
        FROM edits e
        LEFT JOIN labels l ON l.rev_id = e.rev_id
        WHERE l.rev_id IS NULL AND e.edited_at <= %s
        ORDER BY e.edited_at
        LIMIT %s
        """,
        (now - LABEL_WAIT, limit),
    ).fetchall()
    return rows


def run(
    connection: psycopg.Connection[Any],
    client: WikimediaClient,
    publisher: Publisher,
    now: datetime | None = None,
    limit: int = MAX_EDITS_PER_RUN,
) -> Counter[str]:
    """Commit each batch only after Redpanda confirms it. A crash repeats that batch, harmlessly."""
    now = now or datetime.now(UTC)
    results: Counter[str] = Counter()
    for batch in chunks(due_edits(connection, now, limit), BATCH_SIZE):
        states = client.revision_states([rev_id for rev_id, _ in batch])
        for rev_id, edited_at in batch:
            record = decide(rev_id, edited_at, states[rev_id], now)
            inserted = connection.execute(
                """
                INSERT INTO labels (rev_id, label, excluded_reason, decided_at, decided_late)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (rev_id) DO NOTHING
                """,
                (
                    record.rev_id,
                    record.label,
                    record.excluded_reason,
                    record.decided_at,
                    record.decided_late,
                ),
            ).rowcount
            if not inserted:
                results["already_labeled"] += 1
                continue
            publisher.publish(LABELS_FINAL, str(rev_id), record.model_dump_json().encode())
            results[record.excluded_reason or f"label_{record.label}"] += 1
            results["late"] += record.decided_late
        publisher.flush()
        connection.commit()
    return results


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
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        results = run(connection, client, KafkaPublisher(settings.kafka_bootstrap_servers))
    client.close()
    logger.info("Labeler finished: %s", dict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
