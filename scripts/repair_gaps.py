"""Refill edits missed during an outage, from list=recentchanges, into edits.raw.

Replaying is safe because consumers key on rev_id. With DATABASE_URL set, edits that Postgres
already has are skipped, which saves one diff request for each.

The edit count stays missing on these rows. The API only knows today's count, which isn't
what the editor had at edit time, and the model handles a missing count (F-007).

    python scripts/repair_gaps.py --since 2026-11-04T10:00:00Z --until 2026-11-04T12:00:00Z
"""

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from wikiguard_core.config import (
    ConfigError,
    KafkaSettings,
    PrivacySettings,
    WikimediaSettings,
    load_settings,
)
from wikiguard_core.ingest import REASON_API_ERROR, attach_diff
from wikiguard_core.kafka import EDITS_DLQ, EDITS_RAW, KafkaPublisher, Publisher, ensure_topics
from wikiguard_core.schemas import edit_from_recentchange
from wikiguard_core.wikimedia import WikimediaClient, WikimediaError, chunks

# A safety limit: about three days of edits. A longer outage should use the backfill.
MAX_ROWS = 300_000


def known_rev_ids_from_postgres(url: str) -> Callable[[list[int]], set[int]]:
    import psycopg

    def lookup(rev_ids: list[int]) -> set[int]:
        with psycopg.connect(url, prepare_threshold=None, connect_timeout=10) as connection:
            rows = connection.execute(
                "SELECT rev_id FROM edits WHERE rev_id = ANY(%s)", (rev_ids,)
            ).fetchall()
        return {row[0] for row in rows}

    return lookup


def repair(
    client: WikimediaClient,
    publisher: Publisher,
    since: datetime,
    until: datetime,
    hash_key: str,
    known: Callable[[list[int]], Iterable[int]] | None = None,
) -> Counter[str]:
    """Returns what happened to each row, for the report."""
    results: Counter[str] = Counter()
    rows = client.recent_changes(until, MAX_ROWS, end=since)
    results["listed"] = len(rows)
    for batch in chunks(rows, 500):
        already = set(known([row["revid"] for row in batch])) if known else set()
        for row in batch:
            if row["revid"] in already:
                results["already_present"] += 1
                continue
            edit = edit_from_recentchange(row, hash_key)
            try:
                outcome = attach_diff(client, edit)
            except WikimediaError:
                results[REASON_API_ERROR] += 1
                publisher.publish(
                    EDITS_DLQ, str(edit.rev_id), b'{"reason":"api_error","source":"repair"}'
                )
                continue
            if outcome.edit is None:
                results[outcome.failure_reason or REASON_API_ERROR] += 1
                continue
            publisher.publish(EDITS_RAW, str(edit.rev_id), outcome.edit.model_dump_json().encode())
            results["published"] += 1
        publisher.flush()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", type=datetime.fromisoformat, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    since = args.since.astimezone(UTC)
    until = args.until.astimezone(UTC)
    if since >= until:
        print("--since must be before --until")
        return 2

    class Settings(WikimediaSettings, KafkaSettings, PrivacySettings):
        pass

    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        print(error)
        return 2

    database_url = os.environ.get("DATABASE_URL")
    known = known_rev_ids_from_postgres(database_url) if database_url else None
    ensure_topics(settings.kafka_bootstrap_servers)
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    results = repair(
        client,
        KafkaPublisher(settings.kafka_bootstrap_servers),
        since,
        until,
        settings.user_hash_key.get_secret_value(),
        known,
    )
    client.close()
    print(dict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
