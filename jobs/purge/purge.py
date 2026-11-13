"""Daily: remove our copy of anything Wikipedia has since hidden, usually personal information.

Two checks, because neither sees everything:
- The public deletion log finds revision deletions across the whole 30-day archive cheaply.
- Re-checking recent edits directly also catches suppression, whose log is private, and whole
  pages that were deleted.

Only what Wikipedia hid gets cleared, and only the identity of each revision is logged. Rows stay
so counts still add up. Phase 2's snapshots read the same audit entries (F-010).
"""

import logging
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from wikiguard_core.archive import Root, open_archive, purge_references, purge_revisions
from wikiguard_core.config import (
    ArchiveSettings,
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    WikimediaSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.wikimedia import RevisionState, WikimediaClient

logger = logging.getLogger("purge")

RECHECK_WINDOW = timedelta(days=7)
# Longer than the daily schedule, so one failed run loses nothing.
LOG_LOOKBACK = timedelta(days=3)


class Settings(CommonSettings, DatabaseSettings, WikimediaSettings, ArchiveSettings):
    pass


def recent_edits_with_content(connection: psycopg.Connection[Any], now: datetime) -> list[int]:
    rows = connection.execute(
        """
        SELECT rev_id FROM edits
        WHERE purged_at IS NULL AND edited_at >= %s
          AND (added_text IS NOT NULL OR removed_text IS NOT NULL
               OR comment IS NOT NULL OR user_hash IS NOT NULL)
        """,
        (now - RECHECK_WINDOW,),
    ).fetchall()
    return [row[0] for row in rows]


def clear_revision(
    connection: psycopg.Connection[Any], state: RevisionState, now: datetime
) -> bool:
    """Clear what was hidden, and return whether this revision was in the database."""
    gone = state.missing
    updated = connection.execute(
        """
        UPDATE edits SET
            added_text = CASE WHEN %(content)s THEN NULL ELSE added_text END,
            removed_text = CASE WHEN %(content)s THEN NULL ELSE removed_text END,
            comment = CASE WHEN %(comment)s THEN NULL ELSE comment END,
            user_hash = CASE WHEN %(user)s THEN NULL ELSE user_hash END,
            content_hidden = content_hidden OR %(content)s,
            comment_hidden = comment_hidden OR %(comment)s,
            user_hidden = user_hidden OR %(user)s,
            purged_at = %(now)s
        WHERE rev_id = %(rev_id)s
        """,
        {
            "content": gone or state.content_hidden,
            "comment": gone or state.comment_hidden,
            "user": gone or state.user_hidden,
            "now": now,
            "rev_id": state.rev_id,
        },
    ).rowcount
    # An explanation quotes the text it explains, so any hidden part invalidates it.
    connection.execute("DELETE FROM explanations WHERE rev_id = %s", (state.rev_id,))
    return bool(updated)


def hidden_states(client: WikimediaClient, rev_ids: Iterable[int]) -> list[RevisionState]:
    return [state for state in client.revision_states(rev_ids).values() if state.has_hidden_content]


def run(
    connection: psycopg.Connection[Any],
    client: WikimediaClient,
    archive_root: Root,
    now: datetime | None = None,
) -> Counter[str]:
    now = now or datetime.now(UTC)
    results: Counter[str] = Counter()
    candidates = set(recent_edits_with_content(connection, now))
    results["rechecked"] = len(candidates)
    logged = client.revision_deletion_ids(now - LOG_LOOKBACK)
    results["in_deletion_log"] = len(logged)

    states = hidden_states(client, candidates | logged)
    results["hidden"] = len(states)
    for state in states:
        results["cleared_in_database"] += clear_revision(connection, state, now)

    # Content in the archive goes too, whichever check found it.
    purged = sorted(state.rev_id for state in states)
    results["cleared_in_archive"] = purge_revisions(archive_root, set(purged))
    # Drift reference samples hold edit text too, and outlive the 30-day archive.
    results["removed_from_references"] = purge_references(archive_root, set(purged))
    if purged:
        connection.execute(
            "INSERT INTO audit_log (actor, action, details) VALUES ('purge', 'purge.hidden', %s)",
            (Jsonb({"rev_ids": purged}),),
        )
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

    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        results = run(connection, client, open_archive(settings))
    client.close()
    logger.info("Purge finished: %s", dict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
