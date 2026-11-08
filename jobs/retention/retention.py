"""Daily: keep Supabase under its 500 MB limit, shortening the windows as it fills.

Going over the limit can put the free project into read-only mode, so this acts before then. The
archive holds the last 30 days anyway, so what's removed here can still be rebuilt. Reviews,
rollups, model versions, and the audit log are small and stay forever.

Normal rules (docs/capacity.md has the measurements behind them):
- Diff text: 48 hours, or 3 days for high-risk edits, which reviewers may still open.
- Edits, scores, and labels: 3 days, which is also how far back the review queue goes (DEC-038).

Edits under 48 hours old are never touched, because they're still waiting for labels. Normal
rules leave a day for the labeler to catch up after an outage, and the emergency level 12 hours.
"""

import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from wikiguard_core.config import (
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter

logger = logging.getLogger("retention")

LIMIT_BYTES = 500 * 1024 * 1024
WARN_AT = 0.70
SHRINK_AT = 0.85
EMERGENCY_AT = 0.95
# The top 5% of RoBERTa's calibrated test scores start here (docs/metrics.md, section 8). 0.8 was
# the earlier estimate, and it only caught 0.89% of edits. Revisit it if the champion changes.
HIGH_RISK_SCORE = 0.18
# Ended sessions only matter for the guest wait and for tracing, so they don't need to stay long.
SESSION_RETENTION = timedelta(days=7)


@dataclass(frozen=True)
class Windows:
    level: str
    text: timedelta
    high_risk_text: timedelta
    rows: timedelta


NORMAL = Windows("normal", timedelta(hours=48), timedelta(days=3), timedelta(days=3))
WARNING = Windows("warning", timedelta(hours=48), timedelta(days=3), timedelta(days=3))
SHRINK = Windows("shrink", timedelta(hours=48), timedelta(hours=48), timedelta(days=3))
EMERGENCY = Windows("emergency", timedelta(hours=48), timedelta(hours=48), timedelta(hours=60))


class Settings(CommonSettings, DatabaseSettings):
    pass


def windows_for(size_bytes: int, limit: int = LIMIT_BYTES) -> Windows:
    share = size_bytes / limit
    if share >= EMERGENCY_AT:
        return EMERGENCY
    if share >= SHRINK_AT:
        return SHRINK
    if share >= WARN_AT:
        return WARNING
    return NORMAL


def database_size(connection: psycopg.Connection[Any]) -> int:
    row = connection.execute("SELECT pg_database_size(current_database())").fetchone()
    return int(row[0]) if row else 0


def apply(connection: psycopg.Connection[Any], windows: Windows, now: datetime) -> dict[str, int]:
    text_cleared = connection.execute(
        """
        UPDATE edits e SET added_text = NULL, removed_text = NULL
        WHERE (e.added_text IS NOT NULL OR e.removed_text IS NOT NULL)
          AND e.edited_at < %s
          AND (e.edited_at < %s OR NOT EXISTS (
                SELECT 1 FROM scores s WHERE s.rev_id = e.rev_id AND s.score >= %s))
        """,
        (now - windows.text, now - windows.high_risk_text, HIGH_RISK_SCORE),
    ).rowcount
    # Scores, labels, and explanations go with their edit through ON DELETE CASCADE.
    rows_deleted = connection.execute(
        "DELETE FROM edits WHERE edited_at < %s", (now - windows.rows,)
    ).rowcount
    sessions_deleted = connection.execute(
        "DELETE FROM sessions WHERE created_at < %s AND ended_at IS NOT NULL",
        (now - SESSION_RETENTION,),
    ).rowcount
    return {
        "text_cleared": text_cleared,
        "rows_deleted": rows_deleted,
        "sessions_deleted": sessions_deleted,
    }


def run(
    connection: psycopg.Connection[Any], now: datetime | None = None, limit: int = LIMIT_BYTES
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    size = database_size(connection)
    windows = windows_for(size, limit)
    if windows is not NORMAL:
        # Phase 7 routes this to an alert. The log line is the signal until then.
        logger.warning("Database at %.0f%% of its limit: %s", 100 * size / limit, windows.level)
    results: dict[str, Any] = {
        "level": windows.level,
        "size_bytes": size,
        **apply(connection, windows, now),
    }
    connection.execute(
        "INSERT INTO audit_log (actor, action, details) VALUES ('retention', 'retention.run', %s)",
        (Jsonb({**results, "windows": {k: str(v) for k, v in asdict(windows).items()}}),),
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

    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        logger.info("Retention finished: %s", run(connection))
    return 0


if __name__ == "__main__":
    sys.exit(main())
