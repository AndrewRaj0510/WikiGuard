"""The label rule, in one place so the backfill and the labeler can't disagree.

1 if the revision has the mw-reverted tag within 48 hours, 0 if it's still untagged after that,
and no label before then. "Reverted" includes some good-faith mistakes, so the model predicts
reverts, which is close to vandalism but not the same.
"""

from collections.abc import Collection
from datetime import UTC, datetime, timedelta

REVERTED_TAG = "mw-reverted"
LABEL_WAIT = timedelta(hours=48)


def decide_label(
    edited_at: datetime, tags: Collection[str], now: datetime | None = None
) -> int | None:
    """1 or 0 once the label is settled, otherwise None."""
    if REVERTED_TAG in tags:
        return 1
    now = now or datetime.now(UTC)
    return 0 if now - edited_at >= LABEL_WAIT else None
