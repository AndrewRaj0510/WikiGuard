from datetime import UTC, datetime, timedelta
from typing import Any

from plot_snapshot import daily_positive_rates, split_chart
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import Candidate, build_snapshot

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
END = AS_OF - timedelta(hours=48)
SECRET_TEXT = "text-that-must-not-reach-an-image"


def candidate(rev_id: int, at: datetime, label: int) -> Candidate:
    values: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": None,
        "title": "Secret Page",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": "secret-hash",
        "comment": None,
        "old_length": 1,
        "new_length": 2,
        "added_text": f"{SECRET_TEXT} {rev_id}",
    }
    return Candidate(rev_id, EditEvent(**values), label)


def test_daily_rates_group_by_day_in_order() -> None:
    rows = [
        {"timestamp": datetime(2026, 11, 2, 23, tzinfo=UTC), "label": 1},
        {"timestamp": datetime(2026, 11, 1, 1, tzinfo=UTC), "label": 0},
        {"timestamp": datetime(2026, 11, 2, 1, tzinfo=UTC), "label": 0},
    ]

    assert daily_positive_rates(rows) == [("2026-11-01", 0.0), ("2026-11-02", 0.5)]


def test_chart_shows_the_manifest_numbers_and_no_row_content() -> None:
    snapshot = build_snapshot(
        [
            candidate(1, END - timedelta(days=20), 1),
            candidate(2, END - timedelta(days=10), 0),
            candidate(3, END - timedelta(days=1), 0),
        ],
        AS_OF,
        "synthetic",
    )

    svg = split_chart(snapshot.rows, snapshot.manifest).svg

    assert "3 edits, split by time alone" in svg
    assert "100.00% reverted" in svg
    assert SECRET_TEXT not in svg
    assert "Secret Page" not in svg
    assert "secret-hash" not in svg
