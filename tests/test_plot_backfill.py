from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import plot_backfill
import pytest
from measure_capacity import LIMIT_BYTES, project
from plot_backfill import (
    MEGABYTE,
    daily_rates,
    positive_rate_chart,
    retention_chart,
    retention_sizes,
    revert_delay_chart,
    share_within,
)

SIZES = {
    "edit_row_with_text_bytes": 600,
    "edit_row_without_text_bytes": 300,
    "score_row_bytes": 200,
    "label_row_bytes": 100,
    "edits_per_day": 1000,
}


def test_daily_rates_leave_out_edits_with_no_label() -> None:
    day_one = datetime(2026, 10, 6, 5, tzinfo=UTC)
    day_two = datetime(2026, 10, 7, 5, tzinfo=UTC)
    rows = [
        {"timestamp": day_one, "label": 1},
        {"timestamp": day_one, "label": 0},
        {"timestamp": day_one, "label": None},
        {"timestamp": day_two, "label": 0},
    ]
    assert daily_rates(rows) == [("2026-10-06", 0.5, 2), ("2026-10-07", 0.0, 1)]


def test_share_within_includes_the_limit() -> None:
    assert share_within([0.5, 1.0, 48.0, 49.0], 48) == 0.75


def test_longer_retention_costs_more_and_three_days_matches_the_default() -> None:
    sizes = retention_sizes(SIZES, 300)
    assert [size for _, size in sizes] == sorted((size for _, size in sizes), reverse=True)
    assert sizes[-1][1] == project(SIZES, 1000, 300)["steady_state_bytes_normal_rules"] / MEGABYTE


def test_project_counts_every_row_for_its_whole_stay() -> None:
    # 2 days with text, then 1 day with 5% still holding it, plus a score and label for all 3.
    steady = project(SIZES, 1000, 300, 3)["steady_state_bytes_normal_rules"]
    assert steady == 1000 * 2 * 600 + 1000 * 0.05 * 600 + 1000 * 0.95 * 300 + 1000 * 3 * 300


def test_titles_come_from_the_data() -> None:
    days = [("2026-10-06", 0.05, 10), ("2026-10-07", 0.07, 10)]
    assert "6.0% of backfilled edits" in positive_rate_chart(days, 0.06).svg
    assert "25.0% miss the 48-hour cutoff" in revert_delay_chart([0.5, 1.0, 2.0, 50.0]).svg
    share = 100 * MEGABYTE / LIMIT_BYTES
    assert f"{share:.0%} of Supabase" in retention_chart([("a", 300.0), ("b", 100.0)]).svg


def test_chart_text_is_escaped() -> None:
    assert "<script>" not in retention_chart([("<script>", 100.0), ("b", 50.0)]).svg


def test_browser_that_writes_no_image_is_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(plot_backfill.subprocess, "run", lambda *a, **k: CompletedProcess(a, 0))
    with pytest.raises(SystemExit, match="wrote no image"):
        plot_backfill.render(tmp_path / "a.svg", tmp_path / "a.png", "browser")
