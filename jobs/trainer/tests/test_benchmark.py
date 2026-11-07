"""The benchmark's own arithmetic and row loading. Needs no model and no snapshot."""

import json
from pathlib import Path

from trainer.benchmark import benchmark_rows, summarize


def test_a_saved_sample_is_read_without_touching_a_snapshot(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps([{"rev_id": 1}, {"rev_id": 2}, {"rev_id": 3}]), "utf-8")

    assert [row["rev_id"] for row in benchmark_rows(sample, count=2)] == [1, 2]


def test_summary_reports_the_slow_end() -> None:
    summary = summarize([float(value) for value in range(1, 101)])

    assert summary["max_ms"] == 100.0
    assert 50.0 <= summary["p50_ms"] <= 51.0
    assert 95.0 <= summary["p95_ms"] <= 96.0
    assert 99.0 <= summary["p99_ms"] <= 100.0
