from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import (
    ROWS_FILE,
    Candidate,
    Snapshot,
    build_snapshot,
    content_hash,
    write_snapshot,
)
from wikiguard_core.validation import (
    FAIL,
    MIN_TEST_POSITIVES,
    DataValidationError,
    population_stability_index,
    validate_rows,
    validate_snapshot,
)

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
END = AS_OF - timedelta(hours=48)


def edit(rev_id: int, at: datetime, text: str) -> EditEvent:
    values: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": None,
        "title": "Page",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": rev_id % 3 == 0,
        "user_hash": f"user-{rev_id}",
        "comment": None,
        "old_length": 100,
        "new_length": 100 + rev_id % 50,
        "added_text": text,
    }
    return EditEvent(**values)


def candidates(positives_in_test: int = MIN_TEST_POSITIVES) -> list[Candidate]:
    """Spread across all three windows, about 10% positive, with unique text."""
    rows = []
    rev_id = 0
    for days, count in ((20, 500), (10, 500), (1, 5000)):
        for index in range(count):
            rev_id += 1
            at = END - timedelta(days=days) + timedelta(seconds=index)
            if days == 1:
                # Keeps each test day's rate near the mean, whatever the positive count.
                label = int(index < positives_in_test)
                at = END - timedelta(days=1 + index % 3) + timedelta(seconds=index)
            else:
                label = int(index % 10 == 0)
            rows.append(Candidate(rev_id, edit(rev_id, at, f"text number {rev_id}"), label))
    return rows


@lru_cache(maxsize=1)
def built_snapshot() -> Snapshot:
    return build_snapshot(candidates(800), AS_OF, "synthetic")


def snapshot_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copies, so a test can damage them without affecting the next one."""
    snapshot = built_snapshot()
    return [dict(row) for row in snapshot.rows], dict(snapshot.manifest)


def names_failed(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> set[str]:
    manifest = dict(manifest)
    manifest["content_sha256"] = content_hash(rows)
    return {check.name for check in validate_rows(rows, manifest).failures}


def test_a_clean_snapshot_passes() -> None:
    rows, manifest = snapshot_rows()

    report = validate_rows(rows, manifest)

    assert report.failures == []
    report.require_passed()
    assert report.numbers["duplicate_overlap"]["train_rows_with_text_in_later_splits"] == 0


def test_changed_rows_fail_the_hash_check() -> None:
    rows, manifest = snapshot_rows()
    rows[0]["label"] = 1 - rows[0]["label"]

    report = validate_rows(rows, manifest)

    assert "content_hash" in {check.name for check in report.failures}
    with pytest.raises(DataValidationError, match="content_hash"):
        report.require_passed()


def test_repeated_revision_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[1] = dict(rows[1], rev_id=rows[0]["rev_id"])

    assert "unique_rev_id" in names_failed(rows, manifest)


def test_null_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[0] = dict(rows[0], comment=None)

    assert "nulls" in names_failed(rows, manifest)


def test_row_outside_its_window_fails() -> None:
    rows, manifest = snapshot_rows()
    test_row = next(row for row in rows if row["split"] == "test")
    test_row["timestamp"] = END - timedelta(days=30)

    assert "split_integrity" in names_failed(rows, manifest)


def test_row_too_young_for_a_final_label_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[0]["timestamp"] = AS_OF

    assert "label_maturity" in names_failed(rows, manifest)


def test_missing_label_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[0]["label"] = 2

    assert "label_maturity" in names_failed(rows, manifest)


def test_feature_out_of_range_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[0]["uppercase_ratio"] = 1.5

    assert "feature_ranges" in names_failed(rows, manifest)


def test_feature_that_does_not_match_its_text_fails() -> None:
    rows, manifest = snapshot_rows()
    rows[0]["chars_added"] += 1

    assert "derived_features" in names_failed(rows, manifest)


def test_label_poisoning_is_rejected() -> None:
    """Coordinated fake reverts on one day push its rate past 2x the mean."""
    rows, manifest = snapshot_rows()
    day = next(row["timestamp"].date() for row in rows if row["split"] == "train")
    for row in rows:
        if row["timestamp"].date() == day:
            row["label"] = 1

    assert "positive_rate" in names_failed(rows, manifest)


def test_too_few_test_positives_fails() -> None:
    snapshot = build_snapshot(candidates(MIN_TEST_POSITIVES - 1), AS_OF, "synthetic")

    report = validate_rows(snapshot.rows, snapshot.manifest)

    assert "test_positives" in {check.name for check in report.failures}


def test_train_text_repeated_later_fails() -> None:
    rows, manifest = snapshot_rows()
    train_row = next(row for row in rows if row["split"] == "train")
    later = next(row for row in rows if row["split"] == "test")
    train_row["added_text"] = later["added_text"]
    train_row["chars_added"] = later["chars_added"]

    assert "duplicates" in names_failed(rows, manifest)


def test_high_exclusion_rate_fails() -> None:
    rows, manifest = snapshot_rows()
    manifest = dict(manifest, excluded={"hidden": len(rows)})

    assert "excluded_rate" in names_failed(rows, manifest)


def test_a_feature_that_never_varies_is_a_warning_not_a_failure() -> None:
    rows, manifest = snapshot_rows()

    report = validate_rows(rows, manifest)

    assert "user_edit_count_bucket" in report.numbers["constant_features"]
    constant = next(check for check in report.checks if check.name == "constant_features")
    assert constant.status != FAIL


def test_psi_is_zero_for_the_same_data_and_large_for_a_shift() -> None:
    values = [float(index) for index in range(1000)]

    assert population_stability_index(values, values) == pytest.approx(0.0)
    assert population_stability_index(values, [value + 800 for value in values]) > 0.25


def test_extra_column_in_the_file_fails(tmp_path: Path) -> None:
    folder = write_snapshot(built_snapshot(), tmp_path)
    table = pq.read_table(folder / ROWS_FILE)
    with_title = table.append_column("title", pa.array(["Page"] * table.num_rows))
    pq.write_table(with_title, folder / ROWS_FILE)

    report = validate_snapshot(folder)

    assert "schema" in {check.name for check in report.failures}
    assert "title" in next(check.detail for check in report.failures if check.name == "schema")


def test_a_written_snapshot_validates(tmp_path: Path) -> None:
    folder = write_snapshot(built_snapshot(), tmp_path)

    assert validate_snapshot(folder).failures == []
