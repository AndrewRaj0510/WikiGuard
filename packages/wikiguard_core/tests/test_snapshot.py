"""Snapshot rules, on synthetic edits where every right answer is known in advance."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from wikiguard_core.features import FEATURE_VERSION
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import (
    FEATURE_COLUMNS,
    MANIFEST_FILE,
    ROWS_FILE,
    SNAPSHOT_SCHEMA,
    TEST,
    TRAIN,
    VALIDATION,
    Candidate,
    Snapshot,
    SnapshotError,
    SplitWindows,
    build_snapshot,
    content_hash,
    purge_snapshot,
    read_snapshot,
    verify_snapshot,
    write_snapshot,
)

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
# Newest settled moment: as_of minus the 48-hour label wait.
END = AS_OF - timedelta(hours=48)
TEST_START = END - timedelta(days=7)
VALIDATION_START = TEST_START - timedelta(days=7)
OLD = VALIDATION_START - timedelta(days=1)


def candidate(
    rev_id: int,
    at: datetime,
    label: int | None = 0,
    text: str | None = None,
    **edit_fields: Any,
) -> Candidate:
    values: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "title": f"Page {rev_id}",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": f"hash-{rev_id}",
        "comment": "fix",
        "old_length": 100,
        "new_length": 110,
        "added_text": f"unique text {rev_id}" if text is None else text,
        **edit_fields,
    }
    return Candidate(rev_id, EditEvent(**values), label)


def one_per_split() -> list[Candidate]:
    return [
        candidate(1, VALIDATION_START - timedelta(hours=1), 1),
        candidate(2, TEST_START - timedelta(hours=1), 0),
        candidate(3, END - timedelta(hours=1), 1),
    ]


def build(candidates: list[Candidate]) -> Snapshot:
    return build_snapshot(candidates, AS_OF, "synthetic")


def split_by_rev(snapshot: Snapshot) -> dict[int, str]:
    return {row["rev_id"]: row["split"] for row in snapshot.rows}


def test_windows_put_each_boundary_on_the_older_side() -> None:
    windows = SplitWindows.from_as_of(AS_OF)

    assert windows.split_of(VALIDATION_START) == TRAIN
    assert windows.split_of(VALIDATION_START + timedelta(seconds=1)) == VALIDATION
    assert windows.split_of(TEST_START) == VALIDATION
    assert windows.split_of(TEST_START + timedelta(seconds=1)) == TEST
    assert windows.split_of(END) == TEST
    assert windows.split_of(END + timedelta(seconds=1)) is None


def test_windows_need_a_time_zone() -> None:
    with pytest.raises(SnapshotError, match="time zone"):
        SplitWindows.from_as_of(datetime(2026, 11, 20))


def test_rows_land_in_the_split_their_time_says() -> None:
    assert split_by_rev(build(one_per_split())) == {1: TRAIN, 2: VALIDATION, 3: TEST}


def test_unsettled_edit_is_counted_and_left_out() -> None:
    snapshot = build([*one_per_split(), candidate(4, END + timedelta(minutes=1), 1)])

    assert 4 not in split_by_rev(snapshot)
    assert snapshot.manifest["excluded"] == {"unsettled": 1}


def test_excluded_rows_are_counted_by_reason() -> None:
    snapshot = build(
        [
            *one_per_split(),
            Candidate(10, None, None, "hidden"),
            Candidate(11, None, None, "hidden"),
            Candidate(12, None, None, "revert_not_dated"),
            candidate(13, END - timedelta(days=1), None),
        ]
    )

    assert snapshot.manifest["excluded"] == {"hidden": 2, "no_label": 1, "revert_not_dated": 1}
    assert snapshot.manifest["rows"] == 3


def test_split_stats_match_the_rows() -> None:
    splits = build(one_per_split()).manifest["splits"]

    assert splits[TRAIN]["rows"] == 1
    assert splits[TRAIN]["positives"] == 1
    assert splits[TRAIN]["positive_rate"] == 1.0
    assert splits[VALIDATION]["positives"] == 0
    assert splits[TEST]["rows"] == 1


def test_train_row_with_text_from_a_later_split_is_removed() -> None:
    snapshot = build(
        [
            *one_per_split(),
            candidate(4, OLD, 1, text="same as test"),
            candidate(5, OLD, 0, text="same as validation"),
            candidate(6, END - timedelta(hours=2), 1, text="same as test"),
            candidate(7, TEST_START - timedelta(hours=2), 0, text="same as validation"),
        ]
    )

    assert not {4, 5} & split_by_rev(snapshot).keys()
    # The windows being measured never lose rows.
    assert {6, 7} <= split_by_rev(snapshot).keys()
    report = snapshot.manifest["duplicates"]
    assert report["removed_from_train"] == 2
    assert report["positives_removed_from_train"] == 1
    assert report["test_rows_with_text_in_original_train"] == 1
    assert report["validation_rows_with_text_in_original_train"] == 1


def test_repeats_inside_one_split_are_kept_and_reported() -> None:
    snapshot = build(
        [
            *one_per_split(),
            candidate(4, OLD, text="repeat"),
            candidate(5, OLD, text="repeat"),
            candidate(6, END - timedelta(hours=2), text="test repeat"),
            candidate(7, END - timedelta(hours=3), text="test repeat"),
        ]
    )

    assert {4, 5, 6, 7} <= split_by_rev(snapshot).keys()
    assert snapshot.manifest["duplicates"]["repeated_rows_left_inside_train"] == 1


def test_edits_with_no_changed_text_are_not_treated_as_duplicates() -> None:
    snapshot = build(
        [
            *one_per_split(),
            candidate(4, OLD, text=""),
            candidate(5, END - timedelta(hours=2), text=""),
        ]
    )

    assert {4, 5} <= split_by_rev(snapshot).keys()
    assert snapshot.manifest["duplicates"]["removed_from_train"] == 0


def test_a_hidden_character_cannot_disguise_a_duplicate() -> None:
    snapshot = build(
        [
            *one_per_split(),
            candidate(4, OLD, text="vandal​text"),
            candidate(5, END - timedelta(hours=2), text="vandaltext"),
        ]
    )

    assert 4 not in split_by_rev(snapshot)


def test_overlap_reports_shared_pages_and_editors_with_train() -> None:
    recent = END - timedelta(hours=2)
    overlap = build(
        [
            candidate(1, OLD, title="Moon", user_hash="a"),
            candidate(2, TEST_START - timedelta(hours=1)),
            candidate(3, recent, title="Moon", user_hash="b"),
            candidate(4, recent, title="Sun", user_hash="a"),
            candidate(5, recent, title="Mars", user_hash="c"),
            candidate(6, recent, title="Moon", user_hash=None),
        ]
    ).manifest["overlap"]

    assert overlap == {"test_rows_on_train_pages": 0.5, "test_rows_by_train_editors": 0.25}


def test_manifest_carries_the_wikipedia_licence() -> None:
    assert "CC BY-SA 4.0" in build(one_per_split()).manifest["notice"]


def test_a_split_with_no_rows_is_refused() -> None:
    with pytest.raises(SnapshotError, match="validation"):
        build([candidate(1, VALIDATION_START), candidate(3, END)])


def test_a_repeated_revision_is_refused() -> None:
    with pytest.raises(SnapshotError, match="appears twice"):
        build([*one_per_split(), candidate(1, END)])


def test_an_impossible_label_is_refused() -> None:
    with pytest.raises(SnapshotError, match="label"):
        build([*one_per_split(), candidate(4, END, 2)])


def test_hash_ignores_input_order() -> None:
    forward = build(one_per_split()).manifest["content_sha256"]
    backward = build(list(reversed(one_per_split()))).manifest["content_sha256"]

    assert forward == backward


@pytest.mark.parametrize(
    "change",
    [
        {"added_text": "unique text 1!"},
        {"label": 0},
        {"split": VALIDATION},
        {"bytes_delta": 11},
        {"timestamp": VALIDATION_START},
    ],
)
def test_hash_changes_when_any_value_changes(change: dict[str, Any]) -> None:
    rows = build(one_per_split()).rows
    before = content_hash(rows)

    rows[0] = {**rows[0], **change}

    assert content_hash(rows) != before


def test_schema_holds_exactly_the_features_and_no_identity() -> None:
    assert set(SNAPSHOT_SCHEMA.names) == {*FEATURE_COLUMNS, "rev_id", "timestamp", "split", "label"}
    forbidden = {"user_hash", "title", "tagged_reverted", "tags_at_capture"}
    assert not forbidden & set(SNAPSHOT_SCHEMA.names)


def test_identity_is_not_anywhere_in_the_written_files(tmp_path: Path) -> None:
    candidates = [
        candidate(1, VALIDATION_START, 1, title="SecretPage", user_hash="secret-hash"),
        candidate(2, TEST_START),
        candidate(3, END),
    ]
    folder = write_snapshot(build(candidates), tmp_path)

    written = (folder / ROWS_FILE).read_bytes() + (folder / MANIFEST_FILE).read_bytes()
    assert b"SecretPage" not in written
    assert b"secret-hash" not in written


def test_written_snapshot_reads_back_identical(tmp_path: Path) -> None:
    snapshot = build(one_per_split())

    folder = write_snapshot(snapshot, tmp_path)

    assert folder.name == snapshot.manifest["name"]
    assert folder.name.startswith("snapshot-20261120-")
    assert content_hash(read_snapshot(folder)) == snapshot.manifest["content_sha256"]
    assert read_snapshot(folder) == snapshot.rows


def test_two_builds_of_the_same_data_are_byte_identical(tmp_path: Path) -> None:
    first = write_snapshot(build(one_per_split()), tmp_path / "a")
    second = write_snapshot(build(one_per_split()), tmp_path / "b")

    for name in (ROWS_FILE, MANIFEST_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_writing_an_existing_snapshot_again_changes_nothing(tmp_path: Path) -> None:
    snapshot = build(one_per_split())
    folder = write_snapshot(snapshot, tmp_path)
    before = (folder / ROWS_FILE).stat().st_mtime_ns

    assert write_snapshot(snapshot, tmp_path) == folder
    assert (folder / ROWS_FILE).stat().st_mtime_ns == before


def test_a_changed_row_is_caught_on_read(tmp_path: Path) -> None:
    folder = write_snapshot(build(one_per_split()), tmp_path)
    rows = pq.read_table(folder / ROWS_FILE).to_pylist()
    rows[0]["label"] = 1 - rows[0]["label"]
    pq.write_table(pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA), folder / ROWS_FILE)

    with pytest.raises(SnapshotError, match="hash"):
        read_snapshot(folder)


def test_other_feature_versions_are_refused(tmp_path: Path) -> None:
    folder = write_snapshot(build(one_per_split()), tmp_path)
    manifest = json.loads((folder / MANIFEST_FILE).read_text("utf-8"))
    manifest["feature_version"] = FEATURE_VERSION + 1
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest), "utf-8")

    with pytest.raises(SnapshotError, match="feature_version"):
        read_snapshot(folder)


def test_purge_removes_rows_and_records_how_many_but_not_which(tmp_path: Path) -> None:
    snapshot = build([*one_per_split(), candidate(4, END - timedelta(hours=2), 1)])
    folder = write_snapshot(snapshot, tmp_path)
    at = datetime(2026, 11, 21, tzinfo=UTC)

    assert purge_snapshot(folder, {4, 999}, at) == 1

    manifest = verify_snapshot(folder)
    assert {row["rev_id"] for row in read_snapshot(folder)} == {1, 2, 3}
    assert manifest["rows"] == 3
    assert manifest["splits"][TEST]["rows"] == 1
    assert manifest["original_content_sha256"] == snapshot.manifest["content_sha256"]
    assert manifest["content_sha256"] != snapshot.manifest["content_sha256"]
    assert manifest["purges"] == [
        {
            "at": at.isoformat(),
            "rows_removed": 1,
            "positives_removed": 1,
            "previous_content_sha256": snapshot.manifest["content_sha256"],
        }
    ]


def test_purge_removes_the_text_from_the_file_itself(tmp_path: Path) -> None:
    secret = "personal-information-to-remove"
    snapshot = build([*one_per_split(), candidate(4, END - timedelta(hours=2), text=secret)])
    folder = write_snapshot(snapshot, tmp_path)

    purge_snapshot(folder, {4}, AS_OF)

    # Read the rows, because compression means the raw bytes can't prove absence.
    assert all(row["added_text"] != secret for row in read_snapshot(folder))
    assert not (folder / f"{ROWS_FILE}.tmp").exists()


def test_purge_of_nothing_leaves_the_files_alone(tmp_path: Path) -> None:
    folder = write_snapshot(build(one_per_split()), tmp_path)
    before = (folder / ROWS_FILE).read_bytes()

    assert purge_snapshot(folder, {999}, AS_OF) == 0

    assert (folder / ROWS_FILE).read_bytes() == before
    assert verify_snapshot(folder)["purges"] == []


def test_edit_event_refuses_a_revert_tag() -> None:
    """Snapshot rows are built from EditEvents, so the raw tag has no way in."""
    with pytest.raises(ValueError, match="tagged_reverted"):
        EditEvent(
            rev_id=1,
            parent_rev_id=None,
            title="Moon",
            timestamp=AS_OF,
            is_minor=False,
            is_anonymous=False,
            user_hash=None,
            comment=None,
            old_length=None,
            new_length=None,
            tagged_reverted=1,  # type: ignore[call-arg]
        )
