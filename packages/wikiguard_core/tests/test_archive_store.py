"""The archive on an object store: the same functions, with no rename and no local folders.

pyarrow's in-memory filesystem stands in for the bucket, so these run with no network. A real
S3-compatible server was checked by hand (docs/metrics.md, section 19).
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.fs as pafs
import pytest
from pydantic import SecretStr
from wikiguard_core.archive import (
    EDITS,
    LABELS,
    Archive,
    ArchiveConfigError,
    delete_older_than,
    delete_reference,
    files_between,
    mirror,
    open_archive,
    open_bucket,
    purge_references,
    purge_revisions,
    read_reference,
    read_rows,
    reference_versions,
    restore_from_mirror,
    size_in_bytes,
    write_reference,
    write_rows,
)
from wikiguard_core.config import ArchiveSettings
from wikiguard_core.snapshot import SNAPSHOT_SCHEMA


def bucket() -> Archive:
    return Archive(pafs._MockFileSystem(), "wikiguard-archive/archive", local=False)


def edit_row(rev_id: int, day: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "title": f"Page {rev_id}",
        "timestamp": datetime(2026, 11, day, 12, tzinfo=UTC),
        "is_minor": False,
        "is_anonymous": True,
        "user_hash": "h",
        "comment": "c",
        "old_length": 10,
        "new_length": 20,
        "added_text": "added",
        "removed_text": "",
        "user_edit_count": None,
        "user_hidden": False,
        "comment_hidden": False,
        "content_hidden": False,
    }


def reference_row(rev_id: int) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(SNAPSHOT_SCHEMA.names)
    row.update(rev_id=rev_id, score=0.5)
    return row


def test_rows_round_trip_through_the_bucket() -> None:
    archive = bucket()
    written = write_rows(archive, EDITS, [edit_row(1, 3), edit_row(2, 4)])
    assert len(written) == 2
    prefix = "wikiguard-archive/archive/edits/date=2026-11-0"
    assert all(path.startswith(prefix) for path in written)
    assert sorted(row["rev_id"] for row in read_rows(archive, EDITS)) == [1, 2]
    assert size_in_bytes(archive) > 0


def test_no_temporary_name_is_written_to_a_bucket() -> None:
    archive = bucket()
    write_rows(archive, EDITS, [edit_row(1, 3)])
    everything = archive.fs.get_file_info(pafs.FileSelector(archive.base, recursive=True))
    names = [info.path for info in everything if info.type == pafs.FileType.File]
    assert names and not [name for name in names if name.endswith(".tmp")]


def test_date_filter_and_cleanup_work_on_prefixes() -> None:
    archive = bucket()
    write_rows(archive, EDITS, [edit_row(1, 1), edit_row(2, 5), edit_row(3, 9)])
    assert len(files_between(archive, EDITS, since=date(2026, 11, 4))) == 2
    assert delete_older_than(archive, date(2026, 11, 5)) == 1
    assert sorted(row["rev_id"] for row in read_rows(archive, EDITS)) == [2, 3]


def test_a_purge_rewrites_the_object_in_place() -> None:
    archive = bucket()
    write_rows(archive, EDITS, [edit_row(1, 3), edit_row(2, 3)])
    assert purge_revisions(archive, {2}) == 1
    rows = {row["rev_id"]: row for row in read_rows(archive, EDITS)}
    assert rows[2]["added_text"] is None and rows[1]["added_text"] == "added"


def test_reference_samples_live_in_the_bucket_too() -> None:
    archive = bucket()
    write_reference(archive, "7", [reference_row(1), reference_row(2)], {"version": "7"})
    assert reference_versions(archive) == ["7"]
    found = read_reference(archive, "7")
    assert found is not None and found[1] == {"version": "7"} and found[0].num_rows == 2
    assert purge_references(archive, {1}) == 1
    delete_reference(archive, "7")
    assert reference_versions(archive) == [] and read_reference(archive, "7") is None


def test_an_empty_bucket_reads_as_empty() -> None:
    archive = bucket()
    assert read_rows(archive, LABELS) == []
    assert reference_versions(archive) == []
    assert delete_older_than(archive, date(2026, 11, 5)) == 0


def test_the_working_copy_is_a_folder_by_default_even_with_a_bucket(tmp_path: Path) -> None:
    settings = ArchiveSettings(
        archive_path=str(tmp_path), archive_s3_bucket="wikiguard-archive", _env_file=None
    )
    archive = open_archive(settings)
    assert archive.local and archive.base == tmp_path.resolve().as_posix()


def test_s3_as_the_working_copy_needs_a_bucket() -> None:
    with pytest.raises(ArchiveConfigError, match="ARCHIVE_S3_BUCKET"):
        open_archive(ArchiveSettings(archive_store="s3", _env_file=None))


def test_a_bucket_without_its_keys_is_refused_by_name() -> None:
    settings = ArchiveSettings(
        archive_store="s3",
        archive_s3_bucket="wikiguard-archive",
        archive_s3_endpoint="https://example.invalid",
        archive_s3_secret_key=SecretStr("not-shown-anywhere"),
        _env_file=None,
    )
    with pytest.raises(ArchiveConfigError) as error:
        open_archive(settings)
    assert "ARCHIVE_S3_REGION" in str(error.value)
    assert "ARCHIVE_S3_ACCESS_KEY" in str(error.value)
    assert "not-shown-anywhere" not in str(error.value)


def test_a_bucket_uses_path_style_and_the_prefix() -> None:
    settings = ArchiveSettings(
        archive_store="s3",
        archive_path="/archive/",
        archive_s3_bucket="wikiguard-archive",
        archive_s3_endpoint="http://127.0.0.1:9",
        archive_s3_region="eu-frankfurt-1",
        archive_s3_access_key=SecretStr("a"),
        archive_s3_secret_key=SecretStr("b"),
        _env_file=None,
    )
    archive = open_archive(settings)
    assert not archive.local
    assert archive.base == "wikiguard-archive/archive"
    assert isinstance(archive.fs, pafs.S3FileSystem)
    assert archive.fs.region == "eu-frankfurt-1"


# --- The nightly mirror (DEC-083) ---

TODAY = date(2026, 11, 12)


def test_the_mirror_copies_new_files_once(tmp_path: Path) -> None:
    local, remote = Archive.at(tmp_path), bucket()
    write_rows(local, EDITS, [edit_row(1, 10), edit_row(2, 11)])
    assert mirror(local, remote, 30, TODAY)["copied"] == 2
    assert mirror(local, remote, 30, TODAY)["copied"] == 0
    assert sorted(row["rev_id"] for row in read_rows(remote, EDITS)) == [1, 2]


def test_a_purge_reaches_the_mirror(tmp_path: Path) -> None:
    local, remote = Archive.at(tmp_path), bucket()
    write_rows(local, EDITS, [edit_row(1, 10), edit_row(2, 10)])
    mirror(local, remote, 30, TODAY)
    purge_revisions(local, {2})
    assert mirror(local, remote, 30, TODAY)["copied"] == 1
    rows = {row["rev_id"]: row for row in read_rows(remote, EDITS)}
    assert rows[2]["added_text"] is None


def test_an_empty_working_copy_never_empties_the_mirror(tmp_path: Path) -> None:
    """After a VM is rebuilt, its disk is empty, and the mirror is what brings the data back."""
    local, remote = Archive.at(tmp_path / "old"), bucket()
    write_rows(local, EDITS, [edit_row(1, 10)])
    write_reference(local, "7", [reference_row(1)], {"version": "7"})
    mirror(local, remote, 30, TODAY)
    rebuilt = Archive.at(tmp_path / "new")
    result = mirror(rebuilt, remote, 30, TODAY)
    assert result == {"copied": 0, "expired_folders": 0, "dropped_references": 0}
    assert len(read_rows(remote, EDITS)) == 1 and reference_versions(remote) == ["7"]
    assert restore_from_mirror(remote, rebuilt) == 3
    assert len(read_rows(rebuilt, EDITS)) == 1 and reference_versions(rebuilt) == ["7"]


def test_the_mirror_drops_expired_days_and_retired_references(tmp_path: Path) -> None:
    local, remote = Archive.at(tmp_path), bucket()
    write_rows(local, EDITS, [edit_row(1, 1), edit_row(2, 11)])
    write_reference(local, "6", [reference_row(1)], {"version": "6"})
    write_reference(local, "7", [reference_row(1)], {"version": "7"})
    mirror(local, remote, 30, TODAY)
    delete_reference(local, "6")
    result = mirror(local, remote, 5, TODAY)
    assert result["expired_folders"] == 1 and result["dropped_references"] == 1
    assert [row["rev_id"] for row in read_rows(remote, EDITS)] == [2]
    assert reference_versions(remote) == ["7"]


def test_no_bucket_means_no_mirror() -> None:
    assert open_bucket(ArchiveSettings(_env_file=None), "archive") is None
