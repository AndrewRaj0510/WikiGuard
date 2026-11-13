"""The rolling parquet archive: where retraining and monitoring read history, instead of Supabase.

Reads from Supabase count against its monthly egress, so the archive is written directly from the
topics. Files are laid out as <root>/<dataset>/date=YYYY-MM-DD/part-*.parquet, so deleting old
data just means deleting old folders.

Consumers deliver at least once, so the same row can appear twice. Readers keep the last copy of
each key.

Where it lives (ARCHIVE_STORE, DEC-083): a folder on the cluster's disk, mirrored each night to
Oracle Object Storage through its S3-compatible API. Oracle's free tier allows 50,000 requests a
month, and reading parquet from a bucket takes several requests per file, so the jobs that read
weeks of history read the folder, and only the mirror talks to the bucket. ARCHIVE_STORE=s3 puts the
working copy in the bucket instead. Both go through pyarrow's own filesystem layer, so no extra
package is needed, and every function here takes either a folder or an `Archive`.
"""

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from wikiguard_core.snapshot import SNAPSHOT_SCHEMA

if TYPE_CHECKING:
    from wikiguard_core.config import ArchiveSettings

EDITS = "edits"
SCORES = "scores"
LABELS = "labels"
COMPRESSION = "zstd"
SCHEMAS = {
    EDITS: pa.schema(
        [
            ("schema_version", pa.int32()),
            ("rev_id", pa.int64()),
            ("parent_rev_id", pa.int64()),
            ("title", pa.string()),
            ("timestamp", pa.timestamp("s", tz="UTC")),
            ("is_minor", pa.bool_()),
            ("is_anonymous", pa.bool_()),
            ("user_hash", pa.string()),
            ("comment", pa.string()),
            ("old_length", pa.int64()),
            ("new_length", pa.int64()),
            ("added_text", pa.string()),
            ("removed_text", pa.string()),
            ("user_edit_count", pa.int64()),
            ("user_hidden", pa.bool_()),
            ("comment_hidden", pa.bool_()),
            ("content_hidden", pa.bool_()),
        ]
    ),
    SCORES: pa.schema(
        [
            ("schema_version", pa.int32()),
            ("rev_id", pa.int64()),
            ("model_name", pa.string()),
            ("model_version", pa.string()),
            ("score", pa.float32()),
            ("used_fallback", pa.bool_()),
            ("scored_at", pa.timestamp("s", tz="UTC")),
            # Added in Phase 5. Older files read these as null.
            ("threshold", pa.float32()),
            ("alias", pa.string()),
        ]
    ),
    LABELS: pa.schema(
        [
            ("schema_version", pa.int32()),
            ("rev_id", pa.int64()),
            ("label", pa.int8()),
            ("excluded_reason", pa.string()),
            ("decided_at", pa.timestamp("s", tz="UTC")),
            ("decided_late", pa.bool_()),
        ]
    ),
}
PARTITION_FIELD = {EDITS: "timestamp", SCORES: "scored_at", LABELS: "decided_at"}
DEDUPE_KEYS = {EDITS: ("rev_id",), SCORES: ("rev_id", "model_version"), LABELS: ("rev_id",)}
PURGED_EDIT_FIELDS = ("comment", "added_text", "removed_text", "user_hash")


class ArchiveConfigError(RuntimeError):
    """The bucket settings are incomplete. The message names settings, never their values."""


@dataclass(frozen=True)
class Archive:
    """Where the archive lives: a filesystem, and the folder or bucket prefix inside it.

    A local folder writes to a temporary name and renames, so a reader never sees half a file.
    An object store needs no rename: an upload only becomes visible once it's complete.
    """

    fs: pafs.FileSystem
    base: str
    local: bool

    @classmethod
    def at(cls, folder: Path | str) -> "Archive":
        return cls(pafs.LocalFileSystem(), Path(folder).resolve().as_posix(), local=True)

    def join(self, *parts: str) -> str:
        return "/".join([self.base.rstrip("/"), *parts])

    def _children(self, path: str, kind: pafs.FileType) -> list[str]:
        selector = pafs.FileSelector(path, allow_not_found=True, recursive=False)
        return sorted(info.path for info in self.fs.get_file_info(selector) if info.type == kind)

    def folders(self, *parts: str) -> list[str]:
        return self._children(self.join(*parts), pafs.FileType.Directory)

    def files(self, folder: str) -> list[str]:
        return self._children(folder, pafs.FileType.File)

    def is_file(self, path: str) -> bool:
        return bool(self.fs.get_file_info(path).type == pafs.FileType.File)

    def _make_parent(self, path: str) -> None:
        # An object store has keys, not folders, so making one would only add empty marker objects.
        if not isinstance(self.fs, pafs.S3FileSystem):
            self.fs.create_dir(path.rsplit("/", 1)[0], recursive=True)

    def write_parquet(self, table: pa.Table, path: str) -> str:
        self._make_parent(path)
        if not self.local:
            pq.write_table(table, path, filesystem=self.fs, compression=COMPRESSION)
            return path
        temporary = f"{path}.tmp"
        pq.write_table(table, temporary, compression=COMPRESSION)
        os.replace(temporary, path)
        return path

    def read_parquet(self, path: str, **options: Any) -> pa.Table:
        return pq.read_table(path, filesystem=self.fs, **options)

    def write_bytes(self, path: str, data: bytes) -> None:
        self._make_parent(path)
        with self.fs.open_output_stream(path) as stream:
            stream.write(data)

    def read_bytes(self, path: str) -> bytes:
        with self.fs.open_input_stream(path) as stream:
            return bytes(stream.read())

    def write_text(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8"))

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def delete_file(self, path: str) -> None:
        self.fs.delete_file(path)

    def delete_folder(self, path: str) -> None:
        if self.fs.get_file_info(path).type != pafs.FileType.NotFound:
            self.fs.delete_dir(path)

    def size(self) -> int:
        selector = pafs.FileSelector(self.base, allow_not_found=True, recursive=True)
        return sum(
            info.size or 0
            for info in self.fs.get_file_info(selector)
            if info.type == pafs.FileType.File and info.path.endswith(".parquet")
        )


Root = Path | str | Archive


def as_archive(root: Root) -> Archive:
    return root if isinstance(root, Archive) else Archive.at(root)


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    """pyarrow wants the scheme apart from the host."""
    for scheme in ("https", "http"):
        if endpoint.startswith(f"{scheme}://"):
            return scheme, endpoint.removeprefix(f"{scheme}://").rstrip("/")
    return "https", endpoint.rstrip("/")


def open_archive(settings: "ArchiveSettings") -> Archive:
    """The working copy: ARCHIVE_PATH as a folder, or as a prefix in the bucket (ARCHIVE_STORE)."""
    if settings.archive_store == "local":
        return Archive.at(settings.archive_path)
    bucket = open_bucket(settings, settings.archive_path)
    if bucket is None:
        raise ArchiveConfigError("ARCHIVE_STORE is s3, so ARCHIVE_S3_BUCKET is needed")
    return bucket


def open_bucket(settings: "ArchiveSettings", prefix: str) -> Archive | None:
    """The bucket, at a prefix, or None when no bucket is set (local development)."""
    if not settings.archive_s3_bucket:
        return None
    endpoint, region = settings.archive_s3_endpoint, settings.archive_s3_region
    access, secret = settings.archive_s3_access_key, settings.archive_s3_secret_key
    if not (endpoint and region and access and secret):
        missing = [
            name
            for name, value in (
                ("ARCHIVE_S3_ENDPOINT", endpoint),
                ("ARCHIVE_S3_REGION", region),
                ("ARCHIVE_S3_ACCESS_KEY", access),
                ("ARCHIVE_S3_SECRET_KEY", secret),
            )
            if not value
        ]
        raise ArchiveConfigError(f"ARCHIVE_S3_BUCKET is set, so these are needed too: {missing}")
    scheme, host = _split_endpoint(endpoint)
    # The AWS SDK inside pyarrow now adds a checksum to every upload by default, in a streamed
    # format S3-compatible stores often reject (HTTP 411, MissingContentLength). Checksums only
    # when the API requires them, which is the SDK's documented switch. Set before the first
    # client is made, and an operator's own value wins.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "WHEN_REQUIRED")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "WHEN_REQUIRED")
    fs = pafs.S3FileSystem(
        access_key=access.get_secret_value(),
        secret_key=secret.get_secret_value(),
        region=region,
        endpoint_override=host,
        scheme=scheme,
        # Oracle's S3-compatible API takes the bucket in the path, not in the host name.
        force_virtual_addressing=False,
        # The bucket is made by hand, so a typo fails instead of making a new one.
        allow_bucket_creation=False,
        allow_bucket_deletion=False,
        connect_timeout=10,
        request_timeout=60,
    )
    prefix = prefix.strip("/")
    base = f"{settings.archive_s3_bucket}/{prefix}" if prefix else settings.archive_s3_bucket
    return Archive(fs, base, local=False)


MIRROR_MANIFEST = "_mirror.json"


def _relative_files(archive: Archive) -> dict[str, tuple[int, int]]:
    """Every file below the base, by its relative path, with its size and modification time."""
    selector = pafs.FileSelector(archive.base, allow_not_found=True, recursive=True)
    base = archive.base.rstrip("/") + "/"
    return {
        info.path.removeprefix(base): (info.size or 0, info.mtime_ns or 0)
        for info in archive.fs.get_file_info(selector)
        if info.type == pafs.FileType.File
        and not info.path.endswith(".tmp")
        and not info.path.endswith(MIRROR_MANIFEST)
    }


def _read_manifest(target: Archive) -> dict[str, list[int]]:
    path = target.join(MIRROR_MANIFEST)
    if not target.is_file(path):
        return {}
    found: dict[str, list[int]] = json.loads(target.read_text(path))
    return found


def _write_manifest(target: Archive, files: dict[str, tuple[int, int]]) -> None:
    body = {name: list(stats) for name, stats in sorted(files.items())}
    target.write_text(target.join(MIRROR_MANIFEST), json.dumps(body))


def mirror(source: Archive, target: Archive, retention_days: int, today: date) -> dict[str, int]:
    """Copy new and changed files from the working copy to the mirror, and drop expired days.

    The mirror keeps a manifest of each file's size and modification time on the working copy, as
    they were when it was copied. A file goes again when it's missing there or either value
    changed, which includes a purge's rewrite, so hidden text leaves the mirror too. That costs one
    read and one write a night, instead of reading every object back to compare.

    Nothing is deleted just because the working copy lacks it: after a VM is rebuilt the working
    copy starts empty, and the mirror is what brings it back. The mirror only drops date folders
    past the retention window, as the working copy's own cleanup does, and reference samples the
    working copy has dropped, but only while it still holds at least one, so an empty disk can't
    wipe them.
    """
    have = _relative_files(source)
    there = _relative_files(target)
    copied_before = _read_manifest(target)
    copied = 0
    for name, stats in sorted(have.items()):
        if name not in there or copied_before.get(name) != list(stats):
            target.write_bytes(target.join(name), source.read_bytes(source.join(name)))
            copied += 1
    cutoff = today - timedelta(days=retention_days)
    expired = 0
    for dataset in SCHEMAS:
        for folder in target.folders(dataset):
            day = _folder_day(folder)
            if day is not None and day < cutoff:
                target.delete_folder(folder)
                expired += 1
    dropped = 0
    kept = set(reference_versions(source))
    if kept:
        for version in reference_versions(target):
            if version not in kept:
                delete_reference(target, version)
                dropped += 1
    # Entries for files the working copy no longer has are kept, so a rebuilt disk compares right.
    _write_manifest(target, {**{k: (v[0], v[1]) for k, v in copied_before.items()}, **have})
    return {"copied": copied, "expired_folders": expired, "dropped_references": dropped}


def restore_from_mirror(mirror_copy: Archive, target: Archive) -> int:
    """Bring the working copy back from the mirror, after a rebuild. Copies only what's missing.

    The manifest is then updated to the restored files' new times, so the next mirror run doesn't
    send them all back again.
    """
    have = _relative_files(target)
    copied = 0
    for name in sorted(_relative_files(mirror_copy)):
        if name not in have:
            target.write_bytes(target.join(name), mirror_copy.read_bytes(mirror_copy.join(name)))
            copied += 1
    _write_manifest(mirror_copy, {**_relative_files(target)})
    return copied


def _as_datetime(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def write_rows(root: Root, dataset: str, rows: Iterable[dict[str, Any]]) -> list[str]:
    """Write rows into their date folders. Each file appears only once it's complete."""
    archive = as_archive(root)
    by_day: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        moment = _as_datetime(row[PARTITION_FIELD[dataset]]).astimezone(UTC)
        by_day.setdefault(moment.date(), []).append(row)

    written = []
    for day, day_rows in sorted(by_day.items()):
        # A time prefix, so name order is write order and "the last copy wins" means the newest.
        name = f"part-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.parquet"
        path = archive.join(dataset, f"date={day.isoformat()}", name)
        table = pa.Table.from_pylist(day_rows, schema=SCHEMAS[dataset])
        written.append(archive.write_parquet(table, path))
    return written


def read_rows(root: Root, dataset: str) -> list[dict[str, Any]]:
    """Every row in a dataset, with duplicates from redelivery removed."""
    rows: list[dict[str, Any]] = read_table(root, dataset).to_pylist()
    return rows


def _folder_day(folder: str) -> date | None:
    name = folder.rstrip("/").rsplit("/", 1)[-1]
    if not name.startswith("date="):
        return None
    try:
        return date.fromisoformat(name.removeprefix("date="))
    except ValueError:
        return None


def _is_part(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("part-") and name.endswith(".parquet")


def files_between(
    root: Root, dataset: str, since: date | None = None, until: date | None = None
) -> list[str]:
    """Files whose date folder falls in [since, until], oldest first."""
    archive = as_archive(root)
    found: list[str] = []
    for folder in archive.folders(dataset):
        day = _folder_day(folder)
        if day is None or (since and day < since) or (until and day > until):
            continue
        found.extend(path for path in archive.files(folder) if _is_part(path))
    return found


def read_table(
    root: Root,
    dataset: str,
    since: date | None = None,
    until: date | None = None,
    columns: Sequence[str] | None = None,
    keep: Callable[[pa.Array], pa.Array] | None = None,
) -> pa.Table:
    """Rows from the date folders in range, deduplicated, as one Arrow table.

    The live archive holds about 86,000 edits a day (docs/capacity.md), so the monitor and trainer
    read only the days and columns they need. `keep` takes each file's rev_id column and returns
    a mask, so a sample is taken before the rest of a file is read into memory.
    """
    archive = as_archive(root)
    schema = SCHEMAS[dataset]
    keys = list(DEDUPE_KEYS[dataset])
    wanted = list(dict.fromkeys([*keys, *(columns or schema.names)]))
    tables = []
    for path in files_between(archive, dataset, since, until):
        table = archive.read_parquet(path, schema=schema, columns=wanted)
        if keep is not None:
            table = table.filter(keep(table.column("rev_id")))
        if table.num_rows:
            tables.append(table)
    if not tables:
        return schema.empty_table().select(wanted)
    combined = pa.concat_tables(tables)
    # Redelivery can write a row twice. The last copy wins, as files are read oldest first.
    order = pa.array(range(combined.num_rows), pa.int64())
    combined = combined.append_column("__order", order)
    last = combined.group_by(keys).aggregate([("__order", "max")]).column("__order_max")
    rows = pc.take(last, pc.sort_indices(last))
    return combined.take(rows).drop_columns(["__order"])


def sample_mask(rev_ids: pa.Array, fraction: float, salt: str = "") -> pa.Array:
    """Keep about `fraction` of revisions, chosen by a hash of the id.

    The same revision is always in or out, so a rebuild picks the same rows and a sample doesn't
    depend on which file a row landed in.
    """
    if fraction >= 1:
        return pa.array([True] * len(rev_ids))
    limit = int(fraction * 2**32)
    return pa.array(
        [
            int.from_bytes(hashlib.sha256(f"{salt}{value}".encode()).digest()[:4], "big") < limit
            for value in rev_ids.to_pylist()
        ]
    )


def delete_older_than(root: Root, cutoff: date) -> int:
    """Remove date folders before the cutoff, and return how many went."""
    archive = as_archive(root)
    removed = 0
    for dataset in SCHEMAS:
        for folder in archive.folders(dataset):
            day = _folder_day(folder)
            if day is not None and day < cutoff:
                archive.delete_folder(folder)
                removed += 1
    return removed


def purge_revisions(root: Root, rev_ids: set[int]) -> int:
    """Blank the text of hidden revisions in every file that holds them. Returns rows changed."""
    if not rev_ids:
        return 0
    archive = as_archive(root)
    changed = 0
    for path in files_between(archive, EDITS):
        found = archive.read_parquet(path, columns=["rev_id"]).column("rev_id").to_pylist()
        if not rev_ids.intersection(found):
            continue
        rows = archive.read_parquet(path, schema=SCHEMAS[EDITS]).to_pylist()
        for row in rows:
            if row["rev_id"] in rev_ids:
                row.update(dict.fromkeys(PURGED_EDIT_FIELDS))
                changed += 1
        archive.write_parquet(pa.Table.from_pylist(rows, schema=SCHEMAS[EDITS]), path)
    return changed


def size_in_bytes(root: Root) -> int:
    return as_archive(root).size()


# Drift reference samples: kept past the 30 days for as long as their model serves (section 2).
REFERENCE = "reference"
REFERENCE_ROWS_FILE = "reference.parquet"
REFERENCE_META_FILE = "reference.json"
REFERENCE_SCHEMA = SNAPSHOT_SCHEMA.append(pa.field("score", pa.float64()))


def write_reference(
    root: Root, version: str, rows: list[dict[str, Any]], meta: dict[str, Any]
) -> str:
    """rows are snapshot rows with the model's calibrated score added as `score`."""
    archive = as_archive(root)
    table = pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA)
    archive.write_parquet(table, archive.join(REFERENCE, version, REFERENCE_ROWS_FILE))
    archive.write_text(
        archive.join(REFERENCE, version, REFERENCE_META_FILE),
        json.dumps(meta, indent=2, sort_keys=True),
    )
    return archive.join(REFERENCE, version)


def read_reference(root: Root, version: str) -> tuple[pa.Table, dict[str, Any]] | None:
    archive = as_archive(root)
    rows = archive.join(REFERENCE, version, REFERENCE_ROWS_FILE)
    if not archive.is_file(rows):
        return None
    meta: dict[str, Any] = json.loads(
        archive.read_text(archive.join(REFERENCE, version, REFERENCE_META_FILE))
    )
    return archive.read_parquet(rows, schema=REFERENCE_SCHEMA), meta


def reference_versions(root: Root) -> list[str]:
    return sorted(folder.rsplit("/", 1)[-1] for folder in as_archive(root).folders(REFERENCE))


def delete_reference(root: Root, version: str) -> None:
    archive = as_archive(root)
    archive.delete_folder(archive.join(REFERENCE, version))


def purge_references(root: Root, rev_ids: set[int]) -> int:
    """Drop hidden revisions from every reference sample. Whole rows, as in a snapshot."""
    archive = as_archive(root)
    removed = 0
    for version in reference_versions(archive) if rev_ids else []:
        path = archive.join(REFERENCE, version, REFERENCE_ROWS_FILE)
        table = archive.read_parquet(path, schema=REFERENCE_SCHEMA)
        mask = pc.invert(pc.is_in(table.column("rev_id"), value_set=pa.array(sorted(rev_ids))))
        kept = table.filter(mask)
        if kept.num_rows == table.num_rows:
            continue
        removed += table.num_rows - kept.num_rows
        archive.write_parquet(kept, path)
    return removed
