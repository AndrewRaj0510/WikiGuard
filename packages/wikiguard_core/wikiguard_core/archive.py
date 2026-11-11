"""The rolling parquet archive: where retraining and monitoring read history, instead of Supabase.

Reads from Supabase count against its monthly egress, so the archive is written directly from the
topics. Files are laid out as <root>/<dataset>/date=YYYY-MM-DD/part-*.parquet, so deleting old
data just means deleting old folders.

Consumers deliver at least once, so the same row can appear twice. Readers keep the last copy of
each key. A local folder is used until Phase 7, when this moves to Oracle Object Storage.
"""

import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wikiguard_core.snapshot import SNAPSHOT_SCHEMA

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


def _as_datetime(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def write_rows(root: Path, dataset: str, rows: Iterable[dict[str, Any]]) -> list[Path]:
    """Write rows into their date folders. A rename makes each file appear only once complete."""
    by_day: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        moment = _as_datetime(row[PARTITION_FIELD[dataset]]).astimezone(UTC)
        by_day.setdefault(moment.date(), []).append(row)

    written = []
    for day, day_rows in sorted(by_day.items()):
        folder = root / dataset / f"date={day.isoformat()}"
        folder.mkdir(parents=True, exist_ok=True)
        # A time prefix, so name order is write order and "the last copy wins" means the newest.
        name = f"part-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        temporary = folder / f"{name}.tmp"
        pq.write_table(
            pa.Table.from_pylist(day_rows, schema=SCHEMAS[dataset]),
            temporary,
            compression=COMPRESSION,
        )
        written.append(temporary.replace(folder / f"{name}.parquet"))
    return written


def read_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    """Every row in a dataset, with duplicates from redelivery removed."""
    rows: list[dict[str, Any]] = read_table(root, dataset).to_pylist()
    return rows


def _folder_day(folder: Path) -> date | None:
    try:
        return date.fromisoformat(folder.name.removeprefix("date="))
    except ValueError:
        return None


def files_between(
    root: Path, dataset: str, since: date | None = None, until: date | None = None
) -> list[Path]:
    """Files whose date folder falls in [since, until], oldest first."""
    found = []
    for folder in sorted((root / dataset).glob("date=*")):
        day = _folder_day(folder)
        if day is None or (since and day < since) or (until and day > until):
            continue
        found.extend(sorted(folder.glob("part-*.parquet")))
    return found


def read_table(
    root: Path,
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
    schema = SCHEMAS[dataset]
    keys = list(DEDUPE_KEYS[dataset])
    wanted = list(dict.fromkeys([*keys, *(columns or schema.names)]))
    tables = []
    for path in files_between(root, dataset, since, until):
        table = pq.read_table(path, schema=schema, columns=wanted)
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


def delete_older_than(root: Path, cutoff: date) -> int:
    """Remove date folders before the cutoff, and return how many went."""
    removed = 0
    for dataset in SCHEMAS:
        for folder in (root / dataset).glob("date=*"):
            day = _folder_day(folder)
            if day is not None and day < cutoff:
                shutil.rmtree(folder)
                removed += 1
    return removed


def purge_revisions(root: Path, rev_ids: set[int]) -> int:
    """Blank the text of hidden revisions in every file that holds them. Returns rows changed."""
    if not rev_ids:
        return 0
    changed = 0
    for path in sorted((root / EDITS).glob("date=*/part-*.parquet")):
        found = pq.read_table(path, columns=["rev_id"]).column("rev_id").to_pylist()
        if not rev_ids.intersection(found):
            continue
        rows = pq.read_table(path, schema=SCHEMAS[EDITS]).to_pylist()
        for row in rows:
            if row["rev_id"] in rev_ids:
                row.update(dict.fromkeys(PURGED_EDIT_FIELDS))
                changed += 1
        temporary = path.with_suffix(".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SCHEMAS[EDITS]),
            temporary,
            compression=COMPRESSION,
        )
        temporary.replace(path)
    return changed


def size_in_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*.parquet"))


# Drift reference samples: kept past the 30 days for as long as their model serves (section 2).
REFERENCE = "reference"
REFERENCE_ROWS_FILE = "reference.parquet"
REFERENCE_META_FILE = "reference.json"
REFERENCE_SCHEMA = SNAPSHOT_SCHEMA.append(pa.field("score", pa.float64()))


def write_reference(
    root: Path, version: str, rows: list[dict[str, Any]], meta: dict[str, Any]
) -> Path:
    """rows are snapshot rows with the model's calibrated score added as `score`."""
    folder = root / REFERENCE / version
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{REFERENCE_ROWS_FILE}.tmp"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA), temporary, compression=COMPRESSION
    )
    temporary.replace(folder / REFERENCE_ROWS_FILE)
    (folder / REFERENCE_META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), "utf-8")
    return folder


def read_reference(root: Path, version: str) -> tuple[pa.Table, dict[str, Any]] | None:
    folder = root / REFERENCE / version
    if not (folder / REFERENCE_ROWS_FILE).is_file():
        return None
    meta: dict[str, Any] = json.loads((folder / REFERENCE_META_FILE).read_text("utf-8"))
    return pq.read_table(folder / REFERENCE_ROWS_FILE, schema=REFERENCE_SCHEMA), meta


def reference_versions(root: Path) -> list[str]:
    folder = root / REFERENCE
    return (
        sorted(path.name for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
    )


def delete_reference(root: Path, version: str) -> None:
    shutil.rmtree(root / REFERENCE / version, ignore_errors=True)


def purge_references(root: Path, rev_ids: set[int]) -> int:
    """Drop hidden revisions from every reference sample. Whole rows, as in a snapshot."""
    removed = 0
    for version in reference_versions(root) if rev_ids else []:
        path = root / REFERENCE / version / REFERENCE_ROWS_FILE
        table = pq.read_table(path, schema=REFERENCE_SCHEMA)
        mask = pc.invert(pc.is_in(table.column("rev_id"), value_set=pa.array(sorted(rev_ids))))
        kept = table.filter(mask)
        if kept.num_rows == table.num_rows:
            continue
        removed += table.num_rows - kept.num_rows
        temporary = path.with_suffix(".tmp")
        pq.write_table(kept, temporary, compression=COMPRESSION)
        temporary.replace(path)
    return removed
