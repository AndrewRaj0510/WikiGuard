"""The rolling parquet archive: where retraining and monitoring read history, instead of Supabase.

Reads from Supabase count against its monthly egress, so the archive is written directly from the
topics. Files are laid out as <root>/<dataset>/date=YYYY-MM-DD/part-*.parquet, so deleting old
data just means deleting old folders.

Consumers deliver at least once, so the same row can appear twice. Readers keep the last copy of
each key. A local folder is used until Phase 7, when this moves to Oracle Object Storage.
"""

import shutil
import uuid
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

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
        name = f"part-{uuid.uuid4().hex}"
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
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted((root / dataset).glob("date=*/part-*.parquet")):
        for row in pq.read_table(path, schema=SCHEMAS[dataset]).to_pylist():
            rows[tuple(row[key] for key in DEDUPE_KEYS[dataset])] = row
    return list(rows.values())


def _folder_day(folder: Path) -> date | None:
    try:
        return date.fromisoformat(folder.name.removeprefix("date="))
    except ValueError:
        return None


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
