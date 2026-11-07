"""Turn settled edits into the versioned dataset every model trains on.

A snapshot is a folder holding edits.parquet and manifest.json. Three rules shape it:

- Splits come from time alone: the newest 7 days are test, the 7 before are validation, and
  anything older is train. The windows come from as_of rather than from the data, so a few odd
  rows can't move them, and a rebuild with the same inputs gives the same snapshot.
- The content hash covers row values, not file bytes, so upgrading pyarrow doesn't change a
  dataset's identity while changing one character of one row does.
- Rows hold only what training needs. Page titles and editor hashes are used once, to report how
  far the splits overlap, and never written, because snapshots are public (DEC-043).

Sources differ, so this module takes Candidate objects. scripts/build_snapshot.py makes them from
the backfill, and the Phase 5 trainer will make them from the archive.
"""

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wikiguard_core.features import FEATURE_VERSION, EditFeatures, build_features
from wikiguard_core.labels import LABEL_WAIT
from wikiguard_core.schemas import SCHEMA_VERSION, EditEvent

FORMAT_VERSION = 1
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)
EVALUATION_WINDOW = timedelta(days=7)
ROWS_FILE = "edits.parquet"
MANIFEST_FILE = "manifest.json"
COMPRESSION = "zstd"
UNSETTLED = "unsettled"
NO_LABEL = "no_label"
# Snapshots are published, and CC BY-SA requires this to travel with the text (DEC-043).
NOTICE = (
    "Text comes from English Wikipedia, available under CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/). Each revision's authors are listed through "
    "https://en.wikipedia.org/w/index.php?diff=REV_ID. Not affiliated with the Wikimedia "
    "Foundation. It includes vandalism, which can be offensive."
)

SNAPSHOT_SCHEMA = pa.schema(
    [
        ("rev_id", pa.int64()),
        ("timestamp", pa.timestamp("ms", tz="UTC")),
        ("split", pa.string()),
        ("label", pa.int8()),
        ("feature_version", pa.int32()),
        ("added_text", pa.string()),
        ("removed_text", pa.string()),
        ("comment", pa.string()),
        ("is_anonymous", pa.bool_()),
        ("user_edit_count_bucket", pa.int32()),
        ("bytes_delta", pa.int64()),
        ("is_minor", pa.bool_()),
        ("chars_added", pa.int64()),
        ("uppercase_ratio", pa.float64()),
        ("badword_count", pa.int64()),
        ("has_external_link", pa.bool_()),
        ("hour_of_day", pa.int32()),
        ("page_length", pa.int64()),
    ]
)
FEATURE_COLUMNS = tuple(field.name for field in fields(EditFeatures))


class SnapshotError(ValueError):
    """The data can't make a trustworthy snapshot, or a stored one failed its check."""


@dataclass(frozen=True)
class Candidate:
    """One source row. Excluded rows are still passed in, so they get counted."""

    rev_id: int
    edit: EditEvent | None
    label: int | None
    excluded_reason: str | None = None


@dataclass(frozen=True)
class SplitWindows:
    validation_start: datetime
    test_start: datetime
    end: datetime

    @classmethod
    def from_as_of(cls, as_of: datetime) -> "SplitWindows":
        if as_of.tzinfo is None:
            raise SnapshotError("as_of must include a time zone")
        # Newer edits may still be reverted, so they can't have a final label yet.
        end = as_of.astimezone(UTC) - LABEL_WAIT
        test_start = end - EVALUATION_WINDOW
        return cls(test_start - EVALUATION_WINDOW, test_start, end)

    def split_of(self, moment: datetime) -> str | None:
        """Which split an edit belongs to, or None if it's too new to be settled."""
        if moment > self.end:
            return None
        if moment > self.test_start:
            return TEST
        return VALIDATION if moment > self.validation_start else TRAIN


@dataclass(frozen=True)
class Snapshot:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def _text_key(row: dict[str, Any]) -> str | None:
    """Identical changed text, or None for an edit that changed none, which can't leak anything."""
    if not row["added_text"] and not row["removed_text"]:
        return None
    return f"{row['added_text']}\0{row['removed_text']}"


def _split_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        members = [row for row in rows if row["split"] == split]
        if not members:
            stats[split] = {"rows": 0, "positives": 0}
            continue
        positives = sum(row["label"] for row in members)
        stats[split] = {
            "rows": len(members),
            "positives": positives,
            "positive_rate": round(positives / len(members), 4),
            "anonymous_share": round(sum(row["is_anonymous"] for row in members) / len(members), 4),
            "first": min(row["timestamp"] for row in members).isoformat(),
            "last": max(row["timestamp"] for row in members).isoformat(),
        }
    return stats


def _remove_train_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop train rows whose exact text is in validation or test (PLAN.md, section 2).

    Otherwise a model can score well on those windows by having memorized the answer. Only train
    loses rows, so the windows being measured stay exactly what they were.
    """
    keys: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row in rows:
        key = _text_key(row)
        if key is not None:
            keys[row["split"]][key] += 1
    later = keys[VALIDATION].keys() | keys[TEST].keys()

    kept: list[dict[str, Any]] = []
    removed = removed_positives = 0
    for row in rows:
        if row["split"] == TRAIN and _text_key(row) in later:
            removed += 1
            removed_positives += row["label"]
        else:
            kept.append(row)

    report = {
        "removed_from_train": removed,
        "positives_removed_from_train": removed_positives,
        "test_rows_with_text_in_original_train": sum(
            count for key, count in keys[TEST].items() if key in keys[TRAIN]
        ),
        "validation_rows_with_text_in_original_train": sum(
            count for key, count in keys[VALIDATION].items() if key in keys[TRAIN]
        ),
        "test_rows_with_text_in_validation": sum(
            count for key, count in keys[TEST].items() if key in keys[VALIDATION]
        ),
        "repeated_rows_left_inside_train": sum(
            count - 1 for key, count in keys[TRAIN].items() if key not in later
        ),
    }
    return kept, report


def _split_overlap(edits: list[tuple[str, EditEvent]]) -> dict[str, float]:
    """How much test shares pages and editors with train.

    Nothing is removed for this, since real models also meet known pages and editors. It's
    reported because a high share means test scores can't be read as scores on unseen editors,
    and it's worked out here because titles and editor hashes never reach the snapshot.
    """
    train_pages = {edit.title for split, edit in edits if split == TRAIN}
    train_users = {edit.user_hash for split, edit in edits if split == TRAIN and edit.user_hash}
    test = [edit for split, edit in edits if split == TEST]
    return {
        "test_rows_on_train_pages": round(
            sum(edit.title in train_pages for edit in test) / len(test), 4
        ),
        "test_rows_by_train_editors": round(
            sum(edit.user_hash in train_users for edit in test) / len(test), 4
        ),
    }


def _feature_row(candidate: Candidate, edit: EditEvent, split: str) -> dict[str, Any]:
    return {
        "rev_id": candidate.rev_id,
        "timestamp": edit.timestamp,
        "split": split,
        "label": candidate.label,
        **build_features(edit).as_dict(),
    }


def build_snapshot(candidates: Iterable[Candidate], as_of: datetime, source: str) -> Snapshot:
    """Split, deduplicate, hash, and describe. Raises rather than build from impossible data."""
    windows = SplitWindows.from_as_of(as_of)
    excluded: Counter[str] = Counter()
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    edits: list[tuple[str, EditEvent]] = []

    for candidate in candidates:
        if candidate.rev_id in seen:
            raise SnapshotError(f"revision {candidate.rev_id} appears twice")
        seen.add(candidate.rev_id)
        if candidate.label not in (0, 1, None):
            raise SnapshotError(f"revision {candidate.rev_id} has label {candidate.label!r}")

        edit = candidate.edit
        if candidate.excluded_reason or edit is None or candidate.label is None:
            excluded[candidate.excluded_reason or NO_LABEL] += 1
            continue
        split = windows.split_of(edit.timestamp)
        if split is None:
            excluded[UNSETTLED] += 1
            continue
        rows.append(_feature_row(candidate, edit, split))
        edits.append((split, edit))

    rows, duplicates = _remove_train_duplicates(rows)
    rows.sort(key=lambda row: row["rev_id"])
    stats = _split_stats(rows)
    empty = [split for split in SPLITS if not stats[split]["rows"]]
    if empty:
        raise SnapshotError(f"no rows in: {', '.join(empty)}")

    digest = content_hash(rows)
    manifest = {
        "format_version": FORMAT_VERSION,
        "notice": NOTICE,
        "name": f"snapshot-{as_of.astimezone(UTC):%Y%m%d}-{digest[:12]}",
        "source": source,
        "as_of": as_of.astimezone(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "content_sha256": digest,
        "original_content_sha256": digest,
        "windows": {
            "validation_start": windows.validation_start.isoformat(),
            "test_start": windows.test_start.isoformat(),
            "end": windows.end.isoformat(),
        },
        "rows": len(rows),
        "splits": stats,
        "excluded": dict(sorted(excluded.items())),
        "duplicates": duplicates,
        "overlap": _split_overlap(edits),
        # A purge changes the hash, so this links the current content to what models trained on.
        "purges": [],
    }
    return Snapshot(rows, manifest)


def _canonical_rows(rows: Iterable[dict[str, Any]]) -> Iterator[bytes]:
    for row in sorted(rows, key=lambda item: item["rev_id"]):
        values = {column: row[column] for column in SNAPSHOT_SCHEMA.names}
        moment: datetime = values["timestamp"]
        values["timestamp"] = moment.astimezone(UTC).isoformat(timespec="milliseconds")
        yield json.dumps(values, sort_keys=True, ensure_ascii=False).encode() + b"\n"


def content_hash(rows: Iterable[dict[str, Any]]) -> str:
    """SHA-256 of the row values, in rev_id order and independent of how the file was written."""
    digest = hashlib.sha256()
    for line in _canonical_rows(rows):
        digest.update(line)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA)
    pq.write_table(table, path, compression=COMPRESSION)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    # Fixed line endings, so a rebuild on any system gives identical bytes for DVC.
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)


def write_snapshot(snapshot: Snapshot, root: Path) -> Path:
    """Write to a temporary folder and rename, so a crash never leaves half a snapshot."""
    name: str = snapshot.manifest["name"]
    final = root / name
    if final.exists():
        # The name contains the hash, so an intact folder with this name holds this same data.
        verify_snapshot(final)
        return final
    temporary = root / f"{name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    _write_rows(temporary / ROWS_FILE, snapshot.rows)
    _write_manifest(temporary / MANIFEST_FILE, snapshot.manifest)
    temporary.replace(final)
    return final


def read_manifest(folder: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((folder / MANIFEST_FILE).read_text("utf-8"))
    return manifest


def _read_rows(folder: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(
        folder / ROWS_FILE, schema=SNAPSHOT_SCHEMA
    ).to_pylist()
    return rows


def verify_snapshot(folder: Path) -> dict[str, Any]:
    """Check the rows against the manifest, and return the manifest if they match."""
    manifest = read_manifest(folder)
    rows = _read_rows(folder)
    actual = content_hash(rows)
    if actual != manifest["content_sha256"]:
        raise SnapshotError(f"{folder.name}: rows hash to {actual}, not the recorded hash")
    if len(rows) != manifest["rows"]:
        raise SnapshotError(f"{folder.name}: {len(rows)} rows, manifest says {manifest['rows']}")
    return manifest


def read_snapshot(folder: Path) -> list[dict[str, Any]]:
    """Rows for training, refused if damaged or built with different features."""
    manifest = verify_snapshot(folder)
    if manifest["feature_version"] != FEATURE_VERSION:
        raise SnapshotError(
            f"{folder.name} has feature_version {manifest['feature_version']}, "
            f"this code expects {FEATURE_VERSION}"
        )
    return _read_rows(folder)


def purge_snapshot(folder: Path, rev_ids: set[int], at: datetime) -> int:
    """Remove rows Wikipedia has since hidden, and record it. Returns the rows removed.

    The manifest records how many, never which, so a public file doesn't point at hidden
    revisions. This only changes the local copy: the old version stays on the DVC remote until
    `dvc gc --cloud` runs, as README.md explains.
    """
    manifest = verify_snapshot(folder)
    rows = _read_rows(folder)
    kept = [row for row in rows if row["rev_id"] not in rev_ids]
    removed = len(rows) - len(kept)
    if not removed:
        return 0

    previous = manifest["content_sha256"]
    manifest["content_sha256"] = content_hash(kept)
    manifest["rows"] = len(kept)
    manifest["splits"] = _split_stats(kept)
    manifest["purges"].append(
        {
            "at": at.astimezone(UTC).isoformat(),
            "rows_removed": removed,
            "positives_removed": sum(row["label"] for row in rows) - sum(r["label"] for r in kept),
            "previous_content_sha256": previous,
        }
    )
    temporary = folder / f"{ROWS_FILE}.tmp"
    _write_rows(temporary, kept)
    temporary.replace(folder / ROWS_FILE)
    # A crash here leaves new rows with the old hash, which verify_snapshot reports, and hidden
    # text is already gone. The other order could leave the text in place and look clean.
    _write_manifest(folder / MANIFEST_FILE, manifest)
    return removed
