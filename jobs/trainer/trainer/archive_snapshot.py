"""Build a retraining snapshot from the parquet archive (PLAN.md, section 2).

The archive holds about 86,000 edits a day, far more than training needs, so a fixed share is kept
by a hash of each rev_id. The same revision is always in or out, so building twice from the same
archive gives the same snapshot. Rules on top of build_snapshot's own:

- Only final labels count. Edits still inside their 48 hours are left out as unsettled.
- The golden set's revisions never enter training (DEC-051), so it stays a fair test.
- A revision with anything hidden is left out whole, as in DEC-045: snapshots are public.
"""

from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from wikiguard_core.archive import EDITS, LABELS, Root, read_table, sample_mask
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import Candidate, Snapshot, build_snapshot

SOURCE = "archive"
ARCHIVE_DAYS = 30
GOLDEN = "golden_set"
HIDDEN = "hidden"
EDIT_FIELDS = tuple(EditEvent.model_fields)


def _edit(row: dict[str, Any]) -> EditEvent:
    values = {name: row[name] for name in EDIT_FIELDS if name in row}
    values["added_text"] = values.get("added_text") or ""
    values["removed_text"] = values.get("removed_text") or ""
    return EditEvent(**values)


def archive_candidates(
    root: Root, as_of: datetime, target_rows: int, exclude: set[int]
) -> tuple[list[Candidate], int]:
    """The candidates, and how many golden-set revisions were held out."""
    since = (as_of - timedelta(days=ARCHIVE_DAYS)).date()
    labels = read_table(root, LABELS, since=since, columns=["label", "excluded_reason"]).to_pylist()
    final = {row["rev_id"]: row for row in labels if row["rev_id"] not in exclude}
    held_out = len(labels) - len(final)
    labeled = sum(1 for row in final.values() if row["label"] is not None)
    fraction = min(1.0, target_rows / labeled) if labeled else 1.0
    wanted = pa.array(sorted(final), pa.int64())

    def keep(rev_ids: pa.Array) -> pa.Array:
        return pc.and_(pc.is_in(rev_ids, value_set=wanted), sample_mask(rev_ids, fraction))

    candidates = []
    edits = read_table(root, EDITS, since=since, until=as_of.date(), keep=keep)
    for row in edits.to_pylist():
        rev_id = row["rev_id"]
        label = final[rev_id]
        if label["excluded_reason"] or label["label"] is None:
            candidates.append(Candidate(rev_id, None, None, label["excluded_reason"] or "no_label"))
        elif row["user_hidden"] or row["comment_hidden"] or row["content_hidden"]:
            candidates.append(Candidate(rev_id, None, None, HIDDEN))
        else:
            candidates.append(Candidate(rev_id, _edit(row), int(label["label"])))
    return candidates, held_out


def build_from_archive(
    root: Root, as_of: datetime, target_rows: int, exclude: set[int]
) -> Snapshot:
    candidates, held_out = archive_candidates(root, as_of, target_rows, exclude)
    snapshot = build_snapshot(candidates, as_of, SOURCE)
    # A count, never which ones (DEC-045). Not part of the hash, which covers rows only. Kept out
    # of "excluded", which measures censored edits: a held-out row isn't hidden or deleted.
    snapshot.manifest["held_out"] = {GOLDEN: held_out}
    snapshot.manifest["sample_target_rows"] = target_rows
    return snapshot
