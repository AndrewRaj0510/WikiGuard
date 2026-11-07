"""Build the dataset snapshot from the backfill.

Reads only the columns an EditEvent can hold, plus the diff status. The raw tagged_reverted and
tags_at_capture columns are never read, so they can't become the label or a feature (DEC-039).
as_of defaults to when the backfill started, because that's when its labels were settled.

    python scripts/build_snapshot.py data/backfill snapshots
"""

import argparse
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from wikiguard_core.identity import redact_identifiers
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import Candidate, SnapshotError, build_snapshot, write_snapshot

OK_STATUS = "ok"
EDIT_COLUMNS = tuple(EditEvent.model_fields)
SOURCE = "backfill"
TEXT_FIELDS = ("added_text", "removed_text", "comment")


def read_labels(folder: Path) -> dict[int, dict[str, Any]]:
    table = pq.read_table(
        folder / "labels_48h.parquet", columns=["rev_id", "label", "no_label_reason"]
    )
    return {row["rev_id"]: row for row in table.to_pylist()}


def edit_from_row(row: dict[str, Any]) -> EditEvent:
    values = {column: row[column] for column in EDIT_COLUMNS}
    # Applied again because this backfill ran before the redactor's fix (F-066). It's safe to
    # repeat, so later data passes through unchanged.
    for field in TEXT_FIELDS:
        if values[field] is not None:
            values[field] = redact_identifiers(values[field])
    return EditEvent(**values)


def read_candidates(folder: Path) -> Iterator[Candidate]:
    labels = read_labels(folder)
    columns = [*EDIT_COLUMNS, "diff_status"]
    for path in sorted((folder / "parts").glob("part-*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            rev_id = row["rev_id"]
            label_row = labels.get(rev_id)
            if row["diff_status"] != OK_STATUS:
                yield Candidate(rev_id, None, None, row["diff_status"])
            elif label_row is None:
                yield Candidate(rev_id, None, None, "no_label_record")
            elif label_row["label"] is None:
                yield Candidate(rev_id, None, None, label_row["no_label_reason"])
            else:
                yield Candidate(rev_id, edit_from_row(row), label_row["label"])


def backfill_as_of(folder: Path) -> datetime:
    plan: dict[str, Any] = json.loads((folder / "plan.json").read_text("utf-8"))
    return datetime.fromisoformat(plan["created"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="backfill folder")
    parser.add_argument("out", type=Path, help="folder that holds snapshots")
    parser.add_argument("--as-of", type=datetime.fromisoformat, help="defaults to the backfill's")
    args = parser.parse_args()

    as_of = args.as_of or backfill_as_of(args.source)
    try:
        snapshot = build_snapshot(read_candidates(args.source), as_of, SOURCE)
    except SnapshotError as error:
        print(f"Snapshot refused: {error}", file=sys.stderr)
        return 1
    folder = write_snapshot(snapshot, args.out)
    print(folder)
    print(json.dumps(snapshot.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
