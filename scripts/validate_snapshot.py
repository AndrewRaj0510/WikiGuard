"""Run the pre-training checks on a snapshot, and save the reports. Exits 1 if any check fails.

The Evidently report compares train with test on metadata only. It holds no edit text, so a person
can open and share it without meeting vandalism. It's saved beside the other run outputs and not in
the snapshot folder, because a file added there would change what DVC tracks.

    python scripts/validate_snapshot.py snapshots/NAME --out data/reports
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from wikiguard_core.snapshot import ROWS_FILE, TEST, TRAIN
from wikiguard_core.validation import ValidationReport, validate_snapshot

# Columns only, and never the text columns, so no vandal text reaches a report.
REPORT_COLUMNS = (
    "label",
    "bytes_delta",
    "is_anonymous",
    "is_minor",
    "chars_added",
    "uppercase_ratio",
    "badword_count",
    "has_external_link",
    "hour_of_day",
    "page_length",
)
CATEGORY_COLUMNS = ["label", "is_anonymous", "is_minor", "has_external_link"]
NUMERIC_COLUMNS = [column for column in REPORT_COLUMNS if column not in CATEGORY_COLUMNS]


def report_frames(folder: Path) -> tuple[Any, Any]:
    columns = ["split", *REPORT_COLUMNS]
    frame = pq.read_table(folder / ROWS_FILE, columns=columns).to_pandas()
    # Evidently only accepts text and plain numbers as category labels.
    for column in CATEGORY_COLUMNS:
        frame[column] = frame[column].astype(str)
    train = frame[frame["split"] == TRAIN].drop(columns="split")
    test = frame[frame["split"] == TEST].drop(columns="split")
    return train, test


def write_evidently(folder: Path, out: Path) -> Path:
    # Imported here so the checks themselves work in an environment without Evidently.
    from evidently import DataDefinition, Report
    from evidently import Dataset as EvidentlyDataset
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    train, test = report_frames(folder)
    definition = DataDefinition(
        numerical_columns=NUMERIC_COLUMNS, categorical_columns=CATEGORY_COLUMNS
    )
    reference = EvidentlyDataset.from_pandas(train, data_definition=definition)
    current = EvidentlyDataset.from_pandas(test, data_definition=definition)
    result = Report([DataSummaryPreset(), DataDriftPreset()]).run(current, reference)
    path = out / "evidently_data_report.html"
    result.save_html(str(path))
    return path


def write_results(report: ValidationReport, out: Path) -> Path:
    path = out / "validation.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), "utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/reports"))
    parser.add_argument("--skip-evidently", action="store_true")
    args = parser.parse_args()

    report = validate_snapshot(args.folder)
    out = args.out / args.folder.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {write_results(report, out)}")
    for check in report.checks:
        print(f"{check.status:5} {check.name}: {check.detail}")
    if report.failures:
        return 1
    if not args.skip_evidently:
        print(f"Wrote {write_evidently(args.folder, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
