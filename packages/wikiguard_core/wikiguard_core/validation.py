"""Pre-training data checks. A failed check stops the run before any model sees the data.

Silent data problems inflate metrics instead of crashing, so each check below is one that would
otherwise go unnoticed. Checks either fail (the snapshot can't be trusted) or warn (a person should
read the number, but it doesn't make the data wrong). PLAN.md, section 4 lists them.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from wikiguard_core.features import FEATURE_VERSION, count_badwords, uppercase_share
from wikiguard_core.snapshot import (
    ROWS_FILE,
    SNAPSHOT_SCHEMA,
    SPLITS,
    TEST,
    TRAIN,
    VALIDATION,
    SplitWindows,
    content_hash,
    read_manifest,
)

MIN_TEST_POSITIVES = 500
# Coordinated fake reverts would show up as a sudden jump in the positive rate.
MAX_RATE_MULTIPLE = 2.0
MIN_DAY_ROWS_SHARE = 0.1
MAX_EXCLUDED_SHARE = 0.05
# A common rule of thumb: above 0.25, a feature's distribution has moved a lot.
PSI_WARN = 0.25
PSI_BINS = 10
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "hour_of_day": (0, 23),
    "uppercase_ratio": (0.0, 1.0),
    "chars_added": (0, math.inf),
    "badword_count": (0, math.inf),
    "page_length": (0, math.inf),
    "user_edit_count_bucket": (-1, 4),
}
PSI_FEATURES = (
    "bytes_delta",
    "chars_added",
    "uppercase_ratio",
    "badword_count",
    "hour_of_day",
    "page_length",
    "is_anonymous",
    "is_minor",
    "has_external_link",
)
FAIL = "fail"
WARN = "warn"
PASS = "pass"  # noqa: S105 - a check result, not a password


class DataValidationError(ValueError):
    """The snapshot failed a check, so training must not start."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    snapshot: str
    content_sha256: str
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.status == WARN]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"passed": not self.failures}

    def require_passed(self) -> None:
        if self.failures:
            names = "; ".join(f"{check.name}: {check.detail}" for check in self.failures)
            raise DataValidationError(f"{self.snapshot} failed validation: {names}")


def _result(name: str, problems: Sequence[str], passed_detail: str) -> Check:
    if problems:
        shown = ", ".join(problems[:5])
        more = f" and {len(problems) - 5} more" if len(problems) > 5 else ""
        return Check(name, FAIL, f"{shown}{more}")
    return Check(name, PASS, passed_detail)


def _null_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        column: sum(row[column] is None for row in rows) / len(rows)
        for column in SNAPSHOT_SCHEMA.names
    }


def _check_schema(folder: Path, manifest: dict[str, Any]) -> Check:
    actual = pq.read_schema(folder / ROWS_FILE).remove_metadata()
    problems = []
    if not actual.equals(SNAPSHOT_SCHEMA):
        extra = sorted(set(actual.names) - set(SNAPSHOT_SCHEMA.names))
        problems.append(f"schema differs (extra columns: {extra})")
    if manifest["feature_version"] != FEATURE_VERSION:
        problems.append(f"feature_version {manifest['feature_version']}")
    return _result("schema", problems, f"{len(actual.names)} columns as expected")


def _check_hash(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Check:
    matches = content_hash(rows) == manifest["content_sha256"]
    return _result("content_hash", [] if matches else ["rows don't match"], "rows match manifest")


def _check_ids(rows: list[dict[str, Any]]) -> Check:
    counts = Counter(row["rev_id"] for row in rows)
    repeated = [str(rev_id) for rev_id, count in counts.items() if count > 1]
    return _result("unique_rev_id", repeated, f"{len(counts):,} unique")


def _check_splits(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Check:
    windows = SplitWindows(
        *(
            datetime.fromisoformat(manifest["windows"][key])
            for key in ("validation_start", "test_start", "end")
        )
    )
    outside = sum(windows.split_of(row["timestamp"]) != row["split"] for row in rows)
    problems = [f"{outside} rows outside their split's window"] if outside else []
    for split in SPLITS:
        actual = sum(row["split"] == split for row in rows)
        recorded = manifest["splits"][split]["rows"]
        if actual != recorded:
            problems.append(f"{split} has {actual} rows, manifest says {recorded}")
    return _result("split_integrity", problems, "every row fits its window")


def _check_labels(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Check:
    end = datetime.fromisoformat(manifest["windows"]["end"])
    bad = sum(row["label"] not in (0, 1) for row in rows)
    early = sum(row["timestamp"] > end for row in rows)
    problems = []
    if bad:
        problems.append(f"{bad} rows with no valid label")
    if early:
        problems.append(f"{early} rows younger than 48 hours at as_of")
    return _result("label_maturity", problems, "every label is final")


def _check_nulls(rows: list[dict[str, Any]]) -> Check:
    rates = _null_rates(rows)
    problems = [f"{column} {rate:.2%}" for column, rate in rates.items() if rate > 0]
    return _result("nulls", problems, "no nulls in any column")


def _check_ranges(rows: list[dict[str, Any]]) -> Check:
    problems = []
    for column, (low, high) in FEATURE_RANGES.items():
        outside = sum(not low <= row[column] <= high for row in rows)
        if outside:
            problems.append(f"{column}: {outside} rows outside {low} to {high}")
    return _result("feature_ranges", problems, "all within range")


def _check_derived(rows: list[dict[str, Any]]) -> Check:
    """Recompute features from the stored text, which catches a snapshot built with other code."""
    wrong: Counter[str] = Counter()
    for row in rows:
        added = row["added_text"]
        if row["chars_added"] != len(added):
            wrong["chars_added"] += 1
        if row["badword_count"] != count_badwords(added):
            wrong["badword_count"] += 1
        if abs(row["uppercase_ratio"] - uppercase_share(added)) > 1e-9:
            wrong["uppercase_ratio"] += 1
    return _result(
        "derived_features",
        [f"{column}: {count} rows" for column, count in wrong.items()],
        "match the stored text",
    )


def daily_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[str, int] = defaultdict(int)
    positives: dict[str, int] = defaultdict(int)
    for row in rows:
        day = row["timestamp"].strftime("%Y-%m-%d")
        totals[day] += 1
        positives[day] += row["label"]
    return {
        day: {"rows": totals[day], "positive_rate": positives[day] / totals[day]}
        for day in sorted(totals)
    }


def _check_positive_rate(rows: list[dict[str, Any]], days: dict[str, dict[str, float]]) -> Check:
    mean = sum(row["label"] for row in rows) / len(rows)
    limit = MAX_RATE_MULTIPLE * mean
    problems = [
        f"{day} at {stats['positive_rate']:.2%}"
        for day, stats in days.items()
        if stats["positive_rate"] > limit
    ]
    for split in SPLITS:
        members = [row["label"] for row in rows if row["split"] == split]
        rate = sum(members) / len(members) if members else 0.0
        if rate > limit:
            problems.append(f"{split} at {rate:.2%}")
    return _result("positive_rate", problems, f"nothing above 2x the {mean:.2%} mean")


def _check_test_positives(rows: list[dict[str, Any]]) -> Check:
    positives = sum(row["label"] for row in rows if row["split"] == TEST)
    problems = [f"only {positives}"] if positives < MIN_TEST_POSITIVES else []
    return _result("test_positives", problems, f"{positives:,}, minimum {MIN_TEST_POSITIVES}")


def _check_day_rows(days: dict[str, dict[str, float]]) -> Check:
    counts = sorted(stats["rows"] for stats in days.values())
    median = counts[len(counts) // 2]
    thin = [
        f"{day} has {int(stats['rows'])}"
        for day, stats in days.items()
        if stats["rows"] < MIN_DAY_ROWS_SHARE * median
    ]
    if thin:
        return Check("rows_per_day", WARN, f"median {int(median)}; " + ", ".join(thin))
    return Check("rows_per_day", PASS, f"{len(days)} days, median {int(median)} rows")


def _text_key(row: dict[str, Any]) -> str | None:
    if not row["added_text"] and not row["removed_text"]:
        return None
    return f"{row['added_text']}\0{row['removed_text']}"


def duplicate_overlap(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        key = _text_key(row)
        if key is not None:
            keys[row["split"]].add(key)
    later = keys[VALIDATION] | keys[TEST]
    return {
        "train_rows_with_text_in_later_splits": sum(
            row["split"] == TRAIN and _text_key(row) in later for row in rows
        ),
        "test_rows_with_text_in_validation": sum(
            row["split"] == TEST and _text_key(row) in keys[VALIDATION] for row in rows
        ),
    }


def _check_duplicates(overlap: dict[str, int]) -> Check:
    leaked = overlap["train_rows_with_text_in_later_splits"]
    problems = [f"{leaked} train rows repeat later text"] if leaked else []
    detail = f"train clean; {overlap['test_rows_with_text_in_validation']:,} test rows in val"
    return _result("duplicates", problems, detail)


def _check_excluded(manifest: dict[str, Any], rows: int) -> tuple[Check, float]:
    purged = sum(purge["rows_removed"] for purge in manifest["purges"])
    excluded = sum(manifest["excluded"].values()) + purged
    share = excluded / (excluded + rows)
    if share > MAX_EXCLUDED_SHARE:
        return Check("excluded_rate", FAIL, f"{share:.2%} excluded"), share
    return Check("excluded_rate", PASS, f"{share:.2%} excluded"), share


def _bin_edges(values: list[float]) -> list[float]:
    ordered = sorted(values)
    edges = {ordered[len(ordered) * step // PSI_BINS] for step in range(1, PSI_BINS)}
    return sorted(edges)


def _bin_shares(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        low, high = 0, len(edges)
        while low < high:
            middle = (low + high) // 2
            if value < edges[middle]:
                high = middle
            else:
                low = middle + 1
        counts[low] += 1
    # A small floor, so an empty bin doesn't make the score infinite.
    return [max(count / len(values), 1e-4) for count in counts]


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    edges = _bin_edges(expected)
    before = _bin_shares(expected, edges)
    after = _bin_shares(actual, edges)
    return sum((new - old) * math.log(new / old) for old, new in zip(before, after, strict=True))


def _feature_drift(rows: list[dict[str, Any]]) -> dict[str, float]:
    train = [row for row in rows if row["split"] == TRAIN]
    test = [row for row in rows if row["split"] == TEST]
    return {
        feature: round(
            population_stability_index(
                [float(row[feature]) for row in train], [float(row[feature]) for row in test]
            ),
            4,
        )
        for feature in PSI_FEATURES
    }


def _check_drift(drift: dict[str, float]) -> Check:
    moved = [f"{name} {value}" for name, value in drift.items() if value > PSI_WARN]
    if moved:
        return Check("feature_drift", WARN, ", ".join(moved))
    return Check("feature_drift", PASS, f"largest PSI {max(drift.values())}")


def _check_constant_features(rows: list[dict[str, Any]]) -> tuple[Check, list[str]]:
    constant = [
        name
        for name in (*PSI_FEATURES, "user_edit_count_bucket")
        if len({row[name] for row in rows}) == 1
    ]
    if constant:
        return Check("constant_features", WARN, ", ".join(constant) + " never vary"), constant
    return Check("constant_features", PASS, "every feature varies"), constant


def validate_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any], folder: Path | None = None
) -> ValidationReport:
    report = ValidationReport(manifest["name"], manifest["content_sha256"])
    if folder is not None:
        report.checks.append(_check_schema(folder, manifest))
    days = daily_stats(rows)
    overlap = duplicate_overlap(rows)
    drift = _feature_drift(rows)
    excluded, excluded_share = _check_excluded(manifest, len(rows))
    constant, constant_names = _check_constant_features(rows)
    report.checks += [
        _check_hash(rows, manifest),
        _check_ids(rows),
        _check_nulls(rows),
        _check_splits(rows, manifest),
        _check_labels(rows, manifest),
        _check_ranges(rows),
        _check_derived(rows),
        _check_positive_rate(rows, days),
        _check_test_positives(rows),
        _check_day_rows(days),
        _check_duplicates(overlap),
        excluded,
        _check_drift(drift),
        constant,
    ]
    report.numbers = {
        "rows_per_split": {split: manifest["splits"][split]["rows"] for split in SPLITS},
        "null_rates": _null_rates(rows),
        "days": days,
        "duplicate_overlap": overlap,
        "excluded_share": round(excluded_share, 4),
        "psi_train_vs_test": drift,
        "constant_features": constant_names,
    }
    return report


def validate_snapshot(folder: Path) -> ValidationReport:
    """Reads rows directly, not through read_snapshot, so a bad snapshot gets a full report."""
    manifest = read_manifest(folder)
    rows: list[dict[str, Any]] = pq.read_table(folder / ROWS_FILE).to_pylist()
    return validate_rows(rows, manifest, folder)
