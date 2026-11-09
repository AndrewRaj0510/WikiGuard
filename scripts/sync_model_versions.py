"""Copy the trainer's results into model_versions, so the leaderboard has something to show.

Serving reads aliases from the registry. This table is a small copy for the API, because the API's
pages shouldn't depend on DagsHub being up. Phase 5's trainer does this itself after promoting.

    python scripts/sync_model_versions.py --runs data/runs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from wikiguard_core.config import ConfigError, DatabaseSettings, load_settings

ARTIFACT_FOLDER = "model"
# What the leaderboard and model cards show. Measured results only, and nothing with row content.
REPORT_FIELDS = (
    "pr_auc",
    "pr_auc_interval",
    "recall_at_90_precision",
    "recall_at_90_precision_interval",
    "precision_at_50",
    "roc_auc",
    "ece_after",
    "served_bytes",
    "training_warnings",
)
# The rest of a model card (docs/model-cards), as numbers the Models page can lay out itself.
CARD_FIELDS = (
    "threshold_rule",
    "at_threshold",
    "ece_before",
    "train_rows",
    "rows",
    "positives",
    "params",
    "parameters",
    "device_name",
    "snapshot",
    "false_positive_gap_anonymous_minus_registered",
)
# Per slice, only counts and rates: never anything that names an edit or an editor.
SLICE_FIELDS = ("rows", "positives", "pr_auc", "false_positive_rate")
CURVE_FILE = "pr_curve.json"  # from scripts/export_pr_curves.py
UPSERT = """
    INSERT INTO model_versions (model_name, model_version, alias, feature_version, snapshot_hash,
        artifact_sha256, metrics)
    VALUES (%(model_name)s, %(model_version)s, %(alias)s, %(feature_version)s, %(snapshot_hash)s,
        %(artifact_sha256)s, %(metrics)s)
    ON CONFLICT (model_name, model_version) DO UPDATE SET
        alias = EXCLUDED.alias, metrics = EXCLUDED.metrics
"""


def card_of(report: dict[str, Any], folder: Path) -> dict[str, Any]:
    card: dict[str, Any] = {key: report[key] for key in CARD_FIELDS if key in report}
    card["slices"] = {
        name: {key: values[key] for key in SLICE_FIELDS if key in values}
        for name, values in report.get("slices", {}).items()
        if isinstance(values, dict)
    }
    if "fit" in report:
        card["fit_minutes"] = round(report["fit"]["seconds"] / 60, 1)
    if "parity" in report:
        card["parity_gap"] = report["parity"]["test"]["gap"]
    golden = folder / "golden.json"
    if golden.is_file():
        card["golden_pr_auc"] = json.loads(golden.read_text("utf-8"))["pr_auc"]
    curve = folder / CURVE_FILE
    if curve.is_file():
        card["pr_curve"] = json.loads(curve.read_text("utf-8"))
    return card


def read_runs(runs: Path) -> list[dict[str, Any]]:
    aliases = json.loads((runs / "aliases.json").read_text("utf-8"))
    rows = []
    for alias, entry in aliases.items():
        name = entry["model"]
        report = json.loads((runs / name / "report.json").read_text("utf-8"))
        config = json.loads((runs / name / ARTIFACT_FOLDER / "config.json").read_text("utf-8"))
        metrics = {key: report[key] for key in REPORT_FIELDS if key in report}
        metrics["card"] = card_of(report, runs / name)
        benchmark_file = runs / "benchmarks" / f"laptop_{name}.json"
        if benchmark_file.is_file():
            benchmark = json.loads(benchmark_file.read_text("utf-8"))
            metrics["laptop_p50_ms"] = benchmark["single_edit"]["p50_ms"]
            metrics["laptop_p95_ms"] = benchmark["single_edit"]["p95_ms"]
        rows.append(
            {
                "model_name": name,
                "model_version": str(entry["version"]),
                "alias": alias,
                "feature_version": config["feature_version"],
                "snapshot_hash": report["snapshot_content_sha256"],
                "artifact_sha256": report["artifact_sha256"],
                "metrics": Jsonb(metrics),
            }
        )
    return rows


def sync(connection: psycopg.Connection[Any], rows: list[dict[str, Any]]) -> int:
    # An alias belongs to one version, so a version not named here loses any it had.
    connection.execute("UPDATE model_versions SET alias = NULL WHERE alias IS NOT NULL")
    for row in rows:
        connection.execute(UPSERT, row)
    connection.commit()
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("data/runs"))
    args = parser.parse_args()
    try:
        settings = load_settings(DatabaseSettings)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        print(f"Synced {sync(connection, read_runs(args.runs))} model versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
