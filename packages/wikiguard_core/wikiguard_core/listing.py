"""What model_versions holds for each version: the leaderboard's numbers and its model card.

Shared by scripts/sync_model_versions.py and the trainer, so a retrained version is listed exactly
like the first five. Counts and rates only: never anything that names an edit or an editor.
"""

import json
from pathlib import Path
from typing import Any

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
SLICE_FIELDS = ("rows", "positives", "pr_auc", "false_positive_rate")
CURVE_FILE = "pr_curve.json"  # from scripts/export_pr_curves.py
GOLDEN_FILE = "golden.json"


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
    golden = folder / GOLDEN_FILE
    if golden.is_file():
        card["golden_pr_auc"] = json.loads(golden.read_text("utf-8"))["pr_auc"]
    curve = folder / CURVE_FILE
    if curve.is_file():
        card["pr_curve"] = json.loads(curve.read_text("utf-8"))
    return card


def metrics_of(report: dict[str, Any], folder: Path, config: dict[str, Any]) -> dict[str, Any]:
    """The metrics column for one version. folder holds the run, config is the served config."""
    metrics = {key: report[key] for key in REPORT_FIELDS if key in report}
    metrics["card"] = card_of(report, folder)
    # The monitor reads alert rates and false positives at the threshold the model serves with.
    if "threshold_probability" in config:
        metrics["threshold_probability"] = config["threshold_probability"]
    return metrics
