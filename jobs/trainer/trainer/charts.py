"""Every chart the evaluation produces, drawn from report numbers alone.

Only totals reach an image, so no edit text, page, or editor can appear in one. Matplotlib rather
than the browser route the backfill charts use, because the trainer also runs on the VM, which has
no browser. File names follow PLAN.md, section 17.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from wikiguard_core.evaluation import precision_recall  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# A fixed order, so each model keeps its color on every chart.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MODEL_ORDER = ("mlp", "bilstm", "distilbert", "bert", "roberta")
MODEL_LABELS = {
    "mlp": "MLP",
    "bilstm": "BiLSTM",
    "distilbert": "DistilBERT",
    "bert": "BERT",
    "roberta": "RoBERTa",
}
SEQUENTIAL = ("#eaf2fc", "#b9d4f4", "#7fb0ea", "#2a78d6", "#14498b")
SIZE = (9.0, 4.8)
DPI = 150
LINE_WIDTH = 2.0
LEFT = 0.13


def color_of(model: str) -> str:
    return SERIES[MODEL_ORDER.index(model) % len(SERIES)] if model in MODEL_ORDER else MUTED


def label_of(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _figure(title: str, subtitle: str = "") -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=SIZE, dpi=DPI, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    # Wide enough for the longest row labels, such as "Constant score".
    figure.subplots_adjust(top=0.82, left=LEFT, right=0.97, bottom=0.14)
    figure.text(LEFT, 0.93, title, fontsize=14, fontweight="semibold", color=INK)
    if subtitle:
        figure.text(LEFT, 0.875, subtitle, fontsize=10, color=SECONDARY)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=SECONDARY, labelsize=9, length=0)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    return figure, axes


def _axis_labels(axes: Axes, x: str, y: str) -> None:
    axes.set_xlabel(x, color=SECONDARY, fontsize=10)
    axes.set_ylabel(y, color=SECONDARY, fontsize=10)


def _save(figure: Figure, folder: Path, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.png"
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def _legend(axes: Axes, location: str = "best") -> None:
    legend = axes.legend(frameon=False, fontsize=9, loc=location)
    for text in legend.get_texts():
        text.set_color(SECONDARY)


def confusion_matrix(report: Mapping[str, Any], folder: Path, day: date) -> Path:
    model = report["model"]
    cells = report["at_threshold"]
    values = np.array([[cells["tn"], cells["fp"]], [cells["fn"], cells["tp"]]], dtype=float)
    rule = "90% precision" if report["threshold_rule"] == "90_percent_precision" else "best F1"
    figure, axes = _figure(
        f"{label_of(model)}: confusion matrix on test",
        f"{int(values.sum()):,} edits, at the threshold chosen on validation for {rule}",
    )
    axes.grid(False)
    shares = np.log1p(values) / np.log1p(values.max())
    colors = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    axes.imshow(shares, cmap=colors, vmin=0, vmax=1, aspect="auto")
    for row in range(2):
        for column in range(2):
            ink = SURFACE if shares[row, column] > 0.6 else INK
            axes.text(
                column,
                row,
                f"{int(values[row, column]):,}",
                ha="center",
                va="center",
                color=ink,
                fontsize=16,
            )
    axes.set_xticks([0, 1], ["Not flagged", "Flagged"])
    axes.set_yticks([0, 1], ["Not reverted", "Reverted"])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return _save(figure, folder, f"eval_confusion_{model}_{day}")


def learning_curves(report: Mapping[str, Any], folder: Path, day: date) -> Path:
    model = report["model"]
    history = report["fit"]["history"]
    epochs = [entry["epoch"] for entry in history]
    figure, axes = _figure(
        f"{label_of(model)}: PR-AUC by epoch",
        f"Epoch 0 is the untrained network. The kept state is epoch {report['fit']['best_epoch']}.",
    )
    axes.plot(
        epochs,
        [entry["train_sample_pr_auc"] for entry in history],
        color=SERIES[0],
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=5,
        label="Train sample",
    )
    axes.plot(
        epochs,
        [entry["validation_pr_auc"] for entry in history],
        color=SERIES[1],
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=5,
        label="Validation",
    )
    axes.set_xticks(epochs)
    axes.set_ylim(bottom=0)
    _axis_labels(axes, "Epoch", "PR-AUC")
    _legend(axes)
    return _save(figure, folder, f"eval_learning_{model}_{day}")


def reliability(report: Mapping[str, Any], folder: Path, day: date) -> Path:
    model = report["model"]
    figure, axes = _figure(
        f"{label_of(model)}: reliability on test",
        f"ECE {report['ece_before']:.3f} before temperature scaling, "
        f"{report['ece_after']:.3f} after (T = {report['temperature']:.2f})",
    )
    axes.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="Perfect calibration")
    for name, key, color in (
        ("Before", "reliability_before", SERIES[0]),
        ("After", "reliability_after", SERIES[1]),
    ):
        points = report[key]
        axes.plot(
            [point["mean_score"] for point in points],
            [point["positive_rate"] for point in points],
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=5,
            label=name,
        )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    _axis_labels(axes, "Mean score in bin", "Share reverted")
    _legend(axes, "upper left")
    return _save(figure, folder, f"eval_reliability_{model}_{day}")


def slices(report: Mapping[str, Any], folder: Path, day: date) -> Path:
    model = report["model"]
    entries = [
        (name.replace("_", " ").capitalize(), values)
        for name, values in report["slices"].items()
        if values["rows"]
    ]
    positions = np.arange(len(entries))
    figure, axes = _figure(
        f"{label_of(model)}: false positive rate by slice",
        "Share of good edits flagged at the threshold. Edit-count slices can't be measured yet.",
    )
    axes.barh(
        positions,
        [values["false_positive_rate"] * 100 for _, values in entries],
        color=SERIES[0],
        height=0.5,
    )
    for position, (_, values) in zip(positions, entries, strict=True):
        axes.text(
            values["false_positive_rate"] * 100,
            position,
            f"  {values['false_positive_rate']:.2%} (PR-AUC {values.get('pr_auc', 0):.3f})",
            va="center",
            color=SECONDARY,
            fontsize=9,
        )
    axes.set_yticks(positions, [name for name, _ in entries])
    axes.invert_yaxis()
    axes.grid(False, axis="y")
    largest = max(values["false_positive_rate"] * 100 for _, values in entries)
    axes.set_xlim(0, largest * 1.5 or 1)
    _axis_labels(axes, "False positive rate (%)", "")
    return _save(figure, folder, f"eval_slices_{model}_{day}")


def pr_curves(
    scores: Mapping[str, tuple[Any, Any]], prevalence: float, folder: Path, day: date
) -> Path:
    """One curve per model on the same test rows, from each served model's own scores."""
    figure, axes = _figure(
        "Precision and recall on the test window",
        f"Each served model on the same edits. A constant score sits at the {prevalence:.2%} line.",
    )
    for name, (labels, logits) in scores.items():
        curve = precision_recall(labels, logits)
        axes.plot(
            curve.recall,
            curve.precision,
            color=color_of(name),
            linewidth=LINE_WIDTH,
            label=label_of(name),
        )
    axes.axhline(prevalence, color=MUTED, linewidth=1, linestyle="--", label="Constant score")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    _axis_labels(axes, "Recall", "Precision")
    _legend(axes, "upper right")
    return _save(figure, folder, f"eval_pr_curves_all_{day}")


def leaderboard(rows: Sequence[Mapping[str, Any]], folder: Path, day: date) -> Path:
    """PR-AUC with its interval, best first, and the baselines as plain reference points."""
    ordered = sorted(rows, key=lambda row: row["pr_auc"])
    positions = np.arange(len(ordered))
    figure, axes = _figure(
        "Models and baselines on the test window",
        "PR-AUC with its 95% bootstrap interval. Higher is better.",
    )
    for position, row in zip(positions, ordered, strict=True):
        low, high = row["pr_auc_interval"]
        color = color_of(row["name"])
        axes.plot([low, high], [position, position], color=color, linewidth=LINE_WIDTH)
        axes.plot(row["pr_auc"], position, marker="o", markersize=7, color=color)
        axes.text(
            high, position, f"  {row['pr_auc']:.3f}", va="center", color=SECONDARY, fontsize=9
        )
    axes.set_yticks(positions, [row.get("label", label_of(row["name"])) for row in ordered])
    axes.grid(False, axis="y")
    axes.set_xlim(left=0)
    _axis_labels(axes, "PR-AUC", "")
    return _save(figure, folder, f"report_leaderboard_{day}")


def aging(results: Sequence[Mapping[str, Any]], folder: Path, day: date) -> Path:
    figure, axes = _figure(
        "MLP: PR-AUC by weeks since training",
        "Each line is one training window, scored on each later week.",
    )
    windows = sorted({row["train_weeks"] for row in results})
    for index, window in enumerate(windows):
        members = sorted(
            (row for row in results if row["train_weeks"] == window),
            key=lambda row: row["weeks_after_training"],
        )
        axes.plot(
            [row["weeks_after_training"] for row in members],
            [row["pr_auc"] for row in members],
            color=SERIES[index % len(SERIES)],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=6,
            label=f"Trained on {window} week{'s' if window > 1 else ''}",
        )
    axes.set_xticks(sorted({row["weeks_after_training"] for row in results}))
    axes.set_ylim(bottom=0)
    _axis_labels(axes, "Weeks after training", "PR-AUC")
    _legend(axes)
    return _save(figure, folder, f"eval_aging_mlp_{day}")


def robustness(reports: Mapping[str, Mapping[str, Any]], folder: Path, day: date) -> Path:
    figure, axes = _figure(
        "PR-AUC lost to disguised text",
        "500 reverted and 500 good edits, with added text disguised. Higher means more is lost.",
    )
    names = list(next(iter(reports.values()))["changes"])
    positions = np.arange(len(names))
    width = 0.8 / len(reports)
    for index, (model, report) in enumerate(reports.items()):
        axes.bar(
            positions + index * width,
            [report["changes"][name]["pr_auc_drop"] for name in names],
            width=width * 0.9,
            color=color_of(model),
            label=label_of(model),
        )
    axes.set_xticks(
        positions + width * (len(reports) - 1) / 2,
        [name.replace("_", " ").capitalize() for name in names],
    )
    axes.grid(False, axis="x")
    _axis_labels(axes, "", "PR-AUC drop")
    _legend(axes)
    return _save(figure, folder, f"eval_robustness_all_{day}")


def latency(rows: Sequence[Mapping[str, Any]], folder: Path, day: date) -> Path:
    """p50 and p95 from the benchmark on one machine, with a log scale for the wide range."""
    positions = np.arange(len(rows))
    figure, axes = _figure(
        "Time to score one edit",
        "Served int8 model, one edit at a time, one CPU thread. Log scale.",
    )
    axes.barh(
        positions + 0.18,
        [row["single_edit"]["p95_ms"] for row in rows],
        height=0.32,
        color=SERIES[1],
        label="p95",
    )
    axes.barh(
        positions - 0.18,
        [row["single_edit"]["p50_ms"] for row in rows],
        height=0.32,
        color=SERIES[0],
        label="p50",
    )
    axes.set_xscale("log")
    axes.set_yticks(positions, [label_of(row["model"]) for row in rows])
    axes.grid(False, axis="y")
    _axis_labels(axes, "Milliseconds", "")
    _legend(axes, "lower right")
    return _save(figure, folder, f"report_latency_{day}")
