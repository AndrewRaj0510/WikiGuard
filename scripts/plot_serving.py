"""Chart Phase 3's measurements for docs/images/.

python scripts/plot_serving.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGES = Path("docs/images")
BATCHES = Path("data/runs/benchmarks/batch_sizes.json")
PLANS = Path("data/query_plans.json")
DATE = "2026-11-07"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def styled(figure_size: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=figure_size, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=MUTED, length=0, labelsize=10)
    axes.set_axisbelow(True)
    return figure, axes


def plot_batches() -> None:
    data = json.loads(BATCHES.read_text("utf-8"))
    figure, axes = styled((8, 4.6))
    axes.grid(axis="y", color=GRID, linewidth=1)
    series = (("roberta", "RoBERTa", BLUE), ("distilbert", "DistilBERT", ORANGE))
    sizes = data["batch_sizes"]
    positions = list(range(len(sizes)))
    for key, label, color in series:
        values = data[key]["ms_per_edit"]
        axes.plot(
            positions, values, color=color, linewidth=2, marker="o", markersize=8, label=label
        )
        for index in (0, len(values) - 1):
            axes.annotate(
                f"{values[index]:.0f} ms",
                (positions[index], values[index]),
                # Below the lower line and above the upper one, so labels clear each other.
                xytext=(0, -18 if key == "distilbert" and index == 0 else 10),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=10,
            )
    axes.set_xticks(positions, [str(size) for size in sizes])
    axes.set_ylim(0, 125)
    axes.set_xlabel("Edits per ONNX batch", color=MUTED, fontsize=10)
    axes.set_ylabel("Milliseconds per edit", color=MUTED, fontsize=10)
    axes.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=10)
    figure.suptitle(
        "Batching makes each edit slower on one thread",
        x=0.08,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    axes.set_title(
        "So the scorer batches Kafka reads and database writes, and scores edits singly (F-098)",
        loc="left",
        color=MUTED,
        fontsize=10,
    )
    figure.text(
        0.08,
        0.01,
        f"Laptop, one thread, real sample edits, {DATE}. RoBERTa: 64 edits, median of 3 rounds. "
        "DistilBERT: 96 edits, 1 round.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(IMAGES / f"serving_batch_size_{DATE}.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_queries() -> None:
    report = json.loads(PLANS.read_text("utf-8"))
    rows: dict[str, float] = {}
    for plan in report["plans"]:
        rows[plan["query"]] = rows.get(plan["query"], 0.0) + plan["execution_ms"]
    ordered = sorted(rows.items(), key=lambda item: item[1])
    figure, axes = styled((8, 5))
    axes.grid(axis="x", color=GRID, linewidth=1)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    axes.barh(names, values, color=BLUE, height=0.55)
    for index, value in enumerate(values):
        axes.annotate(
            f"{value:.2f} ms",
            (value, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
    axes.set_xlim(0, max(values) * 1.15)
    axes.set_xlabel("Execution time (ms)", color=MUTED, fontsize=10)
    axes.tick_params(axis="y", labelcolor=INK)
    figure.suptitle(
        "A page 5,000 rows deep costs no more than the first",
        x=0.03,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    axes.set_title(
        f"The API's real queries on {report['rows']:,} edits, three days at the measured rate",
        loc="left",
        color=MUTED,
        fontsize=10,
    )
    figure.text(
        0.03,
        0.01,
        f"EXPLAIN ANALYZE, local Postgres 17, one run each, {DATE}. Generated data. "
        "Supabase adds network time. Health is cached 60 s.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(IMAGES / f"serving_query_times_{DATE}.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    plot_batches()
    plot_queries()
