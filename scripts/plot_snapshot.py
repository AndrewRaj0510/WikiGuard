"""Draw a snapshot's split chart from its files, using the same drawing helpers as the backfill.

Every number comes from the snapshot, titles included. Only totals are drawn, so no editor, page,
or edit text can reach the image.

    python scripts/plot_snapshot.py snapshots/NAME docs/images --browser "C:/.../chrome.exe"
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from plot_backfill import (
    AXIS,
    BOTTOM,
    GRID,
    HEIGHT,
    INK,
    LEFT,
    MUTED,
    RIGHT,
    SURFACE,
    TOP,
    WIDTH,
    Chart,
    line,
    page,
    polyline,
    render,
    text,
)
from wikiguard_core.snapshot import SPLITS, TEST, VALIDATION, read_manifest, read_snapshot

TOP_RATE = 0.10
GRID_PERCENTS = (0, 2, 4, 6, 8, 10)
# Neutral bands, so the only color on the chart is the rate itself.
BAND_FILL = {VALIDATION: "#f1f0ec", TEST: "#e7e6e0"}
DAY_LABEL_EVERY = 7


def daily_positive_rates(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """(day, positive rate), for each day that has rows."""
    totals: dict[str, int] = defaultdict(int)
    positives: dict[str, int] = defaultdict(int)
    for row in rows:
        day = row["timestamp"].strftime("%Y-%m-%d")
        totals[day] += 1
        positives[day] += row["label"]
    return [(day, positives[day] / totals[day]) for day in sorted(totals)]


def split_chart(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Chart:
    days = daily_positive_rates(rows)
    first = datetime.strptime(days[0][0], "%Y-%m-%d")
    span = (datetime.strptime(days[-1][0], "%Y-%m-%d") - first).days + 1
    bottom = HEIGHT - BOTTOM
    right = WIDTH - RIGHT
    splits = manifest["splits"]

    def x_at(moment: datetime) -> float:
        share = (moment.replace(tzinfo=None) - first).total_seconds() / timedelta(
            days=span
        ).total_seconds()
        return LEFT + (right - LEFT) * min(max(share, 0.0), 1.0)

    def y_at(rate: float) -> float:
        return bottom - (bottom - TOP) * rate / TOP_RATE

    windows = manifest["windows"]
    validation_start = datetime.fromisoformat(windows["validation_start"])
    test_start = datetime.fromisoformat(windows["test_start"])
    bands = {
        "train": (float(LEFT), x_at(validation_start)),
        VALIDATION: (x_at(validation_start), x_at(test_start)),
        TEST: (x_at(test_start), float(right)),
    }

    body: list[str] = []
    for name, fill in BAND_FILL.items():
        start, end = bands[name]
        body.append(
            f'<rect x="{start:.1f}" y="{TOP}" width="{end - start:.1f}" '
            f'height="{bottom - TOP}" fill="{fill}"/>'
        )
    for percent in GRID_PERCENTS:
        y = y_at(percent / 100)
        body.append(line(LEFT, y, right, y, AXIS if percent == 0 else GRID))
        body.append(text(LEFT - 10, y + 4, f"{percent}%", anchor="end", fill=MUTED))
    for index in range(0, span, DAY_LABEL_EVERY):
        day = first + timedelta(days=index)
        label = f"{day:%b} {day.day}"
        body.append(text(x_at(day), bottom + 22, label, anchor="middle", fill=MUTED))
    for name in SPLITS:
        start, end = bands[name]
        middle = (start + end) / 2
        stats = splits[name]
        body.append(
            text(middle, TOP + 20, name.capitalize(), anchor="middle", fill=INK, weight=600)
        )
        body.append(text(middle, TOP + 38, f"{stats['rows']:,} edits", anchor="middle"))
        body.append(
            text(middle, TOP + 56, f"{stats['positive_rate']:.2%} reverted", anchor="middle")
        )
    points = [
        (x_at(datetime.strptime(day, "%Y-%m-%d") + timedelta(hours=12)), y_at(rate))
        for day, rate in days
    ]
    body.append(polyline(points))
    for x, y in points:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{SURFACE}" stroke="#2a78d6"/>')

    removed = manifest["duplicates"]["removed_from_train"]
    return Chart(
        "snapshot_splits_by_day",
        page(
            f"{manifest['rows']:,} edits, split by time alone",
            f"Share reverted within 48 hours, per day. Train lost {removed:,} rows whose text "
            "repeats in a later split.",
            body,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--browser", help="Chrome, to make a PNG; otherwise SVG only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.folder)
    chart = split_chart(read_snapshot(args.folder), manifest)
    stem = f"{chart.name}_{manifest['name']}"
    svg = args.out / f"{stem}.svg"
    svg.write_text(chart.svg, encoding="utf-8")
    if args.browser:
        render(svg, svg.with_suffix(".png"), args.browser)
        svg.unlink()
    print(f"Wrote {stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
