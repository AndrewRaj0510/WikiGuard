"""Draw the backfill's charts from the files backfill.py and measure_capacity.py saved.

Writes SVG with no plotting library, then uses a browser you already have to make the PNG. Every
number on a chart is computed here, titles included, so a chart can't disagree with its data. Only
totals are drawn: no editor, page, or edit text can reach an image.

    python scripts/plot_backfill.py data/backfill docs/images --browser "C:/.../chrome.exe"
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from measure_capacity import LIMIT_BYTES, project

WIDTH = 900
HEIGHT = 460
LEFT = 70
RIGHT = 40
TOP = 110
BOTTOM = 60
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
FONT = "'Segoe UI', Arial, sans-serif"
LABEL_HOURS = 48
MEGABYTE = 1024 * 1024
# (label, days a row stays). The first is the plan's starting rule, and the last is DEC-038.
RULES = (
    ("14 days (plan's first rule)", 14),
    ("5 days (first proposal)", 5),
    ("3 days (chosen)", 3),
)
HOUR_TICKS = (
    (1 / 60, "1 min"),
    (1 / 6, "10 min"),
    (1, "1 hour"),
    (6, "6 hours"),
    (24, "1 day"),
    (168, "1 week"),
    (672, "4 weeks"),
)


@dataclass
class Chart:
    name: str
    svg: str


def daily_rates(rows: list[dict[str, Any]]) -> list[tuple[str, float, int]]:
    """(day, share reverted within 48 hours, labeled edits), leaving out edits with no label."""
    labeled: dict[str, int] = defaultdict(int)
    positive: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["label"] is None:
            continue
        day = row["timestamp"].strftime("%Y-%m-%d")
        labeled[day] += 1
        positive[day] += row["label"]
    return [(day, positive[day] / labeled[day], labeled[day]) for day in sorted(labeled)]


def share_within(hours: list[float], limit: float) -> float:
    return sum(1 for value in hours if value <= limit) / len(hours)


def retention_sizes(capacity: dict[str, Any], parquet_per_edit: float) -> list[tuple[str, float]]:
    """Megabytes each rule would hold, at the measured row sizes and edit rate."""
    sizes = []
    for label, days in RULES:
        projected = project(capacity, capacity["edits_per_day"], parquet_per_edit, days)
        sizes.append((label, projected["steady_state_bytes_normal_rules"] / MEGABYTE))
    return sizes


def text(x: float, y: float, content: str, **style: Any) -> str:
    size = style.get("size", 13)
    fill = style.get("fill", SECONDARY)
    anchor = style.get("anchor", "start")
    weight = style.get("weight", 400)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(content)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}"/>'


def page(title: str, subtitle: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
            text(30, 40, title, size=20, fill=INK, weight=600),
            text(30, 66, subtitle, size=14),
            *body,
            "</svg>",
        ]
    )


def polyline(points: list[tuple[float, float]]) -> str:
    joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{joined}" fill="none" stroke="{SERIES}" stroke-width="2" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def dot(x: float, y: float) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{SERIES}" '
        f'stroke="{SURFACE}" stroke-width="2"/>'
    )


def positive_rate_chart(days: list[tuple[str, float, int]], overall: float) -> Chart:
    bottom = HEIGHT - BOTTOM
    right = WIDTH - RIGHT
    top_rate = 0.08

    def x_at(index: int) -> float:
        return LEFT + (right - LEFT) * index / (len(days) - 1)

    def y_at(rate: float) -> float:
        return bottom - (bottom - TOP) * rate / top_rate

    body: list[str] = []
    for percent in range(0, 9, 2):
        y = y_at(percent / 100)
        body.append(line(LEFT, y, right, y, AXIS if percent == 0 else GRID))
        body.append(text(LEFT - 10, y + 4, f"{percent}%", anchor="end", fill=MUTED))
    for index, (day, _, _) in enumerate(days):
        if index % 7 == 0:
            parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
            label = f"{parsed:%b} {parsed.day}"
            body.append(text(x_at(index), bottom + 22, label, anchor="middle", fill=MUTED))
    body.append(line(LEFT, y_at(overall), right, y_at(overall), MUTED))
    body.append(polyline([(x_at(i), y_at(rate)) for i, (_, rate, _) in enumerate(days)]))
    low = min(range(len(days)), key=lambda i: days[i][1])
    high = max(range(len(days)), key=lambda i: days[i][1])
    for index, place in ((low, 20), (high, -12)):
        rate = days[index][1]
        body.append(dot(x_at(index), y_at(rate)))
        body.append(text(x_at(index), y_at(rate) + place, f"{rate:.2%}", anchor="middle", fill=INK))
    body.append(text(LEFT + 4, y_at(overall) + 32, f"All days: {overall:.2%}"))
    return Chart(
        "backfill_positive_rate_by_day",
        page(
            f"{overall:.1%} of backfilled edits were reverted within 48 hours",
            f"Share of labeled edits, per day. Daily rates ran from {days[low][1]:.1%} to "
            f"{days[high][1]:.1%}. About 3,600 sampled edits a day.",
            body,
        ),
    )


def revert_delay_chart(hours: list[float]) -> Chart:
    bottom = HEIGHT - BOTTOM
    right = WIDTH - RIGHT
    low_hours, high_hours = 1 / 60, 672.0

    def x_at(value: float) -> float:
        clamped = min(max(value, low_hours), high_hours)
        share = math.log(clamped / low_hours) / math.log(high_hours / low_hours)
        return LEFT + (right - LEFT) * share

    def y_at(share: float) -> float:
        return bottom - (bottom - TOP) * share

    body: list[str] = []
    for percent in range(0, 101, 25):
        y = y_at(percent / 100)
        body.append(line(LEFT, y, right, y, AXIS if percent == 0 else GRID))
        body.append(text(LEFT - 10, y + 4, f"{percent}%", anchor="end", fill=MUTED))
    for value, label in HOUR_TICKS:
        body.append(text(x_at(value), bottom + 22, label, anchor="middle", fill=MUTED))
    ordered = sorted(hours)
    position = 0
    points = []
    for index in range(201):
        step = low_hours * (high_hours / low_hours) ** (index / 200)
        while position < len(ordered) and ordered[position] <= step:
            position += 1
        points.append((x_at(step), y_at(position / len(ordered))))
    cutoff = x_at(LABEL_HOURS)
    body.append(line(cutoff, TOP, cutoff, bottom, MUTED))
    body.append(polyline(points))
    in_hour = share_within(hours, 1)
    in_window = share_within(hours, LABEL_HOURS)
    body.append(dot(x_at(1), y_at(in_hour)))
    body.append(text(x_at(1) + 10, y_at(in_hour) + 18, f"{in_hour:.1%} within an hour", fill=INK))
    body.append(dot(cutoff, y_at(in_window)))
    body.append(
        text(
            cutoff - 10,
            y_at(in_window) - 12,
            f"{in_window:.1%} within 48 hours",
            anchor="end",
            fill=INK,
        )
    )
    body.append(text(cutoff + 8, bottom - 12, "48-hour label cutoff"))
    return Chart(
        "backfill_revert_delay",
        page(
            f"Half of reverts come within an hour, but {1 - in_window:.1%} miss the 48-hour cutoff",
            f"Cumulative share of {len(hours):,} dated reverts, by time from edit to revert. "
            "The log scale spreads out the first hours.",
            body,
        ),
    )


def retention_chart(sizes: list[tuple[str, float]]) -> Chart:
    bottom = HEIGHT - BOTTOM
    right = WIDTH - RIGHT
    start = 210
    limit = LIMIT_BYTES / MEGABYTE
    most = max(limit, *(size for _, size in sizes)) * 1.05

    def x_at(size: float) -> float:
        return start + (right - start) * size / most

    body: list[str] = []
    for mark in range(0, int(most) + 1, 250):
        body.append(line(x_at(mark), TOP, x_at(mark), bottom, AXIS if mark == 0 else GRID))
        body.append(text(x_at(mark), bottom + 22, f"{mark} MB", anchor="middle", fill=MUTED))
    gap = (bottom - TOP) / len(sizes)
    for index, (label, size) in enumerate(sizes):
        y = TOP + gap * index + gap / 2 - 12
        chosen = index == len(sizes) - 1
        end = x_at(size)
        body.append(
            f'<path d="M{start},{y:.1f} H{end - 4:.1f} Q{end:.1f},{y:.1f} {end:.1f},{y + 4:.1f} '
            f'V{y + 20:.1f} Q{end:.1f},{y + 24:.1f} {end - 4:.1f},{y + 24:.1f} H{start} Z" '
            f'fill="{SERIES if chosen else AXIS}"/>'
        )
        body.append(
            text(start - 12, y + 17, label, anchor="end", fill=INK if chosen else SECONDARY)
        )
        beside = f"{size:.0f} MB, {size / limit:.0%}"
        if end + 130 > right:
            body.append(text(end - 8, y + 17, beside, anchor="end", fill=INK))
        else:
            body.append(text(end + 8, y + 17, beside, fill=INK))
    body.append(line(x_at(limit), TOP, x_at(limit), bottom, INK))
    body.append(text(x_at(limit) + 6, TOP + 12, "500 MB free limit", fill=INK))
    final = sizes[-1][1]
    return Chart(
        "capacity_retention_rules",
        page(
            f"Keeping edits for 3 days uses {final / limit:.0%} of Supabase's free limit",
            "Projected database size at the measured row sizes and edit rate. Text stays 48 "
            "hours; assumes 5% high-risk.",
            body,
        ),
    )


def render(svg: Path, png: Path, browser: str) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run(  # noqa: S603
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={WIDTH},{HEIGHT}",
                f"--user-data-dir={profile}",
                f"--screenshot={png.resolve()}",
                svg.resolve().as_uri(),
            ],
            check=True,
            timeout=120,
            capture_output=True,
        )
    # Edge can exit with 0 and write nothing, so success has to mean the file is there.
    if not png.exists():
        raise SystemExit(f"{browser} wrote no image. Try Chrome.")


def build(folder: Path) -> list[Chart]:
    edits = pq.read_table(folder / "parts", columns=["rev_id", "timestamp"]).to_pylist()
    labels = pq.read_table(folder / "labels_48h.parquet").to_pylist()
    by_id = {row["rev_id"]: row for row in labels}
    rows = [{**row, "label": by_id[row["rev_id"]]["label"]} for row in edits]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    capacity = json.loads((folder / "capacity.json").read_text(encoding="utf-8"))
    hours = [row["revert_hours"] for row in labels if row["revert_hours"] is not None]
    return [
        positive_rate_chart(daily_rates(rows), summary["positive_rate_48h"]),
        revert_delay_chart(hours),
        retention_chart(retention_sizes(capacity, summary["parquet_bytes"] / summary["rows"])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--browser", help="Chrome, to make PNGs; otherwise SVG only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    for chart in build(args.folder):
        svg = args.out / f"{chart.name}_{stamp}.svg"
        svg.write_text(chart.svg, encoding="utf-8")
        if args.browser:
            render(svg, svg.with_suffix(".png"), args.browser)
            svg.unlink()
        print(f"Wrote {chart.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
