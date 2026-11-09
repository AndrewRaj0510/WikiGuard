"""Write a small precision-recall curve for each trained model, for the app's Models page.

The app draws its own charts from API data (PLAN section 17), and the API reads model_versions, so
the curve has to travel with the model's metrics. This takes each model's stored test scores
(data/runs/NAME/scores.npz), joins them to the snapshot's test labels by rev_id, and keeps about
100 points: one at each whole percent of recall. sync_model_versions.py copies the result.

Joined by rev_id, not by position: the snapshot was rechecked after training and lost a few rows
(F-092), so a positional match would pair the wrong labels. Rows missing from either side are left
out, and the file says how many were used.

    python scripts/export_pr_curves.py snapshots/snapshot-20261104-28c825af4b19 --runs data/runs
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from wikiguard_core.evaluation import precision_recall
from wikiguard_core.snapshot import read_snapshot

CURVE_FILE = "pr_curve.json"
STEPS = 100


def test_labels(snapshot: Path) -> dict[int, int]:
    return {
        int(row["rev_id"]): int(row["label"])
        for row in read_snapshot(snapshot)
        if row["split"] == "test"
    }


def curve_points(labels: np.ndarray, scores: np.ndarray, steps: int = STEPS) -> list[list[float]]:
    """[recall, precision] at the first point that reaches each 1/steps of recall."""
    curve = precision_recall(labels, scores)
    recall = np.asarray(curve.recall, dtype=float)
    precision = np.asarray(curve.precision, dtype=float)
    points: list[list[float]] = []
    for target in np.linspace(0, 1, steps + 1)[1:]:
        reached = np.flatnonzero(recall >= target - 1e-12)
        if reached.size:
            index = int(reached[0])
            points.append([round(float(recall[index]), 4), round(float(precision[index]), 4)])
    return points


def export(snapshot: Path, runs: Path) -> list[str]:
    labels_by_rev = test_labels(snapshot)
    aliases: dict[str, Any] = json.loads((runs / "aliases.json").read_text("utf-8"))
    written = []
    for entry in aliases.values():
        name = entry["model"]
        stored = np.load(runs / name / "scores.npz", allow_pickle=False)
        rev_ids = stored["test_rev_id"]
        keep = np.array([int(rev) in labels_by_rev for rev in rev_ids])
        labels = np.array([labels_by_rev[int(rev)] for rev in rev_ids[keep]], dtype=np.int64)
        scores = stored["test_logit"][keep]
        result = {
            "points": curve_points(labels, scores),
            "rows": int(keep.sum()),
            "rows_scored": int(rev_ids.size),
            "positives": int(labels.sum()),
            "prevalence": round(float(labels.mean()), 6),
            "snapshot": snapshot.name,
        }
        (runs / name / CURVE_FILE).write_text(json.dumps(result, indent=2) + "\n", "utf-8")
        written.append(f"{name}: {result['rows']} of {result['rows_scored']} test rows")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("data/runs"))
    args = parser.parse_args()
    for line in export(args.snapshot, args.runs):
        print(line)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
