"""How fast a model goes stale: train on older weeks, then score each later week separately.

Uses the MLP, since one fold takes minutes rather than hours. The result tests whether the 7-day
retraining default makes sense. PR-AUC depends on each week's positive rate, so the lift over that
rate is reported too.

    python -m trainer.aging snapshots/NAME --out data/runs/aging.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from wikiguard_core.evaluation import pr_auc

from trainer.data import Dataset, labels_of, load_dataset
from trainer.loop import fit, predict_logits
from trainer.recipes import encode_like, prepare

WEEK = timedelta(days=7)
# The last part of each fold's training weeks stops training, so later weeks stay untouched.
EARLY_STOP_SHARE = 0.15
MIN_WEEK_ROWS = 1000
EPOCHS = 8


def weekly_windows(data: Dataset) -> list[list[dict[str, Any]]]:
    """Rows per week counted back from the end of the data, oldest week first."""
    rows = [*data.train, *data.validation, *data.test]
    end = datetime.fromisoformat(data.manifest["windows"]["end"])
    first = min(row["timestamp"] for row in rows)
    weeks = []
    stop = end
    while stop > first:
        start = stop - WEEK
        members = [row for row in rows if start < row["timestamp"] <= stop]
        if len(members) >= MIN_WEEK_ROWS:
            weeks.append(members)
        stop = start
    return list(reversed(weeks))


def run_fold(
    weeks: list[list[dict[str, Any]]],
    train_weeks: int,
    data: Dataset,
    params: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    history = [row for week in weeks[:train_weeks] for row in week]
    cut = int(len(history) * (1 - EARLY_STOP_SHARE))
    fold = Dataset(data.manifest, history[:cut], history[cut:], [])
    prepared = prepare("mlp", fold, params | {"epochs": EPOCHS}, device, seed)
    result = fit(
        prepared.model,
        prepared.train,
        prepared.train_sample,
        prepared.validation,
        prepared.fit_config,
        device,
    )
    scores = []
    for age, week in enumerate(weeks[train_weeks:], start=1):
        encoded = encode_like(prepared, week, f"week{train_weeks + age}", data)
        logits = predict_logits(prepared.model, encoded, 256, device)
        labels = labels_of(week)
        rate = float(labels.mean())
        score = pr_auc(labels, logits)
        scores.append(
            {
                "train_weeks": train_weeks,
                "train_rows": len(history),
                "weeks_after_training": age,
                "rows": len(week),
                "positive_rate": rate,
                "pr_auc": score,
                "lift_over_positive_rate": score / rate,
                "best_epoch": result.best_epoch,
            }
        )
    return scores


def backtest(
    data: Dataset, params: dict[str, Any], device: torch.device, seed: int = 0
) -> list[dict[str, Any]]:
    weeks = weekly_windows(data)
    results: list[dict[str, Any]] = []
    for train_weeks in range(1, len(weeks)):
        results += run_fold(weeks, train_weeks, data, params, device, seed)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--params", type=json.loads, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/runs/aging.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = backtest(load_dataset(args.snapshot), args.params, torch.device(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), "utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
