"""The numbers every model has to beat, measured before any training.

    python -m trainer.baselines snapshots/NAME
    python -m trainer.baselines snapshots/NAME --fetch-lift-wing

Prevalence is what a constant score gets. The rule baseline is a fixed formula with no fitting.
Lift Wing is Wikimedia's own RevertRisk model. Its results are fetched once and kept as a fixture,
because tests may not call Wikimedia, and the model behind the API changes without notice.
"""

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray
from wikiguard_core.evaluation import (
    bootstrap_interval,
    pr_auc,
    precision_at_k,
    recall_at_precision,
    roc_auc,
)
from wikiguard_core.inputs import Row
from wikiguard_core.snapshot import TEST, read_manifest, read_snapshot

LIFT_WING_URL = (
    "https://api.wikimedia.org/service/lw/inference/v1/models/revertrisk-language-agnostic:predict"
)
LIFT_WING_SAMPLE = 2000
LIFT_WING_FIXTURE = Path("tests/fixtures/liftwing_revertrisk.jsonl")
USER_AGENT_VARIABLE = "WIKIMEDIA_USER_AGENT"
DEFAULT_USER_AGENT = "WikiGuard/0.1 (https://github.com/AndrewRaj0510/WikiGuard)"
# Serial requests with a pause, within Wikimedia's guidance for anonymous clients.
PAUSE_SECONDS = 0.25
MAX_ATTEMPTS = 4
OK = 200
RATE_LIMITED = 429
SAMPLE_SEED = 20261106
BADWORD_WEIGHT = 2.0
ANONYMOUS_WEIGHT = 1.0
# Log of bytes changed, scaled so that it breaks ties more than it decides.
SIZE_WEIGHT = 0.1


def rule_score(row: Row) -> float:
    """Bad words, logged-out, and big changes raise the score. Chosen by hand, and never fitted."""
    return (
        BADWORD_WEIGHT * math.log1p(row["badword_count"])
        + ANONYMOUS_WEIGHT * float(row["is_anonymous"])
        + SIZE_WEIGHT * math.log1p(abs(row["bytes_delta"]))
    )


def _measure(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> dict[str, Any]:
    low, high = bootstrap_interval(pr_auc, labels, scores)
    return {
        "rows": int(labels.size),
        "positives": int(labels.sum()),
        "pr_auc": pr_auc(labels, scores),
        "pr_auc_interval": [low, high],
        "roc_auc": roc_auc(labels, scores),
        "recall_at_90_precision": recall_at_precision(labels, scores),
        "precision_at_50": precision_at_k(labels, scores),
    }


def lift_wing_sample(rows: Sequence[Row], size: int = LIFT_WING_SAMPLE) -> list[Row]:
    """A fixed random sample of test rows, in rev_id order."""
    ordered = sorted((row for row in rows if row["split"] == TEST), key=lambda row: row["rev_id"])
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = rng.choice(len(ordered), min(size, len(ordered)), replace=False)
    return [ordered[index] for index in sorted(chosen.tolist())]


def fetch_lift_wing(rev_ids: Sequence[int], out: Path, client: httpx.Client) -> None:
    """Appends as it goes, so a stopped run resumes from where it was."""
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["rev_id"] for line in out.read_text("utf-8").splitlines()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as file:
        for rev_id in rev_ids:
            if rev_id in done:
                continue
            record = _score_one(rev_id, client)
            file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()
            time.sleep(PAUSE_SECONDS)


def _score_one(rev_id: int, client: httpx.Client) -> dict[str, Any]:
    for attempt in range(MAX_ATTEMPTS):
        response = client.post(LIFT_WING_URL, json={"rev_id": rev_id, "lang": "en"})
        if response.status_code == OK:
            body = response.json()
            return {
                "rev_id": rev_id,
                "score": body["output"]["probabilities"]["true"],
                "model_version": str(body.get("model_version", "")),
            }
        if response.status_code == RATE_LIMITED:
            time.sleep(float(response.headers.get("Retry-After", 2 ** (attempt + 1))))
            continue
        # A deleted or hidden revision gets a client error. It's counted, not retried.
        if response.status_code < 500:
            return {"rev_id": rev_id, "score": None, "status": response.status_code}
        time.sleep(2 ** (attempt + 1))
    return {"rev_id": rev_id, "score": None, "status": "gave_up"}


def read_lift_wing(path: Path = LIFT_WING_FIXTURE) -> dict[int, float | None]:
    return {
        record["rev_id"]: record["score"]
        for record in map(json.loads, path.read_text("utf-8").splitlines())
    }


def lift_wing_report(rows: Sequence[Row], scores: Mapping[int, float | None]) -> dict[str, Any]:
    scored = [row for row in rows if scores.get(row["rev_id"]) is not None]
    labels = np.array([row["label"] for row in scored], dtype=np.int64)
    values = np.array([scores[row["rev_id"]] for row in scored], dtype=np.float64)
    report = _measure(labels, values)
    report["requested"] = len(scores)
    report["unanswered"] = sum(score is None for score in scores.values())
    return report


def measure_baselines(rows: Sequence[Row]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in ("validation", TEST):
        members = [row for row in rows if row["split"] == split]
        labels = np.array([row["label"] for row in members], dtype=np.int64)
        rule = np.array([rule_score(row) for row in members], dtype=np.float64)
        report[split] = {
            "positive_rate": float(labels.mean()),
            "prevalence_pr_auc": pr_auc(labels, np.zeros(labels.size)),
            "rule": _measure(labels, rule),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs/baselines.json"))
    parser.add_argument("--fetch-lift-wing", action="store_true")
    args = parser.parse_args()

    rows = read_snapshot(args.snapshot)
    manifest = read_manifest(args.snapshot)
    report: dict[str, Any] = {
        "snapshot": manifest["name"],
        "snapshot_content_sha256": manifest["content_sha256"],
        **measure_baselines(rows),
    }
    sample = lift_wing_sample(rows)
    if args.fetch_lift_wing:
        agent = os.environ.get(USER_AGENT_VARIABLE, DEFAULT_USER_AGENT)
        with httpx.Client(headers={"User-Agent": agent}, timeout=30) as client:
            fetch_lift_wing([row["rev_id"] for row in sample], LIFT_WING_FIXTURE, client)
    if LIFT_WING_FIXTURE.exists():
        report["lift_wing"] = lift_wing_report(sample, read_lift_wing())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
