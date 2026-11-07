"""Write out edits to read by hand: the worst mistakes, a label-noise sample, golden candidates.

Files go to data/review/, which is git-ignored, because this is raw vandal text. Results that come
out of reading them go in docs/error-analysis.md, in summary form.

    python -m trainer.review snapshots/NAME --model roberta
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from wikiguard_core.snapshot import read_snapshot

REVIEW_FOLDER = Path("data/review")
RUNS_FOLDER = Path("data/runs")
TOP_ERRORS = 50
LABEL_NOISE_ROWS = 200
GOLDEN_CANDIDATES = 300
SEED = 20261106
SHOWN_CHARS = 220
SMALL_EDIT_BYTES = 100


def _clip(text: str) -> str:
    return " ".join(text.split())[:SHOWN_CHARS]


def describe(row: dict[str, Any], score: float | None = None) -> str:
    parts = [
        f"rev={row['rev_id']}",
        f"label={row['label']}",
        f"anon={int(row['is_anonymous'])}",
        f"delta={row['bytes_delta']}",
    ]
    if score is not None:
        parts.append(f"logit={score:.2f}")
    return (
        " ".join(parts)
        + f"\n  comment: {_clip(row['comment'])}"
        + f"\n  added:   {_clip(row['added_text'])}"
        + f"\n  removed: {_clip(row['removed_text'])}\n"
    )


def top_errors(
    rows: list[dict[str, Any]], logits: np.ndarray, count: int = TOP_ERRORS
) -> tuple[list[int], list[int]]:
    """Positions of the highest-scored good edits and the lowest-scored reverted ones."""
    labels = np.array([row["label"] for row in rows])
    negatives = np.flatnonzero(labels == 0)
    positives = np.flatnonzero(labels == 1)
    false_positives = negatives[np.argsort(-logits[negatives], kind="stable")][:count]
    false_negatives = positives[np.argsort(logits[positives], kind="stable")][:count]
    return false_positives.tolist(), false_negatives.tolist()


def golden_candidates(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """An even draw from each group, so rare groups aren't lost to chance.

    Drawn by label, editor type, and edit size, and never by a model's score, so no model gets a
    set that suits it.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, bool, bool], list[dict[str, Any]]] = {}
    for row in rows:
        if not row["added_text"] and not row["removed_text"]:
            continue
        small = abs(row["bytes_delta"]) <= SMALL_EDIT_BYTES
        groups.setdefault((row["label"], row["is_anonymous"], small), []).append(row)
    each = count // len(groups)
    chosen: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        picks = rng.choice(len(members), min(each, len(members)), replace=False)
        chosen += [members[index] for index in sorted(picks.tolist())]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    rows = read_snapshot(args.snapshot)
    test = sorted((row for row in rows if row["split"] == "test"), key=lambda r: r["timestamp"])
    by_id = {row["rev_id"]: row for row in test}
    stored = np.load(RUNS_FOLDER / args.model / "scores.npz", allow_pickle=False)
    ordered = [by_id[int(rev_id)] for rev_id in stored["test_rev_id"]]
    logits = stored["test_logit"]
    REVIEW_FOLDER.mkdir(parents=True, exist_ok=True)

    false_positives, false_negatives = top_errors(ordered, logits)
    with (REVIEW_FOLDER / f"errors_{args.model}.txt").open("w", encoding="utf-8") as file:
        file.write("# Top false positives (not reverted, high score)\n")
        for position in false_positives:
            file.write(describe(ordered[position], float(logits[position])))
        file.write("\n# Top false negatives (reverted, low score)\n")
        for position in false_negatives:
            file.write(describe(ordered[position], float(logits[position])))

    rng = np.random.default_rng(SEED)
    positives = [row for row in test if row["label"] == 1]
    picks = rng.choice(len(positives), LABEL_NOISE_ROWS, replace=False)
    with (REVIEW_FOLDER / "label_noise.txt").open("w", encoding="utf-8") as file:
        for position, index in enumerate(sorted(picks.tolist()), 1):
            file.write(f"[{position}] {describe(positives[index])}")

    candidates = golden_candidates(test, GOLDEN_CANDIDATES, SEED)
    with (REVIEW_FOLDER / "golden_candidates.txt").open("w", encoding="utf-8") as file:
        for position, row in enumerate(candidates, 1):
            file.write(f"[{position}] {describe(row)}")
    (REVIEW_FOLDER / "golden_candidates.json").write_text(
        json.dumps([row["rev_id"] for row in candidates]), "utf-8"
    )
    print(f"Wrote {REVIEW_FOLDER}: {len(candidates)} golden candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
