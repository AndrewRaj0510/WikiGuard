"""Why did the transformers underfit? Change one training setting at a time, on validation only.

    python -m trainer.loss_probe snapshots/NAME

The first runs of DistilBERT, BERT, and RoBERTa ended with their training loss above an untrained
network's, and RoBERTa's validation score matched a metadata-only model's (F-075). Each row here
changes one thing, so the result names the cause. The test window is never read.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from wikiguard_core.tracking import configure

from trainer.data import load_dataset
from trainer.train import RUNS_FOLDER, run_trial

CAP = 20_000
ROBERTA = {"learning_rate": 2e-5, "dropout": 0.2, "epochs": 1}
BERT = dict(ROBERTA)
PLAIN = {"weight_power": 0.0}
FAST_HEAD = PLAIN | {"head_learning_rate": 1e-3}
MLP_BEST = {"hidden": 128, "dropout": 0.3, "learning_rate": 3e-4, "use_comment": True}
BILSTM_BEST = {"hidden": 128, "dropout": 0.4, "learning_rate": 1e-3, "use_comment": True}
# Round 1 asks whether the class weight, gradient clipping, or batch size is at fault. Round 2 asks
# whether the new head learns too slowly, and what the text alone is worth. Each row is
# (name, model, parameters, cap).
Trial = tuple[str, str, dict[str, Any], int | None]
ROUNDS: dict[int, list[Trial]] = {
    1: [
        ("control", "roberta", ROBERTA, CAP),
        ("no_clipping", "roberta", ROBERTA | {"gradient_clip": None}, CAP),
        ("batch_64", "roberta", ROBERTA | {"accumulate": 4}, CAP),
        ("half_weight", "roberta", ROBERTA | {"weight_power": 0.5}, CAP),
        ("no_weight", "roberta", ROBERTA | PLAIN, CAP),
        ("no_weight_bert", "bert", BERT | PLAIN, CAP),
        ("control_bert", "bert", BERT, CAP),
    ],
    2: [
        ("fast_head", "roberta", ROBERTA | FAST_HEAD, CAP),
        ("fast_head_bert", "bert", BERT | FAST_HEAD, CAP),
        ("text_only", "roberta", ROBERTA | FAST_HEAD | {"use_metadata": False}, CAP),
        ("text_only_bert", "bert", BERT | FAST_HEAD | {"use_metadata": False}, CAP),
        ("no_weight_mlp", "mlp", MLP_BEST | PLAIN, None),
        ("no_weight_bilstm", "bilstm", BILSTM_BEST | PLAIN, None),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), required=True)
    args = parser.parse_args()
    configure()
    data = load_dataset(args.snapshot)
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    out = RUNS_FOLDER / f"loss_probe_round{args.round}.json"
    for name, model, params, cap in ROUNDS[args.round]:
        trial = run_trial(model, data, params, device, cap, args.seed)
        results.append({"trial": name, "model": model} | trial)
        print(json.dumps(results[-1]), flush=True)
        out.write_text(json.dumps(results, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
