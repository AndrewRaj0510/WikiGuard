"""Checks that catch a broken pipeline before hours go into training it.

Each one has a known right answer. Wrong initial loss means wrong labels or loss. A model that can't
overfit one batch has a bug in its data flow. A model that learns shuffled labels, or an implausibly
strong metadata-only score, means the answer has leaked into the inputs.

    python -m trainer.sanity snapshots/NAME --out data/runs/sanity.json
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from wikiguard_core.evaluation import pr_auc
from wikiguard_core.inputs import MetadataSpec, TfidfEncoder
from wikiguard_core.models import BILSTM, MLP, MODEL_NAMES, TRANSFORMERS, MLPClassifier
from wikiguard_core.tracking import clean_params, configure, run_tags

from trainer.data import Dataset, TfidfEncoded, encode_tfidf, labels_of, load_dataset, sample_rows
from trainer.loop import (
    FitConfig,
    expected_initial_loss,
    fit,
    positive_weight,
    predict_logits,
    train_step_losses,
)
from trainer.recipes import SEARCH_SPACES, encode_like, prepare

BATCH_ROWS = 16
INITIAL_LOSS_TOLERANCE = 0.5
OVERFIT_MAX_LOSS = 0.1
# Shuffled labels should score near the positive rate, averaged over seeds (F-074).
SHUFFLED_SEEDS = (0, 1, 2, 3, 4, 5)
SHUFFLED_EPOCHS = 4
SHUFFLED_MAX_RATIO = 1.5
METADATA_LEAK_PR_AUC = 0.95
OVERFIT_STEPS = {MLP: 300, BILSTM: 300}
TRANSFORMER_OVERFIT_STEPS = 60
TRANSFORMER_OVERFIT_RATE = 1e-4
MLP_OVERFIT_RATE = 3e-3
SANITY_TRAIN_ROWS = 15_000


class SanityError(RuntimeError):
    """A check failed, which points to a bug or leakage, so training must stop."""


def balanced_rows(data: Dataset, count: int, seed: int) -> list[dict[str, Any]]:
    """Half positive, so the batch has something to learn even with so few rows."""
    rng = np.random.default_rng(seed)
    positives = [row for row in data.train if row["label"] == 1]
    negatives = [row for row in data.train if row["label"] == 0]
    half = count // 2
    chosen = [positives[index] for index in rng.choice(len(positives), half, replace=False)]
    chosen += [negatives[index] for index in rng.choice(len(negatives), half, replace=False)]
    return chosen


def check_model(
    name: str, data: Dataset, device: torch.device, seed: int, pretrained: bool = True
) -> dict[str, Any]:
    """Initial loss and overfitting one batch, for one model."""
    params = dict(SEARCH_SPACES[name][0])
    # Dropout would stop the loss reaching zero, and that's not what this check is about.
    params["dropout"] = 0.0
    prepared = prepare(
        name, data, params, device, seed, cap=SANITY_TRAIN_ROWS, pretrained=pretrained
    )
    rows = balanced_rows(data, BATCH_ROWS, seed)
    encoded = encode_like(prepared, rows, f"sanity{BATCH_ROWS}", data)
    batch, members = next(iter(encoded.batches(BATCH_ROWS, None)))
    labels = encoded.labels[members]

    # The model's real class balance gives the expected starting loss.
    train_labels = prepared.train.labels
    weight = positive_weight(train_labels) ** prepared.fit_config.weight_power
    expected = expected_initial_loss(train_labels, weight)
    logits = predict_logits(prepared.model.to(device), prepared.train_sample, 64, device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    initial = float(
        loss_function(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(prepared.train_sample.labels, dtype=torch.float32),
        )
    )
    steps = OVERFIT_STEPS.get(name, TRANSFORMER_OVERFIT_STEPS)
    rate = TRANSFORMER_OVERFIT_RATE if name in TRANSFORMERS else MLP_OVERFIT_RATE
    losses = train_step_losses(prepared.model, batch, labels, steps, rate, device)
    prepared.model.to("cpu")
    torch.cuda.empty_cache()
    return {
        "model": name,
        "expected_initial_loss": expected,
        "initial_loss": initial,
        "initial_loss_passed": abs(initial - expected) <= INITIAL_LOSS_TOLERANCE * expected,
        "overfit_first_loss": losses[0],
        "overfit_final_loss": losses[-1],
        "overfit_passed": losses[-1] <= OVERFIT_MAX_LOSS,
    }


def metadata_only(data: Dataset, seed: int, device: torch.device) -> float:
    """An MLP that sees only metadata. Above 0.95 would mean the metadata gives away the answer."""
    rows = sample_rows(data.train, SANITY_TRAIN_ROWS, seed)
    metadata = MetadataSpec.fit(rows)
    empty = TfidfEncoder({}, (), False)
    train = encode_tfidf(rows, empty, metadata)
    validation = encode_tfidf(data.validation, empty, metadata)
    model = MLPClassifier(len(metadata.columns), 64, 0.1)
    result = fit(
        model,
        train,
        train,
        validation,
        FitConfig(epochs=8, batch_size=256, learning_rate=1e-3, patience=3, seed=seed),
        device,
    )
    return result.best_validation_pr_auc


def _shuffled_labels(data: Dataset, device: torch.device) -> dict[str, Any]:
    """The same MLP on labels with no connection to the input, over several seeds.

    A random function can correlate with strong metadata by chance, in either direction: untrained
    networks scored from 0.042 to 0.106 here. So one seed proves little, and taking the best epoch
    picks the luckiest (F-074). What's checked is the last epoch, averaged across seeds.

    This mainly catches a target that isn't really shuffled, or scores read from the wrong place.
    It can't catch a feature that gives away the label, since nothing rewards the model for using
    one. The metadata-only check covers that.
    """
    params = SEARCH_SPACES[MLP][1] | {"epochs": SHUFFLED_EPOCHS}
    finals = []
    for seed in SHUFFLED_SEEDS:
        prepared = prepare(MLP, data, params, device, seed, cap=SANITY_TRAIN_ROWS)
        train = prepared.train
        assert isinstance(train, TfidfEncoded)  # noqa: S101 - the MLP recipe always gives this type
        shuffled = replace(train, labels=np.random.default_rng(seed).permutation(train.labels))
        config = replace(prepared.fit_config, seed=seed, patience=SHUFFLED_EPOCHS)
        result = fit(prepared.model, shuffled, shuffled, prepared.validation, config, device)
        finals.append(result.history[-1]["validation_pr_auc"])
    return {
        "shuffled_validation_pr_auc_by_seed": finals,
        "shuffled_validation_pr_auc": float(np.mean(finals)),
        "validation_positive_rate": float(labels_of(data.validation).mean()),
    }


def run_checks(
    data: Dataset,
    device: torch.device,
    models: tuple[str, ...] = MODEL_NAMES,
    seed: int = 0,
    pretrained: bool = True,
) -> dict[str, Any]:
    results: dict[str, Any] = {"models": [], "failures": []}
    for name in models:
        checked = check_model(name, data, device, seed, pretrained)
        results["models"].append(checked)
        print(json.dumps(checked), flush=True)
        if not checked["initial_loss_passed"]:
            results["failures"].append(f"{name}: initial loss {checked['initial_loss']:.3f}")
        if not checked["overfit_passed"]:
            results["failures"].append(f"{name}: final loss {checked['overfit_final_loss']:.3f}")
    shuffled = _shuffled_labels(data, device)
    metadata_score = metadata_only(data, seed, device)
    results |= shuffled | {"metadata_only_validation_pr_auc": metadata_score}
    limit = SHUFFLED_MAX_RATIO * shuffled["validation_positive_rate"]
    if shuffled["shuffled_validation_pr_auc"] > limit:
        results["failures"].append(
            f"shuffled labels scored {shuffled['shuffled_validation_pr_auc']}"
        )
    if metadata_score > METADATA_LEAK_PR_AUC:
        results["failures"].append(f"metadata alone scored {metadata_score}")
    results["passed"] = not results["failures"]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs/sanity.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    args = parser.parse_args()

    configure()
    data = load_dataset(args.snapshot)
    with mlflow.start_run(run_name="sanity", tags=run_tags(data.manifest, "all", "sanity")):
        mlflow.log_params(clean_params({"models": args.models, "device": args.device}))
        results = run_checks(data, torch.device(args.device), tuple(args.models))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), "utf-8")
        mlflow.log_artifact(str(args.out))
        mlflow.log_metrics(
            {
                "shuffled_validation_pr_auc": results["shuffled_validation_pr_auc"],
                "metadata_only_validation_pr_auc": results["metadata_only_validation_pr_auc"],
                "failures": float(len(results["failures"])),
            }
        )
    print(json.dumps({key: results[key] for key in results if key != "models"}, indent=2))
    if not results["passed"]:
        raise SanityError("; ".join(results["failures"]))
    constant = pr_auc(labels_of(data.validation), np.zeros(len(data.validation)))
    print(f"For reference, a constant score gets {constant:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
