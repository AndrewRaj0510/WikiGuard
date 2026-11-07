"""Evaluate the served artifact, because that's what users get (PLAN.md, section 4).

Calibration and the threshold are fit on validation from the quantized model's own scores, test is
only measured, and the PyTorch scores are kept to check the two agree.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from wikiguard_core.evaluation import (
    evaluate,
    expected_calibration_error,
    fit_temperature,
    pr_auc,
    reliability_bins,
    sigmoid,
)
from wikiguard_core.inference import ServedModel

from trainer.data import Dataset, labels_of, masks_of
from trainer.export import update_config

# PLAN.md, section 4: the quantized model must stay within this of the PyTorch model.
MAX_PARITY_GAP = 0.005
# A model that reads the text must beat one that sees only metadata by at least this much.
TEXT_MARGIN = 0.02
SCORING_THREADS = 8
BATCH_CHECK_ROWS = 300


def parity(
    labels: NDArray[np.int64], torch_logits: NDArray[np.float64], served_logits: NDArray[np.float64]
) -> dict[str, float | bool]:
    """How far quantizing moved the scores, overall and per edit."""
    torch_score = pr_auc(labels, torch_logits)
    served_score = pr_auc(labels, served_logits)
    change = np.abs(sigmoid(torch_logits) - sigmoid(served_logits))
    return {
        "pytorch_pr_auc": torch_score,
        "served_pr_auc": served_score,
        "gap": abs(torch_score - served_score),
        "passed": abs(torch_score - served_score) <= MAX_PARITY_GAP,
        "score_change_mean": float(change.mean()),
        "score_change_p99": float(np.percentile(change, 99)),
        "score_change_max": float(change.max()),
    }


def batch_vs_single(
    served: ServedModel, rows: list[dict[str, Any]], count: int = BATCH_CHECK_ROWS, seed: int = 0
) -> dict[str, float | int]:
    """How much a score depends on which edits share its batch.

    Dynamic quantization picks each batch's scale from the values in that batch, so the same edit
    can score differently alone and in a batch. The scorer batches and /predict doesn't (F-073).
    """
    picks = np.random.default_rng(seed).permutation(len(rows))[:count]
    chosen = [rows[index] for index in picks.tolist()]
    batched = sigmoid(served.logits(chosen))
    alone = sigmoid(np.array([served.logits([row])[0] for row in chosen]))
    change = np.abs(batched - alone)
    return {
        "rows": len(chosen),
        "score_change_mean": float(change.mean()),
        "score_change_p99": float(np.percentile(change, 99)),
        "score_change_max": float(change.max()),
    }


def evaluate_served(
    folder: Path,
    data: Dataset,
    torch_validation: NDArray[np.float64],
    torch_test: NDArray[np.float64],
    run_folder: Path,
    seed: int = 0,
) -> tuple[dict[str, Any], str]:
    """Returns the report and the artifact digest, after writing calibration into the folder."""
    served = ServedModel(folder, threads=SCORING_THREADS)
    validation_labels = labels_of(data.validation)
    test_labels = labels_of(data.test)
    validation_logits = served.logits(data.validation)
    test_logits = served.logits(data.test)
    np.savez(
        run_folder / "scores.npz",
        validation_rev_id=np.array([row["rev_id"] for row in data.validation], dtype=np.int64),
        validation_logit=validation_logits,
        validation_torch_logit=torch_validation,
        test_rev_id=np.array([row["rev_id"] for row in data.test], dtype=np.int64),
        test_logit=test_logits,
        test_torch_logit=torch_test,
    )

    temperature = fit_temperature(validation_labels, validation_logits)
    before = sigmoid(test_logits)
    after = sigmoid(test_logits / temperature)
    report = evaluate(
        validation_labels, validation_logits, test_labels, test_logits, masks_of(data.test), seed
    )
    report |= {
        "temperature": temperature,
        "ece_before": expected_calibration_error(test_labels, before),
        "ece_after": expected_calibration_error(test_labels, after),
        "reliability_before": reliability_bins(test_labels, before),
        "reliability_after": reliability_bins(test_labels, after),
        "parity": {
            "validation": parity(validation_labels, torch_validation, validation_logits),
            "test": parity(test_labels, torch_test, test_logits),
        },
        "batch_vs_single": batch_vs_single(served, data.test),
    }
    threshold_logit = report["threshold"]
    digest = update_config(
        folder,
        {
            "temperature": temperature,
            "threshold_logit": threshold_logit,
            "threshold_probability": float(sigmoid(np.array(threshold_logit / temperature))),
            "threshold_rule": report["threshold_rule"],
        },
    )
    return report, digest


def training_warnings(
    history: list[dict[str, float]],
    validation_pr_auc: float,
    metadata_only_pr_auc: float | None,
    parity_passed: bool,
) -> list[str]:
    """What the checks before training can't see, because it only shows in a full run.

    The first transformers passed every sanity check, including overfitting one batch, and still
    ended no better than their metadata (F-075). A flagged model can't become champion.
    """
    warnings = []
    untrained = history[0]["train_loss"]
    final = history[-1]["train_loss"]
    if len(history) > 1 and final >= untrained:
        warnings.append(f"training loss ended at {final:.3f}, above the untrained {untrained:.3f}")
    if metadata_only_pr_auc is not None and validation_pr_auc < metadata_only_pr_auc + TEXT_MARGIN:
        warnings.append(
            f"validation PR-AUC {validation_pr_auc:.4f} is within {TEXT_MARGIN} of metadata alone "
            f"({metadata_only_pr_auc:.4f}), so the model isn't using the text"
        )
    if not parity_passed:
        warnings.append(f"the served model differs from PyTorch by more than {MAX_PARITY_GAP}")
    return warnings


def save_report(run_folder: Path, report: dict[str, Any]) -> Path:
    path = run_folder / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return path
