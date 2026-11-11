"""Every model is measured by this module, so differences come from the model, not the ruler.

Takes plain arrays of labels and scores, with no PyTorch, so baselines, trained models, and served
ONNX files all go through the same code. Ties get a single point on the precision-recall curve:
otherwise the order of equal scores could change the result.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Metric = Callable[[IntArray, FloatArray], float]

TARGET_PRECISION = 0.9
QUEUE_SIZE = 50
ECE_BINS = 15
BOOTSTRAP_SAMPLES = 1000
# A threshold resting on a handful of validation rows would be luck, not a real operating point.
MIN_FLAGGED_FOR_THRESHOLD = 20
# PLAN.md, section 4: slices below this are reported but can't block a promotion.
MIN_SLICE_POSITIVES = 100
SMALL_EDIT_BYTES = 100


@dataclass(frozen=True)
class PrecisionRecall:
    precision: FloatArray
    recall: FloatArray
    thresholds: FloatArray


def _arrays(labels: Any, scores: Any) -> tuple[IntArray, FloatArray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"labels {y.shape} and scores {s.shape} must be the same 1-D shape")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity")
    return y, s


def precision_recall(labels: Any, scores: Any) -> PrecisionRecall:
    """One point per distinct score, from the highest score down."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("no positive labels")
    order = np.argsort(-s, kind="stable")
    ordered = s[order]
    last_of_each_score = np.r_[np.flatnonzero(np.diff(ordered)), ordered.size - 1]
    true_positives = np.cumsum(y[order])[last_of_each_score].astype(np.float64)
    flagged = (last_of_each_score + 1).astype(np.float64)
    return PrecisionRecall(
        true_positives / flagged, true_positives / positives, ordered[last_of_each_score]
    )


def pr_auc(labels: Any, scores: Any) -> float:
    """Average precision. A constant score gets exactly the positive rate."""
    curve = precision_recall(labels, scores)
    steps = np.diff(curve.recall, prepend=0.0)
    return float(np.sum(steps * curve.precision))


def roc_auc(labels: Any, scores: Any) -> float:
    """The chance a positive outranks a negative, with ties counted as half."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC needs both classes")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    average_rank = last_rank - (counts - 1) / 2
    rank_sum = float(average_rank[inverse][y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def precision_at_k(labels: Any, scores: Any, k: int = QUEUE_SIZE) -> float:
    y, s = _arrays(labels, scores)
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def threshold_for_precision(
    labels: Any,
    scores: Any,
    target: float = TARGET_PRECISION,
    min_flagged: int = MIN_FLAGGED_FOR_THRESHOLD,
) -> float | None:
    """The lowest score that still reaches the target precision, or None if nothing does."""
    y, s = _arrays(labels, scores)
    curve = precision_recall(y, s)
    # A threshold flags everything at or above it, so count those rows for each point.
    flagged = np.searchsorted(np.sort(-s), -curve.thresholds, side="right")
    reachable = np.flatnonzero((curve.precision >= target) & (flagged >= min_flagged))
    return None if reachable.size == 0 else float(curve.thresholds[reachable[-1]])


def recall_at_precision(labels: Any, scores: Any, target: float = TARGET_PRECISION) -> float:
    curve = precision_recall(labels, scores)
    reachable = curve.recall[curve.precision >= target]
    return float(reachable.max()) if reachable.size else 0.0


def best_f1_threshold(labels: Any, scores: Any) -> float:
    curve = precision_recall(labels, scores)
    total = curve.precision + curve.recall
    f1 = np.divide(
        2 * curve.precision * curve.recall, total, out=np.zeros_like(total), where=total > 0
    )
    return float(curve.thresholds[int(np.argmax(f1))])


def at_threshold(labels: Any, scores: Any, threshold: float) -> dict[str, float]:
    y, s = _arrays(labels, scores)
    flagged = s >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def sigmoid(logits: Any) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64)
    # Written with logaddexp so a very large logit can't overflow.
    return np.exp(-np.logaddexp(0.0, -values))


def binary_log_loss(labels: Any, logits: Any) -> float:
    y, z = _arrays(labels, logits)
    return float(np.mean(y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)))


def expected_calibration_error(labels: Any, probabilities: Any, bins: int = ECE_BINS) -> float:
    y, p = _arrays(labels, probabilities)
    which = np.minimum((p * bins).astype(np.int64), bins - 1)
    total = 0.0
    for index in range(bins):
        members = which == index
        if members.any():
            total += float(members.mean()) * abs(float(y[members].mean() - p[members].mean()))
    return total


def reliability_bins(
    labels: Any, probabilities: Any, bins: int = ECE_BINS
) -> list[dict[str, float]]:
    y, p = _arrays(labels, probabilities)
    which = np.minimum((p * bins).astype(np.int64), bins - 1)
    points = []
    for index in range(bins):
        members = which == index
        if members.any():
            points.append(
                {
                    "mean_score": float(p[members].mean()),
                    "positive_rate": float(y[members].mean()),
                    "rows": int(members.sum()),
                }
            )
    return points


def fit_temperature(labels: Any, logits: Any) -> float:
    """The T that minimizes log loss of logits / T, by golden-section search on log T.

    One parameter, so it can't change the ranking and can't overfit a validation window.
    """
    y, z = _arrays(labels, logits)
    ratio = (math.sqrt(5) - 1) / 2

    def loss(log_t: float) -> float:
        return binary_log_loss(y, z / math.exp(log_t))

    low, high = math.log(0.05), math.log(50.0)
    first = high - ratio * (high - low)
    second = low + ratio * (high - low)
    first_loss, second_loss = loss(first), loss(second)
    for _ in range(60):
        if first_loss < second_loss:
            high, second, second_loss = second, first, first_loss
            first = high - ratio * (high - low)
            first_loss = loss(first)
        else:
            low, first, first_loss = first, second, second_loss
            second = low + ratio * (high - low)
            second_loss = loss(second)
    return math.exp((low + high) / 2)


def bootstrap_interval(
    metric: Metric,
    labels: Any,
    scores: Any,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    y, s = _arrays(labels, scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        rows = rng.integers(0, y.size, y.size)
        if y[rows].sum() > 0:
            values.append(metric(y[rows], s[rows]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def paired_bootstrap_difference(
    metric: Metric,
    labels: Any,
    scores_a: Any,
    scores_b: Any,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Metric(a) minus metric(b), resampling the same rows for both.

    Pairing matters: the two scores share their hard and easy rows, so comparing two separate
    intervals would hide a real gain inside noise both models have in common.
    """
    y, a = _arrays(labels, scores_a)
    _, b = _arrays(labels, scores_b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        rows = rng.integers(0, y.size, y.size)
        if y[rows].sum() > 0:
            values.append(metric(y[rows], a[rows]) - metric(y[rows], b[rows]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"difference": metric(y, a) - metric(y, b), "low": float(low), "high": float(high)}


def slice_masks(rows: Sequence[Mapping[str, Any]]) -> dict[str, NDArray[np.bool_]]:
    """Groups of edits to read results by.

    Rows with no edit count fall in neither experience group, so on the backfill those two are
    empty, and the report says so instead of showing a number.
    """
    return masks_from_columns(
        np.array([bool(row["is_anonymous"]) for row in rows], dtype=bool),
        np.array([int(row["user_edit_count_bucket"]) for row in rows], dtype=np.int64),
        np.array([int(row["bytes_delta"]) for row in rows], dtype=np.int64),
    )


def masks_from_columns(
    anonymous: NDArray[np.bool_], buckets: NDArray[np.int64], bytes_delta: NDArray[np.int64]
) -> dict[str, NDArray[np.bool_]]:
    """The same slices from columns, so live monitoring reads them exactly as evaluation does."""
    delta = np.abs(bytes_delta)
    return {
        "anonymous": anonymous,
        "registered": ~anonymous,
        # Bucket 0 is a first edit and bucket 1 is up to 10 edits (features.py).
        "new_editor": (buckets >= 0) & (buckets <= 1),
        "experienced_editor": buckets >= 2,
        "small_edit": delta <= SMALL_EDIT_BYTES,
        "large_edit": delta > SMALL_EDIT_BYTES,
    }


def slice_metrics(
    labels: Any, scores: Any, masks: Mapping[str, NDArray[np.bool_]], threshold: float
) -> dict[str, dict[str, Any]]:
    y, s = _arrays(labels, scores)
    report: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        rows = int(mask.sum())
        positives = int(y[mask].sum())
        entry: dict[str, Any] = {
            "rows": rows,
            "positives": positives,
            "gated": positives >= MIN_SLICE_POSITIVES,
        }
        if rows == 0:
            entry["note"] = "not measurable: no rows in this slice"
        else:
            flagged = at_threshold(y[mask], s[mask], threshold)
            entry |= {
                "positive_rate": positives / rows,
                "false_positive_rate": flagged["false_positive_rate"],
                "recall": flagged["recall"],
                "precision": flagged["precision"],
            }
            if positives:
                entry["pr_auc"] = pr_auc(y[mask], s[mask])
        report[name] = entry
    return report


def evaluate(
    validation_labels: Any,
    validation_scores: Any,
    test_labels: Any,
    test_scores: Any,
    test_masks: Mapping[str, NDArray[np.bool_]],
    seed: int = 0,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """The full report for one candidate. Validation picks the threshold; test only measures it."""
    y, s = _arrays(test_labels, test_scores)
    target = threshold_for_precision(validation_labels, validation_scores)
    f1_threshold = best_f1_threshold(validation_labels, validation_scores)
    # If validation never reaches 90% precision, the working point is the best F1 instead, and the
    # report says which one was used.
    threshold = f1_threshold if target is None else target
    pr_low, pr_high = bootstrap_interval(pr_auc, y, s, bootstrap_samples, seed)
    recall_low, recall_high = bootstrap_interval(recall_at_precision, y, s, bootstrap_samples, seed)
    slices = slice_metrics(y, s, test_masks, threshold)
    gap = None
    if slices["anonymous"]["rows"] and slices["registered"]["rows"]:
        gap = (
            slices["anonymous"]["false_positive_rate"] - slices["registered"]["false_positive_rate"]
        )
    return {
        "rows": int(y.size),
        "positives": int(y.sum()),
        "pr_auc": pr_auc(y, s),
        "pr_auc_interval": [pr_low, pr_high],
        "roc_auc": roc_auc(y, s),
        "recall_at_90_precision": recall_at_precision(y, s),
        "recall_at_90_precision_interval": [recall_low, recall_high],
        "precision_at_50": precision_at_k(y, s),
        "validation_pr_auc": pr_auc(validation_labels, validation_scores),
        "threshold": threshold,
        "threshold_rule": "best_f1" if target is None else "90_percent_precision",
        "threshold_90_reached_on_validation": target is not None,
        "at_threshold": at_threshold(y, s, threshold),
        "slices": slices,
        "false_positive_gap_anonymous_minus_registered": gap,
    }
