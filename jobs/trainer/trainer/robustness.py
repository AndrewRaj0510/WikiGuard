"""Re-score edits after the tricks vandals use to hide words, and report what recall is lost.

Each change goes through build_features(), so normalization gets its chance to undo it, exactly as
it would for a live edit. Only added text changes: that's what a vandal controls.
"""

from collections.abc import Callable
from datetime import UTC
from typing import Any

import numpy as np
from wikiguard_core.evaluation import at_threshold, pr_auc
from wikiguard_core.features import build_features
from wikiguard_core.inference import ServedModel
from wikiguard_core.schemas import EditEvent

SAMPLE_POSITIVES = 500
SAMPLE_NEGATIVES = 500
LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
# Cyrillic letters that look the same as Latin ones, which NFKC doesn't fold together.
HOMOGLYPHS = str.maketrans({"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х"})
ZERO_WIDTH = "​"


def leetspeak(text: str) -> str:
    return text.lower().translate(LEET)


def homoglyphs(text: str) -> str:
    return text.translate(HOMOGLYPHS)


def zero_width(text: str) -> str:
    return ZERO_WIDTH.join(text)


def spacing(text: str) -> str:
    return " ".join(text)


def case_flip(text: str) -> str:
    return "".join(char.upper() if index % 2 else char.lower() for index, char in enumerate(text))


OBFUSCATIONS: dict[str, Callable[[str], str]] = {
    "leetspeak": leetspeak,
    "homoglyphs": homoglyphs,
    "zero_width": zero_width,
    "spacing": spacing,
    "case_flip": case_flip,
}


def _rebuild(row: dict[str, Any], added: str) -> dict[str, Any]:
    """The live path: an EditEvent, then build_features. Sizes keep their recorded values."""
    edit = EditEvent(
        rev_id=row["rev_id"],
        parent_rev_id=None,
        title="",
        timestamp=row["timestamp"].astimezone(UTC),
        is_minor=row["is_minor"],
        is_anonymous=row["is_anonymous"],
        user_hash=None,
        comment=row["comment"],
        old_length=row["page_length"] - row["bytes_delta"],
        new_length=row["page_length"],
        added_text=added,
        removed_text=row["removed_text"],
    )
    return row | build_features(edit).as_dict()


def sample(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Positives show the recall lost; negatives show whether a trick also causes false alarms."""
    rng = np.random.default_rng(seed)
    positives = [row for row in rows if row["label"] == 1 and row["added_text"]]
    negatives = [row for row in rows if row["label"] == 0 and row["added_text"]]
    chosen = rng.choice(len(positives), min(SAMPLE_POSITIVES, len(positives)), replace=False)
    others = rng.choice(len(negatives), min(SAMPLE_NEGATIVES, len(negatives)), replace=False)
    return [positives[index] for index in chosen] + [negatives[index] for index in others]


def robustness_report(
    model: ServedModel, rows: list[dict[str, Any]], threshold_logit: float, seed: int = 0
) -> dict[str, Any]:
    chosen = sample(rows, seed)
    labels = np.array([row["label"] for row in chosen], dtype=np.int64)
    # Recall at a strict threshold rests on few edits, and thresholds differ between models, so
    # PR-AUC is the comparison that holds across them. The sample is half reverts, so only the
    # drop means anything, not the level.
    original = model.logits(chosen)
    base = at_threshold(labels, original, threshold_logit)
    base_pr_auc = pr_auc(labels, original)
    report: dict[str, Any] = {
        "rows": len(chosen),
        "positives": int(labels.sum()),
        "threshold_logit": threshold_logit,
        "original_recall": base["recall"],
        "original_false_positive_rate": base["false_positive_rate"],
        "original_pr_auc": base_pr_auc,
        "changes": {},
    }
    for name, change in OBFUSCATIONS.items():
        changed = [_rebuild(row, change(row["added_text"])) for row in chosen]
        logits = model.logits(changed)
        result = at_threshold(labels, logits, threshold_logit)
        score = pr_auc(labels, logits)
        report["changes"][name] = {
            "recall": result["recall"],
            "recall_drop": base["recall"] - result["recall"],
            "false_positive_rate": result["false_positive_rate"],
            "pr_auc": score,
            "pr_auc_drop": base_pr_auc - score,
        }
    return report
