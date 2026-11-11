"""What the daily monitor measures from live labels, and what it decides (PLAN.md, 4 and 5).

Pure numpy, so every rule here is tested in CI. jobs/monitor reads the archive, runs Evidently for
feature drift, and calls these. Every number is about served scores: the score the queue showed
for an edit, from whichever model answered.

Labels arrive 48 hours after an edit, so every live number trails the edits it describes. Each
report carries its label lag next to the number (section 4).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wikiguard_core.evaluation import (
    expected_calibration_error,
    masks_from_columns,
    pr_auc,
    precision_at_k,
    reliability_bins,
    slice_metrics,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

WINDOW = timedelta(days=7)
# Section 5: a drop this far below the champion's offline result triggers a retrain...
PERFORMANCE_DROP = 0.03
# ...once this many labels exist, because PR-AUC on fewer is mostly noise.
MIN_LABELS = 500
# Section 5: this share of drifted metadata features is drift.
DRIFT_SHARE = 0.30
RETRAIN_GAP = timedelta(hours=24)
POST_CHECK_AFTER = timedelta(days=7)
POST_CHECK_LABELS = 500
ROLLBACK_DROP = 0.03
# The same rule of thumb the snapshot checks use (validation.PSI_WARN).
SCORE_PSI_WARN = 0.25
PSI_BINS = 10
# Statuses of a retrain that really trained, which restarts the schedule's clock. A rejected
# snapshot or a crash didn't, so the next day's monitor may try again.
TRAINED_STATUSES = frozenset({"running", "shadow", "promoted", "not_promoted", "rolled_back"})
CONFIRMED_DECISIONS = frozenset({"vandalism", "good_faith_mistake"})


@dataclass(frozen=True)
class Observed:
    """Served scores of edits with final labels, as columns. One entry per edit."""

    rev_id: IntArray
    edited_at: NDArray[np.datetime64]
    label: IntArray
    score: FloatArray
    # NaN where the score record didn't carry it (archive files from before Phase 5).
    threshold: FloatArray
    used_fallback: BoolArray
    model_version: NDArray[np.object_]
    is_anonymous: BoolArray
    edit_count_bucket: IntArray
    bytes_delta: IntArray

    @property
    def size(self) -> int:
        return int(self.rev_id.size)

    def where(self, mask: BoolArray) -> "Observed":
        return Observed(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})

    def days(self) -> NDArray[np.datetime64]:
        return self.edited_at.astype("datetime64[D]")


def empty_observed() -> Observed:
    return Observed(
        np.array([], dtype=np.int64),
        np.array([], dtype="datetime64[s]"),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=bool),
        np.array([], dtype=object),
        np.array([], dtype=bool),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )


def _has_both_classes(labels: IntArray) -> bool:
    return bool(labels.size and 0 < labels.sum() < labels.size)


def summarize(observed: Observed, threshold: float | None) -> dict[str, Any]:
    """Every live number for one group of edits. PR-AUC needs both classes, or it's None."""
    labels = observed.label
    scores = observed.score
    report: dict[str, Any] = {
        "labeled": observed.size,
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else None,
        "pr_auc": pr_auc(labels, scores) if _has_both_classes(labels) else None,
        "precision_at_50": precision_at_k(labels, scores) if labels.size else None,
        "fallback_rate": float(observed.used_fallback.mean()) if labels.size else None,
    }
    if labels.size:
        report["ece"] = expected_calibration_error(labels, scores)
        report["reliability"] = reliability_bins(labels, scores)
    if threshold is not None and labels.size:
        flagged = scores >= threshold
        report["threshold"] = threshold
        report["alert_rate"] = float(flagged.mean())
        report["precision_at_threshold"] = float(labels[flagged].mean()) if flagged.any() else None
        masks = masks_from_columns(
            observed.is_anonymous, observed.edit_count_bucket, observed.bytes_delta
        )
        report["slices"] = slice_metrics(labels, scores, masks, threshold)
    return report


def daily(observed: Observed, threshold: float | None) -> dict[str, dict[str, Any]]:
    """The Health page's chart: one point per edit day. Slices stay in the weekly report."""
    out: dict[str, dict[str, Any]] = {}
    days = observed.days()
    for day in np.unique(days):
        summary = summarize(observed.where(days == day), threshold)
        summary.pop("reliability", None)
        summary.pop("slices", None)
        out[str(day)] = summary
    return out


def score_psi(reference: FloatArray, live: FloatArray, bins: int = PSI_BINS) -> float | None:
    """How far live scores moved from the reference sample's, in bins set by the reference.

    Scores move before labels arrive, so this is the earliest sign a model sees different data.
    """
    if reference.size == 0 or live.size == 0:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    expected = np.bincount(
        np.searchsorted(edges, reference, side="right"), minlength=edges.size + 1
    )
    actual = np.bincount(np.searchsorted(edges, live, side="right"), minlength=edges.size + 1)
    # A small floor, so an empty bin gives a large number rather than infinity.
    e = np.maximum(expected / reference.size, 1e-4)
    a = np.maximum(actual / live.size, 1e-4)
    return float(np.sum((a - e) * np.log(a / e)))


def reviewer_precision(
    decisions: list[str], scores: FloatArray, threshold: float | None
) -> dict[str, Any]:
    """How often the owner agreed with a flag. Skipped edits say nothing either way.

    This comes from people acting on WikiGuard's own queue, so it's reported beside the revert
    labels and never used to train (the plan's feedback-loop rule).
    """
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision] = counts.get(decision, 0) + 1
    judged = np.array([decision != "skip" for decision in decisions], dtype=bool)
    confirmed = np.array([decision in CONFIRMED_DECISIONS for decision in decisions], dtype=bool)
    report: dict[str, Any] = {"reviewed": len(decisions), "decisions": counts}
    if threshold is not None and scores.size:
        flagged = (scores >= threshold) & judged
        report["flagged_and_judged"] = int(flagged.sum())
        report["precision"] = float(confirmed[flagged].mean()) if flagged.any() else None
    judged_count = int(judged.sum())
    report["confirmed_share"] = float(confirmed[judged].mean()) if judged_count else None
    return report


@dataclass(frozen=True)
class TriggerInputs:
    now: datetime
    live_pr_auc: float | None
    live_labels: int
    offline_pr_auc: float | None
    drift_share: float | None
    text_drift: bool
    # The newest retrain that started, whatever happened to it: the 24-hour guard.
    last_retrain_started: datetime | None
    # The newest retrain that really trained, or the champion's registration: the schedule.
    schedule_since: datetime | None
    interval_days: int
    on_drift: bool


@dataclass
class Decision:
    retrain: bool = False
    trigger: str | None = None
    reasons: list[str] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)


def decide(inputs: TriggerInputs) -> Decision:
    """Section 5's three triggers, the 24-hour guard, and the RETRAIN_ON_DRIFT switch."""
    decision = Decision()
    fired: list[tuple[str, str]] = []

    if inputs.live_labels < MIN_LABELS:
        decision.alerts.append(
            (
                "Too few labels",
                f"{inputs.live_labels} labeled edits in the last 7 days, below {MIN_LABELS}, so "
                "the performance trigger can't be checked",
            )
        )
    elif inputs.live_pr_auc is not None and inputs.offline_pr_auc is not None:
        if inputs.live_pr_auc < inputs.offline_pr_auc - PERFORMANCE_DROP:
            fired.append(
                (
                    "performance",
                    f"live PR-AUC {inputs.live_pr_auc:.4f} is more than {PERFORMANCE_DROP} below "
                    f"the offline {inputs.offline_pr_auc:.4f}, on {inputs.live_labels} labels",
                )
            )

    drift_reasons = []
    if inputs.drift_share is not None and inputs.drift_share >= DRIFT_SHARE:
        drift_reasons.append(f"{inputs.drift_share:.0%} of metadata features drifted")
    if inputs.text_drift:
        drift_reasons.append("text drift found")
    if drift_reasons:
        fired.append(("drift", "; ".join(drift_reasons)))

    if not inputs.on_drift and fired:
        for name, reason in fired:
            decision.alerts.append((f"{name.capitalize()} trigger (retraining is off)", reason))
        fired = []

    since = inputs.schedule_since
    if since is None or inputs.now - since >= timedelta(days=inputs.interval_days):
        age = "never retrained" if since is None else f"{(inputs.now - since).days} days since"
        fired.append(("schedule", f"{age}, interval {inputs.interval_days} days"))

    if not fired:
        return decision
    decision.reasons = [reason for _, reason in fired]
    started = inputs.last_retrain_started
    if started is not None and inputs.now - started < RETRAIN_GAP:
        decision.reasons.append("not started: a retrain began within the last 24 hours")
        return decision
    decision.retrain = True
    # The first in section 5's order names the run.
    decision.trigger = fired[0][0]
    return decision


@dataclass(frozen=True)
class PostCheckInputs:
    now: datetime
    promoted_at: datetime
    new_pr_auc: float | None
    new_labels: int
    previous_pr_auc: float | None
    previous_labels: int


@dataclass(frozen=True)
class PostCheck:
    due: bool
    rollback: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"due": self.due, "rollback": self.rollback, "reason": self.reason}


def post_promotion(inputs: PostCheckInputs) -> PostCheck:
    """Section 4: at 500 new labels or after 7 days, compare the new champion with the old one.

    Labels arrive too late for the shadow gate, so this is the first check on real outcomes. The
    previous champion is measured on the week before its replacement, since it no longer answers.
    """
    waited = inputs.now - inputs.promoted_at
    if inputs.new_labels < POST_CHECK_LABELS and waited < POST_CHECK_AFTER:
        return PostCheck(False, False, f"waiting: {inputs.new_labels} labels so far")
    if inputs.new_pr_auc is None or inputs.previous_pr_auc is None:
        return PostCheck(True, False, "not measurable: one side has no labeled edits of both kinds")
    drop = inputs.previous_pr_auc - inputs.new_pr_auc
    numbers = (
        f"live PR-AUC {inputs.new_pr_auc:.4f} on {inputs.new_labels} labels, against "
        f"{inputs.previous_pr_auc:.4f} for the previous champion"
    )
    if drop > ROLLBACK_DROP:
        return PostCheck(True, True, f"rolled back: {numbers}, a drop of {drop:.4f}")
    return PostCheck(True, False, f"kept: {numbers}")


def utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
