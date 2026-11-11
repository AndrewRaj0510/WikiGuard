"""The promotion gate, the chain order, the shadow health check, and the before vs after report.

PLAN.md, section 4 sets every rule here. Pure numpy, so CI tests each rule, and the trainer and
the shadow check call the same code. Scores are logits: ranking metrics don't care, and each
model's threshold is in the same units as its own scores.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wikiguard_core.evaluation import (
    MIN_SLICE_POSITIVES,
    bootstrap_interval,
    paired_bootstrap_difference,
    pr_auc,
    slice_metrics,
)
from wikiguard_core.monitoring import score_psi

FloatArray = NDArray[np.float64]

MIN_GAIN = 0.005
MAX_SLICE_PR_AUC_DROP = 0.02
MAX_SLICE_FPR_RISE = 0.02
MAX_P95_MS = 150.0
# DEC-050: RoBERTa's 126.5 MB served folder fit Render's 512 MB at a 428 MB peak (F-105).
# Anything much larger leaves no room for the API, so this is the budget (DEC-069).
RENDER_MAX_SERVED_BYTES = 200_000_000
# The golden set is 200 edits, so a tiny drop is noise. The same allowance as a slice (DEC-069).
MAX_GOLDEN_DROP = 0.02
LAST_MODEL = "mlp"
SHADOW_MIN_EDITS = 500
SHADOW_MAX_ERROR_RATE = 0.01
SHADOW_MAX_SCORE_PSI = 0.25


@dataclass(frozen=True)
class Evaluated:
    """One model measured on the newest test window, ready to compare."""

    model: str
    # None for a candidate that isn't registered yet.
    version: str | None
    logits: FloatArray
    threshold: float
    is_new: bool
    p95_ms: float | None = None
    served_bytes: int | None = None
    golden_pr_auc: float | None = None
    warnings: tuple[str, ...] = ()
    smoke_passed: bool = True
    checksum_ok: bool = True
    feature_version_ok: bool = True

    @property
    def label(self) -> str:
        return f"{self.model} v{self.version}" if self.version else f"{self.model} (new)"

    def same_as(self, other: "Evaluated | None") -> bool:
        return other is not None and (self.model, self.version) == (other.model, other.version)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class GateResult:
    checks: list[Check] = field(default_factory=list)
    difference: dict[str, float] | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "difference": self.difference,
        }


def artifact_checks(model: Evaluated) -> list[Check]:
    """What any version must pass to serve at all, even as a fallback."""
    return [
        Check("checksum", model.checksum_ok, "artifact matches its recorded SHA-256"),
        Check("feature_version", model.feature_version_ok, "built for this feature_version"),
        Check("smoke_test", model.smoke_passed, "scored the fixture edits without error"),
        Check(
            "training_warnings",
            not model.warnings,
            "; ".join(model.warnings) if model.warnings else "none",
        ),
    ]


def offline_gate(
    candidate: Evaluated,
    champion: Evaluated | None,
    labels: NDArray[np.int64],
    masks: Mapping[str, NDArray[np.bool_]],
    rule_interval_high: float | None = None,
    samples: int = 1000,
    seed: int = 0,
) -> GateResult:
    """Whether a candidate may replace the champion. Both are scored on the same rows."""
    result = GateResult(artifact_checks(candidate))
    p95 = candidate.p95_ms
    result.checks.append(
        Check(
            "latency",
            p95 is not None and p95 <= MAX_P95_MS,
            "not measured" if p95 is None else f"p95 {p95:.1f} ms, limit {MAX_P95_MS:.0f} ms",
        )
    )
    size = candidate.served_bytes
    result.checks.append(
        Check(
            "size",
            size is not None and size <= RENDER_MAX_SERVED_BYTES,
            "not measured"
            if size is None
            else f"{size / 1e6:.1f} MB, Render budget {RENDER_MAX_SERVED_BYTES / 1e6:.0f} MB",
        )
    )
    after = pr_auc(labels, candidate.logits)

    if champion is None:
        # The first champion has nothing to beat, so it must clearly beat the rule baseline.
        low, _ = bootstrap_interval(pr_auc, labels, candidate.logits, samples, seed)
        result.checks.append(
            Check(
                "beats_rule_baseline",
                rule_interval_high is not None and low > rule_interval_high,
                f"interval starts at {low:.4f}, rule baseline's ends at {rule_interval_high}",
            )
        )
        return result

    before = pr_auc(labels, champion.logits)
    paired = paired_bootstrap_difference(
        pr_auc, labels, candidate.logits, champion.logits, samples, seed
    )
    result.difference = {
        "mean": paired["difference"],
        "low": paired["low"],
        "high": paired["high"],
    }
    result.checks += [
        Check(
            "pr_auc_gain",
            after >= before + MIN_GAIN,
            f"{after:.4f} against {before:.4f}, needs +{MIN_GAIN}",
        ),
        Check(
            "paired_bootstrap",
            paired["low"] > 0,
            f"95% interval of the difference {paired['low']:+.4f} to {paired['high']:+.4f}",
        ),
    ]
    result.checks += slice_checks(candidate, champion, labels, masks)
    if candidate.golden_pr_auc is None or champion.golden_pr_auc is None:
        result.checks.append(Check("golden_set", False, "not measured"))
    else:
        result.checks.append(
            Check(
                "golden_set",
                candidate.golden_pr_auc >= champion.golden_pr_auc - MAX_GOLDEN_DROP,
                f"{candidate.golden_pr_auc:.4f} against {champion.golden_pr_auc:.4f}",
            )
        )
    return result


def slice_checks(
    candidate: Evaluated,
    champion: Evaluated,
    labels: NDArray[np.int64],
    masks: Mapping[str, NDArray[np.bool_]],
) -> list[Check]:
    """No gated slice may lose more than 0.02 PR-AUC or gain more than 0.02 false positive rate.

    Slices under 100 positives are reported but can't block (section 4), and each model is read at
    its own threshold, because that's where it flags edits in the queue.
    """
    after = slice_metrics(labels, candidate.logits, masks, candidate.threshold)
    before = slice_metrics(labels, champion.logits, masks, champion.threshold)
    checks = []
    for name, old in before.items():
        new = after[name]
        if not old.get("gated") or "pr_auc" not in old or "pr_auc" not in new:
            continue
        drop = old["pr_auc"] - new["pr_auc"]
        rise = new["false_positive_rate"] - old["false_positive_rate"]
        checks.append(
            Check(
                f"slice_{name}",
                drop <= MAX_SLICE_PR_AUC_DROP and rise <= MAX_SLICE_FPR_RISE,
                f"PR-AUC {old['pr_auc']:.4f} to {new['pr_auc']:.4f}, false positive rate "
                f"{old['false_positive_rate']:.4f} to {new['false_positive_rate']:.4f}",
            )
        )
    return checks


def chain_order(models: Sequence[Evaluated], labels: NDArray[np.int64]) -> list[Evaluated]:
    """Best PR-AUC first, flagged models behind healthy ones, and the MLP always last."""
    scores = {id(model): pr_auc(labels, model.logits) for model in models}
    ranked = sorted(
        (model for model in models if model.model != LAST_MODEL),
        key=lambda model: (not model.warnings, scores[id(model)]),
        reverse=True,
    )
    return ranked + [model for model in models if model.model == LAST_MODEL]


@dataclass
class Proposal:
    """The chain after this retrain, and whether the champion changes (which needs a shadow)."""

    order: list[Evaluated]
    current_champion: Evaluated | None
    challenger: Evaluated | None
    gate: GateResult | None
    champion_change: bool
    reason: str
    replaced: list[str] = field(default_factory=list)

    def aliases(self) -> list[tuple[str, Evaluated]]:
        return [
            ("champion" if index == 0 else f"fallback-{index}", model)
            for index, model in enumerate(self.order)
        ]


def propose(
    current: Sequence[Evaluated],
    candidates: Sequence[Evaluated],
    labels: NDArray[np.int64],
    masks: Mapping[str, NDArray[np.bool_]],
    rule_interval_high: float | None = None,
    samples: int = 1000,
    seed: int = 0,
) -> Proposal:
    """Choose one version per architecture, order them, and gate any change of champion.

    A retrained version replaces the serving one of its architecture when it passes the artifact
    checks and scores at least as well on the same rows: a fallback needs no margin, only to be no
    worse (DEC-069). The champion changes only through the full gate.
    """
    champion = current[0] if current else None
    chosen: dict[str, Evaluated] = {model.model: model for model in current}
    replaced = []
    for candidate in candidates:
        if not all(check.passed for check in artifact_checks(candidate)):
            continue
        existing = chosen.get(candidate.model)
        if existing is None or pr_auc(labels, candidate.logits) >= pr_auc(labels, existing.logits):
            if existing is not None:
                replaced.append(f"{existing.label} by {candidate.label}")
            chosen[candidate.model] = candidate
    order = chain_order(list(chosen.values()), labels)
    if not order:
        return Proposal([], champion, None, None, False, "no usable model")

    top = order[0]
    if top.same_as(champion):
        best_new = max(candidates, key=lambda model: pr_auc(labels, model.logits), default=None)
        gate = (
            offline_gate(best_new, champion, labels, masks, rule_interval_high, samples, seed)
            if best_new is not None
            else None
        )
        reason = "the champion still ranks first on the newest window"
        return Proposal(order, champion, best_new, gate, False, reason, replaced)

    gate = offline_gate(top, champion, labels, masks, rule_interval_high, samples, seed)
    if gate.passed:
        reason = f"{top.label} passed the promotion gate"
        return Proposal(order, champion, top, gate, True, reason, replaced)
    failed = ", ".join(check.name for check in gate.failures)
    if champion is None:
        return Proposal([], None, top, gate, False, f"no first champion: failed {failed}", replaced)
    # The champion stays, and the rest keep their new order behind it. A retrained version of the
    # champion's own model goes too: one version per model in the chain.
    rest = [model for model in order if model.model != champion.model]
    reason = f"{top.label} failed the promotion gate ({failed}), so the champion stays"
    return Proposal([champion, *rest], champion, top, gate, False, reason, replaced)


@dataclass(frozen=True)
class ShadowStats:
    attempts: int
    errors: int
    p95_ms: float | None
    candidate_scores: FloatArray
    served_scores: FloatArray
    started_at: datetime
    now: datetime


def shadow_gate(stats: ShadowStats, hours: int) -> GateResult:
    """Section 4: after the shadow period, error rate, latency, and score distribution.

    Labels arrive too late to gate on, so the scores are compared with what the serving chain gave
    the same edits: a very different distribution means the candidate sees live data differently
    from how it saw the test window.
    """
    elapsed = stats.now - stats.started_at
    error_rate = stats.errors / stats.attempts if stats.attempts else None
    psi = score_psi(stats.served_scores, stats.candidate_scores)
    p95 = stats.p95_ms
    return GateResult(
        [
            Check(
                "shadow_period",
                elapsed >= timedelta(hours=hours),
                f"{elapsed.total_seconds() / 3600:.1f} hours of {hours}",
            ),
            Check(
                "shadow_volume",
                stats.attempts >= SHADOW_MIN_EDITS,
                f"{stats.attempts} live edits, minimum {SHADOW_MIN_EDITS}",
            ),
            Check(
                "shadow_errors",
                error_rate is not None and error_rate <= SHADOW_MAX_ERROR_RATE,
                "no attempts" if error_rate is None else f"{error_rate:.2%} of edits failed",
            ),
            Check(
                "shadow_latency",
                p95 is not None and p95 <= MAX_P95_MS,
                "not measured" if p95 is None else f"p95 {p95:.1f} ms",
            ),
            Check(
                "shadow_scores",
                psi is not None and psi <= SHADOW_MAX_SCORE_PSI,
                "not measured" if psi is None else f"PSI {psi:.3f} against the served scores",
            ),
        ]
    )


def side(model: Evaluated | None, labels: NDArray[np.int64], seed: int = 0) -> dict[str, Any]:
    """One side of DEC-066's before vs after report."""
    if model is None:
        return {"model": None, "version": None, "pr_auc": None, "pr_auc_interval": None}
    low, high = bootstrap_interval(pr_auc, labels, model.logits, seed=seed)
    return {
        "model": model.model,
        "version": model.version,
        "pr_auc": pr_auc(labels, model.logits),
        "pr_auc_interval": [low, high],
    }


def slice_changes(
    before: Evaluated | None,
    after: Evaluated | None,
    labels: NDArray[np.int64],
    masks: Mapping[str, NDArray[np.bool_]],
) -> dict[str, Any]:
    if before is None or after is None:
        return {}
    old = slice_metrics(labels, before.logits, masks, before.threshold)
    new = slice_metrics(labels, after.logits, masks, after.threshold)
    keys = ("rows", "positives", "gated", "pr_auc", "false_positive_rate")
    return {
        name: {
            "before": {key: old[name].get(key) for key in keys},
            "after": {key: new[name].get(key) for key in keys},
        }
        for name in old
    }


def report(
    proposal: Proposal,
    labels: NDArray[np.int64],
    masks: Mapping[str, NDArray[np.bool_]],
    min_positives: int = MIN_SLICE_POSITIVES,
) -> dict[str, Any]:
    """DEC-066's shape, which the Health page reads, plus what the gate looked at."""
    before, after = proposal.current_champion, proposal.challenger
    gate = proposal.gate
    return {
        "before": side(before, labels),
        "after": side(after, labels),
        "difference": gate.difference if gate else None,
        # Decided only after the shadow period when the champion would change.
        "promoted": None if proposal.champion_change else False,
        "reason": proposal.reason,
        "gate": gate.as_dict() if gate else None,
        "slices": slice_changes(before, after, labels, masks),
        "slice_min_positives": min_positives,
        "latency_p95_ms": {
            "before": before.p95_ms if before else None,
            "after": after.p95_ms if after else None,
        },
        "served_bytes": {
            "before": before.served_bytes if before else None,
            "after": after.served_bytes if after else None,
        },
        "proposed_order": [model.label for model in proposal.order],
        "replaced": proposal.replaced,
    }
