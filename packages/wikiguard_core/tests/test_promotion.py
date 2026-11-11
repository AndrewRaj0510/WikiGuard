"""The promotion gate and chain order, with scores built to pass or fail each rule."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from wikiguard_core.evaluation import masks_from_columns
from wikiguard_core.promotion import (
    MAX_P95_MS,
    RENDER_MAX_SERVED_BYTES,
    SHADOW_MIN_EDITS,
    Evaluated,
    ShadowStats,
    chain_order,
    offline_gate,
    propose,
    report,
    shadow_gate,
)

ROWS = 4000
RNG = np.random.default_rng(7)
LABELS = (RNG.random(ROWS) < 0.1).astype(np.int64)
MASKS = masks_from_columns(
    np.arange(ROWS) % 2 == 0,
    np.full(ROWS, -1, dtype=np.int64),
    np.where(np.arange(ROWS) % 3 == 0, 500, 10).astype(np.int64),
)


def scores(signal: float, seed: int) -> np.ndarray:
    """Logits whose ranking quality grows with signal."""
    noise = np.random.default_rng(seed).normal(0, 1, ROWS)
    return signal * LABELS + noise


def model(
    name: str, version: str | None, signal: float, seed: int = 1, **changes: object
) -> Evaluated:
    base = Evaluated(
        name,
        version,
        scores(signal, seed),
        threshold=1.0,
        is_new=version is None,
        p95_ms=40.0,
        served_bytes=60_000_000,
        golden_pr_auc=0.7,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


CHAMPION = model("roberta", "1", 1.0, seed=1)


def gate(candidate: Evaluated, champion: Evaluated | None = CHAMPION) -> dict[str, bool]:
    result = offline_gate(candidate, champion, LABELS, MASKS, rule_interval_high=0.2, samples=200)
    return {check.name: check.passed for check in result.checks}


def test_a_clearly_better_candidate_passes_every_check() -> None:
    checks = gate(model("distilbert", None, 2.0, seed=2))
    assert all(checks.values()), checks


def test_the_same_model_again_fails_the_gain_and_the_interval() -> None:
    checks = gate(replace(CHAMPION, version=None, is_new=True))
    assert not checks["pr_auc_gain"] and not checks["paired_bootstrap"]


def test_an_equally_good_model_has_an_interval_that_includes_zero() -> None:
    # Same signal, its own noise: any difference is chance, and the paired interval shows it.
    result = offline_gate(model("bilstm", None, 1.0, seed=9), CHAMPION, LABELS, MASKS, samples=200)
    assert result.difference is not None
    assert result.difference["low"] <= 0 <= result.difference["high"]
    assert not result.passed


def test_pairing_finds_a_small_gain_shared_noise_would_hide() -> None:
    # The same noise with a little more signal: a real gain, and pairing keeps the interval tight.
    result = offline_gate(model("bilstm", None, 1.03, seed=1), CHAMPION, LABELS, MASKS, samples=200)
    assert result.difference is not None and result.difference["low"] > 0


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"p95_ms": MAX_P95_MS + 1}, "latency"),
        ({"p95_ms": None}, "latency"),
        ({"served_bytes": RENDER_MAX_SERVED_BYTES + 1}, "size"),
        ({"golden_pr_auc": 0.6}, "golden_set"),
        ({"golden_pr_auc": None}, "golden_set"),
        ({"checksum_ok": False}, "checksum"),
        ({"feature_version_ok": False}, "feature_version"),
        ({"smoke_passed": False}, "smoke_test"),
        ({"warnings": ("loss went up",)}, "training_warnings"),
    ],
)
def test_each_rule_can_block_a_better_model(change: dict[str, object], check: str) -> None:
    candidate = replace(model("distilbert", None, 2.0, seed=2), **change)  # type: ignore[arg-type]
    checks = gate(candidate)
    assert not checks[check]


def test_a_small_golden_drop_inside_the_allowance_passes() -> None:
    assert gate(model("distilbert", None, 2.0, seed=2, golden_pr_auc=0.69))["golden_set"]


def test_a_slice_that_gets_worse_blocks_promotion() -> None:
    candidate = model("distilbert", None, 2.0, seed=2)
    hurt = candidate.logits.copy()
    anonymous = MASKS["anonymous"]
    # Ruin the ranking for logged-out editors only.
    hurt[anonymous] = np.random.default_rng(3).normal(0, 1, int(anonymous.sum()))
    checks = gate(replace(candidate, logits=hurt))
    assert not checks["slice_anonymous"]
    assert checks["slice_registered"]


def test_a_first_champion_must_clear_the_rule_baseline() -> None:
    assert gate(model("bilstm", None, 2.0), champion=None)["beats_rule_baseline"]
    weak = offline_gate(model("bilstm", None, 0.0), None, LABELS, MASKS, 0.2, samples=200)
    assert not weak.passed


def test_the_mlp_is_always_last_and_flagged_models_rank_behind_healthy_ones() -> None:
    order = chain_order(
        [
            model("mlp", "5", 3.0),
            model("bert", "4", 2.5, warnings=("parity",)),
            model("bilstm", "3", 1.5),
            model("roberta", "1", 2.0),
        ],
        LABELS,
    )
    assert [m.model for m in order] == ["roberta", "bilstm", "bert", "mlp"]


def current_chain() -> list[Evaluated]:
    return [CHAMPION, model("bilstm", "3", 0.7, seed=4), model("mlp", "5", 0.5, seed=5)]


def test_a_retrained_fallback_replaces_its_older_version_without_touching_the_champion() -> None:
    proposal = propose(
        current_chain(), [model("bilstm", "6", 0.9, seed=6)], LABELS, MASKS, 0.2, samples=200
    )
    assert not proposal.champion_change
    assert [m.label for m in proposal.order] == ["roberta v1", "bilstm v6", "mlp v5"]
    assert proposal.replaced == ["bilstm v3 by bilstm v6"]


def test_a_worse_retrained_fallback_keeps_the_old_one() -> None:
    proposal = propose(
        current_chain(), [model("bilstm", "6", 0.3, seed=6)], LABELS, MASKS, 0.2, samples=200
    )
    assert [m.label for m in proposal.order] == ["roberta v1", "bilstm v3", "mlp v5"]


def test_a_candidate_that_wins_the_gate_becomes_the_proposed_champion() -> None:
    winner = model("distilbert", "6", 2.0, seed=2)
    proposal = propose(current_chain(), [winner], LABELS, MASKS, 0.2, samples=200)
    assert proposal.champion_change and proposal.challenger is winner
    assert [alias for alias, _ in proposal.aliases()] == [
        "champion",
        "fallback-1",
        "fallback-2",
        "fallback-3",
    ]
    assert proposal.order[-1].model == "mlp"


def test_a_candidate_that_ranks_first_but_fails_the_gate_leaves_the_champion() -> None:
    slow = model("distilbert", "6", 2.0, seed=2, p95_ms=900.0)
    proposal = propose(current_chain(), [slow], LABELS, MASKS, 0.2, samples=200)
    assert not proposal.champion_change
    assert proposal.order[0] is CHAMPION
    assert "latency" in proposal.reason


def test_a_candidate_failing_artifact_checks_is_never_used() -> None:
    broken = model("bilstm", "6", 3.0, seed=6, checksum_ok=False)
    proposal = propose(current_chain(), [broken], LABELS, MASKS, 0.2, samples=200)
    assert all(m.version != "6" for m in proposal.order)


def test_the_report_has_dec_066s_shape() -> None:
    proposal = propose(
        current_chain(), [model("distilbert", "6", 2.0, seed=2)], LABELS, MASKS, 0.2, samples=200
    )
    body = report(proposal, LABELS, MASKS)
    assert body["before"]["model"] == "roberta" and body["before"]["version"] == "1"
    assert body["after"]["version"] == "6"
    assert len(body["after"]["pr_auc_interval"]) == 2
    assert set(body["difference"]) == {"mean", "low", "high"}
    # The champion would change, so the decision waits for the shadow period.
    assert body["promoted"] is None
    assert body["gate"]["passed"] is True


def stats(**changes: object) -> ShadowStats:
    rng = np.random.default_rng(0)
    served = rng.beta(1, 8, 2000)
    base = {
        "attempts": 2000,
        "errors": 0,
        "p95_ms": 35.0,
        "candidate_scores": np.clip(served + rng.normal(0, 0.01, 2000), 0, 1),
        "served_scores": served,
        "started_at": datetime(2026, 11, 1, tzinfo=UTC),
        "now": datetime(2026, 11, 2, 1, tzinfo=UTC),
    }
    return ShadowStats(**(base | changes))  # type: ignore[arg-type]


def test_a_healthy_shadow_passes() -> None:
    assert shadow_gate(stats(), hours=24).passed


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"now": datetime(2026, 11, 1, 23, tzinfo=UTC)}, "shadow_period"),
        ({"attempts": SHADOW_MIN_EDITS - 1}, "shadow_volume"),
        ({"errors": 30}, "shadow_errors"),
        ({"p95_ms": 400.0}, "shadow_latency"),
        ({"candidate_scores": np.random.default_rng(1).beta(5, 2, 2000)}, "shadow_scores"),
    ],
)
def test_each_shadow_rule_can_block(change: dict[str, object], check: str) -> None:
    result = shadow_gate(stats(**change), hours=24)
    assert [failure.name for failure in result.failures] == [check]


def test_a_zero_hour_shadow_is_allowed_for_replays() -> None:
    assert shadow_gate(stats(now=datetime(2026, 11, 1, tzinfo=UTC) + timedelta(0)), 0).passed


def test_a_retrained_champion_model_that_fails_the_gate_leaves_one_version_of_it() -> None:
    again = model("roberta", "6", 1.0, seed=1)
    proposal = propose(current_chain(), [again], LABELS, MASKS, 0.2, samples=200)
    assert not proposal.champion_change
    assert [m.label for m in proposal.order] == ["roberta v1", "bilstm v3", "mlp v5"]
