"""The monitor's numbers and decisions, on arrays with known answers."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from wikiguard_core.monitoring import (
    MIN_LABELS,
    Observed,
    PostCheckInputs,
    TriggerInputs,
    daily,
    decide,
    empty_observed,
    post_promotion,
    reviewer_precision,
    score_psi,
    summarize,
)

NOW = datetime(2026, 11, 20, 6, tzinfo=UTC)


def observed(labels: list[int], scores: list[float], days: list[str] | None = None) -> Observed:
    count = len(labels)
    stamps = days or ["2026-11-10"] * count
    return Observed(
        rev_id=np.arange(count, dtype=np.int64),
        edited_at=np.array([f"{day}T12:00:00" for day in stamps], dtype="datetime64[s]"),
        label=np.array(labels, dtype=np.int64),
        score=np.array(scores, dtype=np.float64),
        threshold=np.full(count, 0.5),
        used_fallback=np.array([index % 4 == 0 for index in range(count)]),
        model_version=np.array(["1"] * count, dtype=object),
        is_anonymous=np.array([index % 2 == 0 for index in range(count)]),
        edit_count_bucket=np.full(count, -1, dtype=np.int64),
        bytes_delta=np.array([10 if index % 3 else 500 for index in range(count)], dtype=np.int64),
    )


def test_a_perfect_ranking_scores_one_and_its_rates_are_counted() -> None:
    report = summarize(observed([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), threshold=0.5)
    assert report["pr_auc"] == pytest.approx(1.0)
    assert report["labeled"] == 4 and report["positives"] == 2
    assert report["alert_rate"] == pytest.approx(0.5)
    assert report["precision_at_threshold"] == pytest.approx(1.0)
    assert report["fallback_rate"] == pytest.approx(0.25)
    assert set(report["slices"]) >= {"anonymous", "registered", "small_edit", "large_edit"}


def test_one_class_has_no_pr_auc_rather_than_a_wrong_one() -> None:
    assert summarize(observed([0, 0], [0.4, 0.6]), 0.5)["pr_auc"] is None
    assert summarize(empty_observed(), 0.5)["labeled"] == 0


def test_without_a_threshold_there_are_no_alert_rates() -> None:
    report = summarize(observed([1, 0], [0.9, 0.1]), threshold=None)
    assert "alert_rate" not in report and "slices" not in report


def test_daily_gives_one_point_per_edit_day() -> None:
    days = daily(
        observed(
            [1, 0, 1, 0],
            [0.9, 0.1, 0.2, 0.8],
            ["2026-11-10", "2026-11-10", "2026-11-11", "2026-11-11"],
        ),
        0.5,
    )
    assert list(days) == ["2026-11-10", "2026-11-11"]
    assert days["2026-11-10"]["pr_auc"] == pytest.approx(1.0)
    assert days["2026-11-11"]["pr_auc"] == pytest.approx(0.5)


def test_score_psi_is_zero_for_the_same_scores_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.beta(1, 8, 5000)
    assert score_psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    assert score_psi(reference, rng.beta(1, 8, 5000)) < 0.05
    assert score_psi(reference, rng.beta(4, 2, 5000)) > 1.0
    assert score_psi(reference, np.array([])) is None


def test_reviewer_precision_ignores_skips_and_counts_good_faith_as_confirmed() -> None:
    decisions = ["vandalism", "good_faith_mistake", "ok", "skip", "vandalism"]
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.1])
    report = reviewer_precision(decisions, scores, threshold=0.5)
    # Flagged and judged: the first three. Two of them were confirmed.
    assert report["flagged_and_judged"] == 3
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["decisions"]["skip"] == 1
    assert report["confirmed_share"] == pytest.approx(3 / 4)


def inputs(**changes: object) -> TriggerInputs:
    base = {
        "now": NOW,
        "live_pr_auc": 0.36,
        "live_labels": 10_000,
        "offline_pr_auc": 0.3641,
        "drift_share": 0.1,
        "text_drift": False,
        "last_retrain_started": NOW - timedelta(days=2),
        "schedule_since": NOW - timedelta(days=2),
        "interval_days": 7,
        "on_drift": True,
    }
    return TriggerInputs(**(base | changes))  # type: ignore[arg-type]


def test_nothing_fires_on_a_healthy_day() -> None:
    decision = decide(inputs())
    assert not decision.retrain and decision.reasons == [] and decision.alerts == []


def test_a_performance_drop_past_the_margin_retrains() -> None:
    assert not decide(inputs(live_pr_auc=0.3641 - 0.029)).retrain
    decision = decide(inputs(live_pr_auc=0.3641 - 0.031))
    assert decision.retrain and decision.trigger == "performance"


def test_too_few_labels_alerts_and_cant_trigger_on_performance() -> None:
    decision = decide(inputs(live_labels=MIN_LABELS - 1, live_pr_auc=0.1))
    assert not decision.retrain
    assert decision.alerts[0][0] == "Too few labels"


@pytest.mark.parametrize(
    ("share", "text", "fires"),
    [
        (0.29, False, False),
        (0.30, False, True),
        (0.0, True, True),
    ],
)
def test_drift_fires_at_thirty_percent_or_on_text(share: float, text: bool, fires: bool) -> None:
    decision = decide(inputs(drift_share=share, text_drift=text))
    assert decision.retrain is fires
    assert (decision.trigger == "drift") is fires


def test_with_retrain_on_drift_off_drift_and_performance_only_alert() -> None:
    decision = decide(inputs(on_drift=False, drift_share=0.5, live_pr_auc=0.1))
    assert not decision.retrain
    assert [title for title, _ in decision.alerts] == [
        "Performance trigger (retraining is off)",
        "Drift trigger (retraining is off)",
    ]


def test_the_schedule_still_runs_with_retrain_on_drift_off() -> None:
    decision = decide(
        inputs(
            on_drift=False,
            drift_share=0.9,
            schedule_since=NOW - timedelta(days=8),
            last_retrain_started=NOW - timedelta(days=8),
        )
    )
    assert decision.retrain and decision.trigger == "schedule"


@pytest.mark.parametrize("interval", [1, 7, 14])
def test_the_schedule_follows_retrain_interval_days(interval: int) -> None:
    just_before = NOW - timedelta(days=interval) + timedelta(hours=1)
    just_after = NOW - timedelta(days=interval)
    assert not decide(
        inputs(interval_days=interval, schedule_since=just_before, last_retrain_started=just_before)
    ).retrain
    due = decide(
        inputs(interval_days=interval, schedule_since=just_after, last_retrain_started=just_after)
    )
    assert due.retrain and due.trigger == "schedule"


def test_never_retrained_is_due() -> None:
    decision = decide(inputs(schedule_since=None, last_retrain_started=None))
    assert decision.retrain and "never retrained" in decision.reasons[0]


def test_at_most_one_retrain_a_day_whatever_fires() -> None:
    decision = decide(
        inputs(drift_share=0.9, live_pr_auc=0.1, last_retrain_started=NOW - timedelta(hours=23))
    )
    assert not decision.retrain
    assert decision.reasons[-1].startswith("not started")


def test_performance_names_the_run_when_several_fire() -> None:
    decision = decide(
        inputs(drift_share=0.9, live_pr_auc=0.1, schedule_since=None, last_retrain_started=None)
    )
    assert decision.trigger == "performance" and len(decision.reasons) == 3


def check(**changes: object) -> PostCheckInputs:
    base = {
        "now": NOW,
        "promoted_at": NOW - timedelta(days=3),
        "new_pr_auc": 0.35,
        "new_labels": 600,
        "previous_pr_auc": 0.36,
        "previous_labels": 5000,
    }
    return PostCheckInputs(**(base | changes))  # type: ignore[arg-type]


def test_the_post_check_waits_for_500_labels_or_7_days() -> None:
    assert not post_promotion(check(new_labels=499)).due
    assert post_promotion(check(new_labels=499, promoted_at=NOW - timedelta(days=7))).due
    assert post_promotion(check()).due


def test_a_drop_over_003_rolls_back_and_a_small_one_doesnt() -> None:
    assert not post_promotion(check(new_pr_auc=0.36 - 0.029)).rollback
    result = post_promotion(check(new_pr_auc=0.36 - 0.031))
    assert result.rollback and result.reason.startswith("rolled back")


def test_an_unmeasurable_side_never_rolls_back() -> None:
    result = post_promotion(check(new_pr_auc=None))
    assert result.due and not result.rollback and "not measurable" in result.reason
