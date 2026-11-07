"""Known answers worked out by hand, so the ruler itself is checked before anything is measured."""

import math

import numpy as np
import pytest
from wikiguard_core.evaluation import (
    at_threshold,
    best_f1_threshold,
    binary_log_loss,
    bootstrap_interval,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    paired_bootstrap_difference,
    pr_auc,
    precision_at_k,
    recall_at_precision,
    reliability_bins,
    roc_auc,
    sigmoid,
    slice_masks,
    slice_metrics,
    threshold_for_precision,
)

LABELS = np.array([1, 0, 1, 0, 0, 1])
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def test_pr_auc_matches_a_hand_calculation() -> None:
    # Recall rises by 1/3 at ranks 1, 3, and 6, where precision is 1, 2/3, and 1/2.
    assert pr_auc(LABELS, SCORES) == pytest.approx((1 + 2 / 3 + 1 / 2) / 3)


def test_pr_auc_of_a_constant_score_is_the_positive_rate() -> None:
    labels = np.array([0] * 94 + [1] * 6)

    assert pr_auc(labels, np.zeros(100)) == pytest.approx(0.06)


def test_pr_auc_of_a_perfect_ranking_is_one() -> None:
    assert pr_auc([1, 1, 0, 0], [4, 3, 2, 1]) == pytest.approx(1.0)


def test_tied_scores_can_not_be_ordered_to_look_better() -> None:
    assert pr_auc([1, 0], [0.5, 0.5]) == pr_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_matches_a_hand_calculation() -> None:
    # Of 9 positive and negative pairs, the positive ranks higher in 3 + 2 + 0 = 5.
    assert roc_auc(LABELS, SCORES) == pytest.approx(5 / 9)


def test_roc_auc_counts_a_tie_as_half() -> None:
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_scores_with_nan_are_refused() -> None:
    with pytest.raises(ValueError, match="NaN"):
        pr_auc([1, 0], [float("nan"), 0.1])


def test_precision_at_k_reads_the_top_of_the_queue() -> None:
    assert precision_at_k(LABELS, SCORES, 3) == pytest.approx(2 / 3)


def test_threshold_is_the_lowest_score_that_reaches_the_precision() -> None:
    labels = [1] * 30 + [0] * 3 + [1] * 5 + [0] * 50
    scores = np.arange(len(labels), 0, -1, dtype=float)

    threshold = threshold_for_precision(labels, scores, target=0.9)

    # The first 33 rows hold 30 positives (90.9%). Row 34 onward stays above 90% until row 38.
    assert threshold is not None
    flagged = scores >= threshold
    assert np.array(labels)[flagged].mean() >= 0.9
    assert flagged.sum() == 38


def test_threshold_is_none_when_precision_is_never_reached() -> None:
    labels = [0, 1] * 50

    assert threshold_for_precision(labels, np.arange(100, 0, -1.0)) is None


def test_a_threshold_resting_on_a_few_rows_is_refused() -> None:
    labels = [1, 1] + [0] * 98

    assert threshold_for_precision(labels, np.arange(100, 0, -1.0), min_flagged=20) is None


def test_recall_at_precision_is_zero_when_unreachable() -> None:
    assert recall_at_precision([0, 1], [0.9, 0.1]) == 0.0


def test_at_threshold_counts_each_cell() -> None:
    result = at_threshold(LABELS, SCORES, 0.65)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)


def test_best_f1_threshold_picks_the_best_cut() -> None:
    assert best_f1_threshold([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.8)


def test_sigmoid_does_not_overflow() -> None:
    assert sigmoid(np.array([-1000.0, 0.0, 1000.0])) == pytest.approx([0.0, 0.5, 1.0])


def test_log_loss_of_a_zero_logit_is_log_two() -> None:
    assert binary_log_loss([0, 1], [0.0, 0.0]) == pytest.approx(math.log(2))


def test_ece_is_zero_when_scores_match_rates() -> None:
    labels = np.array([1, 0, 0, 0] * 25)

    assert expected_calibration_error(labels, np.full(100, 0.25)) == pytest.approx(0.0)


def test_ece_is_the_gap_for_one_bin() -> None:
    assert expected_calibration_error([0, 0, 0, 0], np.full(4, 0.5)) == pytest.approx(0.5)


def test_reliability_bins_skip_empty_ones_and_count_rows() -> None:
    bins = reliability_bins([0, 1, 1], [0.01, 0.99, 0.98])

    assert [item["rows"] for item in bins] == [1, 2]
    assert bins[1]["positive_rate"] == 1.0


def test_temperature_recovers_how_overconfident_scores_were() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 1.5, 40_000)
    labels = (rng.random(40_000) < sigmoid(true_logits)).astype(int)

    assert fit_temperature(labels, true_logits * 3) == pytest.approx(3.0, abs=0.15)


def test_temperature_does_not_change_the_ranking() -> None:
    temperature = fit_temperature(LABELS, SCORES * 4 - 2)

    assert pr_auc(LABELS, (SCORES * 4 - 2) / temperature) == pytest.approx(pr_auc(LABELS, SCORES))


def test_bootstrap_interval_contains_the_estimate_and_repeats() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 500)
    scores = labels + rng.normal(0, 1, 500)

    first = bootstrap_interval(pr_auc, labels, scores, samples=200)

    assert first[0] < pr_auc(labels, scores) < first[1]
    assert first == bootstrap_interval(pr_auc, labels, scores, samples=200)


def test_paired_difference_sees_a_gain_shared_noise_would_hide() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 2000)
    noise = rng.normal(0, 1, 2000)
    weak = labels * 0.6 + noise
    strong = labels * 0.9 + noise

    result = paired_bootstrap_difference(pr_auc, labels, strong, weak, samples=200)

    assert result["low"] > 0
    assert result["difference"] > 0


def _row(anonymous: bool, bucket: int, delta: int) -> dict[str, object]:
    return {"is_anonymous": anonymous, "user_edit_count_bucket": bucket, "bytes_delta": delta}


def test_slices_leave_a_missing_edit_count_out_of_both_groups() -> None:
    masks = slice_masks([_row(True, -1, 5), _row(False, 0, -500), _row(False, 3, 101)])

    assert masks["new_editor"].tolist() == [False, True, False]
    assert masks["experienced_editor"].tolist() == [False, False, True]
    assert masks["small_edit"].tolist() == [True, False, False]
    assert masks["anonymous"].tolist() == [True, False, False]


def test_an_empty_slice_says_so_rather_than_showing_a_number() -> None:
    masks = {"nobody": np.zeros(6, dtype=bool)}

    entry = slice_metrics(LABELS, SCORES, masks, 0.5)["nobody"]

    assert entry["rows"] == 0
    assert "not measurable" in entry["note"]
    assert "pr_auc" not in entry


def test_small_slices_are_reported_but_not_gated() -> None:
    masks = {"everyone": np.ones(6, dtype=bool)}

    assert slice_metrics(LABELS, SCORES, masks, 0.5)["everyone"]["gated"] is False


def test_report_measures_test_with_a_threshold_from_validation() -> None:
    rng = np.random.default_rng(3)
    val_labels = rng.integers(0, 2, 1000)
    val_scores = val_labels * 6 + rng.normal(0, 1, 1000)
    test_labels = rng.integers(0, 2, 1000)
    test_scores = test_labels * 6 + rng.normal(0, 1, 1000)
    rows = [_row(bool(index % 2), -1, 5) for index in range(1000)]

    report = evaluate(
        val_labels, val_scores, test_labels, test_scores, slice_masks(rows), bootstrap_samples=50
    )

    assert report["threshold_rule"] == "90_percent_precision"
    assert report["threshold"] == threshold_for_precision(val_labels, val_scores)
    # Close to 90%, not exactly: the threshold came from another window.
    assert report["at_threshold"]["precision"] > 0.8
    assert report["slices"]["new_editor"]["rows"] == 0
    assert report["false_positive_gap_anonymous_minus_registered"] is not None


def test_report_says_when_90_percent_precision_was_out_of_reach() -> None:
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 500)
    scores = rng.normal(0, 1, 500)
    rows = [_row(False, -1, 5)] * 500

    report = evaluate(labels, scores, labels, scores, slice_masks(rows), bootstrap_samples=20)

    assert report["threshold_rule"] == "best_f1"
    assert report["threshold_90_reached_on_validation"] is False
