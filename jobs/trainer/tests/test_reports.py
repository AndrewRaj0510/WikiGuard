"""Baselines, robustness, aging, review files, charts, and the alias order, on tiny data."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from trainer import aging as aging_module  # noqa: E402
from trainer import charts, finish  # noqa: E402
from trainer import robustness as robustness_module  # noqa: E402
from trainer.baselines import (  # noqa: E402
    fetch_lift_wing,
    lift_wing_report,
    lift_wing_sample,
    measure_baselines,
    read_lift_wing,
    rule_score,
)
from trainer.data import load_dataset  # noqa: E402
from trainer.finish import alias_order, clears_baselines, leaderboard_rows, model_card  # noqa: E402
from trainer.review import describe, golden_candidates, top_errors  # noqa: E402
from trainer.robustness import (  # noqa: E402
    OBFUSCATIONS,
    _rebuild,
    case_flip,
    homoglyphs,
    leetspeak,
    spacing,
    zero_width,
)
from wikiguard_core.features import normalize_text  # noqa: E402

SECRET = "text-that-must-not-reach-an-image"


@pytest.fixture(scope="module")
def data(snapshot_folder: Path) -> Any:
    return load_dataset(snapshot_folder)


def test_rule_score_rises_with_each_signal() -> None:
    base = {"badword_count": 0, "is_anonymous": False, "bytes_delta": 10}

    assert rule_score(base | {"badword_count": 2}) > rule_score(base)
    assert rule_score(base | {"is_anonymous": True}) > rule_score(base)
    assert rule_score(base | {"bytes_delta": -5000}) > rule_score(base)


def test_prevalence_baseline_equals_the_positive_rate(data: Any) -> None:
    rows = [*data.validation, *data.test]

    report = measure_baselines(rows)

    assert report["test"]["prevalence_pr_auc"] == pytest.approx(report["test"]["positive_rate"])
    assert report["test"]["rule"]["pr_auc_interval"][0] < report["test"]["rule"]["pr_auc"]


def test_lift_wing_sample_is_fixed_and_comes_only_from_test(data: Any) -> None:
    rows = [*data.train, *data.test]

    first = lift_wing_sample(rows, 50)

    assert {row["split"] for row in first} == {"test"}
    assert first == lift_wing_sample(list(reversed(rows)), 50)


def _lift_wing_client(failures: dict[int, int]) -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        rev_id = json.loads(request.content)["rev_id"]
        if rev_id in failures:
            return httpx.Response(failures[rev_id])
        body = {"model_version": "3", "output": {"probabilities": {"true": rev_id / 100}}}
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_lift_wing_fetch_records_scores_and_counts_a_refused_revision(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl"

    fetch_lift_wing([10, 20], out, _lift_wing_client({20: 404}))

    assert read_lift_wing(out) == {10: 0.1, 20: None}


def test_lift_wing_fetch_resumes_without_asking_again(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl"
    fetch_lift_wing([10], out, _lift_wing_client({}))

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("asked for a revision it already has")

    fetch_lift_wing([10], out, httpx.Client(transport=httpx.MockTransport(fail)))

    assert read_lift_wing(out) == {10: 0.1}


def test_lift_wing_report_leaves_out_unanswered_revisions() -> None:
    rows = [{"rev_id": 1, "label": 1}, {"rev_id": 2, "label": 0}, {"rev_id": 3, "label": 0}]

    report = lift_wing_report(rows, {1: 0.9, 2: 0.1, 3: None})

    assert report["rows"] == 2
    assert report["unanswered"] == 1
    assert report["pr_auc"] == pytest.approx(1.0)


def test_each_disguise_changes_the_text() -> None:
    for name, change in OBFUSCATIONS.items():
        assert change("bad words") != "bad words", name


def test_normalization_undoes_zero_width_but_not_the_others() -> None:
    """Which tricks build_features removes, as the robustness report needs to show."""
    text = "bad words"

    assert normalize_text(zero_width(text)) == text
    assert normalize_text(leetspeak(text)) != text
    assert normalize_text(homoglyphs(text)) != text
    assert normalize_text(spacing(text)) != text
    assert normalize_text(case_flip(text)) != text


def test_a_disguised_row_is_rebuilt_through_build_features(data: Any) -> None:
    row = data.test[0]

    rebuilt = _rebuild(row, "HELLO http://example.com")

    assert rebuilt["chars_added"] == len("HELLO http://example.com")
    assert rebuilt["has_external_link"] is True
    assert rebuilt["bytes_delta"] == row["bytes_delta"]
    assert rebuilt["hour_of_day"] == row["hour_of_day"]
    assert rebuilt["label"] == row["label"]


def test_weekly_windows_never_mix_weeks_and_end_at_the_data_end(data: Any) -> None:
    weeks = aging_module.weekly_windows(data)

    assert len(weeks) >= 1
    last_of_each = [max(row["timestamp"] for row in week) for week in weeks]
    first_of_each = [min(row["timestamp"] for row in week) for week in weeks]
    assert all(
        earlier < later for earlier, later in zip(last_of_each, first_of_each[1:], strict=False)
    )


def test_top_errors_picks_the_worst_of_each_kind() -> None:
    rows = [{"label": label} for label in (0, 0, 1, 1)]
    logits = np.array([0.1, 0.9, -5.0, 3.0])

    false_positives, false_negatives = top_errors(rows, logits, 1)

    assert false_positives == [1]
    assert false_negatives == [2]


def test_review_text_is_clipped() -> None:
    row = {
        "rev_id": 1,
        "label": 1,
        "is_anonymous": False,
        "bytes_delta": 1,
        "comment": "c",
        "added_text": "x" * 1000,
        "removed_text": "",
    }

    assert len(describe(row)) < 400


def test_golden_candidates_cover_each_group_without_a_model_score(data: Any) -> None:
    chosen = golden_candidates(data.test, 40, 1)

    assert len({row["rev_id"] for row in chosen}) == len(chosen)
    assert {row["label"] for row in chosen} == {0, 1}
    assert chosen == golden_candidates(data.test, 40, 1)


def report_stub(model: str, score: float) -> dict[str, Any]:
    return {
        "model": model,
        "pr_auc": score,
        "pr_auc_interval": [score - 0.01, score + 0.01],
        "roc_auc": 0.8,
        "recall_at_90_precision": 0.1,
        "recall_at_90_precision_interval": [0.05, 0.15],
        "precision_at_50": 0.6,
        "threshold_rule": "best_f1",
        "at_threshold": {
            "tp": 10,
            "fp": 5,
            "fn": 20,
            "tn": 100,
            "precision": 0.66,
            "recall": 0.33,
            "f1": 0.44,
        },
        "ece_before": 0.3,
        "ece_after": 0.02,
        "temperature": 2.0,
        "reliability_before": [{"mean_score": 0.5, "positive_rate": 0.1, "rows": 10}],
        "reliability_after": [{"mean_score": 0.1, "positive_rate": 0.1, "rows": 10}],
        "parity": {"test": {"gap": 0.001}},
        "slices": {
            "anonymous": {
                "rows": 5,
                "positives": 2,
                "pr_auc": 0.5,
                "false_positive_rate": 0.1,
            },
            "new_editor": {"rows": 0, "positives": 0, "note": "not measurable"},
        },
        "false_positive_gap_anonymous_minus_registered": 0.01,
        "fit": {
            "best_epoch": 1,
            "seconds": 60,
            "history": [
                {"epoch": 0.0, "train_sample_pr_auc": 0.1, "validation_pr_auc": 0.1},
                {"epoch": 1.0, "train_sample_pr_auc": 0.5, "validation_pr_auc": 0.4},
            ],
        },
        "rows": 135,
        "positives": 30,
        "run_id": "run",
        "artifact_sha256": "digest",
        "params": {"hidden": 8},
        "train_rows": 100,
        "device_name": "cpu",
        "parameters": {"total": 10, "trainable": 10},
        "served_bytes": 1024 * 1024,
    }


def test_the_best_model_is_champion_and_the_mlp_is_always_last() -> None:
    reports = {
        "mlp": report_stub("mlp", 0.9),
        "bilstm": report_stub("bilstm", 0.3),
        "roberta": report_stub("roberta", 0.5),
    }

    assert alias_order(reports) == ["roberta", "bilstm", "mlp"]


def test_a_chart_is_written_and_holds_only_totals(tmp_path: Path) -> None:
    report = report_stub("mlp", 0.5)
    day = date(2026, 11, 6)
    labels = np.array([1, 0, 1, 0])
    logits = np.array([2.0, 1.0, 0.5, -1.0])

    paths = [
        charts.confusion_matrix(report, tmp_path, day),
        charts.learning_curves(report, tmp_path, day),
        charts.reliability(report, tmp_path, day),
        charts.slices(report, tmp_path, day),
        charts.pr_curves({"mlp": (labels, logits)}, 0.5, tmp_path, day),
        charts.leaderboard(leaderboard_rows({"mlp": report}, None), tmp_path, day),
    ]

    assert [path.name for path in paths][0] == "eval_confusion_mlp_2026-11-06.png"
    assert all(path.stat().st_size > 1000 for path in paths)
    assert all(SECRET.encode() not in path.read_bytes() for path in paths)


def test_a_model_card_states_what_could_not_be_measured() -> None:
    manifest = {
        "name": "snapshot-x",
        "content_sha256": "hash-after-a-later-purge",
        "windows": {
            "validation_start": "2026-10-01T00",
            "test_start": "2026-10-08T00",
            "end": "2026-10-15T00",
        },
    }
    report = report_stub("mlp", 0.5) | {"snapshot_content_sha256": "hash-when-trained"}

    card = model_card(report, manifest, None, None, None, "3")

    assert "hash-when-trained" in card
    assert "hash-after-a-later-purge" not in card
    assert "version 3" in card
    assert "| new_editor | 0 | 0 | not measured | not measured |" in card
    assert "It never reverts anything" in card
    assert "snapshot-x" in card


def _scored(pr_auc: float, warnings: list[str] | None = None) -> dict[str, Any]:
    return {"pr_auc": pr_auc, "training_warnings": warnings or []}


def test_a_flagged_model_cant_become_champion_however_high_it_scores() -> None:
    reports = {
        "mlp": _scored(0.25),
        "bert": _scored(0.40, ["the served model differs from PyTorch"]),
        "bilstm": _scored(0.31),
    }

    assert alias_order(reports) == ["bilstm", "bert", "mlp"]


def test_first_champion_needs_an_interval_clear_of_the_rule_baseline() -> None:
    baselines = {"test": {"rule": {"pr_auc_interval": [0.17, 0.20]}}}

    assert clears_baselines({"pr_auc_interval": [0.25, 0.30]}, baselines)
    assert not clears_baselines({"pr_auc_interval": [0.19, 0.30]}, baselines)


def test_a_rerun_without_register_keeps_the_registered_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aliases = {
        "champion": {"version": "5", "model": "roberta"},
        "fallback-1": {"version": "3", "model": "distilbert"},
    }
    (tmp_path / "aliases.json").write_text(json.dumps(aliases), "utf-8")
    monkeypatch.setattr(finish, "RUNS_FOLDER", tmp_path)

    assert finish.registered_versions() == {"roberta": "5", "distilbert": "3"}


def test_no_registration_yet_means_no_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finish, "RUNS_FOLDER", tmp_path)

    assert finish.registered_versions() == {}


class _WordScorer:
    """Scores an edit by whether its added text still holds a word the disguise breaks."""

    def logits(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.array([5.0 if "bad" in row["added_text"] else -5.0 for row in rows])


def test_robustness_reports_the_pr_auc_a_disguise_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"label": 1, "added_text": "bad"},
        {"label": 1, "added_text": "bad"},
        {"label": 0, "added_text": "fine"},
        {"label": 0, "added_text": "fine"},
    ]
    monkeypatch.setattr(robustness_module, "sample", lambda chosen, seed: chosen)
    monkeypatch.setattr(
        robustness_module, "_rebuild", lambda row, added: row | {"added_text": added}
    )
    monkeypatch.setattr(
        robustness_module,
        "OBFUSCATIONS",
        {"harmless": lambda text: text, "breaks_word": lambda text: "b a d"},
    )

    report = robustness_module.robustness_report(_WordScorer(), rows, 0.0)  # type: ignore[arg-type]

    assert report["original_pr_auc"] == 1.0
    assert report["changes"]["harmless"]["pr_auc_drop"] == 0.0
    assert report["changes"]["breaks_word"]["pr_auc_drop"] > 0.0
    assert report["changes"]["breaks_word"]["recall_drop"] == 1.0


def test_robustness_chart_draws_the_pr_auc_drop(tmp_path: Path) -> None:
    report = {"changes": {"spacing": {"pr_auc_drop": 0.02}, "case_flip": {"pr_auc_drop": -0.01}}}

    path = charts.robustness({"mlp": report, "roberta": report}, tmp_path, date(2026, 11, 7))

    assert path.exists()
