"""Alerts, the DVC guard, the archive's filtered reads and reference samples, and new settings."""

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pytest
from wikiguard_core.alerts import Alerter, ping
from wikiguard_core.archive import (
    EDITS,
    SCORES,
    purge_references,
    read_reference,
    read_rows,
    read_table,
    reference_versions,
    sample_mask,
    write_reference,
    write_rows,
)
from wikiguard_core.config import ConfigError, RetrainSettings, StorageSettings, load_settings
from wikiguard_core.dvcops import DvcRepo

# --- Alerts ---


def test_an_alert_goes_to_discord_with_no_mentions() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(204)

    alerter = Alerter("monitor", "https://discord.test/hook", httpx.MockTransport(handler))
    assert alerter.send("Drift", "40% of features drifted")
    assert sent[0]["allowed_mentions"] == {"parse": []}
    assert "Drift" in str(sent[0]["content"]) and "monitor" in str(sent[0]["content"])


def test_an_alert_never_raises_and_is_recorded_without_a_webhook() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    failing = Alerter("trainer", "https://discord.test/hook", httpx.MockTransport(refuse))
    assert not failing.send("x", "y")
    quiet = Alerter("trainer")
    assert not quiet.send("Too few labels", "12 labels", "warning")
    assert [alert.title for alert in quiet.sent] == ["Too few labels"]


def test_a_long_alert_is_cut_to_discords_limit() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["content"])
        return httpx.Response(204)

    Alerter("x", "https://discord.test/h", httpx.MockTransport(handler)).send("t", "a" * 5000)
    assert len(sent[0]) <= 1900


def test_the_heartbeat_pings_only_when_set() -> None:
    hits: list[str] = []
    transport = httpx.MockTransport(
        lambda request: hits.append(str(request.url)) or httpx.Response(200)
    )
    assert not ping(None, transport)
    assert ping("https://hc.test/abc", transport) and hits == ["https://hc.test/abc"]


# --- DVC guard ---


class FakeRunner:
    def __init__(self, status: str = "", published: str = "") -> None:
        self.calls: list[list[str]] = []
        self.status = status
        self.published = published

    def __call__(self, args: Sequence[str], cwd: Path) -> str:
        self.calls.append(list(args))
        if args[:2] == ["git", "status"]:
            return self.status
        if args[:2] == ["git", "diff"]:
            return self.published
        return ""


def test_gc_is_off_unless_enabled(tmp_path: Path) -> None:
    runner = FakeRunner()
    assert DvcRepo(tmp_path, runner).gc_cloud(enabled=False).startswith("skipped")
    assert runner.calls == []


def test_gc_waits_while_a_pointer_differs_from_the_checkout(tmp_path: Path) -> None:
    runner = FakeRunner(status=" M snapshots/snapshot-1.dvc\n")
    assert "differ" in DvcRepo(tmp_path, runner).gc_cloud(enabled=True)
    assert not any(call[:2] == ["dvc", "gc"] for call in runner.calls)


def test_gc_waits_while_a_pointer_differs_from_github(tmp_path: Path) -> None:
    runner = FakeRunner(published="snapshots/snapshot-1.dvc\n")
    assert "differ" in DvcRepo(tmp_path, runner).gc_cloud(enabled=True)
    assert not any(call[:2] == ["dvc", "gc"] for call in runner.calls)


def test_gc_runs_once_everything_is_published(tmp_path: Path) -> None:
    runner = FakeRunner()
    assert DvcRepo(tmp_path, runner).gc_cloud(enabled=True) == "done"
    assert runner.calls[-1] == ["dvc", "gc", "--workspace", "--cloud", "--force"]


# --- Archive ---


def score_row(rev_id: int, version: str, day: int, score: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "model_name": "mlp",
        "model_version": version,
        "score": score,
        "used_fallback": False,
        "scored_at": datetime(2026, 11, day, 12, tzinfo=UTC),
        "threshold": 0.5,
        "alias": "champion",
    }


def test_read_table_keeps_days_columns_and_the_last_copy(tmp_path: Path) -> None:
    write_rows(tmp_path, SCORES, [score_row(1, "1", 1, 0.2), score_row(2, "1", 3, 0.4)])
    # Redelivered later with a new value: the last copy wins.
    write_rows(tmp_path, SCORES, [score_row(2, "1", 3, 0.9)])
    table = read_table(tmp_path, SCORES, since=date(2026, 11, 2), columns=["score"])
    assert table.column_names == ["rev_id", "model_version", "score"]
    assert table.to_pylist() == [{"rev_id": 2, "model_version": "1", "score": pytest.approx(0.9)}]
    assert len(read_rows(tmp_path, SCORES)) == 2


def test_older_score_files_read_threshold_and_alias_as_null(tmp_path: Path) -> None:
    folder = tmp_path / SCORES / "date=2026-11-01"
    folder.mkdir(parents=True)
    import pyarrow.parquet as pq

    old = score_row(1, "1", 1, 0.2)
    del old["threshold"], old["alias"]
    pq.write_table(pa.Table.from_pylist([old]), folder / "part-old.parquet")
    assert read_rows(tmp_path, SCORES)[0]["threshold"] is None


def test_the_sample_is_stable_and_near_its_fraction() -> None:
    ids = pa.array(range(20_000), pa.int64())
    first = sample_mask(ids, 0.25).to_pylist()
    assert first == sample_mask(ids, 0.25).to_pylist()
    assert 0.23 < sum(first) / len(first) < 0.27
    assert sample_mask(ids, 0.25, salt="drift").to_pylist() != first


def test_read_table_samples_before_reading_the_rest(tmp_path: Path) -> None:
    write_rows(tmp_path, SCORES, [score_row(rev, "1", 1, 0.1) for rev in range(1000)])
    table = read_table(tmp_path, SCORES, keep=lambda ids: sample_mask(ids, 0.1))
    assert 60 < table.num_rows < 140


def reference_row(rev_id: int) -> dict[str, object]:
    return {
        "rev_id": rev_id,
        "timestamp": datetime(2026, 11, 1, tzinfo=UTC),
        "split": "train",
        "label": 0,
        "feature_version": 1,
        "added_text": "words",
        "removed_text": "",
        "comment": "",
        "is_anonymous": False,
        "user_edit_count_bucket": -1,
        "bytes_delta": 5,
        "is_minor": False,
        "chars_added": 5,
        "uppercase_ratio": 0.0,
        "badword_count": 0,
        "has_external_link": False,
        "hour_of_day": 3,
        "page_length": 100,
        "score": 0.1,
    }


def test_reference_samples_round_trip_and_lose_hidden_revisions(tmp_path: Path) -> None:
    write_reference(tmp_path, "7", [reference_row(1), reference_row(2)], {"version": "7"})
    assert reference_versions(tmp_path) == ["7"]
    found = read_reference(tmp_path, "7")
    assert found is not None and found[0].num_rows == 2 and found[1]["version"] == "7"
    assert purge_references(tmp_path, {2, 99}) == 1
    found = read_reference(tmp_path, "7")
    assert found is not None and found[0].column("rev_id").to_pylist() == [1]
    assert read_reference(tmp_path, "8") is None
    # Older than the 30 days doesn't apply: cleanup deletes only edits, scores, and labels.
    assert EDITS != "reference"


# --- Settings ---


def test_retraining_settings_have_the_plans_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RETRAIN_INTERVAL_DAYS",
        "RETRAIN_ON_DRIFT",
        "RETRAIN_STARTUP_DELAY_SECONDS",
        "RETRAIN_MODELS",
        "SHADOW_HOURS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(RetrainSettings, env_file=None)
    assert settings.retrain_interval_days == 7
    assert settings.retrain_on_drift is True
    assert settings.retrain_startup_delay_seconds == 300
    assert settings.model_names == ["mlp", "bilstm", "distilbert"]
    assert settings.shadow_hours == 24


@pytest.mark.parametrize("models", ["roberta", "mlp,bert", ""])
def test_gpu_models_cant_be_retrained_automatically(
    monkeypatch: pytest.MonkeyPatch, models: str
) -> None:
    monkeypatch.setenv("RETRAIN_MODELS", models or " ,")
    with pytest.raises(ConfigError, match="RETRAIN_MODELS"):
        load_settings(RetrainSettings, env_file=None)


def test_off_switches_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRAIN_ON_DRIFT", "false")
    monkeypatch.setenv("RETRAIN_STARTUP_DELAY_SECONDS", "0")
    settings = load_settings(RetrainSettings, env_file=None)
    assert settings.retrain_on_drift is False and settings.retrain_startup_delay_seconds == 0


def test_snapshot_gc_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAPSHOT_GC_ENABLED", raising=False)
    assert load_settings(StorageSettings, env_file=None).snapshot_gc_enabled is False
