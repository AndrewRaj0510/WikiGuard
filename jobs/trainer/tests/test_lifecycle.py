"""Continuous training end to end on the tiny dataset: snapshot from the archive, real MLP training,
the gate, the shadow check, promotion, rollback, a poisoned snapshot, and storage retention.

MLflow logs to a throwaway SQLite file and the registry is a temporary folder, so nothing here can
touch data/runs, data/mlflow, or DagsHub.
"""

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from trainer.archive_snapshot import build_from_archive  # noqa: E402
from trainer.data import Dataset, load_dataset  # noqa: E402
from trainer.lifecycle import Lifecycle, LifecycleError  # noqa: E402
from trainer.pipeline import Paths, Pipeline, startup_wait  # noqa: E402
from trainer.records import MemoryRecorder, ShadowRow  # noqa: E402
from trainer.registry import LocalRegistry, RegistryError  # noqa: E402
from trainer.storage import retain, snapshots_to_delete, versions_to_delete  # noqa: E402
from trainer.train import train_final  # noqa: E402
from wikiguard_core.alerts import Alerter  # noqa: E402
from wikiguard_core.archive import EDITS, LABELS, read_reference, write_rows  # noqa: E402
from wikiguard_core.chain import LocalRuns  # noqa: E402
from wikiguard_core.dvcops import DvcRepo  # noqa: E402
from wikiguard_core.schemas import EditEvent  # noqa: E402
from wikiguard_core.snapshot import Candidate, read_manifest, write_snapshot  # noqa: E402

# The trainer's own conftest, by path: "conftest" alone finds jobs/conftest.py first.
_spec = importlib.util.spec_from_file_location(
    "trainer_fixtures", Path(__file__).with_name("conftest.py")
)
assert _spec is not None and _spec.loader is not None
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
AS_OF = _fixtures.AS_OF
END = AS_OF - timedelta(hours=48)


def tiny_candidates() -> list[Candidate]:
    """Like conftest's, but the same words mean the same thing in every window, and logged-out
    editors make good edits too, so a working model wins in every slice."""
    rows = []
    rev_id = 0
    for days, count in ((20, 600), (10, 400), (1, 1800)):
        for index in range(count):
            rev_id += 1
            bad = index % 3 == 0
            words = _fixtures.BAD_WORDS if bad else _fixtures.GOOD_WORDS
            edit = EditEvent(
                rev_id=rev_id,
                parent_rev_id=None,
                title="Page",
                timestamp=END - timedelta(days=days) + timedelta(seconds=index * 7),
                is_minor=False,
                is_anonymous=index % 2 == 0,
                user_hash=None,
                comment="" if index % 5 == 0 else "copyedit",
                old_length=1000,
                new_length=1000 + (index % 40) - 20,
                added_text=f"{words[(index // 3) % 3]} {rev_id}",
                removed_text="",
            )
            rows.append(Candidate(rev_id, edit, int(bad)))
    return rows


CPU = torch.device("cpu")
MLP_PARAMS = {"hidden": 32, "dropout": 0.1, "learning_rate": 3e-3, "use_comment": True}
# The recipe's batch of 128 gives 3 steps an epoch on 300 rows, so the candidate needs more epochs
# to learn the tiny data's signal and clearly beat the weak champion in every slice.
CANDIDATE_PARAMS = MLP_PARAMS | {"epochs": 40}


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("wikiguard", (tmp_path / "artifacts").resolve().as_uri())
    mlflow.set_experiment("wikiguard")


def write_archive(root: Path, poison_day: bool = False, leak: bool = False) -> None:
    edits, labels = [], []
    # conftest's rows give bad edits an empty comment and only bad edits are logged out: a
    # metadata leak, which the leak test needs.
    for candidate in _fixtures.tiny_candidates() if leak else tiny_candidates():
        assert candidate.edit is not None
        edits.append(candidate.edit.model_dump())
        label = candidate.label
        # A coordinated burst of fake reverts: every edit of the 10-days-back day "reverted".
        if poison_day and AS_OF - timedelta(days=13) < candidate.edit.timestamp < AS_OF - timedelta(
            days=11
        ):
            label = 1
        labels.append(
            {
                "schema_version": 1,
                "rev_id": candidate.rev_id,
                "label": label,
                "excluded_reason": None,
                "decided_at": candidate.edit.timestamp + timedelta(hours=48),
                "decided_late": False,
            }
        )
    write_rows(root, EDITS, edits)
    write_rows(root, LABELS, labels)


def shuffled(data: Dataset) -> Dataset:
    """Labels at random: a model trained on this is the deliberately bad one."""
    rng = np.random.default_rng(1)

    def mix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = rng.permutation([row["label"] for row in rows])
        return [row | {"label": int(label)} for row, label in zip(rows, labels, strict=True)]

    return dataclasses.replace(data, train=mix(data.train), validation=mix(data.validation))


class World:
    """A sandbox: archive, snapshots, golden set, a registry holding a weak champion."""

    def __init__(self, tmp: Path, poison_day: bool = False, leak: bool = False) -> None:
        self.tmp = tmp
        self.archive = tmp / "archive"
        write_archive(self.archive, poison_day, leak)
        self.paths = Paths(
            workdir=tmp / "retrain",
            snapshots=tmp / "snapshots",
            archive=self.archive,
            golden=tmp / "golden.json",
            images=tmp / "images",
            cache=tmp / "cache",
        )
        self.registry = LocalRegistry.create(tmp / "registry")
        self.recorder = MemoryRecorder()
        self.alerter = Alerter("trainer")
        self.settings = SimpleNamespace(
            retrain_timeout_minutes=60,
            retrain_startup_delay_seconds=0,
            retrain_threads=2,
            model_names=["mlp"],
            retrain_snapshot_rows=100_000,
            shadow_hours=0,
        )
        self.pipeline = Pipeline(
            self.settings,
            self.registry,
            self.recorder,
            self.alerter,
            self.paths,
            clock=lambda: AS_OF,
            train=lambda name, data, out, meta: train_final(
                name, data, CANDIDATE_PARAMS, CPU, None, 0, out, "retrain", meta
            ),
            # The tiny data plants a metadata signal on purpose, which the real leak check stops
            # (see the test below), so the other tests pass the checks they aren't about.
            sanity=lambda data, models: {
                "passed": True,
                "failures": [],
                "shuffled_validation_pr_auc": 0.33,
                "metadata_only_validation_pr_auc": 0.5,
            },
        )
        self.lifecycle = Lifecycle(
            self.settings, self.registry, self.recorder, self.alerter, self.pipeline
        )

    def seed_weak_champion(self) -> str:
        snapshot = write_snapshot(
            build_from_archive(self.archive, AS_OF, 100_000, set()), self.paths.snapshots
        )
        data = load_dataset(snapshot)
        golden = [row | {"checked_label": row["label"]} for row in data.test[:200]]
        self.paths.golden.write_text(json.dumps({"rows": golden}, default=str), "utf-8")
        out = self.tmp / "seed"
        report = train_final("mlp", shuffled(data), MLP_PARAMS, CPU, None, 0, out, "seed", 0.0)
        version = self.registry.adopt(
            out / "model",
            "mlp",
            {"pr_auc": report["pr_auc"]},
            [],
            report["run_id"],
            data.manifest["content_sha256"],
        )
        self.registry.set_aliases({"champion": version})
        return version

    def retrain(self, trigger: str = "drift") -> tuple[int, str]:
        run_id = self.recorder.start_run(trigger, "running", {})
        return run_id, self.pipeline.retrain(run_id)

    def good_shadow(self, version: str, rows: int = 800) -> None:
        rng = np.random.default_rng(0)
        served = rng.beta(1, 3, rows)
        self.recorder.shadow[version] = [
            ShadowRow(float(np.clip(score + rng.normal(0, 0.01), 0, 1)), None, 3.0, float(score))
            for score in served
        ]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


class FakeRunner:
    """DVC and git, as far as retention needs: everything succeeds, nothing is published."""

    def __call__(self, args: Any, cwd: Path) -> str:
        # A pointer that differs from git, so gc always waits here.
        return " M snapshots/x.dvc" if list(args[:2]) == ["git", "status"] else ""


# --- Registry ---


def test_a_folder_without_versions_json_is_never_written(tmp_path: Path) -> None:
    (tmp_path / "aliases.json").write_text('{"champion": {"version": 1, "model": "mlp"}}')
    registry = LocalRegistry(tmp_path)
    assert registry.aliases() == {"champion": "1"}
    with pytest.raises(RegistryError, match="isn't a registry"):
        registry.set_aliases({"champion": "2"})
    assert json.loads((tmp_path / "aliases.json").read_text())["champion"]["version"] == 1


def test_startup_wait_lets_the_scorer_load_first() -> None:
    assert startup_wait(300, uptime=60) == 240
    assert startup_wait(300, uptime=900) == 0
    assert startup_wait(300, uptime=None) == 300
    assert startup_wait(0, uptime=None) == 0


def test_the_pipeline_waits_after_a_restart(world: World) -> None:
    slept: list[float] = []
    world.settings.retrain_startup_delay_seconds = 300
    world.pipeline.uptime = lambda: 100.0
    world.pipeline.sleep = slept.append
    # Stop right after the wait: this test is only about the wait.
    world.pipeline.sanity = lambda data, models: {"passed": False, "failures": ["stopped"]}
    run_id = world.recorder.start_run("schedule", "running", {})
    assert world.pipeline.retrain(run_id) == "rejected"
    assert slept == [200.0]
    assert world.recorder.run(run_id)["details"]["startup_wait_seconds"] == 200.0


# --- The snapshot ---


def test_the_archive_snapshot_is_repeatable_and_holds_out_the_golden_set(world: World) -> None:
    golden = {1, 2, 3}
    first = build_from_archive(world.archive, AS_OF, 100_000, golden)
    second = build_from_archive(world.archive, AS_OF, 100_000, golden)
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    assert first.manifest["held_out"]["golden_set"] == 3
    assert not {row["rev_id"] for row in first.rows} & golden
    sampled = build_from_archive(world.archive, AS_OF, 1_000, set())
    assert sampled.manifest["rows"] < first.manifest["rows"]


def test_a_poisoned_snapshot_is_rejected_and_nothing_changes(tmp_path: Path) -> None:
    world = World(tmp_path, poison_day=True)
    world.registry.set_aliases({})
    before = world.registry.aliases()
    run_id, status = world.retrain()
    assert status == "rejected"
    run = world.recorder.run(run_id)
    assert run["details"]["rejected_at"] == "validation"
    assert any(problem.startswith("positive_rate") for problem in run["details"]["problems"])
    assert world.registry.aliases() == before
    assert "Retraining stopped" in [alert.title for alert in world.alerter.sent]


def test_the_real_leak_check_stops_a_retrain_before_any_training(tmp_path: Path) -> None:
    from trainer.pipeline import sanity_checks

    world = World(tmp_path, leak=True)
    world.pipeline.sanity = sanity_checks
    trained: list[str] = []
    world.pipeline.train = lambda name, *rest: trained.append(name) or {}
    run_id, status = world.retrain()
    assert status == "rejected" and trained == []
    assert world.recorder.run(run_id)["details"]["rejected_at"] == "sanity"
    assert "metadata alone" in world.recorder.run(run_id)["details"]["problems"][0]


# --- Retrain, shadow, promote, roll back ---


def test_retrain_beats_a_weak_champion_then_shadow_promotes_then_rollback_restores(
    world: World,
) -> None:
    weak = world.seed_weak_champion()
    run_id, status = world.retrain()
    run = world.recorder.run(run_id)
    details = run["details"]
    assert status == "shadow", details.get("reason")
    candidate = details["candidate_version"]
    # DEC-066's report, on the same rows, with a clear win.
    assert details["before"]["version"] == weak and details["after"]["version"] == candidate
    assert details["difference"]["low"] > 0
    assert details["promoted"] is None and details["gate"]["passed"]
    # In shadow: scored live, serving nothing.
    assert world.registry.aliases() == {"champion": weak, "candidate": candidate}
    assert [ref.version for ref in LocalRuns(world.registry.root).resolve()] == [weak]
    assert read_reference(world.archive, candidate) is not None
    assert list(world.paths.images.glob("report_before_after_run*.png"))

    world.good_shadow(candidate)
    assert world.lifecycle.shadow_check(run_id) == "promoted"
    assert world.registry.aliases() == {"champion": candidate}
    promoted = world.recorder.run(run_id)
    assert promoted["details"]["promoted"] is True
    assert promoted["details"]["aliases_before"] == {"champion": weak}
    assert world.recorder.listed[candidate]["alias"] == "champion"
    assert ("model.promoted", candidate) in [(a, t) for a, t, _ in world.recorder.audits]

    assert world.lifecycle.rollback(None, "post-promotion check: drop of 0.2", "monitor") == (
        "rolled_back"
    )
    assert world.registry.aliases() == {"champion": weak}
    assert world.recorder.run(run_id)["status"] == "rolled_back"
    assert world.recorder.listed[weak]["alias"] == "champion"
    with pytest.raises(LifecycleError, match="no promotion"):
        world.lifecycle.rollback(None, "again", "user")


def test_a_failed_shadow_removes_the_candidate_and_keeps_the_champion(world: World) -> None:
    weak = world.seed_weak_champion()
    run_id, _ = world.retrain()
    candidate = world.recorder.run(run_id)["details"]["candidate_version"]
    world.recorder.shadow[candidate] = [ShadowRow(None, "timeout", None, 0.2)] * 600
    assert world.lifecycle.shadow_check(run_id) == "not_promoted"
    assert world.registry.aliases() == {"champion": weak}
    assert "shadow_errors" in world.recorder.run(run_id)["details"]["reason"]


def test_retraining_against_an_equal_champion_keeps_it_and_alerts_on_a_drift_run(
    world: World,
) -> None:
    world.seed_weak_champion()
    first, _ = world.retrain()
    candidate = world.recorder.run(first)["details"]["candidate_version"]
    world.good_shadow(candidate)
    world.lifecycle.shadow_check(first)
    # The same recipe on the same data can't clear the gate against itself.
    second, status = world.retrain("drift")
    assert status == "not_promoted"
    assert world.registry.aliases()["champion"] == candidate
    assert world.recorder.run(second)["details"]["promoted"] is False
    assert "Retraining found nothing better" in [alert.title for alert in world.alerter.sent]


def test_a_manual_promotion_needs_a_reason_and_is_audited(world: World) -> None:
    weak = world.seed_weak_champion()
    run_id, _ = world.retrain()
    candidate = world.recorder.run(run_id)["details"]["candidate_version"]
    world.recorder.requests[7] = {
        "id": 7,
        "kind": "promote",
        "reason": "demo",
        "requested_by": "user",
        "details": {"model_version": candidate},
    }
    assert world.lifecycle.promote(7) == "promoted"
    assert world.registry.aliases() == {"champion": candidate}
    action, target, details = world.recorder.audits[-1]
    assert (action, target, details["force"], details["reason"]) == (
        "model.promoted",
        candidate,
        True,
        "demo",
    )
    # A rollback undoes an override like any other promotion.
    world.lifecycle.rollback(8, "changed my mind", "user")
    assert world.registry.aliases() == {"champion": weak}


def test_an_override_still_checks_the_artifact(world: World) -> None:
    world.seed_weak_champion()
    run_id, _ = world.retrain()
    candidate = world.recorder.run(run_id)["details"]["candidate_version"]
    folder = world.registry.root / f"v{candidate}" / "model"
    (folder / "config.json").write_text("{}")
    shutil.rmtree(world.paths.cache, ignore_errors=True)
    world.recorder.requests[9] = {
        "id": 9,
        "kind": "promote",
        "reason": "x",
        "requested_by": "user",
        "details": {"model_version": candidate},
    }
    with pytest.raises(Exception, match="checksum|digest|match"):
        world.lifecycle.promote(9)


# --- Storage ---


def test_retention_keeps_aliases_and_the_newest_three() -> None:
    from trainer.registry import VersionInfo

    versions = [VersionInfo(str(n), "mlp", "d") for n in range(1, 7)]
    versions.append(VersionInfo("7", "bilstm", "d"))
    assert versions_to_delete(versions, aliased={"1"}) == ["2", "3"]


def test_snapshots_in_use_by_any_of_their_hashes_are_kept(world: World) -> None:
    folder = write_snapshot(
        build_from_archive(world.archive, AS_OF, 100_000, set()), world.paths.snapshots
    )
    manifest = read_manifest(folder)
    assert snapshots_to_delete(world.paths.snapshots, {manifest["original_content_sha256"]}) == []
    assert snapshots_to_delete(world.paths.snapshots, {"other"}) == [folder]


def test_retention_deletes_unused_versions_snapshots_and_references(world: World) -> None:
    weak = world.seed_weak_champion()
    run_id, _ = world.retrain()
    candidate = world.recorder.run(run_id)["details"]["candidate_version"]
    world.good_shadow(candidate)
    world.lifecycle.shadow_check(run_id)
    result = retain(
        world.registry,
        DvcRepo(world.tmp, FakeRunner()),
        world.archive,
        gc_enabled=False,
        alert_bytes=1,
        alerter=world.alerter,
    )
    # The weak seed is the newest-but-one MLP, so it's kept; its reference never existed.
    assert weak not in result["versions_deleted"]
    assert result["gc"].startswith("skipped")
    assert "Storage nearly full" in [alert.title for alert in world.alerter.sent]


# --- The real DVC guard ---


def git(*args: str, cwd: Path) -> str:
    # A throwaway repo in the test's own folder: git from PATH, with fixed arguments.
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_gc_frees_the_remote_only_once_pointers_are_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A throwaway repo, a throwaway DVC remote, and a throwaway "GitHub", never the real ones."""
    if shutil.which("git") is None:
        pytest.skip("git isn't installed")
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    if shutil.which("dvc") is None:
        pytest.skip("dvc isn't installed")
    origin, repo, remote = tmp_path / "origin.git", tmp_path / "repo", tmp_path / "dvc-remote"
    git("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    git("clone", str(origin), str(repo), cwd=tmp_path)
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "test")):
        git("config", key, value, cwd=repo)
    dvc = DvcRepo(repo)
    dvc._run("dvc", "init", "-q")
    dvc._run("dvc", "remote", "add", "-d", "local", str(remote))
    snapshot = repo / "snapshots" / "snapshot-x"
    snapshot.mkdir(parents=True)
    (snapshot / "rows.txt").write_text("version one")
    dvc.add(snapshot)
    pointer = snapshot.with_name("snapshot-x.dvc")
    dvc.push(pointer)
    git("add", "-A", cwd=repo)
    git("commit", "-qm", "v1", cwd=repo)
    git("push", "-q", "origin", "main", cwd=repo)
    # A purge rewrites the data: a second version on the remote.
    (snapshot / "rows.txt").write_text("version two, a hidden row removed")
    dvc.add(snapshot)
    dvc.push(pointer)

    def remote_files() -> int:
        return sum(1 for path in remote.rglob("*") if path.is_file())

    held = remote_files()
    assert "differ" in dvc.gc_cloud(enabled=True)
    git("commit", "-qam", "v2", cwd=repo)
    # Committed but not on "GitHub": a published commit still names version one.
    assert "differ" in dvc.gc_cloud(enabled=True)
    assert remote_files() == held
    git("push", "-q", "origin", "main", cwd=repo)
    assert dvc.gc_cloud(enabled=True) == "done"
    assert remote_files() < held
