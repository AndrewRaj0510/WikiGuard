"""The retraining pipeline (PLAN.md, section 10), run by trainer.lifecycle for one retrain request.

    snapshot -> validate -> baselines + sanity -> train the CPU models -> evaluate every served
    artifact on the newest test window -> register -> gate -> shadow (or keep the champion)

Promotion itself waits for the shadow check (trainer.lifecycle). Rules that aren't obvious:

- Every model, old and new, is scored on the same newest test window, because "before" and "after"
  only compare on the same rows (section 4).
- BERT and RoBERTa aren't retrained here: their recipe needs a GPU, which nothing can start
  automatically (DEC-022). They stay in the chain, are measured on the new window like the rest,
  and an alert asks for a notebook run.
- A failure anywhere before the gate leaves every alias untouched. The champion only changes after
  a passed gate and a passed shadow period.
"""

import dataclasses
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from wikiguard_core.alerts import Alerter
from wikiguard_core.artifacts import ArtifactError, verify_artifact
from wikiguard_core.chain import CANDIDATE_ALIAS, SMOKE_EDIT, ModelCache
from wikiguard_core.features import FEATURE_VERSION, build_features
from wikiguard_core.inference import ServedModel, read_config
from wikiguard_core.listing import metrics_of
from wikiguard_core.promotion import Evaluated, Proposal, propose
from wikiguard_core.promotion import report as gate_report
from wikiguard_core.snapshot import SnapshotError, write_snapshot
from wikiguard_core.validation import validate_snapshot

from trainer import charts
from trainer.archive_snapshot import build_from_archive
from trainer.baselines import measure_baselines
from trainer.benchmark import latency_ms
from trainer.data import Dataset, labels_of, load_dataset, masks_of
from trainer.finish import golden_report
from trainer.records import Listing, Recorder
from trainer.reference import save_for_model
from trainer.registry import Registry, chain_mapping, chain_of, snapshot_of_aliases
from trainer.sanity import run_checks
from trainer.train import search, train_final

GPU_MODELS = ("bert", "roberta")
LATENCY_ROWS = 100
SMOKE_ROWS = 20
EVALUATION_THREADS = 4
# DEC-072: DistilBERT keeps its CPU recipe (DEC-052), with the parameters its Phase 2 search chose
# on a GPU. Searching again on the VM's CPU would add hours to every retrain.
FIXED_PARAMS: dict[str, dict[str, Any]] = {
    "distilbert": {
        "dropout": 0.2,
        "epochs": 1,
        "head_learning_rate": 0.001,
        "learning_rate": 0.0001,
        "trainable_blocks": 2,
    }
}
FIXED_CAP = {"distilbert": 30_000}
TRAIN_SEED = 0

TrainFn = Callable[[str, Dataset, Path, float | None], dict[str, Any]]
SanityFn = Callable[[Dataset, tuple[str, ...]], dict[str, Any]]


class RetrainError(RuntimeError):
    pass


def read_uptime() -> float | None:
    """Seconds since the machine started. Containers share the host's /proc/uptime."""
    try:
        return float(Path("/proc/uptime").read_text("utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def startup_wait(delay_seconds: int, uptime: float | None) -> float:
    """RETRAIN_STARTUP_DELAY_SECONDS: after a restart, let the scorer load and catch up first.

    With no way to tell how long the machine has been up, wait the whole delay.
    """
    if delay_seconds <= 0:
        return 0.0
    if uptime is None:
        return float(delay_seconds)
    return max(0.0, delay_seconds - uptime)


def train_candidate(
    name: str, data: Dataset, out: Path, metadata_only: float | None
) -> dict[str, Any]:
    """Search then train, like the first training, on the CPU (DEC-044)."""
    device = torch.device("cpu")
    if name in FIXED_PARAMS:
        params, cap = FIXED_PARAMS[name], FIXED_CAP.get(name)
    else:
        params, cap = search(name, data, device, TRAIN_SEED, out, {}), None
    return train_final(name, data, params, device, cap, TRAIN_SEED, out, "retrain", metadata_only)


def sanity_checks(data: Dataset, models: tuple[str, ...]) -> dict[str, Any]:
    return run_checks(data, torch.device("cpu"), models, TRAIN_SEED)


@dataclass(frozen=True)
class Paths:
    workdir: Path
    snapshots: Path
    archive: Path
    golden: Path
    images: Path
    cache: Path


@dataclass
class Pipeline:
    settings: Any
    registry: Registry
    recorder: Recorder
    alerter: Alerter
    paths: Paths
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    uptime: Callable[[], float | None] = read_uptime
    train: TrainFn = train_candidate
    sanity: SanityFn = sanity_checks
    after_run: Callable[[], dict[str, Any]] | None = None

    # --- The run ---

    def retrain(
        self, run_id: int, snapshot: Path | None = None, as_of: datetime | None = None
    ) -> str:
        now = self.clock()
        deadline = now + timedelta(minutes=self.settings.retrain_timeout_minutes)
        wait = startup_wait(self.settings.retrain_startup_delay_seconds, self.uptime())
        if wait:
            self.recorder.update_run(run_id, details={"startup_wait_seconds": wait})
            self.sleep(wait)
        torch.set_num_threads(self.settings.retrain_threads)
        if hasattr(os, "nice"):
            # Below the scorer, which must keep up with the stream on the same VM (section 3).
            os.nice(10)
        workdir = self.paths.workdir / f"run-{run_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            folder = snapshot or self._build_snapshot(as_of or now)
        except SnapshotError as error:
            return self._reject(run_id, "snapshot", [str(error)])
        validation = validate_snapshot(folder)
        (workdir / "validation.json").write_text(
            json.dumps(validation.as_dict(), indent=2, default=str), "utf-8"
        )
        details: dict[str, Any] = {
            "snapshot": validation.snapshot,
            "snapshot_content_sha256": validation.content_sha256,
            "validation_warnings": [check.detail for check in validation.warnings],
        }
        self.recorder.update_run(run_id, details=details)
        if validation.failures:
            # Section 4: a failed check stops the run, and this is where poisoned labels end.
            problems = [f"{check.name}: {check.detail}" for check in validation.failures]
            return self._reject(run_id, "validation", problems)

        data = load_dataset(folder)
        rows = [*data.train, *data.validation, *data.test]
        baselines = measure_baselines(rows)
        models = tuple(self.settings.model_names)
        sanity = self.sanity(data, models)
        (workdir / "sanity.json").write_text(json.dumps(sanity, indent=2, default=str), "utf-8")
        if not sanity["passed"]:
            return self._reject(run_id, "sanity", list(sanity["failures"]))
        details["baselines"] = {
            "prevalence_pr_auc": baselines["test"]["prevalence_pr_auc"],
            "rule_pr_auc": baselines["test"]["rule"]["pr_auc"],
            "rule_pr_auc_interval": baselines["test"]["rule"]["pr_auc_interval"],
        }
        details["sanity"] = {
            "shuffled_validation_pr_auc": sanity["shuffled_validation_pr_auc"],
            "metadata_only_validation_pr_auc": sanity["metadata_only_validation_pr_auc"],
        }

        reports, failed, skipped = {}, {}, []
        for name in models:
            if self.clock() > deadline:
                skipped.append(name)
                continue
            try:
                reports[name] = self.train(
                    name, data, workdir / name, sanity["metadata_only_validation_pr_auc"]
                )
            except Exception as error:  # noqa: BLE001 - one model failing doesn't stop the rest
                failed[name] = type(error).__name__
                self.alerter.send("A model failed to retrain", f"{name}: {type(error).__name__}")
        details |= {"trained": sorted(reports), "failed": failed, "skipped_out_of_time": skipped}

        labels = labels_of(data.test)
        masks = masks_of(data.test)
        current = self._evaluate_current(data)
        candidates = [
            self._evaluate_new(name, report, data, workdir) for name, report in reports.items()
        ]
        candidates = [self._register(model, reports[model.model], workdir) for model in candidates]
        listings = {
            model.version: self._listing(model, reports[model.model], data, workdir)
            for model in candidates
            if model.version
        }
        rule_high = baselines["test"]["rule"]["pr_auc_interval"][1]
        proposal = propose(current, candidates, labels, masks, rule_high)
        report = gate_report(proposal, labels, masks)
        details |= report
        details["listings"] = {key: dataclasses.asdict(value) for key, value in listings.items()}
        self._chart(run_id, report)
        status = self._act(run_id, proposal, details, data, workdir)
        self._remind_gpu_models(current, validation.snapshot)
        if status == "not_promoted" and self.recorder.run(run_id)["trigger"] in (
            "drift",
            "performance",
        ):
            self.alerter.send(
                "Retraining found nothing better",
                f"run {run_id}: {proposal.reason}. The problem may be the data, not the model.",
            )
        if self.after_run is not None:
            details["storage"] = self.after_run()
        self.recorder.update_run(run_id, status, details, finished=status != "shadow")
        return status

    # --- Steps ---

    def _build_snapshot(self, as_of: datetime) -> Path:
        golden = self._golden()
        exclude = {int(row["rev_id"]) for row in golden["rows"]} if golden else set()
        built = build_from_archive(
            self.paths.archive, as_of, self.settings.retrain_snapshot_rows, exclude
        )
        return write_snapshot(built, self.paths.snapshots)

    def _reject(self, run_id: int, stage: str, problems: list[str]) -> str:
        self.alerter.send(
            "Retraining stopped", f"run {run_id}: {stage} failed: {'; '.join(problems)[:500]}"
        )
        self.recorder.update_run(
            run_id,
            "rejected",
            {
                "rejected_at": stage,
                "problems": problems,
                "promoted": False,
                "reason": f"{stage} failed, so nothing was trained and the champion stays",
            },
            finished=True,
        )
        return "rejected"

    def _golden(self) -> dict[str, Any] | None:
        if not self.paths.golden.is_file():
            return None
        golden: dict[str, Any] = json.loads(self.paths.golden.read_text("utf-8"))
        return golden

    def _measure(
        self,
        model: str,
        version: str | None,
        folder: Path,
        digest: str,
        data: Dataset,
        warnings: tuple[str, ...],
        is_new: bool,
        logits: np.ndarray | None = None,
    ) -> Evaluated:
        """Everything the gate needs about one served artifact."""
        try:
            verify_artifact(folder, digest)
        except (ArtifactError, OSError, KeyError, ValueError):
            self.alerter.send("Artifact checksum failed", f"{model} v{version}", "critical")
            return Evaluated(
                model, version, np.zeros(len(data.test)), 0.0, is_new, checksum_ok=False
            )
        config = read_config(folder)
        try:
            served = ServedModel(folder, threads=EVALUATION_THREADS, expected_digest=digest)
            if logits is None:
                logits = served.logits(data.test)
            single = ServedModel(folder, threads=1, expected_digest=digest)
        except Exception as error:  # noqa: BLE001 - a model that can't score can't serve
            self.alerter.send(
                "A model wouldn't load", f"{model} v{version}: {type(error).__name__}"
            )
            return Evaluated(
                model, version, np.zeros(len(data.test)), 0.0, is_new, smoke_passed=False
            )
        sample = data.test[:: max(1, len(data.test) // LATENCY_ROWS)][:LATENCY_ROWS]
        p95 = float(np.percentile(latency_ms(single, sample), 95))
        golden_data = self._golden()
        golden_pr_auc = None
        if golden_data:
            golden = golden_report(served, golden_data, float(config["threshold_logit"]))
            golden_pr_auc = float(golden["pr_auc"])
            if is_new:
                (folder.parent / "golden.json").write_text(json.dumps(golden, indent=2), "utf-8")
        smoke_rows = [build_features(SMOKE_EDIT).as_dict(), *data.test[:SMOKE_ROWS]]
        try:
            scores = single.probabilities(smoke_rows, batch_size=1)
            smoke = bool(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all())
        except Exception:  # noqa: BLE001 - any failure here is a failed smoke test
            smoke = False
        return Evaluated(
            model,
            version,
            np.asarray(logits, dtype=np.float64),
            float(config["threshold_logit"]),
            is_new,
            p95_ms=p95,
            served_bytes=sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()),
            golden_pr_auc=golden_pr_auc,
            warnings=warnings,
            smoke_passed=smoke,
            feature_version_ok=config.get("feature_version") == FEATURE_VERSION,
        )

    def _evaluate_current(self, data: Dataset) -> list[Evaluated]:
        source = self.registry.source()
        cache = ModelCache(self.paths.cache)
        listed = self.recorder.model_metrics()
        found = []
        for alias, version in chain_of(self.registry.aliases()):
            ref = source.resolve_alias(alias)
            if ref is None:
                continue
            info = self.registry.info(version)
            # BERT's flag (F-082) lives in model_versions for versions registered before Phase 5.
            warnings = info.warnings or tuple(listed.get(version, {}).get("training_warnings", []))
            folder = cache.ensure(ref, source)
            found.append(
                self._measure(
                    ref.model_name,
                    version,
                    folder,
                    ref.artifact_sha256,
                    data,
                    warnings,
                    is_new=False,
                )
            )
        return found

    def _evaluate_new(
        self, name: str, report: dict[str, Any], data: Dataset, workdir: Path
    ) -> Evaluated:
        stored = np.load(workdir / name / "scores.npz", allow_pickle=False)
        expected = np.array([row["rev_id"] for row in data.test], dtype=np.int64)
        if not np.array_equal(stored["test_rev_id"], expected):
            raise RetrainError(f"{name}: stored test scores aren't for this snapshot's test rows")
        return self._measure(
            name,
            None,
            workdir / name / "model",
            report["artifact_sha256"],
            data,
            tuple(report.get("training_warnings", [])),
            is_new=True,
            logits=stored["test_logit"],
        )

    def _register(self, model: Evaluated, report: dict[str, Any], workdir: Path) -> Evaluated:
        """Only versions that could serve get registered. Registering moves no alias."""
        if not (model.checksum_ok and model.feature_version_ok and model.smoke_passed):
            return model
        version = self.registry.register(
            workdir / model.model / "model",
            model.model,
            report["run_id"],
            report["artifact_sha256"],
            {
                "pr_auc": report["pr_auc"],
                "recall_at_90_precision": report["recall_at_90_precision"],
            },
            list(report.get("training_warnings", [])),
            report["snapshot_content_sha256"],
        )
        self.recorder.audit(
            "model.registered", version, {"model": model.model, "run_id": report["run_id"]}
        )
        return dataclasses.replace(model, version=version)

    def _listing(
        self, model: Evaluated, report: dict[str, Any], data: Dataset, workdir: Path
    ) -> Listing:
        folder = workdir / model.model
        config = read_config(folder / "model")
        return Listing(
            model.model,
            str(model.version),
            FEATURE_VERSION,
            data.manifest["content_sha256"],
            report["artifact_sha256"],
            metrics_of(report, folder, config),
        )

    def listings_for(self, versions: list[str], known: dict[str, Any]) -> list[Listing]:
        """Rows for the API's model list. Versions already listed keep their metrics."""
        rows = []
        for version in dict.fromkeys(versions):
            if version in known:
                rows.append(Listing(**known[version]))
                continue
            info = self.registry.info(version)
            rows.append(Listing(info.model, version, FEATURE_VERSION, "", info.artifact_sha256))
        return rows

    def _act(
        self,
        run_id: int,
        proposal: Proposal,
        details: dict[str, Any],
        data: Dataset,
        workdir: Path,
    ) -> str:
        before = snapshot_of_aliases(self.registry)
        order = [str(model.version) for model in proposal.order]
        details["aliases_before"] = before
        details["proposed_aliases"] = {
            alias: str(model.version) for alias, model in proposal.aliases()
        }
        known = details["listings"]
        if proposal.champion_change and proposal.challenger is not None:
            candidate = str(proposal.challenger.version)
            # Saved now, and kept if it's promoted (section 4, "with each promoted model").
            save_for_model(
                self.paths.archive,
                candidate,
                proposal.challenger.model,
                workdir / proposal.challenger.model / "model",
                [*data.train, *data.validation],
                data.manifest["name"],
            )
            self.registry.set_aliases({CANDIDATE_ALIAS: candidate})
            aliases = {
                **{k: v for k, v in before.items() if k != CANDIDATE_ALIAS},
                CANDIDATE_ALIAS: candidate,
            }
            self.recorder.sync_models(aliases, self.listings_for(list(aliases.values()), known))
            details |= {
                "candidate_version": candidate,
                "shadow_started_at": self.clock().isoformat(),
            }
            self.recorder.audit("model.shadow_started", candidate, {"run_id": run_id})
            return "shadow"
        current_order = [version for _, version in chain_of(before)]
        if order and order != current_order:
            # The champion stays, but a retrained fallback replaces its older version.
            self.registry.set_aliases(chain_mapping(order))
            after = snapshot_of_aliases(self.registry)
            self.recorder.sync_models(after, self.listings_for(list(after.values()), known))
            details["aliases_after"] = after
            self.recorder.audit(
                "model.fallbacks_refreshed", None, {"run_id": run_id, "order": order}
            )
        return "not_promoted"

    def _chart(self, run_id: int, report: dict[str, Any]) -> None:
        if report["before"]["pr_auc"] is None or report["after"]["pr_auc"] is None:
            return
        day = self.clock().date()
        charts.before_after(
            report, self.paths.images, day, f"report_before_after_run{run_id}_{day}"
        )

    def _remind_gpu_models(self, current: list[Evaluated], snapshot: str) -> None:
        waiting = sorted({model.model for model in current if model.model in GPU_MODELS})
        if waiting:
            self.alerter.send(
                "GPU models need a notebook run",
                f"{', '.join(waiting)} weren't retrained (DEC-022). Run "
                f"notebooks/train_transformers_gpu.ipynb on {snapshot}.",
                "info",
            )
