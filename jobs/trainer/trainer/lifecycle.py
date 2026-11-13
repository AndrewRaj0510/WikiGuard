"""Every change to what serves goes through here, run by the retrain-request poller (DEC-071).

    python -m trainer.lifecycle retrain --request 12 --run 4 [--snapshot PATH] [--as-of ISO]
    python -m trainer.lifecycle shadow-check --request 13 --run 4
    python -m trainer.lifecycle promote --request 14
    python -m trainer.lifecycle rollback --request 15

- retrain: the pipeline (trainer.pipeline), ending in a shadow period or with the champion kept.
- shadow-check: after SHADOW_HOURS, the live health gate. Passing promotes the whole proposed
  chain at once, and saves the aliases it replaced so a rollback can restore them exactly.
- promote: the owner's manual override, with the request's reason in the audit log (section 4).
  The artifact checks still apply: an override skips the gate, never the checksum.
- rollback: puts back the aliases from before the newest promotion. The monitor asks for this when
  the post-promotion check fails, and the owner can from the Health page.

Each step records its outcome on the run and in audit_log, and alerts on anything a person should
know about. The API holds no registry token that can write, so it can only ask (section 8).
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from wikiguard_core.alerts import Alerter
from wikiguard_core.archive import open_archive
from wikiguard_core.artifacts import verify_artifact
from wikiguard_core.chain import CANDIDATE_ALIAS, ModelCache, smoke_check
from wikiguard_core.config import (
    AlertSettings,
    ArchiveSettings,
    CommonSettings,
    DatabaseSettings,
    RetrainSettings,
    ServingSettings,
    StorageSettings,
    load_settings,
)
from wikiguard_core.features import FEATURE_VERSION
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.inference import OneAtATime, ServedModel, read_config
from wikiguard_core.promotion import LAST_MODEL, ShadowStats, shadow_gate
from wikiguard_core.tracking import configure

from trainer.pipeline import Paths, Pipeline
from trainer.records import DbRecorder, Listing, Recorder
from trainer.registry import (
    Registry,
    chain_mapping,
    chain_of,
    registry_from_settings,
    snapshot_of_aliases,
)

logger = logging.getLogger("trainer.lifecycle")


class Settings(
    CommonSettings,
    DatabaseSettings,
    RetrainSettings,
    ServingSettings,
    ArchiveSettings,
    AlertSettings,
    StorageSettings,
):
    pass


class LifecycleError(RuntimeError):
    pass


def parse_time(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


@dataclass
class Lifecycle:
    settings: Any
    registry: Registry
    recorder: Recorder
    alerter: Alerter
    pipeline: Pipeline

    def now(self) -> datetime:
        return self.pipeline.clock()

    def _sync(self, extra: dict[str, Any] | None = None) -> dict[str, str]:
        aliases = snapshot_of_aliases(self.registry)
        known = extra or {}
        self.recorder.sync_models(
            aliases, self.pipeline.listings_for(list(aliases.values()), known)
        )
        return aliases

    # --- Shadow check ---

    def shadow_check(self, run_id: int) -> str:
        run = self.recorder.run(run_id)
        details = run["details"]
        if run["status"] != "shadow":
            raise LifecycleError(f"run {run_id} is {run['status']}, not in shadow")
        candidate = str(details["candidate_version"])
        started = parse_time(details["shadow_started_at"])
        aliases = snapshot_of_aliases(self.registry)
        champion_then = details["aliases_before"].get("champion")
        if aliases.get("champion") != champion_then or aliases.get(CANDIDATE_ALIAS) != candidate:
            # Someone promoted or rolled back meanwhile, so this run's plan is out of date.
            return self._end_shadow(
                run_id,
                candidate,
                False,
                "the chain changed during the shadow period, so this plan no longer applies",
                None,
            )
        rows = self.recorder.shadow_rows(candidate, started)
        scored = [row for row in rows if row.score is not None]
        latencies = [row.latency_ms for row in scored if row.latency_ms is not None]
        stats = ShadowStats(
            attempts=len(rows),
            errors=sum(row.error is not None for row in rows),
            p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
            candidate_scores=np.array([row.score for row in scored], dtype=np.float64),
            served_scores=np.array(
                [row.served_score for row in scored if row.served_score is not None],
                dtype=np.float64,
            ),
            started_at=started,
            now=self.now(),
        )
        gate = shadow_gate(stats, self.settings.shadow_hours)
        if not gate.passed:
            failed = ", ".join(check.name for check in gate.failures)
            return self._end_shadow(
                run_id, candidate, False, f"failed the shadow check ({failed})", gate.as_dict()
            )
        order = [
            details["proposed_aliases"][alias] for alias, _ in chain_of(details["proposed_aliases"])
        ]
        before = snapshot_of_aliases(self.registry)
        self.registry.set_aliases(chain_mapping(order) | {CANDIDATE_ALIAS: None})
        after = self._sync(details.get("listings"))
        reason = f"version {candidate} passed the promotion gate and {stats.attempts} live edits"
        self.recorder.update_run(
            run_id,
            "promoted",
            {
                "promoted": True,
                "promoted_at": self.now().isoformat(),
                "reason": reason,
                "shadow": gate.as_dict(),
                "aliases_before": {k: v for k, v in before.items() if k != CANDIDATE_ALIAS},
                "aliases_after": after,
            },
            finished=True,
        )
        self.recorder.audit("model.promoted", candidate, {"run_id": run_id, "aliases": after})
        self.alerter.send("New champion", f"version {candidate} is champion: {reason}", "info")
        return "promoted"

    def _end_shadow(
        self,
        run_id: int,
        candidate: str,
        promoted: bool,
        reason: str,
        gate: dict[str, Any] | None,
    ) -> str:
        if snapshot_of_aliases(self.registry).get(CANDIDATE_ALIAS) == candidate:
            self.registry.set_aliases({CANDIDATE_ALIAS: None})
        self._sync()
        self.recorder.update_run(
            run_id,
            "not_promoted",
            {"promoted": promoted, "reason": reason, "shadow": gate},
            finished=True,
        )
        self.recorder.audit("model.shadow_ended", candidate, {"run_id": run_id, "reason": reason})
        self.alerter.send("Candidate not promoted", f"version {candidate}: {reason}")
        return "not_promoted"

    # --- Manual promotion and rollback ---

    def check_artifact(self, version: str) -> str:
        """The checks no override skips: checksum, feature version, and a scored smoke edit."""
        info = self.registry.info(version)
        cache = ModelCache(self.pipeline.paths.cache)
        folder = cache.ensure(self.registry.ref_for(version), self.registry.source())
        verify_artifact(folder, info.artifact_sha256)
        if read_config(folder).get("feature_version") != FEATURE_VERSION:
            raise LifecycleError(f"version {version} was built for another feature_version")
        smoke_check("override", info.model, version, OneAtATime(ServedModel(folder)))
        return info.model

    def promote(self, request_id: int) -> str:
        request = self.recorder.request(request_id)
        version = str((request["details"] or {}).get("model_version", ""))
        reason = str(request["reason"])
        if not version:
            raise LifecycleError("a promotion needs a model_version")
        model = self.check_artifact(version)
        current = chain_of(snapshot_of_aliases(self.registry))
        if model == LAST_MODEL and len(current) > 1:
            raise LifecycleError("the MLP is always last in the chain, so it can't be champion")
        # The chosen version first. It replaces any other version of the same model.
        rest = [v for _, v in current if v != version and self.registry.info(v).model != model]
        before = snapshot_of_aliases(self.registry)
        self.registry.set_aliases(chain_mapping([version, *rest]) | {CANDIDATE_ALIAS: None})
        after = self._sync()
        champion_before = before.get("champion")
        run_id = self.recorder.start_run(
            "manual_promote",
            "promoted",
            {
                "request_id": request_id,
                "promoted": True,
                "forced": True,
                "promoted_at": self.now().isoformat(),
                "reason": f"manual override: {reason}",
                "before": {
                    "model": self.registry.info(champion_before).model if champion_before else None,
                    "version": champion_before,
                },
                "after": {"model": model, "version": version},
                "aliases_before": {k: v for k, v in before.items() if k != CANDIDATE_ALIAS},
                "aliases_after": after,
            },
        )
        self.recorder.update_run(run_id, finished=True)
        # Section 4: an override needs force and a written reason, and goes in the audit log.
        self.recorder.audit(
            "model.promoted",
            version,
            {"force": True, "reason": reason, "request_id": request_id, "aliases": after},
        )
        self.alerter.send("Manual promotion", f"version {version} ({model}) by override: {reason}")
        return "promoted"

    def rollback(self, request_id: int | None, reason: str, by: str) -> str:
        promoted = [run for run in self.recorder.promotions() if run["status"] == "promoted"]
        if not promoted:
            raise LifecycleError("no promotion to roll back")
        latest = promoted[0]
        restore = latest["details"]["aliases_before"]
        wanted = chain_mapping([version for _, version in chain_of(restore)])
        self.registry.set_aliases(wanted | {CANDIDATE_ALIAS: None})
        after = self._sync()
        self.recorder.update_run(
            latest["id"],
            "rolled_back",
            {
                "rolled_back_at": self.now().isoformat(),
                "rollback_reason": reason,
                "rolled_back_by": by,
                "rollback_request_id": request_id,
            },
        )
        self.recorder.audit(
            "model.rolled_back",
            str(latest["id"]),
            {"reason": reason, "by": by, "request_id": request_id, "aliases": after},
        )
        self.alerter.send(
            "Rolled back",
            f"run {latest['id']} undone, champion is v{after.get('champion')}: {reason}",
            "critical" if by == "monitor" else "warning",
        )
        return "rolled_back"


def build(settings: Any, connection: psycopg.Connection[Any]) -> Lifecycle:
    configure(settings.mlflow_tracking_uri if settings.model_source == "mlflow" else None)
    registry = registry_from_settings(settings)
    recorder = DbRecorder(connection)
    webhook = settings.discord_webhook_url
    alerter = Alerter("trainer", webhook.get_secret_value() if webhook else None)
    paths = Paths(
        workdir=Path("data/retrain"),
        snapshots=Path(settings.snapshot_repo_path) / "snapshots",
        archive=open_archive(settings),
        golden=Path(settings.snapshot_repo_path) / "snapshots/golden-set-v1/golden.json",
        images=Path("docs/images"),
        cache=Path(f"{settings.model_cache_dir}-trainer"),
    )
    pipeline = Pipeline(settings, registry, recorder, alerter, paths)
    from trainer.storage import retention_after_retrain

    pipeline.after_run = lambda: retention_after_retrain(settings, registry, recorder, alerter)
    return Lifecycle(settings, registry, recorder, alerter, pipeline)


def main(argv: list[str] | None = None) -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["retrain", "shadow-check", "promote", "rollback"])
    parser.add_argument("--request", type=int, required=True)
    parser.add_argument("--run", type=int)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--as-of", type=parse_time)
    args = parser.parse_args(argv)
    settings = load_settings(Settings)
    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        lifecycle = build(settings, connection)
        if args.kind in ("retrain", "shadow-check") and args.run is None:
            raise LifecycleError(f"{args.kind} needs --run")
        if args.kind == "retrain":
            outcome = lifecycle.pipeline.retrain(args.run, args.snapshot, args.as_of)
        elif args.kind == "shadow-check":
            outcome = lifecycle.shadow_check(args.run)
        elif args.kind == "promote":
            outcome = lifecycle.promote(args.request)
        else:
            request = lifecycle.recorder.request(args.request)
            outcome = lifecycle.rollback(
                args.request, str(request["reason"]), str(request["requested_by"])
            )
    logger.info("%s finished: %s", args.kind, outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Lifecycle", "LifecycleError", "Listing", "main"]
