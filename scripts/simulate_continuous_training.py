"""Replay Phase 5's loops on real data and real models, in a sandbox (PLAN.md, Phase 5).

    .venv-train\\Scripts\\python scripts\\simulate_continuous_training.py drift --admin-url URL
    .venv-train\\Scripts\\python scripts\\simulate_continuous_training.py rollback --admin-url URL
    .venv-train\\Scripts\\python scripts\\simulate_continuous_training.py poisoning --admin-url URL

URL is a throwaway Postgres (the test container, never Supabase): each scenario creates its own
database there. The shadow replay runs the scorer's own code, so .venv-train also needs the
scorer's two packages: pip install confluent-kafka==2.16.1 prometheus-client==0.26.0.

Everything else lives in data/sim/SCENARIO: a parquet archive built from the backfill, a local
registry seeded with copies of the Phase 2 BiLSTM and MLP, and its own MLflow file. Settings are
built here, not read from .env, so nothing can reach DagsHub, Discord, or the real registry, and
data/runs is only read.

The scenarios drive the real code: jobs/monitor's Monitor, jobs/retrain_poller's run_once, and
trainer.lifecycle, with the poller's launchers calling the lifecycle in this process instead of a
subprocess. "Now" is the backfill's as_of, because the archive ends there, and shadow periods are
replayed rather than waited out (SHADOW_HOURS=0).

- drift: a synthetic spam campaign starts 20 days before as_of in 10% of edits. The monitor
  should flag drift and ask for a retrain; the retrain should beat the sandbox champion through
  the gate; the replayed shadow should pass; the candidate should be promoted.
- rollback: a BiLSTM trained on shuffled labels is force-promoted 9 days before as_of and serves
  from then. The monitor's post-promotion check should ask for a rollback, and the lifecycle
  should restore the chain.
- poisoning: a burst of fake reverts on one day. The retrain's snapshot should be rejected and
  nothing should change.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for extra in ("scripts", "jobs/monitor", "jobs/retrain_poller", "jobs/trainer", "services/scorer"):
    sys.path.insert(0, str(ROOT / extra))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import psycopg  # noqa: E402
import torch  # noqa: E402
from build_snapshot import backfill_as_of, read_candidates  # noqa: E402
from drift import evidently_drift  # noqa: E402
from monitor import Monitor  # noqa: E402
from poller import request_shadow_checks, run_once  # noqa: E402
from trainer.data import load_dataset  # noqa: E402
from trainer.lifecycle import Lifecycle  # noqa: E402
from trainer.pipeline import Paths, Pipeline, sanity_checks  # noqa: E402
from trainer.records import DbRecorder, Listing  # noqa: E402
from trainer.reference import save_for_model  # noqa: E402
from trainer.registry import LocalRegistry, snapshot_of_aliases  # noqa: E402
from trainer.storage import retain  # noqa: E402
from trainer.train import train_final  # noqa: E402
from wikiguard_core.alerts import Alerter  # noqa: E402
from wikiguard_core.archive import EDITS, LABELS, SCORES, write_rows  # noqa: E402
from wikiguard_core.chain import ModelCache, ServingChain, ShadowSlot  # noqa: E402
from wikiguard_core.dvcops import DvcRepo  # noqa: E402
from wikiguard_core.features import FEATURE_VERSION, build_features  # noqa: E402
from wikiguard_core.inference import ServedModel, read_config  # noqa: E402
from wikiguard_core.listing import metrics_of  # noqa: E402
from wikiguard_core.schemas import EditEvent  # noqa: E402

BACKFILL = ROOT / "data/backfill"
RUNS = ROOT / "data/runs"
SNAPSHOT = ROOT / "snapshots/snapshot-20261104-28c825af4b19"
GOLDEN = ROOT / "snapshots/golden-set-v1/golden.json"
SIM = ROOT / "data/sim"
LOGIN = "sim_trainer"
LOGIN_PASSWORD = "sim-only-not-a-secret"  # noqa: S105 - a throwaway login on a throwaway database
CPU = torch.device("cpu")
# The campaign: synthetic text on a reserved domain, so nothing real is named or linked.
SPAM = "BUY CHEAP REPLICA WATCHES NOW at http://replica-deals.invalid best prices guaranteed"
SPAM_SHARE = 0.10
SHIFT_DAYS = 20
SCORED_DAYS = 16
# Phase 2's chosen parameters, so the replay fits a laptop run. The pipeline's own path searches.
PHASE2_PARAMS = {
    name: json.loads((RUNS / name / "report.json").read_text("utf-8"))["params"]
    for name in ("mlp", "bilstm")
}


def log(message: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {message}", flush=True)


def chosen(rev_id: int, share: float, salt: str) -> bool:
    digest = hashlib.sha256(f"{salt}{rev_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") < share * 2**32


# --- The sandbox ---


def make_database(admin_url: str, name: str) -> str:
    """A fresh, migrated database, and a URL that connects as the trainer role."""
    import os

    from alembic import command
    from alembic.config import Config

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {name}")
    # A URL, not key=value conninfo, because Alembic parses it with SQLAlchemy.
    owner_url = f"{admin_url.rsplit('/', 1)[0]}/{name}"
    os.environ["DATABASE_MIGRATIONS_URL"] = owner_url
    command.upgrade(Config(str(ROOT / "jobs/migrate/alembic.ini")), "head")
    with psycopg.connect(owner_url, autocommit=True) as owner:
        owner.execute(
            f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{LOGIN}') THEN "
            f"CREATE ROLE {LOGIN} LOGIN PASSWORD '{LOGIN_PASSWORD}'; END IF; END $$"
        )
        owner.execute(f"GRANT wikiguard_trainer TO {LOGIN}")
    return owner_url


def isolate_mlflow(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(folder / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("wikiguard") is None:
        mlflow.create_experiment("wikiguard", (folder / "artifacts").resolve().as_uri())
    mlflow.set_experiment("wikiguard")


def seed_registry(root: Path) -> LocalRegistry:
    """Copies of the Phase 2 BiLSTM (champion here) and MLP. data/runs is only read."""
    registry = LocalRegistry.create(root)
    versions = []
    for name in ("bilstm", "mlp"):
        report = json.loads((RUNS / name / "report.json").read_text("utf-8"))
        versions.append(
            registry.adopt(
                RUNS / name / "model",
                name,
                {"pr_auc": report["pr_auc"]},
                report["training_warnings"],
                report["run_id"],
                report["snapshot_content_sha256"],
            )
        )
    registry.set_aliases({"champion": versions[0], "fallback-1": versions[1]})
    return registry


def seed_listing(recorder: DbRecorder, registry: LocalRegistry) -> None:
    listings = []
    for info in registry.versions():
        folder = RUNS / info.model
        report = json.loads((folder / "report.json").read_text("utf-8"))
        config = read_config(folder / "model")
        listings.append(
            Listing(
                info.model,
                info.version,
                FEATURE_VERSION,
                info.snapshot_hash,
                info.artifact_sha256,
                metrics_of(report, folder, config),
            )
        )
    recorder.sync_models(snapshot_of_aliases(registry), listings)


def campaign(edit: EditEvent) -> EditEvent:
    """One edit of the spam campaign: logged out, a link, shouting, and much longer."""
    return edit.model_copy(
        update={
            "added_text": f"{SPAM} {edit.added_text[:200]}",
            "is_anonymous": True,
            "comment": "",
            "new_length": (edit.new_length or 0) + 400,
        }
    )


def build_archive(
    root: Path, as_of: datetime, shift: bool = False, poison_day: datetime | None = None
) -> dict[str, int]:
    edits, labels = [], []
    counts = {"edits": 0, "campaign": 0, "poisoned": 0}
    shift_start = as_of - timedelta(days=SHIFT_DAYS)
    for candidate in read_candidates(BACKFILL):
        if candidate.edit is None or candidate.label is None:
            continue
        edit, label = candidate.edit, candidate.label
        if shift and edit.timestamp >= shift_start and chosen(edit.rev_id, SPAM_SHARE, "spam"):
            edit, label = campaign(edit), 1
            counts["campaign"] += 1
        if poison_day is not None and poison_day <= edit.timestamp < poison_day + timedelta(1):
            if chosen(edit.rev_id, 0.6, "poison") and label == 0:
                label = 1
                counts["poisoned"] += 1
        edits.append(edit.model_dump())
        labels.append(
            {
                "schema_version": 1,
                "rev_id": edit.rev_id,
                "label": label,
                "excluded_reason": None,
                "decided_at": edit.timestamp + timedelta(hours=48),
                "decided_late": False,
            }
        )
        counts["edits"] += 1
    write_rows(root, EDITS, edits)
    write_rows(root, LABELS, labels)
    return counts


def served_folder(registry: LocalRegistry, version: str) -> Path:
    return registry.root / f"v{version}" / "model"


def score_archive(
    root: Path, as_of: datetime, registry: LocalRegistry, serving: list[tuple[datetime, str]]
) -> int:
    """What the scorer would have written: each edit scored by the version serving at that time.

    serving lists (from, version) in time order. Scored in batches here, while the live scorer
    scores singly (DEC-056), so these scores can differ slightly from served ones (F-107).
    """
    from wikiguard_core.archive import read_rows

    since = as_of - timedelta(days=SCORED_DAYS)
    rows = [row for row in read_rows(root, EDITS) if row["timestamp"] >= since]
    by_version: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        version = next(v for start, v in reversed(serving) if row["timestamp"] >= start)
        by_version.setdefault(version, []).append(row)
    written = []
    for version, members in by_version.items():
        folder = served_folder(registry, version)
        model = ServedModel(folder, threads=4)
        features = [build_features(EditEvent(**_edit_fields(row))).as_dict() for row in members]
        scores = model.probabilities(features)
        info = registry.info(version)
        for row, score in zip(members, scores, strict=True):
            written.append(
                {
                    "schema_version": 1,
                    "rev_id": row["rev_id"],
                    "model_name": info.model,
                    "model_version": version,
                    "score": float(round(score, 7)),
                    "used_fallback": False,
                    "scored_at": row["timestamp"] + timedelta(seconds=5),
                    "threshold": model.threshold,
                    "alias": "champion",
                }
            )
    write_rows(root, SCORES, written)
    return len(written)


def _edit_fields(row: dict[str, Any]) -> dict[str, Any]:
    values = {name: row[name] for name in EditEvent.model_fields}
    values["added_text"] = values["added_text"] or ""
    values["removed_text"] = values["removed_text"] or ""
    return values


class Sandbox:
    def __init__(self, name: str, admin_url: str, as_of: datetime, models: list[str]) -> None:
        self.folder = SIM / name
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        self.as_of = as_of
        self.owner_url = make_database(admin_url, f"wikiguard_sim_{name}")
        self.trainer_url = psycopg.conninfo.make_conninfo(
            self.owner_url, user=LOGIN, password=LOGIN_PASSWORD
        )
        self.connection = psycopg.connect(self.trainer_url)
        isolate_mlflow(self.folder / "mlflow")
        self.archive = self.folder / "archive"
        self.repo = self.folder / "repo"
        golden = self.repo / "snapshots/golden-set-v1"
        golden.mkdir(parents=True)
        shutil.copy2(GOLDEN, golden / "golden.json")
        self.registry = seed_registry(self.folder / "registry")
        self.recorder = DbRecorder(self.connection)
        seed_listing(self.recorder, self.registry)
        self.alerter = Alerter("sim")
        self.settings = SimpleNamespace(
            retrain_timeout_minutes=600,
            retrain_startup_delay_seconds=0,
            retrain_threads=4,
            model_names=models,
            retrain_snapshot_rows=100_000,
            shadow_hours=0,
            retrain_interval_days=7,
            retrain_on_drift=True,
        )
        self.paths = Paths(
            workdir=self.folder / "retrain",
            snapshots=self.repo / "snapshots",
            archive=self.archive,
            golden=golden / "golden.json",
            images=self.folder / "images",
            cache=self.folder / "cache",
        )
        self.clock = as_of
        self.pipeline = Pipeline(
            self.settings,
            self.registry,
            self.recorder,
            self.alerter,
            self.paths,
            clock=lambda: self.clock,
            train=lambda name, data, out, meta: train_final(
                name, data, PHASE2_PARAMS[name], CPU, None, 0, out, "retrain", meta
            ),
            sanity=sanity_checks,
            after_run=lambda: retain(
                self.registry,
                DvcRepo(self.repo),
                self.archive,
                False,
                10 * 1024**3,
                self.alerter,
            ),
        )
        self.lifecycle = Lifecycle(
            self.settings, self.registry, self.recorder, self.alerter, self.pipeline
        )
        self.timings: dict[str, float] = {}

    def launchers(self) -> dict[str, Any]:
        """The poller's launchers, calling the lifecycle here instead of in a subprocess."""

        def rollback(request: dict[str, Any]) -> None:
            self.lifecycle.rollback(request["id"], request["reason"], request["requested_by"])

        return {
            "retrain": lambda request: self.pipeline.retrain(request["run_id"], None, self.as_of),
            "shadow_check": lambda request: self.lifecycle.shadow_check(request["run_id"]),
            "promote": lambda request: self.lifecycle.promote(request["id"]),
            "rollback": rollback,
        }

    def poll(self) -> list[dict[str, Any]]:
        handled = []
        request_shadow_checks(self.connection, self.settings.shadow_hours)
        while (result := run_once(self.connection, self.launchers())) is not None:
            handled.append({key: result[key] for key in ("id", "kind", "status")})
        return handled

    def monitor(self, now: datetime) -> Any:
        return Monitor(
            self.settings, self.connection, self.alerter, evidently_drift, self.archive
        ).run(now)

    def runs(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT id, trigger, status, details - 'listings' - 'slices' FROM retrain_runs "
            "ORDER BY id"
        )
        rows = [
            {"id": row[0], "trigger": row[1], "status": row[2], "details": row[3]}
            for row in cursor.fetchall()
        ]
        self.connection.commit()
        return rows

    def write(self, summary: dict[str, Any]) -> None:
        summary |= {
            "timings_seconds": self.timings,
            "alerts": [f"[{a.severity}] {a.title}: {a.detail}" for a in self.alerter.sent],
            "aliases": self.registry.aliases(),
        }
        (self.folder / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), "utf-8"
        )
        log(f"Summary in {self.folder / 'summary.json'}")

    def timed(self, step: str, work: Any) -> Any:
        started = time.perf_counter()
        result = work()
        self.timings[step] = round(time.perf_counter() - started, 1)
        log(f"{step}: {self.timings[step]} s")
        return result


def replay_shadow(box: Sandbox, days: float = 1.0) -> int:
    """The shadow period, replayed: the scorer's own shadow code on the archive's newest edits."""
    from scorer import Item, ScoringWorker
    from wikiguard_core.archive import read_rows

    newest = box.as_of - timedelta(hours=48)
    edits = [
        EditEvent(**_edit_fields(row))
        for row in read_rows(box.archive, EDITS)
        if newest - timedelta(days=days) <= row["timestamp"] < newest
    ]
    source = box.registry.source()
    chain = ServingChain(source, ModelCache(box.folder / "serve-cache"), 5000)
    chain.load()
    shadow = ShadowSlot(source, ModelCache(box.folder / "serve-cache-shadow"), 5000)
    shadow.refresh()
    worker = ScoringWorker(None, None, chain, lambda: None, shadow=shadow)  # type: ignore[arg-type]
    items = worker.score([Item(edit, build_features(edit).as_dict()) for edit in edits])
    rows = worker.shadow_rows(items)
    with psycopg.connect(box.owner_url) as owner:
        with owner.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO edits (rev_id, schema_version, title, edited_at, is_minor, "
                "is_anonymous) VALUES (%s, 1, '', %s, %s, %s) ON CONFLICT DO NOTHING",
                [(e.rev_id, e.timestamp, e.is_minor, e.is_anonymous) for e in edits],
            )
            from scorer import INSERT_SHADOW

            cursor.executemany(INSERT_SHADOW, rows)
        owner.commit()
    return len(rows)


# --- Scenarios ---


def drift(admin_url: str) -> None:
    as_of = backfill_as_of(BACKFILL)
    box = Sandbox("drift", admin_url, as_of, ["mlp", "bilstm"])
    counts = box.timed("archive", lambda: build_archive(box.archive, as_of, shift=True))
    champion = box.registry.aliases()["champion"]
    box.timed(
        "scores",
        lambda: score_archive(
            box.archive, as_of, box.registry, [(datetime.min.replace(tzinfo=UTC), champion)]
        ),
    )
    rows = [row for row in load_dataset(SNAPSHOT).train]
    rows += load_dataset(SNAPSHOT).validation
    box.timed(
        "reference",
        lambda: save_for_model(
            box.archive,
            champion,
            "bilstm",
            served_folder(box.registry, champion),
            rows,
            SNAPSHOT.name,
        ),
    )
    report = box.timed("monitor", lambda: box.monitor(as_of))
    handled = box.timed("retrain", box.poll)
    runs = box.runs()
    summary: dict[str, Any] = {
        "scenario": "drift",
        "as_of": as_of,
        "archive": counts,
        "monitor": {
            "drift": report.drift,
            "decision": report.decision,
            "champion_window": {
                k: report.champion_window.get(k)
                for k in ("labeled", "positives", "pr_auc", "precision_at_50")
            },
        },
        "poller_after_monitor": handled,
        "runs_after_retrain": runs,
    }
    if runs and runs[-1]["status"] == "shadow":
        summary["shadow_rows"] = box.timed("shadow", lambda: replay_shadow(box))
        summary["poller_after_shadow"] = box.timed("shadow_check", box.poll)
        summary["runs_after_shadow"] = box.runs()
    box.write(summary)


def bad_model(box: Sandbox, data: Any) -> str:
    """A BiLSTM trained on shuffled labels: a valid artifact that ranks no better than chance."""
    rng = np.random.default_rng(3)

    def mix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = rng.permutation([row["label"] for row in rows])
        return [row | {"label": int(label)} for row, label in zip(rows, labels, strict=True)]

    shuffled = replace(data, train=mix(data.train), validation=mix(data.validation))
    out = box.folder / "bad-model"
    report = train_final(
        "bilstm",
        shuffled,
        PHASE2_PARAMS["bilstm"] | {"epochs": 2},
        CPU,
        8_000,
        0,
        out,
        "deliberately-bad",
        None,
    )
    version = box.registry.register(
        out / "model",
        "bilstm",
        report["run_id"],
        report["artifact_sha256"],
        {"pr_auc": report["pr_auc"]},
        [],
        report["snapshot_content_sha256"],
    )
    config = read_config(out / "model")
    box.recorder.sync_models(
        snapshot_of_aliases(box.registry),
        box.pipeline.listings_for(list(snapshot_of_aliases(box.registry).values()), {})
        + [
            Listing(
                "bilstm",
                version,
                FEATURE_VERSION,
                report["snapshot_content_sha256"],
                report["artifact_sha256"],
                metrics_of(report, out, config),
            )
        ],
    )
    return version


def rollback(admin_url: str) -> None:
    as_of = backfill_as_of(BACKFILL)
    promoted_at = as_of - timedelta(days=9)
    box = Sandbox("rollback", admin_url, as_of, ["mlp", "bilstm"])
    counts = box.timed("archive", lambda: build_archive(box.archive, as_of))
    data = load_dataset(SNAPSHOT)
    champion = box.registry.aliases()["champion"]
    bad = box.timed("bad_model", lambda: bad_model(box, data))
    # The owner forces the bad model in, at replay time promoted_at.
    box.clock = promoted_at
    box.connection.execute(
        "INSERT INTO retrain_requests (kind, reason, requested_by, details) VALUES "
        "('promote', 'rollback rehearsal: a deliberately bad model', 'user', %s)",
        (psycopg.types.json.Jsonb({"model_version": bad}),),
    )
    box.connection.commit()
    promoted = box.timed("promote", box.poll)
    after_promotion = box.registry.aliases()
    # It serves from then on: what the scorer would have written.
    box.timed(
        "scores",
        lambda: score_archive(
            box.archive,
            as_of,
            box.registry,
            [(datetime.min.replace(tzinfo=UTC), champion), (promoted_at, bad)],
        ),
    )
    box.clock = as_of
    report = box.timed("monitor", lambda: box.monitor(as_of))
    started = time.perf_counter()
    handled = box.poll()
    box.timings["rollback_request_to_restored"] = round(time.perf_counter() - started, 1)
    box.write(
        {
            "scenario": "rollback",
            "as_of": as_of,
            "promoted_at": promoted_at,
            "archive": counts,
            "champion_before": champion,
            "bad_version": bad,
            "aliases_after_promotion": after_promotion,
            "poller_promote": promoted,
            "post_check": report.post_check,
            "monitor_decision": report.decision,
            "poller_after_monitor": handled,
            "runs": box.runs(),
        }
    )


def poisoning(admin_url: str) -> None:
    as_of = backfill_as_of(BACKFILL)
    day = (as_of - timedelta(days=12)).replace(hour=0, minute=0, second=0, microsecond=0)
    box = Sandbox("poisoning", admin_url, as_of, ["mlp"])
    counts = box.timed("archive", lambda: build_archive(box.archive, as_of, poison_day=day))
    before = box.registry.aliases()
    box.connection.execute(
        "INSERT INTO retrain_requests (kind, reason, requested_by) VALUES "
        "('retrain', 'poisoning rehearsal', 'user')"
    )
    box.connection.commit()
    handled = box.timed("retrain", box.poll)
    box.write(
        {
            "scenario": "poisoning",
            "as_of": as_of,
            "poisoned_day": day,
            "archive": counts,
            "aliases_before": before,
            "poller": handled,
            "runs": box.runs(),
            "unchanged": box.registry.aliases() == before,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=["drift", "rollback", "poisoning"])
    parser.add_argument("--admin-url", required=True, help="a throwaway Postgres, never Supabase")
    args = parser.parse_args()
    if "supabase" in args.admin_url:
        raise SystemExit("refusing: this creates and drops databases")
    {"drift": drift, "rollback": rollback, "poisoning": poisoning}[args.scenario](args.admin_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
