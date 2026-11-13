"""Daily: measure the live model, check drift, and decide whether to retrain (PLAN.md, section 5).

    python monitor.py                       # in the cluster, as a CronJob
    python monitor.py --now 2026-11-03T12:00:00+00:00 --charts docs/images   # a replay

What it reads: edits, scores, and labels from the parquet archive, never Supabase history (D1),
and the champion from model_versions. What it writes: one live_metrics row per labeled day, one
drift_reports row, and, when a trigger fires, a retrain request the poller picks up, the same way
the API asks (DEC-071). It also runs the post-promotion check, asking for a rollback when the new
champion does worse live than the one it replaced.

When it can't run, or has too few labels to judge, it alerts instead of quietly doing nothing.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pyarrow.compute as pc
from psycopg.types.json import Jsonb
from wikiguard_core.alerts import Alerter, ping
from wikiguard_core.archive import (
    EDITS,
    LABELS,
    SCORES,
    Root,
    open_archive,
    read_reference,
    read_table,
    sample_mask,
)
from wikiguard_core.config import (
    AlertSettings,
    ArchiveSettings,
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    RetrainSettings,
    load_settings,
)
from wikiguard_core.features import build_features, edit_count_bucket
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.monitoring import (
    TRAINED_STATUSES,
    WINDOW,
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
from wikiguard_core.schemas import EditEvent

logger = logging.getLogger("monitor")

# Labels are final 48 hours after an edit, so the newest labeled day is two days back.
LABEL_WAIT = timedelta(hours=48)
# Enough history for the 7-day window, the post-promotion comparison, and the label wait.
READ_DAYS = 16
DRIFT_ROWS = 2_000
SKIP_TRIGGERS = ("manual_promote",)
EDIT_COLUMNS = [
    "timestamp",
    "is_anonymous",
    "user_edit_count",
    "old_length",
    "new_length",
]
DriftFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


class Settings(CommonSettings, DatabaseSettings, ArchiveSettings, RetrainSettings, AlertSettings):
    pass


@dataclass
class Champion:
    version: str
    model: str
    offline_pr_auc: float | None
    threshold: float | None
    registered_at: datetime


@dataclass
class Report:
    now: datetime
    champion: Champion | None = None
    window: dict[str, Any] = field(default_factory=dict)
    champion_window: dict[str, Any] = field(default_factory=dict)
    days: dict[str, dict[str, Any]] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    reviewers: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    post_check: dict[str, Any] | None = None
    label_lag_hours: float | None = None


# --- Reading ---


def observed_from_archive(root: Root, now: datetime) -> Observed:
    """Every edit with a final label, its newest served score, and the columns for slices."""
    since = (now - timedelta(days=READ_DAYS)).date()
    labels = read_table(root, LABELS, since=since, columns=["label", "excluded_reason"])
    labels = labels.filter(pc.is_valid(labels.column("label")))
    if not labels.num_rows:
        return empty_observed()
    wanted = labels.column("rev_id").combine_chunks()
    scores = read_table(
        root,
        SCORES,
        since=since,
        columns=["model_version", "score", "used_fallback", "scored_at", "threshold"],
        keep=lambda ids: pc.is_in(ids, value_set=wanted),
    ).to_pylist()
    newest: dict[int, dict[str, Any]] = {}
    for row in scores:
        held = newest.get(row["rev_id"])
        if held is None or row["scored_at"] >= held["scored_at"]:
            newest[row["rev_id"]] = row
    edits = {
        row["rev_id"]: row
        for row in read_table(
            root,
            EDITS,
            since=since,
            columns=EDIT_COLUMNS,
            keep=lambda ids: pc.is_in(ids, value_set=wanted),
        ).to_pylist()
    }
    rows = [
        (label, newest[label["rev_id"]], edits[label["rev_id"]])
        for label in labels.to_pylist()
        if label["rev_id"] in newest and label["rev_id"] in edits
    ]
    if not rows:
        return empty_observed()
    return Observed(
        rev_id=np.array([label["rev_id"] for label, _, _ in rows], dtype=np.int64),
        edited_at=np.array(
            [edit["timestamp"].replace(tzinfo=None) for _, _, edit in rows], dtype="datetime64[s]"
        ),
        label=np.array([label["label"] for label, _, _ in rows], dtype=np.int64),
        score=np.array([score["score"] for _, score, _ in rows], dtype=np.float64),
        threshold=np.array(
            [np.nan if score["threshold"] is None else score["threshold"] for _, score, _ in rows],
            dtype=np.float64,
        ),
        used_fallback=np.array([bool(score["used_fallback"]) for _, score, _ in rows]),
        model_version=np.array([str(score["model_version"]) for _, score, _ in rows], dtype=object),
        is_anonymous=np.array([bool(edit["is_anonymous"]) for _, _, edit in rows]),
        edit_count_bucket=np.array(
            [edit_count_bucket(edit["user_edit_count"]) for _, _, edit in rows], dtype=np.int64
        ),
        bytes_delta=np.array(
            [(edit["new_length"] or 0) - (edit["old_length"] or 0) for _, _, edit in rows],
            dtype=np.int64,
        ),
    )


def current_sample(root: Root, now: datetime, rows: int = DRIFT_ROWS) -> list[dict[str, Any]]:
    """Features of a sample of the last 7 days' edits. Labels aren't needed for drift."""
    since = (now - WINDOW).date()
    count = read_table(root, EDITS, since=since, until=now.date(), columns=[]).num_rows
    fraction = min(1.0, rows * 1.2 / count) if count else 1.0
    table = read_table(
        root,
        EDITS,
        since=since,
        until=now.date(),
        keep=lambda ids: sample_mask(ids, fraction, salt="drift"),
    )
    found = []
    for row in table.to_pylist()[:rows]:
        values = {name: row[name] for name in EditEvent.model_fields if name in row}
        values["added_text"] = values.get("added_text") or ""
        values["removed_text"] = values.get("removed_text") or ""
        found.append(build_features(EditEvent(**values)).as_dict())
    return found


def champion_of(connection: psycopg.Connection[Any]) -> Champion | None:
    row = connection.execute(
        "SELECT model_version, model_name, metrics, registered_at FROM model_versions "
        "WHERE alias = 'champion'"
    ).fetchone()
    connection.commit()
    if row is None:
        return None
    version, model, metrics, registered = row
    return Champion(
        str(version),
        model,
        metrics.get("pr_auc"),
        metrics.get("threshold_probability"),
        registered,
    )


def retrain_history(
    connection: psycopg.Connection[Any],
) -> tuple[datetime | None, datetime | None]:
    """When the newest retrain started at all, and when the newest one that trained started."""
    rows = connection.execute(
        "SELECT status, started_at FROM retrain_runs WHERE trigger <> ALL(%s) "
        "ORDER BY started_at DESC LIMIT 50",
        (list(SKIP_TRIGGERS),),
    ).fetchall()
    connection.commit()
    started = rows[0][1] if rows else None
    trained = next((at for status, at in rows if status in TRAINED_STATUSES), None)
    return started, trained


def reviews_of(connection: psycopg.Connection[Any], now: datetime) -> list[tuple[int, str]]:
    rows = connection.execute(
        "SELECT rev_id, decision FROM reviews WHERE reviewed_at > %s",
        (now - timedelta(days=30),),
    ).fetchall()
    connection.commit()
    return [(int(rev_id), str(decision)) for rev_id, decision in rows]


# --- Measuring ---


def in_window(observed: Observed, start: datetime, end: datetime) -> Observed:
    begin = np.datetime64(start.astimezone(UTC).replace(tzinfo=None), "s")
    finish = np.datetime64(end.astimezone(UTC).replace(tzinfo=None), "s")
    return observed.where((observed.edited_at >= begin) & (observed.edited_at < finish))


def day_rows(observed: Observed, threshold: float | None) -> dict[str, dict[str, Any]]:
    """One point per day, for whichever version served most of that day's edits."""
    out = {}
    days = observed.days()
    for day in np.unique(days):
        members = observed.where(days == day)
        versions, counts = np.unique(members.model_version, return_counts=True)
        version = str(versions[int(np.argmax(counts))])
        summary = daily(members, threshold)[str(day)]
        out[str(day)] = summary | {"model_version": version}
    return out


def post_check(
    connection: psycopg.Connection[Any], observed: Observed, now: datetime
) -> dict[str, Any] | None:
    """Section 4's post-promotion check on the newest promotion not yet decided."""
    row = connection.execute(
        "SELECT id, details FROM retrain_runs WHERE status = 'promoted' "
        "AND NOT (details ? 'post_check' AND (details->'post_check'->>'due')::boolean) "
        "ORDER BY (details->>'promoted_at') DESC LIMIT 1"
    ).fetchone()
    connection.commit()
    if row is None:
        return None
    run_id, details = row
    promoted_at = datetime.fromisoformat(details["promoted_at"])
    new_version = details["aliases_after"].get("champion")
    old_version = details["aliases_before"].get("champion")
    new = observed.where(observed.model_version == new_version)
    new = in_window(new, promoted_at, now)
    old = observed.where(observed.model_version == old_version)
    old = in_window(old, promoted_at - WINDOW, promoted_at)
    new_summary, old_summary = summarize(new, None), summarize(old, None)
    check = post_promotion(
        PostCheckInputs(
            now=now,
            promoted_at=promoted_at,
            new_pr_auc=new_summary["pr_auc"],
            new_labels=new.size,
            previous_pr_auc=old_summary["pr_auc"],
            previous_labels=old.size,
        )
    )
    result = check.as_dict() | {
        "run_id": run_id,
        "checked_at": now.isoformat(),
        "new_version": new_version,
        "previous_version": old_version,
        "new_pr_auc": new_summary["pr_auc"],
        "new_labels": new.size,
        "previous_pr_auc": old_summary["pr_auc"],
        "previous_labels": old.size,
    }
    connection.execute(
        "UPDATE retrain_runs SET details = details || %s WHERE id = %s",
        (Jsonb({"post_check": result}), run_id),
    )
    if check.rollback:
        connection.execute(
            "INSERT INTO retrain_requests (kind, reason, requested_by, details) "
            "VALUES ('rollback', %s, 'monitor', %s)",
            (f"post-promotion check: {check.reason}"[:500], Jsonb({"run_id": run_id})),
        )
    connection.commit()
    return result


# --- The run ---


@dataclass
class Monitor:
    settings: Any
    connection: psycopg.Connection[Any]
    alerter: Alerter
    drift: DriftFn
    archive_root: Root

    def run(self, now: datetime) -> Report:
        report = Report(now)
        champion = champion_of(self.connection)
        report.champion = champion
        if champion is None:
            self.alerter.send("Monitor has nothing to measure", "no model has the champion alias")
            return report
        observed = observed_from_archive(self.archive_root, now)
        window = in_window(observed, now - LABEL_WAIT - WINDOW, now)
        report.window = summarize(window, champion.threshold)
        served_by_champion = window.where(window.model_version == champion.version)
        report.champion_window = summarize(served_by_champion, champion.threshold)
        if observed.size:
            newest = observed.edited_at.max().item().replace(tzinfo=UTC)
            report.label_lag_hours = (now - newest).total_seconds() / 3600
        report.days = day_rows(observed, champion.threshold)
        self._write_days(report)

        report.drift = self._drift(champion, now, served_by_champion)
        report.reviewers = self._reviewers(observed, champion, now)
        report.post_check = post_check(self.connection, observed, now)
        if report.post_check and report.post_check["rollback"]:
            self.alerter.send("Rolling back", report.post_check["reason"], "critical")

        started, trained = retrain_history(self.connection)
        decision = decide(
            TriggerInputs(
                now=now,
                live_pr_auc=report.champion_window["pr_auc"],
                live_labels=served_by_champion.size,
                offline_pr_auc=champion.offline_pr_auc,
                drift_share=report.drift.get("metadata_share_drifted"),
                text_drift=bool(report.drift.get("text_drift")),
                last_retrain_started=started,
                schedule_since=trained or champion.registered_at,
                interval_days=self.settings.retrain_interval_days,
                on_drift=self.settings.retrain_on_drift,
            )
        )
        report.decision = {
            "retrain": decision.retrain,
            "trigger": decision.trigger,
            "reasons": decision.reasons,
            "alerts": [title for title, _ in decision.alerts],
        }
        for title, detail in decision.alerts:
            self.alerter.send(title, detail)
        if decision.retrain:
            self._ask_for_retrain(decision.trigger or "schedule", decision.reasons)
        self._write_drift(report)
        return report

    def _write_days(self, report: Report) -> None:
        for day, summary in report.days.items():
            version = summary.pop("model_version")
            self.connection.execute(
                """
                INSERT INTO live_metrics (day, model_version, metrics) VALUES (%s, %s, %s)
                ON CONFLICT (day, model_version) DO UPDATE SET metrics = EXCLUDED.metrics
                """,
                (date.fromisoformat(day), version, Jsonb(_clean(summary))),
            )
            summary["model_version"] = version
        self.connection.commit()

    def _drift(self, champion: Champion, now: datetime, served: Observed) -> dict[str, Any]:
        reference = read_reference(self.archive_root, champion.version)
        if reference is None:
            self.alerter.send(
                "No drift reference",
                f"version {champion.version} has no reference sample, so drift can't be checked",
            )
            return {"checked": False, "reason": "no reference sample"}
        table, meta = reference
        rows = table.to_pylist()
        current = current_sample(self.archive_root, now)
        if not current:
            self.alerter.send("No live edits", "the archive has no edits from the last 7 days")
            return {"checked": False, "reason": "no live edits"}
        result = self.drift(rows, current)
        psi = score_psi(np.array([row["score"] for row in rows]), served.score)
        result |= {"checked": True, "reference_version": meta["version"], "score_psi": psi}
        if psi is not None and psi > 0.25:
            self.alerter.send("Live scores moved", f"score PSI {psi:.3f} against the reference")
        return result

    def _reviewers(self, observed: Observed, champion: Champion, now: datetime) -> dict[str, Any]:
        reviews = reviews_of(self.connection, now)
        if not reviews:
            return {"reviewed": 0}
        positions = {int(rev): index for index, rev in enumerate(observed.rev_id.tolist())}
        scored = [(decision, positions[rev]) for rev, decision in reviews if rev in positions]
        decisions = [decision for decision, _ in scored]
        scores = observed.score[[index for _, index in scored]] if scored else np.array([])
        result = reviewer_precision(decisions, scores, champion.threshold)
        result["reviewed_total"] = len(reviews)
        return result

    def _ask_for_retrain(self, trigger: str, reasons: list[str]) -> None:
        waiting = self.connection.execute(
            "SELECT 1 FROM retrain_requests WHERE kind = 'retrain' "
            "AND status IN ('pending', 'running') LIMIT 1"
        ).fetchone()
        if waiting is None:
            self.connection.execute(
                "INSERT INTO retrain_requests (kind, reason, requested_by, details) "
                "VALUES ('retrain', %s, 'monitor', %s)",
                ("; ".join(reasons)[:500], Jsonb({"trigger": trigger, "reasons": reasons})),
            )
        self.connection.commit()

    def _write_drift(self, report: Report) -> None:
        drift = report.drift
        drifted = bool(
            drift.get("checked")
            and ((drift.get("metadata_share_drifted") or 0) >= 0.30 or drift.get("text_drift"))
        )
        body = {
            "drift": drift,
            "window": _clean(report.window),
            "champion_window": _clean(report.champion_window),
            "reviewers": report.reviewers,
            "decision": report.decision,
            "post_check": report.post_check,
            "label_lag_hours": report.label_lag_hours,
            "champion": report.champion.version if report.champion else None,
            "measured_at": report.now.isoformat(),
        }
        self.connection.execute(
            "INSERT INTO drift_reports (created_at, drifted, report) VALUES (%s, %s, %s)",
            (report.now, drifted, Jsonb(_clean(body))),
        )
        self.connection.commit()


def _clean(value: Any) -> Any:
    """JSON can't hold NaN, and numpy numbers aren't JSON types."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_charts(report: Report, folder: Path) -> list[Path]:
    """monitor_<chart>_<date>.png (PLAN.md, section 17), from the report's numbers only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    day = report.now.date().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    points = [(d, s["pr_auc"]) for d, s in sorted(report.days.items()) if s.get("pr_auc")]
    if points:
        figure, axes = plt.subplots(figsize=(9, 4.8), dpi=150)
        axes.plot([p[0] for p in points], [p[1] for p in points], marker="o", label="Live PR-AUC")
        if report.champion and report.champion.offline_pr_auc:
            axes.axhline(
                report.champion.offline_pr_auc,
                linestyle="--",
                color="#898781",
                label="Offline test PR-AUC",
            )
        axes.set_title("Live PR-AUC by edit day")
        axes.set_ylabel("PR-AUC")
        axes.tick_params(axis="x", rotation=45)
        axes.legend()
        figure.tight_layout()
        path = folder / f"monitor_live_pr_auc_{day}.png"
        figure.savefig(path)
        plt.close(figure)
        saved.append(path)
    statistics = report.drift.get("metadata_statistics") or {}
    if statistics:
        figure, axes = plt.subplots(figsize=(9, 4.8), dpi=150)
        names = sorted(statistics)
        # Each column's statistic over its own threshold, so one scale works for every test:
        # distances drift above 1, p-values below it (F-124).
        ratios, labels = [], []
        for name in names:
            entry = statistics[name]
            value, threshold = float(entry["value"]), float(entry["threshold"] or 1)
            p_value = "p_value" in entry["method"]
            ratios.append(threshold / max(value, 1e-12) if p_value else value / threshold)
            labels.append(f"{name} ({entry['method'].split(' ')[0]})")
        colors = ["#e34948" if ratio > 1 else "#2a78d6" for ratio in ratios]
        axes.barh(labels, ratios, color=colors)
        axes.axvline(1.0, linestyle="--", color="#898781", label="Drift threshold")
        axes.set_xscale("log")
        axes.set_title(
            f"Feature drift: {report.drift.get('metadata_share_drifted', 0):.0%} of features "
            f"drifted, text drift {'found' if report.drift.get('text_drift') else 'not found'}"
        )
        axes.set_xlabel("Strength against the column's own threshold (above 1 is drift)")
        axes.legend()
        figure.tight_layout()
        path = folder / f"monitor_drift_{day}.png"
        figure.savefig(path)
        plt.close(figure)
        saved.append(path)
    return saved


def main() -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--now", type=datetime.fromisoformat, help="for a replay")
    parser.add_argument("--charts", type=Path, help="save monitor_*.png charts here")
    args = parser.parse_args()
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    webhook = settings.discord_webhook_url
    alerter = Alerter("monitor", webhook.get_secret_value() if webhook else None)
    now = args.now or datetime.now(UTC)
    try:
        from drift import evidently_drift

        with psycopg.connect(
            settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
        ) as connection:
            report = Monitor(
                settings, connection, alerter, evidently_drift, open_archive(settings)
            ).run(now)
    except Exception as error:
        # Section 5: a monitor that fails must say so, not quietly skip a day.
        alerter.send("Monitor failed", type(error).__name__, "critical")
        raise
    if args.charts:
        for path in save_charts(report, args.charts):
            logger.info("Saved %s", path)
    logger.info("Monitor finished: %s", report.decision)
    ping_url = settings.healthchecks_ping_url
    ping(ping_url.get_secret_value() if ping_url else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
