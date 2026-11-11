"""The daily monitor against real Postgres, as the role it runs as, with a synthetic archive."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest
from monitor import Monitor, observed_from_archive, save_charts
from psycopg.types.json import Jsonb
from wikiguard_core.alerts import Alerter
from wikiguard_core.archive import EDITS, LABELS, SCORES, write_reference, write_rows

NOW = datetime(2026, 11, 20, 6, tzinfo=UTC)
PASSWORD = "test-only-not-a-secret"


def edit(rev_id: int, at: datetime, anonymous: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "title": "Page",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": anonymous,
        "user_hash": None,
        "comment": "c",
        "old_length": 1000,
        "new_length": 1000 + rev_id % 300,
        "added_text": f"words {rev_id}",
        "removed_text": "",
        "user_edit_count": 5,
        "user_hidden": False,
        "comment_hidden": False,
        "content_hidden": False,
    }


def fill_archive(
    root: Path,
    versions: dict[int, str] | None = None,
    per_day: int = 300,
    quality: float = 3.0,
    bad_after: datetime | None = None,
) -> None:
    """Twelve days of edits, scored by version "1" with a clear signal, and final labels."""
    rng = np.random.default_rng(0)
    edits, scores, labels = [], [], []
    rev_id = 0
    for days_back in range(12, 0, -1):
        day = NOW - timedelta(days=days_back)
        for index in range(per_day):
            rev_id += 1
            at = day + timedelta(seconds=index * 60)
            label = int(rng.random() < 0.1)
            version = "1"
            signal = quality
            if bad_after is not None and at >= bad_after:
                version, signal = "2", 0.0
            logit = signal * label + rng.normal(0, 1)
            edits.append(edit(rev_id, at, index % 2 == 0))
            scores.append(
                {
                    "schema_version": 1,
                    "rev_id": rev_id,
                    "model_name": "mlp",
                    "model_version": (versions or {}).get(rev_id, version),
                    "score": float(1 / (1 + np.exp(-logit))),
                    "used_fallback": False,
                    "scored_at": at + timedelta(seconds=1),
                    "threshold": 0.5,
                    "alias": "champion",
                }
            )
            if at <= NOW - timedelta(hours=48):
                labels.append(
                    {
                        "schema_version": 1,
                        "rev_id": rev_id,
                        "label": label,
                        "excluded_reason": None,
                        "decided_at": at + timedelta(hours=48),
                        "decided_late": False,
                    }
                )
    write_rows(root, EDITS, edits)
    write_rows(root, SCORES, scores)
    write_rows(root, LABELS, labels)
    reference = [
        {
            "rev_id": 10_000_000 + index,
            "timestamp": NOW - timedelta(days=40),
            "split": "train",
            "label": 0,
            "feature_version": 1,
            "added_text": "words",
            "removed_text": "",
            "comment": "c",
            "is_anonymous": False,
            "user_edit_count_bucket": -1,
            "bytes_delta": index % 300,
            "is_minor": False,
            "chars_added": 5,
            "uppercase_ratio": 0.0,
            "badword_count": 0,
            "has_external_link": False,
            "hour_of_day": index % 24,
            "page_length": 1000,
            # Scores like the live ones, so a healthy day shows no score drift.
            "score": float(1 / (1 + np.exp(-(quality * (index % 10 == 0) + rng.normal(0, 1))))),
        }
        for index in range(500)
    ]
    write_reference(root, "1", reference, {"version": "1"})


@pytest.fixture
def monitor_db(db: psycopg.Connection[Any], migrated_url: str) -> Iterator[psycopg.Connection[Any]]:
    """The monitor runs as the trainer role (migration 0004), so a missing grant fails here."""
    with psycopg.connect(migrated_url, autocommit=True) as owner:
        owner.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = "
            f"'test_login_trainer') THEN CREATE ROLE test_login_trainer LOGIN PASSWORD "
            f"'{PASSWORD}'; END IF; END $$"
        )
        owner.execute("GRANT wikiguard_trainer TO test_login_trainer")
    db.execute(
        "INSERT INTO model_versions (model_name, model_version, alias, feature_version, "
        "snapshot_hash, artifact_sha256, metrics, registered_at) VALUES "
        "('roberta', '1', 'champion', 1, 'h', 'd', %s, %s)",
        (Jsonb({"pr_auc": 0.36, "threshold_probability": 0.5}), NOW - timedelta(days=3)),
    )
    db.commit()
    url = psycopg.conninfo.make_conninfo(migrated_url, user="test_login_trainer", password=PASSWORD)
    with psycopg.connect(url) as connection:
        yield connection


class Settings:
    retrain_interval_days = 7
    retrain_on_drift = True


def no_drift(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata_share_drifted": 0.1,
        "text_drift": False,
        "metadata_statistics": {
            "x": {"value": 0.05, "method": "Wasserstein distance (normed)", "threshold": 0.1}
        },
    }


def drifted(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata_share_drifted": 0.4,
        "text_drift": True,
        "metadata_statistics": {"x": {"value": 1e-9, "method": "K-S p_value", "threshold": 0.05}},
    }


def run(
    connection: psycopg.Connection[Any], root: Path, drift: Any = no_drift, settings: Any = None
) -> tuple[Any, Alerter]:
    alerter = Alerter("monitor")
    report = Monitor(settings or Settings(), connection, alerter, drift, root).run(NOW)
    return report, alerter


def requests(db: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    rows = db.execute(
        "SELECT kind, requested_by, details FROM retrain_requests ORDER BY id"
    ).fetchall()
    db.commit()
    return rows


def test_the_archive_gives_one_served_score_per_labeled_edit(tmp_path: Path) -> None:
    fill_archive(tmp_path)
    observed = observed_from_archive(tmp_path, NOW)
    # Ten labeled days of 300, since the last two days are still inside their 48 hours.
    assert 2900 <= observed.size <= 3100
    assert set(observed.model_version.tolist()) == {"1"}


pytestmark = pytest.mark.integration


def test_a_healthy_day_writes_metrics_and_asks_for_nothing(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)
    report, alerter = run(monitor_db, tmp_path)
    assert report.champion_window["pr_auc"] > 0.4
    assert report.champion_window["labeled"] >= 1500
    assert report.label_lag_hours is not None and report.label_lag_hours >= 48
    days = db.execute("SELECT day, model_version, metrics FROM live_metrics").fetchall()
    db.commit()
    assert len(days) >= 9 and {version for _, version, _ in days} == {"1"}
    assert all(metrics["labeled"] > 0 for _, _, metrics in days)
    drift = db.execute("SELECT drifted, report FROM drift_reports").fetchone()
    db.commit()
    assert drift is not None and drift[0] is False
    assert drift[1]["drift"]["reference_version"] == "1"
    assert requests(db) == []
    assert alerter.sent == []


def test_drift_asks_the_poller_for_a_retrain_with_its_reasons(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)
    run(monitor_db, tmp_path, drift=drifted)
    found = requests(db)
    assert [(kind, by) for kind, by, _ in found] == [("retrain", "monitor")]
    assert found[0][2]["trigger"] == "drift"
    assert "40% of metadata features drifted; text drift found" in found[0][2]["reasons"]
    # A second run the same day doesn't add another while one is waiting.
    run(monitor_db, tmp_path, drift=drifted)
    assert len(requests(db)) == 1


def test_with_retrain_on_drift_off_drift_only_alerts(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)

    class Off(Settings):
        retrain_on_drift = False

    _, alerter = run(monitor_db, tmp_path, drift=drifted, settings=Off())
    assert requests(db) == []
    assert "Drift trigger (retraining is off)" in [alert.title for alert in alerter.sent]


def test_the_schedule_triggers_after_the_interval(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)
    db.execute(
        "INSERT INTO retrain_runs (trigger, status, started_at) VALUES ('schedule', "
        "'not_promoted', %s)",
        (NOW - timedelta(days=8),),
    )
    db.commit()
    run(monitor_db, tmp_path)
    assert requests(db)[0][2]["trigger"] == "schedule"


def test_a_performance_drop_triggers(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path, quality=0.0)
    run(monitor_db, tmp_path)
    assert requests(db)[0][2]["trigger"] == "performance"


def test_too_few_labels_alerts(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path, per_day=20)
    _, alerter = run(monitor_db, tmp_path)
    assert "Too few labels" in [alert.title for alert in alerter.sent]


def test_no_reference_sample_alerts_instead_of_skipping_quietly(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)
    db.execute("UPDATE model_versions SET model_version = '9'")
    db.commit()
    _, alerter = run(monitor_db, tmp_path)
    assert "No drift reference" in [alert.title for alert in alerter.sent]


def test_reviewer_precision_reads_the_owners_reviews(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    fill_archive(tmp_path)
    db.execute(
        "INSERT INTO reviews (rev_id, decision, reviewed_at) VALUES "
        "(10, 'vandalism', %s), (11, 'ok', %s), (12, 'skip', %s)",
        (NOW, NOW, NOW),
    )
    db.commit()
    report, _ = run(monitor_db, tmp_path)
    assert report.reviewers["reviewed_total"] == 3
    assert report.reviewers["decisions"] == {"vandalism": 1, "ok": 1, "skip": 1}


def test_a_bad_promotion_is_rolled_back_after_500_labels(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    promoted_at = NOW - timedelta(days=6)
    # Version 2 took over six days ago and ranks no better than chance.
    fill_archive(tmp_path, bad_after=promoted_at)
    db.execute(
        "INSERT INTO retrain_runs (trigger, status, details) VALUES ('drift', 'promoted', %s)",
        (
            Jsonb(
                {
                    "promoted_at": promoted_at.isoformat(),
                    "aliases_before": {"champion": "1"},
                    "aliases_after": {"champion": "2"},
                }
            ),
        ),
    )
    db.commit()
    report, alerter = run(monitor_db, tmp_path)
    assert report.post_check is not None and report.post_check["rollback"]
    assert ("rollback", "monitor") in [(kind, by) for kind, by, _ in requests(db)]
    assert "Rolling back" in [alert.title for alert in alerter.sent]
    # Decided once: the next run doesn't ask again.
    run(monitor_db, tmp_path)
    assert [kind for kind, _, _ in requests(db)].count("rollback") == 1


def test_charts_come_from_the_report(
    tmp_path: Path, db: psycopg.Connection[Any], monitor_db: psycopg.Connection[Any]
) -> None:
    # Charts are drawn only when asked, from the trainer's tools, so the job doesn't need them.
    pytest.importorskip("matplotlib")
    fill_archive(tmp_path)
    report, _ = run(monitor_db, tmp_path)
    saved = save_charts(report, tmp_path / "images")
    assert [path.name for path in saved] == [
        "monitor_live_pr_auc_2026-11-20.png",
        "monitor_drift_2026-11-20.png",
    ]
