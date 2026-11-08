from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from conftest import add_edit
from retention import EMERGENCY, LIMIT_BYTES, NORMAL, SHRINK, WARNING, run, windows_for

NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)


def ago(**delta: float) -> datetime:
    return NOW - timedelta(**delta)


def add_score(db: psycopg.Connection[Any], rev_id: int, score: float) -> None:
    db.execute(
        "INSERT INTO scores (rev_id, model_name, model_version, score) VALUES (%s, 'm', '1', %s)",
        (rev_id, score),
    )
    db.commit()


def text_of(db: psycopg.Connection[Any], rev_id: int) -> str | None:
    row = db.execute("SELECT added_text FROM edits WHERE rev_id = %s", (rev_id,)).fetchone()
    assert row is not None
    return row[0]


def rows_left(db: psycopg.Connection[Any]) -> set[int]:
    return {row[0] for row in db.execute("SELECT rev_id FROM edits").fetchall()}


@pytest.mark.parametrize(
    ("share", "windows"),
    [(0.1, NORMAL), (0.7, WARNING), (0.85, SHRINK), (0.95, EMERGENCY), (1.2, EMERGENCY)],
)
def test_windows_shorten_as_the_database_fills(share: float, windows: object) -> None:
    assert windows_for(int(LIMIT_BYTES * share)) is windows


@pytest.mark.parametrize("windows", [NORMAL, WARNING, SHRINK, EMERGENCY])
def test_no_level_can_reach_edits_still_waiting_for_labels(windows: Any) -> None:
    assert windows.text >= timedelta(hours=48)
    assert windows.high_risk_text >= timedelta(hours=48)
    assert windows.rows >= timedelta(hours=60)


@pytest.mark.integration
def test_normal_rules(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, ago(hours=47))
    add_edit(db, 2, ago(hours=49))
    add_edit(db, 3, ago(days=2, hours=12))
    add_score(db, 3, 0.9)
    add_edit(db, 4, ago(days=2, hours=12))
    add_score(db, 4, 0.1)
    add_edit(db, 5, ago(days=3, hours=1))

    results = run(db, NOW)

    assert results["level"] == "normal"
    assert text_of(db, 1) == "words"
    assert text_of(db, 2) is None
    assert text_of(db, 3) == "words"
    assert text_of(db, 4) is None
    assert rows_left(db) == {1, 2, 3, 4}


@pytest.mark.integration
def test_high_risk_text_also_goes_after_its_window(
    db: psycopg.Connection[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under normal rules the row goes at the same moment, so this needs the shrink level.
    monkeypatch.setattr("retention.windows_for", lambda *_: SHRINK)
    add_edit(db, 1, ago(hours=49))
    add_score(db, 1, 0.9)
    add_edit(db, 2, ago(hours=47))
    add_score(db, 2, 0.9)

    run(db, NOW)

    assert text_of(db, 1) is None
    assert text_of(db, 2) == "words"


@pytest.mark.integration
def test_deleting_an_edit_takes_its_score_and_label(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, ago(days=6))
    add_score(db, 1, 0.5)
    db.execute("INSERT INTO labels (rev_id, label) VALUES (1, 1)")
    db.execute("INSERT INTO reviews (rev_id, decision) VALUES (1, 'vandalism')")
    db.commit()

    run(db, NOW)

    assert db.execute("SELECT count(*) FROM scores").fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM labels").fetchone() == (0,)
    # Reviews stay forever, as the plan says.
    assert db.execute("SELECT count(*) FROM reviews").fetchone() == (1,)


@pytest.mark.integration
def test_a_nearly_full_database_shortens_the_windows_but_keeps_unlabeled_edits(
    db: psycopg.Connection[Any],
) -> None:
    add_edit(db, 1, ago(hours=47))
    add_edit(db, 2, ago(hours=59))
    add_score(db, 2, 0.9)
    add_edit(db, 3, ago(hours=61))

    # A limit equal to a tiny fraction of the real size means "over the emergency level".
    results = run(db, NOW, limit=1024)

    assert results["level"] == "emergency"
    assert rows_left(db) == {1, 2}
    assert text_of(db, 1) == "words"
    assert text_of(db, 2) is None


@pytest.mark.integration
def test_each_run_is_recorded(db: psycopg.Connection[Any]) -> None:
    run(db, NOW)

    entry = db.execute("SELECT action, details FROM audit_log").fetchone()
    assert entry is not None
    assert entry[0] == "retention.run"
    assert entry[1]["level"] == "normal"


@pytest.mark.integration
def test_the_maintenance_role_has_every_permission_it_needs(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, ago(days=6))
    add_edit(db, 2, ago(days=2))
    db.execute("SET ROLE wikiguard_maintenance")
    try:
        results = run(db, NOW)
    finally:
        db.rollback()
        db.execute("RESET ROLE")
    assert results["rows_deleted"] == 1


def test_old_ended_sessions_are_removed_and_live_ones_kept(db: psycopg.Connection[Any]) -> None:
    for session_id, created, ended in (
        ("old-ended", ago(days=8), ago(days=8)),
        ("recent-ended", ago(days=1), ago(days=1)),
        ("old-live", ago(days=8), None),
    ):
        db.execute(
            "INSERT INTO sessions (id, role, csrf_token, client_hash, expires_at, created_at, "
            "ended_at) VALUES (%s, 'user', 'c', 'h', %s, %s, %s)",
            (session_id, NOW, created, ended),
        )
    db.commit()

    results = run(db, NOW)

    assert results["sessions_deleted"] == 1
    rows = db.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    db.commit()
    assert [row[0] for row in rows] == ["old-live", "recent-ended"]
