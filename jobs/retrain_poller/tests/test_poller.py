"""The API's only way into the cluster, against real Postgres as the real trainer role."""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from poller import command_launcher, not_built_launcher, run_once

pytestmark = pytest.mark.integration


def add_request(db: psycopg.Connection[Any], kind: str = "retrain", status: str = "pending") -> int:
    row = db.execute(
        "INSERT INTO retrain_requests (kind, reason, requested_by, status) "
        "VALUES (%s, 'test', 'user', %s) RETURNING id",
        (kind, status),
    ).fetchone()
    db.commit()
    assert row is not None
    return int(row[0])


def status_of(db: psycopg.Connection[Any], request_id: int) -> tuple[Any, ...]:
    row = db.execute(
        "SELECT status, details, picked_up_at IS NOT NULL, finished_at IS NOT NULL "
        "FROM retrain_requests WHERE id = %s",
        (request_id,),
    ).fetchone()
    db.commit()
    assert row is not None
    return tuple(row)


@pytest.fixture
def trainer(db: psycopg.Connection[Any], migrated_url: str) -> Iterator[psycopg.Connection[Any]]:
    """Connected as the role the job really uses, so a missing grant fails here."""
    with psycopg.connect(migrated_url, autocommit=True) as owner:
        owner.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = "
            "'test_login_trainer') THEN CREATE ROLE test_login_trainer LOGIN "
            "PASSWORD 'test-only-not-a-secret'; END IF; END $$"
        )
        owner.execute("GRANT wikiguard_trainer TO test_login_trainer")
    url = psycopg.conninfo.make_conninfo(
        migrated_url, user="test_login_trainer", password="test-only-not-a-secret"
    )
    with psycopg.connect(url) as connection:
        yield connection


def test_nothing_waiting_returns_none(trainer: psycopg.Connection[Any]) -> None:
    assert run_once(trainer, {}) is None


def test_a_retrain_runs_and_is_recorded(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    request_id = add_request(db)
    seen: list[dict[str, Any]] = []
    result = run_once(trainer, {"retrain": seen.append})
    assert result is not None and result["status"] == "done"
    assert seen[0]["id"] == request_id
    assert status_of(db, request_id) == ("done", {}, True, True)
    run = db.execute("SELECT trigger, status FROM retrain_runs").fetchone()
    db.commit()
    assert run == ("manual", "done")


def test_a_failing_trainer_is_recorded_as_failed(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    request_id = add_request(db)

    def boom(_: dict[str, Any]) -> None:
        raise RuntimeError("out of memory")

    run_once(trainer, {"retrain": boom})
    status, details, _, _ = status_of(db, request_id)
    assert status == "failed"
    assert details["error"] == "RuntimeError"
    run = db.execute("SELECT status FROM retrain_runs").fetchone()
    db.commit()
    assert run == ("failed",)


def test_only_one_retrain_a_day(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    first = add_request(db)
    run_once(trainer, {"retrain": lambda _: None})
    second = add_request(db)
    calls: list[int] = []
    run_once(trainer, {"retrain": lambda r: calls.append(r["id"])})
    assert calls == []
    assert status_of(db, first)[0] == "done"
    assert status_of(db, second)[0] == "failed"
    assert "one retrain per" in status_of(db, second)[1]["error"]


@pytest.mark.parametrize("kind", ["promote", "rollback"])
def test_promote_and_rollback_say_they_arent_built_yet(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any], kind: str
) -> None:
    request_id = add_request(db, kind)
    run_once(trainer, {"retrain": lambda _: None})
    status, details, _, _ = status_of(db, request_id)
    assert status == "failed"
    assert "Phase 5" in details["message"]


def test_a_claimed_row_isnt_taken_twice(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any], migrated_url: str
) -> None:
    add_request(db)
    other = psycopg.connect(migrated_url)
    try:
        # Another run holds this row's lock mid-claim.
        other.execute("SELECT id FROM retrain_requests FOR UPDATE")
        assert run_once(trainer, {"retrain": lambda _: None}) is None
    finally:
        other.rollback()
        other.close()


def test_a_stuck_run_is_failed_after_the_limit(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    request_id = add_request(db, status="running")
    db.execute(
        "UPDATE retrain_requests SET picked_up_at = now() - interval '13 hours' WHERE id = %s",
        (request_id,),
    )
    db.commit()
    run_once(trainer, {})
    assert status_of(db, request_id)[0] == "failed"


def test_a_running_retrain_blocks_another(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    running = add_request(db, status="running")
    db.execute(
        "UPDATE retrain_requests SET picked_up_at = now() WHERE id = %s",
        (running,),
    )
    db.commit()
    waiting = add_request(db)
    run_once(trainer, {"retrain": lambda _: None})
    assert status_of(db, waiting)[0] == "failed"


def test_the_oldest_request_goes_first(
    db: psycopg.Connection[Any], trainer: psycopg.Connection[Any]
) -> None:
    first = add_request(db, "rollback")
    add_request(db, "promote")
    result = run_once(trainer, {})
    assert result is not None and result["id"] == first


def test_the_launcher_uses_no_shell() -> None:
    launch = command_launcher("python -c pass")
    launch({"reason": "; rm -rf /"})


def test_no_trainer_command_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="TRAINER_COMMAND"):
        command_launcher(None)({})


def test_unbuilt_kinds_name_themselves() -> None:
    with pytest.raises(RuntimeError, match="rollback"):
        not_built_launcher({"kind": "rollback"})
