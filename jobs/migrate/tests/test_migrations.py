"""Migrations against real Postgres. Skipped unless TEST_DATABASE_URL names a throwaway database.

Each test starts from an empty schema, so never point this at Supabase or anything you want to keep.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config

pytestmark = pytest.mark.integration

URL = os.environ.get("TEST_DATABASE_URL", "")
INI = Path(__file__).parents[1] / "alembic.ini"
EXPECTED_TABLES = {
    "edits",
    "scores",
    "labels",
    "model_versions",
    "drift_reports",
    "live_metrics",
    "retrain_runs",
    "explanations",
    "reviews",
    "retrain_requests",
    "llm_usage",
    "audit_log",
    "active_session",
}
ROLES = {
    "wikiguard_api",
    "wikiguard_scorer",
    "wikiguard_labeler",
    "wikiguard_trainer",
    "wikiguard_maintenance",
}


def alembic_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_MIGRATIONS_URL", URL)
    return Config(str(INI))


def public_tables(connection: psycopg.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
        "AND tablename <> 'alembic_version'"
    ).fetchall()
    return {row[0] for row in rows}


def reset(connection: psycopg.Connection) -> None:
    connection.execute("DROP SCHEMA public CASCADE")
    connection.execute("CREATE SCHEMA public")
    for role in ROLES:
        connection.execute(
            f"DO $$ BEGIN IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN "
            f"DROP OWNED BY {role}; DROP ROLE {role}; END IF; END $$"
        )


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[Config]:
    if not URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    with psycopg.connect(URL, autocommit=True) as connection:
        reset(connection)
    yield alembic_config(monkeypatch)
    with psycopg.connect(URL, autocommit=True) as connection:
        reset(connection)


def migrated(database: Config) -> psycopg.Connection:
    command.upgrade(database, "head")
    return psycopg.connect(URL, autocommit=True)


def test_up_creates_every_table_and_down_removes_them(database: Config) -> None:
    with migrated(database) as connection:
        assert public_tables(connection) == EXPECTED_TABLES

    command.downgrade(database, "base")

    with psycopg.connect(URL, autocommit=True) as connection:
        assert public_tables(connection) == set()
        assert not connection.execute(
            "SELECT rolname FROM pg_roles WHERE rolname LIKE 'wikiguard\\_%'"
        ).fetchall()


def test_up_down_up_works(database: Config) -> None:
    command.upgrade(database, "head")
    command.downgrade(database, "base")
    command.upgrade(database, "head")

    with psycopg.connect(URL, autocommit=True) as connection:
        assert public_tables(connection) == EXPECTED_TABLES


def test_every_table_has_row_level_security(database: Config) -> None:
    """The check the plan asks CI to run. It covers any table added later too."""
    with migrated(database) as connection:
        rows = connection.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND NOT c.relrowsecurity "
            "AND c.relname <> 'alembic_version'"
        ).fetchall()

    assert rows == []


def as_role(connection: psycopg.Connection, role: str) -> None:
    connection.execute(f"SET ROLE {role}")


def add_edit(connection: psycopg.Connection, rev_id: int = 1) -> None:
    connection.execute(
        "INSERT INTO edits (rev_id, schema_version, title, edited_at, is_minor, is_anonymous) "
        "VALUES (%s, 1, 'Moon', now(), false, false)",
        (rev_id,),
    )


@pytest.mark.parametrize(
    ("role", "statement"),
    [
        ("wikiguard_api", "UPDATE labels SET label = 1"),
        ("wikiguard_api", "INSERT INTO labels (rev_id, label) VALUES (1, 1)"),
        ("wikiguard_api", "UPDATE model_versions SET alias = 'champion'"),
        ("wikiguard_api", "DELETE FROM edits"),
        ("wikiguard_api", "UPDATE audit_log SET actor = 'x'"),
        ("wikiguard_scorer", "SELECT * FROM reviews"),
        ("wikiguard_scorer", "DELETE FROM edits"),
        ("wikiguard_labeler", "UPDATE edits SET title = 'x'"),
        ("wikiguard_labeler", "SELECT * FROM active_session"),
        ("wikiguard_trainer", "SELECT * FROM edits"),
        ("wikiguard_trainer", "DELETE FROM model_versions"),
        ("wikiguard_maintenance", "SELECT * FROM reviews"),
        ("wikiguard_maintenance", "DELETE FROM audit_log"),
    ],
)
def test_roles_are_refused_outside_their_job(database: Config, role: str, statement: str) -> None:
    with migrated(database) as connection:
        as_role(connection, role)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(statement)


@pytest.mark.parametrize(
    ("role", "statement"),
    [
        ("wikiguard_api", "SELECT * FROM labels"),
        ("wikiguard_api", "INSERT INTO reviews (rev_id, decision) VALUES (1, 'unsure')"),
        ("wikiguard_labeler", "INSERT INTO labels (rev_id, label) VALUES (1, 1)"),
        ("wikiguard_scorer", "UPDATE edits SET title = 'x'"),
        ("wikiguard_maintenance", "DELETE FROM edits"),
    ],
)
def test_roles_can_do_their_job_through_rls(database: Config, role: str, statement: str) -> None:
    with migrated(database) as connection:
        add_edit(connection)
        as_role(connection, role)
        connection.execute(statement)


def test_a_role_without_a_policy_sees_no_rows(database: Config) -> None:
    """Stands in for Supabase's anon role, in case a grant is ever added by mistake."""
    with migrated(database) as connection:
        add_edit(connection)
        connection.execute("CREATE ROLE wikiguard_outsider NOLOGIN")
        connection.execute("GRANT USAGE ON SCHEMA public TO wikiguard_outsider")
        connection.execute("GRANT SELECT ON edits TO wikiguard_outsider")
        try:
            as_role(connection, "wikiguard_outsider")
            assert connection.execute("SELECT * FROM edits").fetchall() == []
        finally:
            connection.execute("RESET ROLE")
            connection.execute("DROP OWNED BY wikiguard_outsider")
            connection.execute("DROP ROLE wikiguard_outsider")


def test_audit_log_refuses_changes_even_from_the_owner(database: Config) -> None:
    with migrated(database) as connection:
        connection.execute("INSERT INTO audit_log (actor, action) VALUES ('t', 't')")
        with pytest.raises(psycopg.errors.RaiseException):
            connection.execute("DELETE FROM audit_log")
        with pytest.raises(psycopg.errors.RaiseException):
            connection.execute("UPDATE audit_log SET actor = 'x'")


def test_the_seat_table_allows_one_row(database: Config) -> None:
    with migrated(database) as connection:
        take_seat = (
            "INSERT INTO active_session (session_id, role, tab_id, lease_expires_at) "
            "VALUES (%s, 'guest', 't', now() + interval '45 seconds')"
        )
        connection.execute(take_seat, ("first",))
        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(take_seat, ("second",))
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                "INSERT INTO active_session (seat, session_id, role, tab_id, lease_expires_at) "
                "VALUES (2, 'x', 'guest', 't', now())"
            )


def test_a_label_is_either_a_value_or_an_exclusion(database: Config) -> None:
    with migrated(database) as connection:
        add_edit(connection)
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute("INSERT INTO labels (rev_id) VALUES (1)")
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                "INSERT INTO labels (rev_id, label, excluded_reason) VALUES (1, 1, 'hidden')"
            )
        connection.execute("INSERT INTO labels (rev_id, excluded_reason) VALUES (1, 'hidden')")


def test_scores_must_be_probabilities(database: Config) -> None:
    with migrated(database) as connection:
        add_edit(connection)
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                "INSERT INTO scores (rev_id, model_name, model_version, score) "
                "VALUES (1, 'mlp', '1', 1.5)"
            )


def test_queue_query_uses_its_index(database: Config) -> None:
    with migrated(database) as connection:
        connection.execute("SET enable_seqscan = off")
        plan = connection.execute(
            "EXPLAIN SELECT rev_id FROM scores ORDER BY score DESC, rev_id DESC LIMIT 50"
        ).fetchall()

    assert "scores_queue" in str(plan)
