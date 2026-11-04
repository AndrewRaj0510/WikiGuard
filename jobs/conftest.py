"""A migrated throwaway database for job tests. Skipped unless TEST_DATABASE_URL is set."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config

URL = os.environ.get("TEST_DATABASE_URL", "")
INI = Path(__file__).parent / "migrate" / "alembic.ini"
ROLES = (
    "wikiguard_api",
    "wikiguard_scorer",
    "wikiguard_labeler",
    "wikiguard_trainer",
    "wikiguard_maintenance",
)
TABLES = (
    "audit_log",
    "reviews",
    "explanations",
    "labels",
    "scores",
    "edits",
)


@pytest.fixture
def migrated_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Migrate if needed. test_migrations.py resets the schema, so this can't assume it's there."""
    if not URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    with psycopg.connect(URL, autocommit=True) as connection:
        row = connection.execute("SELECT to_regclass('public.audit_log')").fetchone()
        if row is not None and row[0] is not None:
            return URL
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
        for role in ROLES:
            connection.execute(
                f"DO $$ BEGIN IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
                f"THEN DROP OWNED BY {role}; DROP ROLE {role}; END IF; END $$"
            )
    monkeypatch.setenv("DATABASE_MIGRATIONS_URL", URL)
    command.upgrade(Config(str(INI)), "head")
    return URL


@pytest.fixture
def db(migrated_url: str) -> Iterator[psycopg.Connection[Any]]:
    """A clean set of tables, and a connection that doesn't commit unless the code does."""
    with psycopg.connect(migrated_url, autocommit=True) as cleaner:
        # audit_log is append-only for DELETE, but TRUNCATE is a test-only reset.
        cleaner.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    with psycopg.connect(migrated_url) as connection:
        yield connection


def add_edit(
    connection: psycopg.Connection[Any],
    rev_id: int,
    edited_at: Any,
    text: str | None = "words",
) -> None:
    connection.execute(
        """
        INSERT INTO edits (rev_id, schema_version, title, edited_at, is_minor, is_anonymous,
                           user_hash, comment, added_text, removed_text)
        VALUES (%s, 1, 'Moon', %s, false, false, 'hash', 'a comment', %s, %s)
        """,
        (rev_id, edited_at, text, text),
    )
    connection.commit()
