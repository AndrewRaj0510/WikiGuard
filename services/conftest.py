"""Real-Postgres fixtures for services: the same ones the jobs use, plus a login for one role.

Tests connect as a real least-privilege login, not the database owner, so a missing grant fails here
rather than on Supabase.
"""

import importlib.util
from pathlib import Path

import psycopg
import pytest

spec = importlib.util.spec_from_file_location(
    "jobs_conftest", Path(__file__).parents[1] / "jobs" / "conftest.py"
)
assert spec is not None and spec.loader is not None  # noqa: S101 - the file is in this repo
jobs_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs_conftest)

migrated_url = jobs_conftest.migrated_url
db = jobs_conftest.db
add_edit = jobs_conftest.add_edit
LOGIN_PASSWORD = "test-only-not-a-secret"  # noqa: S105 - a throwaway test login


def login_for(owner_url: str, role: str) -> str:
    """A login in that role, created if needed, and the URL to connect as it."""
    login = f"test_login_{role.removeprefix('wikiguard_')}"
    info = psycopg.conninfo.conninfo_to_dict(owner_url)
    with psycopg.connect(owner_url, autocommit=True) as connection:
        connection.execute(
            f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{login}') THEN "
            f"CREATE ROLE {login} LOGIN PASSWORD '{LOGIN_PASSWORD}'; END IF; END $$"
        )
        connection.execute(f"GRANT {role} TO {login}")
    return psycopg.conninfo.make_conninfo(owner_url, user=login, password=LOGIN_PASSWORD) or str(
        info
    )


@pytest.fixture
def scorer_url(migrated_url: str) -> str:
    return login_for(migrated_url, "wikiguard_scorer")


@pytest.fixture
def api_url(migrated_url: str) -> str:
    return login_for(migrated_url, "wikiguard_api")
