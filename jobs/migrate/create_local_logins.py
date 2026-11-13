"""Docker Compose only: give each service a real least-privilege login on the local Postgres.

On Supabase the owner creates these by hand (README.md), so no password passes through the repo.
Locally they use COMPOSE_LOGIN_PASSWORD, so Compose runs with the same roles production will. It
refuses any host but Compose's own, so it can't be pointed at Supabase by mistake.
"""

import os
import sys
from urllib.parse import urlparse

import psycopg
from psycopg import sql

COMPOSE_HOST = "postgres"
LOGINS = {
    "api_login": "wikiguard_api",
    "scorer_login": "wikiguard_scorer",
    "labeler_login": "wikiguard_labeler",
    "trainer_login": "wikiguard_trainer",
    "maintenance_login": "wikiguard_maintenance",
    "backup_login": "wikiguard_backup",
    "observer_login": "wikiguard_observer",
}


def main() -> int:
    url = os.environ["DATABASE_MIGRATIONS_URL"]
    if urlparse(url).hostname != COMPOSE_HOST:
        print("This is for Docker Compose's local Postgres only.", file=sys.stderr)
        return 2
    password = os.environ["COMPOSE_LOGIN_PASSWORD"]
    with psycopg.connect(url, autocommit=True) as connection:
        for login, role in LOGINS.items():
            exists = connection.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s", (login,)
            ).fetchone()
            action = "ALTER" if exists else "CREATE"
            connection.execute(
                sql.SQL("{} ROLE {} LOGIN PASSWORD {}").format(
                    sql.SQL(action), sql.Identifier(login), sql.Literal(password)
                )
            )
            connection.execute(
                sql.SQL("GRANT {} TO {}").format(sql.Identifier(role), sql.Identifier(login))
            )
    print(f"Local logins ready: {sorted(LOGINS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
