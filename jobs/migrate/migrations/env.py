"""Alembic's entry point. Migrations are plain SQL, so no models need importing here."""

import os

from alembic import context
from sqlalchemy import create_engine, pool


def database_url() -> str:
    """A single-session connection: the direct URL or session pooler, not the transaction pooler."""
    url = os.environ.get("DATABASE_MIGRATIONS_URL")
    if not url:
        raise SystemExit("DATABASE_MIGRATIONS_URL is not set")
    # SQLAlchemy has to be told to use psycopg 3 rather than the older driver.
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run() -> None:
    # A timeout, so a migration can't wait forever behind a long-running query on a lock.
    engine = create_engine(
        database_url(),
        poolclass=pool.NullPool,
        connect_args={"options": "-c lock_timeout=30s"},
    )
    with engine.connect() as connection:
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
