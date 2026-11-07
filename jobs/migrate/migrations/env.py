"""Alembic's entry point. Migrations are plain SQL, so no models need importing here."""

from alembic import context
from sqlalchemy import create_engine, make_url, pool, text
from wikiguard_core.config import ConfigError, MigrationSettings, load_settings


def database_url() -> str:
    """A single-session connection: the direct URL or session pooler, not the transaction pooler."""
    try:
        settings = load_settings(MigrationSettings)
    except ConfigError as error:
        raise SystemExit(str(error)) from error
    # SQLAlchemy has to be told to use psycopg 3 rather than the older driver.
    url = settings.database_migrations_url.get_secret_value()
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run() -> None:
    url = database_url()
    # A Supabase username ends in the project ref, so this shows staging or prod before anything
    # changes. It leaves out the password.
    print(f"Migrating as {make_url(url).username} on {make_url(url).host}")
    engine = create_engine(url, poolclass=pool.NullPool, connect_args={"connect_timeout": 10})
    with engine.connect() as connection:
        # So a migration can't wait forever behind a long-running query. Set here rather than as a
        # startup option, which poolers may refuse.
        connection.execute(text("SET lock_timeout = '30s'"))
        connection.commit()
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
