"""Two read-only roles: one for the nightly backup, one for Grafana's dashboards.

wikiguard_backup (jobs/backup, section 14 of the plan): only what can't be rebuilt is backed up:
reviews, the model history, retraining, the audit log, usage counters, and the daily rollups.
Edits, scores, and labels come back from the parquet archive, and sessions are meant to be lost.
So the role reads exactly those tables and nothing else, and a leaked backup login can't change
data or read diff text.

wikiguard_observer (Grafana, DEC-085): the Security and Free-tier Usage dashboards need what only
the API knows, such as sign-ins, denied actions, and LLM calls, and the API runs on Render, which
Prometheus must not scrape (F-015). The audit log and the usage counter already hold it, so
Grafana reads them through this role. No diff text, sessions, or hashed client addresses.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

BACKUP = "wikiguard_backup"
# Must match BACKUP_TABLES in jobs/backup/backup.py, which a test checks.
TABLES = (
    "reviews",
    "model_versions",
    "retrain_runs",
    "retrain_requests",
    "audit_log",
    "llm_usage",
    "live_metrics",
    "drift_reports",
)
OBSERVER = "wikiguard_observer"
OBSERVER_TABLES = (
    "audit_log",
    "llm_usage",
    "retrain_requests",
    "retrain_runs",
    "model_versions",
    "live_metrics",
)


def create_role(role: str) -> None:
    op.execute(
        f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
                CREATE ROLE {role} NOLOGIN;
            END IF;
        END $$;
        GRANT USAGE ON SCHEMA public TO {role};
        """
    )


def grant_reads(role: str, tables: tuple[str, ...]) -> None:
    for table in tables:
        op.execute(f"GRANT SELECT ON {table} TO {role}")
        # Read only: no WITH CHECK, because the role has no write grant to check.
        op.execute(f"CREATE POLICY {role}_access ON {table} FOR SELECT TO {role} USING (true)")


def upgrade() -> None:
    create_role(BACKUP)
    grant_reads(BACKUP, TABLES)
    # The schema's revision goes in the backup's manifest, so a restore can refuse an older schema.
    op.execute(f"GRANT SELECT ON alembic_version TO {BACKUP}")
    op.execute(
        f"CREATE POLICY {BACKUP}_access ON alembic_version FOR SELECT TO {BACKUP} USING (true)"
    )
    create_role(OBSERVER)
    grant_reads(OBSERVER, OBSERVER_TABLES)


def revoke_reads(role: str, tables: tuple[str, ...]) -> None:
    for table in tables:
        op.execute(f"DROP POLICY {role}_access ON {table}")
        op.execute(f"REVOKE ALL ON {table} FROM {role}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {role}")
    op.execute(f"DROP ROLE {role}")


def downgrade() -> None:
    op.execute(f"DROP POLICY {BACKUP}_access ON alembic_version")
    op.execute(f"REVOKE ALL ON alembic_version FROM {BACKUP}")
    revoke_reads(OBSERVER, OBSERVER_TABLES)
    revoke_reads(BACKUP, TABLES)
