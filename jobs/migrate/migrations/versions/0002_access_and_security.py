"""Reviews, the single-seat gate, usage counters, audit log, indexes, roles, and RLS.

Roles get only what they need, so a leaked credential can do limited damage. They are created
here without passwords and can't log in. In each environment the owner creates a login for each
role by hand, so no password ever passes through this repo:

    CREATE ROLE api_login LOGIN PASSWORD '...' IN ROLE wikiguard_api;

Row Level Security is a second layer. The app connects only through Postgres and the Data API is
off, but if REST were ever turned on, Supabase's anon and authenticated roles match no policy and
see nothing.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

ROLES = [
    "wikiguard_api",
    "wikiguard_scorer",
    "wikiguard_labeler",
    "wikiguard_trainer",
    "wikiguard_maintenance",
]
TABLES = [
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
]
GRANTS = {
    # The API reads almost everything, but can't touch model tables or labels (section 8).
    "wikiguard_api": {
        "SELECT": TABLES,
        "INSERT": [
            "reviews",
            "retrain_requests",
            "llm_usage",
            "audit_log",
            "active_session",
            "explanations",
        ],
        "UPDATE": ["llm_usage", "active_session"],
        "DELETE": ["active_session"],
    },
    "wikiguard_scorer": {
        "SELECT": ["edits", "scores", "model_versions"],
        "INSERT": ["edits", "scores", "audit_log"],
        "UPDATE": ["edits", "scores"],
    },
    "wikiguard_labeler": {
        "SELECT": ["edits", "labels"],
        "INSERT": ["labels", "audit_log"],
    },
    "wikiguard_trainer": {
        "SELECT": [
            "scores",
            "labels",
            "model_versions",
            "drift_reports",
            "live_metrics",
            "retrain_runs",
            "retrain_requests",
        ],
        "INSERT": ["model_versions", "drift_reports", "live_metrics", "retrain_runs", "audit_log"],
        "UPDATE": ["model_versions", "live_metrics", "retrain_runs", "retrain_requests"],
    },
    # Retention and purge: the only role that deletes data.
    "wikiguard_maintenance": {
        "SELECT": ["edits", "scores", "labels", "explanations"],
        "UPDATE": ["edits"],
        "DELETE": ["edits", "scores", "labels", "explanations"],
        "INSERT": ["audit_log"],
    },
}


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE reviews (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            -- Kept after the edit is deleted, so feedback-loop analysis still has its numbers.
            rev_id      bigint NOT NULL,
            decision    text NOT NULL CHECK (decision IN ('vandalism', 'not_vandalism', 'unsure')),
            note        text,
            reviewed_at timestamptz NOT NULL DEFAULT now()
        );

        -- Render holds no cluster credentials, so the API asks by row and the cluster polls.
        CREATE TABLE retrain_requests (
            id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            requested_at timestamptz NOT NULL DEFAULT now(),
            reason       text NOT NULL,
            status       text NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'running', 'done', 'failed')),
            picked_up_at timestamptz
        );

        -- A durable counter, so a restart can't reset the daily LLM cap.
        CREATE TABLE llm_usage (
            day      date NOT NULL,
            provider text NOT NULL,
            calls    integer NOT NULL DEFAULT 0 CHECK (calls >= 0),
            PRIMARY KEY (day, provider)
        );

        CREATE TABLE audit_log (
            id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            at         timestamptz NOT NULL DEFAULT now(),
            actor      text NOT NULL,
            action     text NOT NULL,
            target     text,
            request_id text,
            details    jsonb NOT NULL DEFAULT '{}'
        );

        -- The single seat. The primary key and CHECK allow one row, which makes taking the seat
        -- an atomic insert or update.
        CREATE TABLE active_session (
            seat             smallint PRIMARY KEY DEFAULT 1 CHECK (seat = 1),
            session_id       text NOT NULL,
            role             text NOT NULL CHECK (role IN ('guest', 'user')),
            tab_id           text NOT NULL,
            started_at       timestamptz NOT NULL DEFAULT now(),
            last_activity_at timestamptz NOT NULL DEFAULT now(),
            lease_expires_at timestamptz NOT NULL
        );

        -- Grants can be changed by whoever owns the table, so the rule is also enforced here.
        CREATE FUNCTION refuse_audit_change() RETURNS trigger
        LANGUAGE plpgsql SET search_path = '' AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only';
        END;
        $$;
        CREATE TRIGGER audit_log_append_only
            BEFORE UPDATE OR DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION refuse_audit_change();

        -- The live feed, and retention's time-based deletes.
        CREATE INDEX edits_edited_at ON edits (edited_at DESC);
        -- The review queue: riskiest first, with rev_id for keyset pagination.
        CREATE INDEX scores_queue ON scores (score DESC, rev_id DESC);
        CREATE INDEX scores_scored_at ON scores (scored_at DESC);
        -- Foreign keys need no extra index, because rev_id leads each primary key.
        CREATE INDEX reviews_rev_id ON reviews (rev_id);
        CREATE INDEX retrain_requests_pending ON retrain_requests (requested_at)
            WHERE status = 'pending';
        CREATE INDEX audit_log_at ON audit_log (at DESC);
        """
    )

    for role in ROLES:
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

    for table in TABLES:
        op.execute(f"REVOKE ALL ON {table} FROM PUBLIC")
        # Supabase's REST roles. They don't exist on plain Postgres, so check first.
        op.execute(
            f"""
            DO $$
            DECLARE r text;
            BEGIN
                FOREACH r IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
                    IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                        EXECUTE format('REVOKE ALL ON {table} FROM %I', r);
                    END IF;
                END LOOP;
            END $$;
            """
        )
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    for role, grants in GRANTS.items():
        tables_for_role: set[str] = set()
        for privilege, tables in grants.items():
            for table in tables:
                op.execute(f"GRANT {privilege} ON {table} TO {role}")
                tables_for_role.add(table)
        # RLS needs a policy to allow anything. Grants above decide what each role may do.
        for table in sorted(tables_for_role):
            op.execute(
                f"CREATE POLICY {role}_access ON {table} TO {role} USING (true) WITH CHECK (true)"
            )


def downgrade() -> None:
    for role, grants in GRANTS.items():
        tables_for_role = {table for tables in grants.values() for table in tables}
        for table in sorted(tables_for_role):
            op.execute(f"DROP POLICY {role}_access ON {table}")
            op.execute(f"REVOKE ALL ON {table} FROM {role}")
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.execute(
        """
        DROP TABLE active_session;
        DROP TABLE audit_log;
        DROP FUNCTION refuse_audit_change();
        DROP TABLE llm_usage;
        DROP TABLE retrain_requests;
        DROP TABLE reviews;
        DROP INDEX edits_edited_at;
        DROP INDEX scores_queue;
        DROP INDEX scores_scored_at;
        """
    )
    for role in ROLES:
        op.execute(f"REVOKE USAGE ON SCHEMA public FROM {role}")
        op.execute(f"DROP ROLE {role}")
