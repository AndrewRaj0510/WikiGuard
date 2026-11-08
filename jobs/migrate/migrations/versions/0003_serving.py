"""What serving needs: score details, the queue's columns, server-side sessions, and request kinds.

- edits.latest_* holds the score the queue ranks by. An edit can have several scores, one per model
  version, so sorting scores directly would list it twice and paging would need DISTINCT.
- reviews takes the plan's four decisions, with one review per edit, so a second click edits it.
- sessions is separate from active_session because the gate can be switched off (GATE_ENABLED)
  while sign-in still has to work, and because a cookie must be refused after a session has ended.
- retrain_requests gets a kind, because promote and rollback also have to go through the database:
  Render holds no cluster credentials and only a read-only registry token.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

NEW_TABLES = ["sessions", "login_throttle"]
NEW_GRANTS = [
    ("wikiguard_api", "SELECT, INSERT, UPDATE", "sessions"),
    ("wikiguard_api", "SELECT, INSERT, UPDATE, DELETE", "login_throttle"),
    ("wikiguard_maintenance", "SELECT, DELETE", "sessions"),
]


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE scores
            ADD COLUMN alias text,
            ADD COLUMN threshold real,
            ADD COLUMN latency_ms real,
            -- The scorer was behind and used a faster model on purpose.
            ADD COLUMN load_shed boolean NOT NULL DEFAULT false,
            ADD COLUMN forced boolean NOT NULL DEFAULT false;

        ALTER TABLE edits
            ADD COLUMN latest_score real CHECK (latest_score >= 0 AND latest_score <= 1),
            ADD COLUMN latest_model_version text,
            ADD COLUMN latest_scored_at timestamptz;

        ALTER TABLE reviews DROP CONSTRAINT reviews_decision_check;
        UPDATE reviews SET decision = CASE decision
            WHEN 'not_vandalism' THEN 'ok' WHEN 'unsure' THEN 'skip' ELSE decision END;
        ALTER TABLE reviews ADD CONSTRAINT reviews_decision_check
            CHECK (decision IN ('vandalism', 'good_faith_mistake', 'ok', 'skip'));
        DELETE FROM reviews a USING reviews b WHERE a.rev_id = b.rev_id AND a.id < b.id;
        DROP INDEX reviews_rev_id;
        ALTER TABLE reviews ADD CONSTRAINT reviews_one_per_edit UNIQUE (rev_id);
        ALTER TABLE reviews ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();

        ALTER TABLE retrain_requests
            ADD COLUMN kind text NOT NULL DEFAULT 'retrain'
                CHECK (kind IN ('retrain', 'promote', 'rollback')),
            ADD COLUMN requested_by text NOT NULL DEFAULT 'unknown',
            ADD COLUMN finished_at timestamptz,
            ADD COLUMN details jsonb NOT NULL DEFAULT '{}';

        CREATE TABLE sessions (
            -- An HMAC of the cookie, so a database leak can't be replayed as a sign-in.
            id                text PRIMARY KEY,
            role              text NOT NULL CHECK (role IN ('guest', 'user')),
            csrf_token        text NOT NULL,
            -- An HMAC of the IP, only for the wait between guest sign-ins.
            client_hash       text NOT NULL,
            created_at        timestamptz NOT NULL DEFAULT now(),
            last_activity_at  timestamptz NOT NULL DEFAULT now(),
            expires_at        timestamptz NOT NULL,
            ended_at          timestamptz,
            ended_reason      text,
            fresh_explains    integer NOT NULL DEFAULT 0 CHECK (fresh_explains >= 0)
        );
        CREATE INDEX sessions_guest_cooldown ON sessions (client_hash, ended_at DESC)
            WHERE role = 'guest';
        CREATE INDEX sessions_created_at ON sessions (created_at);

        -- In Postgres because Render wipes memory when it sleeps, which would reset a lockout.
        CREATE TABLE login_throttle (
            scope        text PRIMARY KEY,
            failures     integer NOT NULL DEFAULT 0,
            window_start timestamptz NOT NULL DEFAULT now(),
            locked_until timestamptz
        );

        -- The queue and feed, sorted the same way as these indexes so paging can stop early.
        CREATE INDEX edits_queue ON edits (latest_score DESC, rev_id DESC)
            WHERE latest_score IS NOT NULL;
        CREATE INDEX edits_queue_by_editor ON edits (is_anonymous, latest_score DESC, rev_id DESC)
            WHERE latest_score IS NOT NULL;
        CREATE INDEX edits_unscored ON edits (edited_at) WHERE latest_score IS NULL;
        CREATE INDEX edits_feed ON edits (edited_at DESC, rev_id DESC);
        -- Replaced by the ones above. Each index costs space on a 500 MB database.
        DROP INDEX edits_edited_at;
        DROP INDEX scores_queue;
        """
    )
    for table in NEW_TABLES:
        op.execute(f"REVOKE ALL ON {table} FROM PUBLIC")
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
    for role, privileges, table in NEW_GRANTS:
        op.execute(f"GRANT {privileges} ON {table} TO {role}")
        op.execute(
            f"CREATE POLICY {role}_access ON {table} TO {role} USING (true) WITH CHECK (true)"
        )
    op.execute("GRANT UPDATE ON reviews TO wikiguard_api")


def downgrade() -> None:
    op.execute("REVOKE UPDATE ON reviews FROM wikiguard_api")
    for role, _, table in NEW_GRANTS:
        op.execute(f"DROP POLICY {role}_access ON {table}")
        op.execute(f"REVOKE ALL ON {table} FROM {role}")
    op.execute(
        """
        CREATE INDEX scores_queue ON scores (score DESC, rev_id DESC);
        CREATE INDEX edits_edited_at ON edits (edited_at DESC);
        DROP INDEX edits_feed;
        DROP INDEX edits_unscored;
        DROP INDEX edits_queue_by_editor;
        DROP INDEX edits_queue;
        DROP TABLE login_throttle;
        DROP TABLE sessions;

        ALTER TABLE retrain_requests
            DROP COLUMN details,
            DROP COLUMN finished_at,
            DROP COLUMN requested_by,
            DROP COLUMN kind;

        ALTER TABLE reviews DROP COLUMN updated_at;
        ALTER TABLE reviews DROP CONSTRAINT reviews_one_per_edit;
        CREATE INDEX reviews_rev_id ON reviews (rev_id);
        ALTER TABLE reviews DROP CONSTRAINT reviews_decision_check;
        UPDATE reviews SET decision = CASE decision
            WHEN 'ok' THEN 'not_vandalism' WHEN 'skip' THEN 'unsure'
            WHEN 'good_faith_mistake' THEN 'not_vandalism' ELSE decision END;
        ALTER TABLE reviews ADD CONSTRAINT reviews_decision_check
            CHECK (decision IN ('vandalism', 'not_vandalism', 'unsure'));

        ALTER TABLE edits
            DROP COLUMN latest_scored_at,
            DROP COLUMN latest_model_version,
            DROP COLUMN latest_score;

        ALTER TABLE scores
            DROP COLUMN forced,
            DROP COLUMN load_shed,
            DROP COLUMN latency_ms,
            DROP COLUMN threshold,
            DROP COLUMN alias;
        """
    )
