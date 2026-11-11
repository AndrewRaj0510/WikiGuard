"""What Phase 5's monitor, trainer, and shadow scoring need.

- shadow_scores holds a candidate's scores on live edits during its shadow period. It's a separate
  table so a shadow score can never reach the queue, the feed, or /predict, which read scores.
  Rows go with their edit (ON DELETE CASCADE), so retention needs no change.
- retrain_requests gets a shadow_check kind: the poller adds one when a shadow period ends, so the
  check and its promotion go through the same claimed, audited path as every other model change.
- The monitor runs as the trainer role. It reads reviews for reviewer precision, and asks for a
  retrain or a rollback by adding a request, the way the API does, so the poller stays the only
  thing that starts the trainer.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

NEW_TABLE = "shadow_scores"
NEW_GRANTS = [
    ("wikiguard_scorer", "SELECT, INSERT", "shadow_scores"),
    ("wikiguard_trainer", "SELECT", "shadow_scores"),
    ("wikiguard_trainer", "SELECT", "reviews"),
    ("wikiguard_trainer", "INSERT", "retrain_requests"),
]
KINDS = "('retrain', 'promote', 'rollback', 'shadow_check')"
OLD_KINDS = "('retrain', 'promote', 'rollback')"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE shadow_scores (
            rev_id        bigint NOT NULL REFERENCES edits (rev_id) ON DELETE CASCADE,
            model_name    text NOT NULL,
            model_version text NOT NULL,
            -- NULL with an error when the candidate failed on this edit, so errors are counted.
            score         real CHECK (score >= 0 AND score <= 1),
            error         text,
            latency_ms    real,
            -- What the serving chain gave the same edit, so the two can be compared on one edit.
            served_score  real CHECK (served_score >= 0 AND served_score <= 1),
            scored_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (rev_id, model_version),
            CHECK ((score IS NULL) <> (error IS NULL))
        );
        CREATE INDEX shadow_scores_version ON shadow_scores (model_version, scored_at);
        """
    )
    op.execute(f"REVOKE ALL ON {NEW_TABLE} FROM PUBLIC")
    op.execute(
        f"""
        DO $$
        DECLARE r text;
        BEGIN
            FOREACH r IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
                IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('REVOKE ALL ON {NEW_TABLE} FROM %I', r);
                END IF;
            END LOOP;
        END $$;
        """
    )
    op.execute(f"ALTER TABLE {NEW_TABLE} ENABLE ROW LEVEL SECURITY")
    for role, privileges, table in NEW_GRANTS:
        op.execute(f"GRANT {privileges} ON {table} TO {role}")
    # One policy per role and table. The trainer already has one on retrain_requests (0002).
    op.execute(
        "CREATE POLICY wikiguard_scorer_access ON shadow_scores TO wikiguard_scorer "
        "USING (true) WITH CHECK (true)"
    )
    op.execute(
        "CREATE POLICY wikiguard_trainer_access ON shadow_scores TO wikiguard_trainer "
        "USING (true) WITH CHECK (true)"
    )
    op.execute(
        "CREATE POLICY wikiguard_trainer_access ON reviews TO wikiguard_trainer "
        "USING (true) WITH CHECK (true)"
    )
    op.execute(
        f"""
        ALTER TABLE retrain_requests DROP CONSTRAINT retrain_requests_kind_check;
        ALTER TABLE retrain_requests ADD CONSTRAINT retrain_requests_kind_check
            CHECK (kind IN {KINDS});
        """
    )


def downgrade() -> None:
    op.execute(
        f"""
        DELETE FROM retrain_requests WHERE kind = 'shadow_check';
        ALTER TABLE retrain_requests DROP CONSTRAINT retrain_requests_kind_check;
        ALTER TABLE retrain_requests ADD CONSTRAINT retrain_requests_kind_check
            CHECK (kind IN {OLD_KINDS});
        """
    )
    op.execute("DROP POLICY wikiguard_trainer_access ON reviews")
    for role, privileges, table in NEW_GRANTS:
        op.execute(f"REVOKE {privileges} ON {table} FROM {role}")
    op.execute(f"DROP TABLE {NEW_TABLE}")
