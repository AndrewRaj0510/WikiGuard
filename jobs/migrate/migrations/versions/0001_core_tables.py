"""The tables that hold edits, scores, labels, models, and monitoring results.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE edits (
            rev_id          bigint PRIMARY KEY,
            parent_rev_id   bigint,
            schema_version  integer NOT NULL,
            title           text NOT NULL,
            edited_at       timestamptz NOT NULL,
            is_minor        boolean NOT NULL,
            is_anonymous    boolean NOT NULL,
            -- HMAC only. The raw username is never stored.
            user_hash       text,
            comment         text,
            old_length      integer,
            new_length      integer,
            -- Text is cleared by the retention job, and by the purge job when Wikipedia hides it.
            added_text      text,
            removed_text    text,
            user_edit_count integer,
            user_hidden     boolean NOT NULL DEFAULT false,
            comment_hidden  boolean NOT NULL DEFAULT false,
            content_hidden  boolean NOT NULL DEFAULT false,
            ingested_at     timestamptz NOT NULL DEFAULT now(),
            purged_at       timestamptz
        );

        CREATE TABLE scores (
            rev_id        bigint NOT NULL REFERENCES edits (rev_id) ON DELETE CASCADE,
            model_name    text NOT NULL,
            model_version text NOT NULL,
            score         real NOT NULL CHECK (score >= 0 AND score <= 1),
            -- True when the champion failed and a fallback answered.
            used_fallback boolean NOT NULL DEFAULT false,
            scored_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (rev_id, model_version)
        );

        CREATE TABLE labels (
            rev_id          bigint PRIMARY KEY REFERENCES edits (rev_id) ON DELETE CASCADE,
            -- NULL with a reason when the revision was hidden or deleted before labeling.
            label           smallint CHECK (label IN (0, 1)),
            excluded_reason text CHECK (excluded_reason IN ('hidden', 'deleted')),
            decided_at      timestamptz NOT NULL DEFAULT now(),
            -- True when downtime delayed the decision, so a revert after 48 hours may be included.
            decided_late    boolean NOT NULL DEFAULT false,
            CHECK ((label IS NULL) <> (excluded_reason IS NULL))
        );

        CREATE TABLE model_versions (
            id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            model_name      text NOT NULL,
            model_version   text NOT NULL,
            alias           text,
            feature_version integer NOT NULL,
            snapshot_hash   text NOT NULL,
            artifact_sha256 text NOT NULL,
            metrics         jsonb NOT NULL DEFAULT '{}',
            registered_at   timestamptz NOT NULL DEFAULT now(),
            UNIQUE (model_name, model_version)
        );

        CREATE TABLE drift_reports (
            id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now(),
            drifted    boolean NOT NULL,
            report     jsonb NOT NULL
        );

        -- Daily rollups, which are small and kept forever.
        CREATE TABLE live_metrics (
            day           date NOT NULL,
            model_version text NOT NULL,
            metrics       jsonb NOT NULL,
            PRIMARY KEY (day, model_version)
        );

        CREATE TABLE retrain_runs (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            trigger     text NOT NULL,
            status      text NOT NULL,
            started_at  timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            details     jsonb NOT NULL DEFAULT '{}'
        );

        CREATE TABLE explanations (
            rev_id         bigint NOT NULL REFERENCES edits (rev_id) ON DELETE CASCADE,
            prompt_version text NOT NULL,
            llm_model      text NOT NULL,
            explanation    text NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (rev_id, prompt_version, llm_model)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE explanations;
        DROP TABLE retrain_runs;
        DROP TABLE live_metrics;
        DROP TABLE drift_reports;
        DROP TABLE model_versions;
        DROP TABLE labels;
        DROP TABLE scores;
        DROP TABLE edits;
        """
    )
