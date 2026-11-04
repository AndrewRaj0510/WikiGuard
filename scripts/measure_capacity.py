"""Measure what an edit costs to store, by loading real backfill rows into a throwaway Postgres.

The plan estimated 1 KB per row, which matters because Supabase's limit is 500 MB. Each edit is
measured with its indexes and out-of-line text, both with its diff text and after retention has
cleared it, since most rows spend most of their life cleared. Only the empty tables in
TEST_DATABASE_URL are touched, and they're emptied at the end.

    python scripts/measure_capacity.py data/backfill_pilot
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
import pyarrow.parquet as pq

LIMIT_BYTES = 500 * 1024 * 1024
TEXT_DAYS = 2
ROW_DAYS = 5
# Share of edits whose text stays as long as the row. A guess until Phase 3 has real scores.
HIGH_RISK_SHARE = 0.05
TABLES = ("edits", "scores", "labels")


def total_bytes(connection: psycopg.Connection[Any], table: str) -> int:
    """The table, its indexes, and its out-of-line text."""
    row = connection.execute(f"SELECT pg_total_relation_size('{table}')").fetchone()  # noqa: S608
    return int(row[0]) if row else 0


def compact(connection: psycopg.Connection[Any]) -> None:
    """Rewrite the tables, so dead rows from loading and updating aren't counted."""
    for table in TABLES:
        connection.execute(f"VACUUM FULL ANALYZE {table}")


def load(connection: psycopg.Connection[Any], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        connection.execute(
            """
            INSERT INTO edits (rev_id, parent_rev_id, schema_version, title, edited_at, is_minor,
                               is_anonymous, user_hash, comment, old_length, new_length,
                               added_text, removed_text, user_hidden, comment_hidden,
                               content_hidden)
            VALUES (%(rev_id)s, %(parent_rev_id)s, %(schema_version)s, %(title)s, %(timestamp)s,
                    %(is_minor)s, %(is_anonymous)s, %(user_hash)s, %(comment)s, %(old_length)s,
                    %(new_length)s, %(added_text)s, %(removed_text)s, %(user_hidden)s,
                    %(comment_hidden)s, %(content_hidden)s)
            ON CONFLICT DO NOTHING
            """,
            row,
        )
        connection.execute(
            "INSERT INTO scores (rev_id, model_name, model_version, score) "
            "VALUES (%s, 'roberta', 'v1', 0.5) ON CONFLICT DO NOTHING",
            (row["rev_id"],),
        )
        connection.execute(
            "INSERT INTO labels (rev_id, label) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (row["rev_id"], row["tagged_reverted"]),
        )


def measure(connection: psycopg.Connection[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    load(connection, rows)
    compact(connection)
    count = len(rows)
    with_text = total_bytes(connection, "edits") / count
    scores = total_bytes(connection, "scores") / count
    labels = total_bytes(connection, "labels") / count
    connection.execute("UPDATE edits SET added_text = NULL, removed_text = NULL")
    compact(connection)
    without_text = total_bytes(connection, "edits") / count
    for table in reversed(TABLES):
        connection.execute(f"TRUNCATE {table} CASCADE")
    return {
        "rows": count,
        "edit_row_with_text_bytes": round(with_text),
        "edit_row_without_text_bytes": round(without_text),
        "score_row_bytes": round(scores),
        "label_row_bytes": round(labels),
    }


def project(sizes: dict[str, Any], edits_per_day: float, parquet_per_edit: float) -> dict[str, Any]:
    """What the retention rules add up to at the measured rate."""
    other = sizes["score_row_bytes"] + sizes["label_row_bytes"]
    text = sizes["edit_row_with_text_bytes"]
    bare = sizes["edit_row_without_text_bytes"]
    with_text_rows = edits_per_day * TEXT_DAYS
    high_risk_rows = edits_per_day * (ROW_DAYS - TEXT_DAYS) * HIGH_RISK_SHARE
    bare_rows = edits_per_day * (ROW_DAYS - TEXT_DAYS) * (1 - HIGH_RISK_SHARE)
    steady = (with_text_rows + high_risk_rows) * text + bare_rows * bare
    steady += edits_per_day * ROW_DAYS * other
    naive_days = LIMIT_BYTES / (edits_per_day * (text + other))
    return {
        "edits_per_day": round(edits_per_day),
        "days_that_fit_if_every_row_kept_text": round(naive_days, 1),
        "steady_state_bytes_normal_rules": round(steady),
        "steady_state_share_of_limit": round(steady / LIMIT_BYTES, 3),
        "parquet_bytes_per_edit": round(parquet_per_edit),
        "archive_30_day_bytes": round(parquet_per_edit * edits_per_day * 30),
        "high_risk_share_assumed": HIGH_RISK_SHARE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        print("Set TEST_DATABASE_URL to a throwaway, migrated database")
        return 2

    rows = pq.read_table(args.folder / "parts").to_pylist()
    summary = json.loads((args.folder / "summary.json").read_text(encoding="utf-8"))
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as connection:
        sizes = measure(connection, rows)
    report = {
        **sizes,
        **project(
            sizes, summary["estimated_edits_per_day"], summary["parquet_bytes"] / summary["rows"]
        ),
    }
    (args.folder / "capacity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
