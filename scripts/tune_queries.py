"""Run the API's real queries with EXPLAIN ANALYZE on a realistic volume, and report the plans.

Seeds 3 days of edits at the measured rate (docs/capacity.md: 86,637 a day), then runs each query
through the same functions the API calls. It deletes what's there, so it only runs against a local
throwaway database.

    python scripts/tune_queries.py --url postgresql://... --out data/query_plans.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row
from wikiguard_api import queries
from wikiguard_api.foundations import Tx

EDITS_PER_DAY = 86_637
DAYS = 3
STATEMENT_BREAK = ";\n\n"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "postgres"}
SEED = """
INSERT INTO edits (rev_id, parent_rev_id, schema_version, title, edited_at, is_minor, is_anonymous,
    user_hash, comment, old_length, new_length, added_text, removed_text, user_edit_count,
    latest_score, latest_model_version, latest_scored_at)
SELECT n, n - 1, 1, 'Article ' || (n %% 5000), edited, n %% 5 = 0, n %% 7 = 0,
    md5(n::text), 'a comment of ordinary length', 1000, 1000 + (n %% 200),
    -- 620 bytes a row with indexes was measured with real text. This is about that much.
    CASE WHEN edited > now() - interval '48 hours' THEN repeat('added ', 40) END,
    CASE WHEN edited > now() - interval '48 hours' THEN repeat('old ', 20) END,
    n %% 1000,
    CASE WHEN n %% 500 = 0 THEN NULL ELSE greatest(power(random(), 8), 0.00001) END,
    CASE WHEN n %% 500 = 0 THEN NULL ELSE '1' END,
    CASE WHEN n %% 500 = 0 THEN NULL ELSE edited END
FROM (
    SELECT n, now() - (n * (%(days)s * interval '1 day') / %(rows)s) AS edited
    FROM generate_series(1, %(rows)s) AS n
) AS numbered;

INSERT INTO scores (rev_id, model_name, model_version, score, alias, threshold, scored_at,
    used_fallback)
SELECT rev_id, 'roberta', '1', latest_score, 'champion', 0.97, latest_scored_at, rev_id %% 50 = 0
FROM edits WHERE latest_score IS NOT NULL;

INSERT INTO labels (rev_id, label)
SELECT rev_id, CASE WHEN rev_id %% 15 = 0 THEN 1 ELSE 0 END FROM edits
WHERE edited_at < now() - interval '48 hours';

INSERT INTO reviews (rev_id, decision)
SELECT rev_id, 'ok' FROM edits ORDER BY latest_score DESC NULLS LAST LIMIT 300;
"""


class ExplainTx(Tx):
    """Runs every query as EXPLAIN ANALYZE, and hands the caller no rows."""

    def __init__(self, connection: psycopg.Connection[Any], name: str, out: list[Any]) -> None:
        super().__init__(connection, None)
        self.name = name
        self.out = out

    def _explain(self, query: str, params: Any) -> None:
        row = self.connection.execute(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}", params
        ).fetchone()
        assert row is not None  # noqa: S101 - EXPLAIN always returns a row
        plan = row["QUERY PLAN"][0]
        top = plan["Plan"]
        nodes: list[str] = []

        def walk(node: dict[str, Any]) -> None:
            label = node["Node Type"]
            if "Index Name" in node:
                label += f" using {node['Index Name']}"
            nodes.append(label)
            for child in node.get("Plans", []):
                walk(child)

        walk(top)
        self.out.append(
            {
                "query": self.name,
                "execution_ms": round(plan["Execution Time"], 2),
                "planning_ms": round(plan["Planning Time"], 2),
                "rows": top["Actual Rows"],
                "shared_buffers_read": top.get("Shared Read Blocks", 0),
                "shared_buffers_hit": top.get("Shared Hit Blocks", 0),
                "nodes": nodes,
            }
        )

    def one(self, query: str, params: Any = None) -> dict[str, Any] | None:
        self._explain(query, params)
        return None

    def all(self, query: str, params: Any = None) -> list[dict[str, Any]]:
        self._explain(query, params)
        return []


def check_local(url: str) -> None:
    if urlparse(url).hostname not in LOCAL_HOSTS:
        raise SystemExit("This deletes data, so it only runs against a local database.")


def seed(connection: psycopg.Connection[Any]) -> int:
    rows = EDITS_PER_DAY * DAYS
    connection.execute(
        "TRUNCATE reviews, labels, scores, edits, explanations RESTART IDENTITY CASCADE"
    )
    for statement in SEED.split(STATEMENT_BREAK):
        connection.execute(statement, {"rows": rows, "days": DAYS})
    connection.commit()
    connection.autocommit = True
    connection.execute("VACUUM ANALYZE edits")
    connection.execute("VACUUM ANALYZE scores")
    connection.execute("VACUUM ANALYZE labels")
    connection.execute("VACUUM ANALYZE reviews")
    connection.autocommit = False
    return rows


def run_all(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    middle = connection.execute(
        "SELECT latest_score, rev_id FROM edits WHERE latest_score IS NOT NULL "
        "ORDER BY latest_score DESC, rev_id DESC OFFSET 5000 LIMIT 1"
    ).fetchone()
    assert middle is not None  # noqa: S101 - seeded just above
    deep = queries.encode_cursor([middle["latest_score"], middle["rev_id"]])
    feed_row = connection.execute(
        "SELECT edited_at, rev_id FROM edits ORDER BY edited_at DESC, rev_id DESC "
        "OFFSET 5000 LIMIT 1"
    ).fetchone()
    assert feed_row is not None  # noqa: S101
    deep_feed = queries.encode_cursor([feed_row["edited_at"].isoformat(), feed_row["rev_id"]])

    def tx(name: str) -> ExplainTx:
        return ExplainTx(connection, name, out)

    queries.review_queue(tx("queue, first page"), "all", 50, None, False)
    queries.review_queue(tx("queue, page after 5,000 rows"), "all", 50, deep, False)
    queries.review_queue(tx("queue, logged-out filter"), "logged_out", 50, None, False)
    queries.review_queue(tx("queue, registered filter"), "registered", 50, None, False)
    queries.review_queue(tx("queue, including reviewed"), "all", 50, None, True)
    queries.live_feed(tx("feed, first page"), 50, None)
    queries.live_feed(tx("feed, page after 5,000 rows"), 50, deep_feed)
    queries.edit_detail(tx("edit detail"), 1000)
    queries.stored_score(tx("stored score"), 1000)
    queries.model_list(tx("leaderboard"))
    health = tx("health summary")
    original_one = health.one

    def health_one(query: str, params: Any = None) -> dict[str, Any] | None:
        original_one(query, params)
        return {"scored": 1, "fallback": 0, "shed": 0, "last_scored_at": None, "waiting": 0}

    health.one = health_one  # type: ignore[method-assign]
    queries.health_summary(health)
    connection.rollback()
    return out


def sizes(connection: psycopg.Connection[Any]) -> dict[str, float]:
    row = connection.execute(
        """
        SELECT pg_database_size(current_database()) AS database,
               pg_total_relation_size('edits') AS edits,
               pg_indexes_size('edits') AS edits_indexes,
               pg_total_relation_size('scores') AS scores
        """
    ).fetchone()
    assert row is not None  # noqa: S101
    connection.rollback()
    return {key: round(value / (1024 * 1024), 1) for key, value in row.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()
    check_local(args.url)
    with psycopg.connect(args.url, row_factory=dict_row) as connection:
        rows = seed(connection) if not args.skip_seed else None
        plans = run_all(connection)
        report = {"rows": rows, "sizes_mb": sizes(connection), "plans": plans}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
