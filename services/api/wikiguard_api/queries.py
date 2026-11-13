"""The read queries. Named columns only, so a new column can't reach a client by accident.

Paging is by keyset, not OFFSET: the last row's sort values say where to continue, so the database
walks the index from there and page 500 costs the same as page 1.
"""

import base64
import binascii
import json
from datetime import datetime
from typing import Any

from wikiguard_api.foundations import Problem, Tx

WIKIPEDIA_DIFF = "https://en.wikipedia.org/w/index.php?diff={rev_id}"
# The review queue goes back as far as the edits are kept (DEC-038).
WINDOW = "3 days"

SUMMARY_COLUMNS = """
    e.rev_id, e.title, e.edited_at, e.is_minor, e.is_anonymous, e.comment,
    e.new_length - e.old_length AS bytes_delta,
    e.latest_score AS score, s.model_name, s.model_version, s.threshold,
    s.used_fallback, s.load_shed, r.decision, e.content_hidden
"""
JOIN_SCORE_AND_REVIEW = """
    LEFT JOIN scores s ON s.rev_id = e.rev_id AND s.model_version = e.latest_model_version
    LEFT JOIN reviews r ON r.rev_id = e.rev_id
"""


def encode_cursor(values: list[Any]) -> str:
    text = json.dumps(values, default=str)
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def decode_cursor(cursor: str, kinds: tuple[type, ...]) -> list[Any]:
    """Cursors come from the client, so they're checked as carefully as any other input."""
    bad = Problem(400, "bad_cursor", "That page cursor isn't valid")
    try:
        padding = "=" * (-len(cursor) % 4)
        values = json.loads(base64.urlsafe_b64decode(cursor + padding))
    except (ValueError, binascii.Error) as error:
        raise bad from error
    if not isinstance(values, list) or len(values) != len(kinds):
        raise bad
    parsed: list[Any] = []
    for value, kind in zip(values, kinds, strict=True):
        try:
            parsed.append(datetime.fromisoformat(value) if kind is datetime else kind(value))
        except (TypeError, ValueError) as error:
            raise bad from error
    return parsed


def with_link(row: dict[str, Any]) -> dict[str, Any]:
    # Built from an integer, so a title can't turn it into a link somewhere else.
    return row | {"wikipedia_url": WIKIPEDIA_DIFF.format(rev_id=int(row["rev_id"]))}


def _page(rows: list[dict[str, Any]], limit: int, cursor_of: Any) -> dict[str, Any]:
    more = len(rows) > limit
    kept = rows[:limit]
    return {
        "items": [with_link(row) for row in kept],
        "next_cursor": encode_cursor(cursor_of(kept[-1])) if more and kept else None,
    }


def _window(conditions: list[str], params: dict[str, Any], snapshot: bool) -> None:
    """Live data is limited to the edits still kept. A snapshot is shown whole (DEC-091)."""
    if not snapshot:
        conditions.append("e.edited_at >= now() - %(window)s::interval")
        params["window"] = WINDOW


def review_queue(
    tx: Tx,
    editor: str,
    limit: int,
    cursor: str | None,
    include_reviewed: bool,
    snapshot: bool = False,
) -> dict[str, Any]:
    """Riskiest first. Editor type is a filter only: it never changes the order (DEC-053)."""
    conditions = ["e.latest_score IS NOT NULL"]
    params: dict[str, Any] = {"limit": limit + 1}
    _window(conditions, params, snapshot)
    if editor != "all":
        conditions.append("e.is_anonymous = %(anonymous)s")
        params["anonymous"] = editor == "logged_out"
    if not include_reviewed:
        conditions.append("r.rev_id IS NULL")
    if cursor:
        score, rev_id = decode_cursor(cursor, (float, int))
        conditions.append("(e.latest_score, e.rev_id) < (%(score)s::real, %(rev_id)s)")
        params |= {"score": score, "rev_id": rev_id}
    rows = tx.all(
        f"""
        SELECT {SUMMARY_COLUMNS}
        FROM edits e {JOIN_SCORE_AND_REVIEW}
        WHERE {" AND ".join(conditions)}
        ORDER BY e.latest_score DESC, e.rev_id DESC
        LIMIT %(limit)s
        """,
        params,
    )
    return _page(rows, limit, lambda row: [row["score"], row["rev_id"]])


def live_feed(tx: Tx, limit: int, cursor: str | None, snapshot: bool = False) -> dict[str, Any]:
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit + 1}
    _window(conditions, params, snapshot)
    if cursor:
        edited_at, rev_id = decode_cursor(cursor, (datetime, int))
        conditions.append("(e.edited_at, e.rev_id) < (%(edited_at)s, %(rev_id)s)")
        params |= {"edited_at": edited_at, "rev_id": rev_id}
    rows = tx.all(
        f"""
        SELECT {SUMMARY_COLUMNS}
        FROM edits e {JOIN_SCORE_AND_REVIEW}
        {"WHERE " + " AND ".join(conditions) if conditions else ""}
        ORDER BY e.edited_at DESC, e.rev_id DESC
        LIMIT %(limit)s
        """,
        params,
    )
    return _page(rows, limit, lambda row: [row["edited_at"].isoformat(), row["rev_id"]])


def edit_detail(tx: Tx, rev_id: int) -> dict[str, Any] | None:
    row = tx.one(
        f"""
        SELECT {SUMMARY_COLUMNS}, e.parent_rev_id, e.added_text, e.removed_text, e.comment_hidden,
               e.user_hidden, e.user_edit_count, l.label,
               l.excluded_reason AS label_excluded_reason,
               EXISTS (SELECT 1 FROM explanations x WHERE x.rev_id = e.rev_id) AS has_explanation
        FROM edits e {JOIN_SCORE_AND_REVIEW}
        LEFT JOIN labels l ON l.rev_id = e.rev_id
        WHERE e.rev_id = %s
        """,
        (rev_id,),
    )
    if row is None:
        return None
    scores = tx.all(
        """
        SELECT model_name, model_version, alias, score, threshold, used_fallback, load_shed,
               forced, latency_ms, scored_at
        FROM scores WHERE rev_id = %s ORDER BY scored_at DESC
        """,
        (rev_id,),
    )
    return with_link(row) | {"scores": scores}


def stored_score(tx: Tx, rev_id: int) -> dict[str, Any] | None:
    return tx.one(
        """
        SELECT s.score, s.threshold, s.model_name, s.model_version, s.used_fallback, s.forced,
               s.latency_ms
        FROM edits e JOIN scores s
          ON s.rev_id = e.rev_id AND s.model_version = e.latest_model_version
        WHERE e.rev_id = %s
        """,
        (rev_id,),
    )


def model_list(tx: Tx) -> list[dict[str, Any]]:
    # Champion first, then the fallbacks in order, then anything without an alias.
    return tx.all(
        """
        SELECT model_name, model_version, alias, feature_version, snapshot_hash, metrics,
               registered_at
        FROM model_versions
        ORDER BY CASE WHEN alias = 'champion' THEN 0 WHEN alias IS NOT NULL THEN 1 ELSE 2 END,
                 alias, registered_at DESC
        LIMIT 50
        """
    )


# Keys of retrain_runs.details the page doesn't use, removed in SQL so they never leave Supabase.
RUN_DETAILS_DROPPED = (
    "listings",
    "slices",
    "aliases_before",
    "aliases_after",
    "proposed_aliases",
    "validation_warnings",
    "storage",
)
RUN_DETAILS_KEPT = "".join(f" - '{key}'" for key in RUN_DETAILS_DROPPED)


def health_summary(tx: Tx, snapshot: bool = False) -> dict[str, Any]:
    """The last 24 hours of scoring, live. A snapshot was scored once, so it's counted whole, and
    the edits' first and last dates say what the data covers (DEC-091)."""
    recent = "" if snapshot else "WHERE scored_at > now() - interval '24 hours'"
    counts = tx.one(
        f"""
        SELECT count(*) AS scored,
               count(*) FILTER (WHERE used_fallback) AS fallback,
               count(*) FILTER (WHERE load_shed) AS shed,
               max(scored_at) AS last_scored_at
        FROM scores {recent}
        """
    )
    assert counts is not None  # noqa: S101 - an aggregate always returns one row
    kept = "" if snapshot else " AND edited_at > now() - interval '3 days'"
    unscored = tx.one(f"SELECT count(*) AS waiting FROM edits WHERE latest_score IS NULL{kept}")
    assert unscored is not None  # noqa: S101
    covered = (
        tx.one("SELECT min(edited_at) AS first, max(edited_at) AS last FROM edits")
        if snapshot
        else None
    )
    total = counts["scored"]
    return {
        "data_mode": "snapshot" if snapshot else "live",
        "data_from": covered["first"] if covered else None,
        "data_to": covered["last"] if covered else None,
        "scored_last_24h": total,
        "fallback_rate_24h": counts["fallback"] / total if total else None,
        "load_shed_rate_24h": counts["shed"] / total if total else None,
        "last_scored_at": counts["last_scored_at"],
        "unscored_edits": unscored["waiting"],
        "live_metrics": tx.all(
            "SELECT day, model_version, metrics FROM live_metrics ORDER BY day DESC LIMIT 30"
        ),
        "drift_reports": tx.all(
            "SELECT created_at, drifted FROM drift_reports ORDER BY created_at DESC LIMIT 10"
        ),
        # details holds the trainer's before vs after report, which the Health page shows. The
        # trainer's own bookkeeping (model listings, slice tables, alias maps) stays behind:
        # every byte read from Supabase counts against its 5 GB a month.
        "retrain_runs": tx.all(
            f"SELECT id, trigger, status, started_at, finished_at, details {RUN_DETAILS_KEPT} "
            "AS details FROM retrain_runs ORDER BY started_at DESC LIMIT 10"
        ),
    }


def audit_page(tx: Tx, limit: int, cursor: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit + 1}
    where = ""
    if cursor:
        params["id"] = decode_cursor(cursor, (int,))[0]
        where = "WHERE id < %(id)s"
    rows = tx.all(
        f"""
        SELECT id, at, actor, action, target, request_id, details
        FROM audit_log {where} ORDER BY id DESC LIMIT %(limit)s
        """,
        params,
    )
    more = len(rows) > limit
    kept = rows[:limit]
    return {
        "items": kept,
        "next_cursor": encode_cursor([kept[-1]["id"]]) if more and kept else None,
    }


REQUEST_COLUMNS = (
    "id, kind, status, reason, requested_by, requested_at, picked_up_at, finished_at, details"
)
