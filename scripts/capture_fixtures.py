"""Record real stream and API responses, to learn the field formats and build test fixtures.

Everything goes to data/fixtures_raw/, which is git-ignored because it holds real usernames and
vandal text. scripts/scrub_fixtures.py then writes the public copies to tests/fixtures/wikimedia/.
Keeping the two apart means re-scrubbing never calls Wikimedia again.

Run it by hand from the repo root. It keeps requests serial and honours maxlag and Retry-After,
as Wikimedia's policy asks, and prints counts and field names only, never usernames or text:

    python scripts/capture_fixtures.py --stream-seconds 600

To fetch one more kind of diff without repeating the rest:

    python scripts/capture_fixtures.py --only-diff unicode
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from wikiguard_core.config import ConfigError, WikimediaSettings, load_settings

RAW_DIR = Path("data/fixtures_raw")
MAXLAG_SECONDS = "5"
SECONDS_BETWEEN_REQUESTS = 1.0
MAX_ATTEMPTS = 5
STREAM_READ_TIMEOUT_SECONDS = 60
# Labels are final after 48 hours, so older rows have settled tags.
LABEL_WINDOW = timedelta(hours=48)
BACKFILL_PAGES = 4
PAGE_SIZE = 500
MAX_LARGE_DIFF_CHARS = 200_000
RC_PROPS = "ids|title|user|userid|comment|parsedcomment|sizes|flags|tags|timestamp|sha1"
REV_PROPS = "ids|flags|timestamp|user|userid|size|sha1|comment|parsedcomment|tags"
COMPARE_PROPS = "diff|diffsize|ids|title|user|comment|parsedcomment|size|timestamp"
REVERT_TAGS = {"mw-reverted", "mw-undo", "mw-rollback", "mw-manual-revert"}
# Edits the community or an edit filter flagged stay out of the kept-text fixtures.
FLAGGED_TAG_PREFIXES = ("possible", "changing height", "mw-blank", "mw-replace")
HIDDEN_FLAGS = ("commenthidden", "userhidden", "sha1hidden", "suppressed")
STREAM_SAMPLES_PER_CATEGORY = 20
NONEXISTENT_REVID = 4_000_000_000


class Wikimedia:
    """Serial, polite access to the Action API, keeping any maxlag response it happens to see."""

    def __init__(self, settings: WikimediaSettings) -> None:
        self.settings = settings
        self.request_count = 0
        self.maxlag_response: dict[str, Any] | None = None

    def query(self, **params: str) -> dict[str, Any]:
        full = {"format": "json", "formatversion": "2", "maxlag": MAXLAG_SECONDS, **params}
        url = f"{self.settings.mediawiki_api_url}?{urllib.parse.urlencode(full)}"
        request = urllib.request.Request(  # noqa: S310 - the URL comes from our own settings
            url, headers={"User-Agent": self.settings.wikimedia_user_agent}
        )
        for attempt in range(1, MAX_ATTEMPTS + 1):
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            self.request_count += 1
            try:
                with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                    body: dict[str, Any] = json.load(response)
                    retry_after = response.headers.get("Retry-After")
            except urllib.error.HTTPError as error:
                if error.code not in (429, 503) or attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(int(error.headers.get("Retry-After", 2**attempt)))
                continue
            if body.get("error", {}).get("code") != "maxlag":
                return body
            self.maxlag_response = self.maxlag_response or body
            time.sleep(int(retry_after or 2**attempt))
        raise RuntimeError(f"still lagged after {MAX_ATTEMPTS} attempts")


def read_stream(settings: WikimediaSettings, last_event_id: str | None = None) -> Iterator[Any]:
    """Yield (sse_id, event) pairs. Server-sent events are plain text, so this needs no library."""
    headers = {"User-Agent": settings.wikimedia_user_agent, "Accept": "text/event-stream"}
    if last_event_id:
        headers["Last-Event-ID"] = last_event_id
    request = urllib.request.Request(  # noqa: S310 - the URL comes from our own settings
        settings.wikimedia_stream_url, headers=headers
    )
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=STREAM_READ_TIMEOUT_SECONDS
    ) as response:
        sse_id = None
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("id: "):
                sse_id = line[4:]
            elif line.startswith("data: "):
                yield sse_id, json.loads(line[6:])


def stream_category(event: dict[str, Any]) -> str:
    """Sort an event the way the ingestor will have to, so every excluded kind gets a sample."""
    if event.get("meta", {}).get("domain") == "canary":
        return "canary"
    if event.get("wiki") != "enwiki":
        return "other_wiki"
    if event.get("type") != "edit":
        return f"enwiki_type_{event.get('type')}"
    if event.get("namespace") != 0:
        return "enwiki_other_namespace"
    if event.get("bot"):
        return "enwiki_bot_edit"
    if str(event.get("user", "")).startswith("~"):
        return "in_scope_temporary"
    return "in_scope_registered"


def field_types(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Field names and value types across rows. Safe to print, because it holds no values."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        for key, value in row.items():
            seen.setdefault(key, set()).add(type(value).__name__)
    return {key: sorted(types) for key, types in sorted(seen.items())}


def capture_stream(settings: WikimediaSettings, seconds: int) -> dict[str, Any]:
    """Count every event for the window. Keep all in-scope edits, plus samples of the rest."""
    counts: Counter[str] = Counter()
    fields: dict[str, list[dict[str, Any]]] = {}
    kept: list[dict[str, Any]] = []
    new_page_revids: list[int] = []
    first_id = None
    first_ids_after: list[str] = []
    started = time.monotonic()
    for sse_id, event in read_stream(settings):
        category = stream_category(event)
        counts[category] += 1
        first_id = first_id or sse_id
        if len(fields.setdefault(category, [])) < STREAM_SAMPLES_PER_CATEGORY:
            fields[category].append(event)
        in_scope = category.startswith("in_scope")
        if in_scope or counts[category] <= STREAM_SAMPLES_PER_CATEGORY:
            kept.append({"category": category, "sse_id": sse_id, "event": event})
        if category == "enwiki_type_new" and event.get("namespace") == 0:
            new_page_revids.append(event["revision"]["new"])
        if len(first_ids_after) < 3:
            first_ids_after.append(event["meta"]["id"])
        if time.monotonic() - started >= seconds:
            break

    with (RAW_DIR / "stream_events.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for row in kept:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "seconds": round(time.monotonic() - started),
        "counts": dict(counts),
        "fields": {category: field_types(rows) for category, rows in sorted(fields.items())},
        "first_sse_id": first_id,
        "new_page_revids": new_page_revids,
    }


def check_resume(settings: WikimediaSettings, kept_path: Path) -> dict[str, Any]:
    """Reconnect with an id from the capture and see whether the stream replays what followed."""
    rows = [json.loads(line) for line in kept_path.read_text(encoding="utf-8").splitlines()]
    resume_from = rows[len(rows) // 2]
    resumed_at = datetime.fromisoformat(resume_from["event"]["meta"]["dt"].replace("Z", "+00:00"))
    for _, event in read_stream(settings, last_event_id=resume_from["sse_id"]):
        first_dt = datetime.fromisoformat(event["meta"]["dt"].replace("Z", "+00:00"))
        return {
            "requested_id_shape": sorted(json.loads(resume_from["sse_id"])[0]),
            "first_event_seconds_after_resume_point": round(
                (first_dt - resumed_at).total_seconds()
            ),
            "seconds_behind_now_on_arrival": round((datetime.now(UTC) - first_dt).total_seconds()),
        }
    return {"error": "the stream ended before an event arrived"}


def save_case(case: str, treatment: str, what: str, params: dict[str, str], response: Any) -> None:
    """Save one response. `treatment` tells the scrubber whether its text may be kept."""
    body = {
        "case": case,
        "treatment": treatment,
        "what": what,
        "request": params,
        "response": response,
    }
    (RAW_DIR / "cases" / f"{case}.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def fetch_and_save(api: Wikimedia, case: str, treatment: str, what: str, **params: str) -> Any:
    response = api.query(**params)
    save_case(case, treatment, what, params, response)
    return response


def backfill_rows(api: Wikimedia) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Page through settled edits, with the same filter scripts/backfill.py will use."""
    rcstart = (datetime.now(UTC) - LABEL_WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "action": "query",
        "list": "recentchanges",
        "rcstart": rcstart,
        "rcdir": "older",
        "rctype": "edit",
        "rcnamespace": "0",
        "rcshow": "!bot",
        "rclimit": str(PAGE_SIZE),
        "rcprop": RC_PROPS,
    }
    rows: list[dict[str, Any]] = []
    extra: dict[str, str] = {}
    for _ in range(BACKFILL_PAGES):
        body = api.query(**params, **extra)
        rows.extend(body["query"]["recentchanges"])
        extra = body.get("continue", {})
        if not extra:
            break
    return rows, params


def has_hidden_flag(row: dict[str, Any]) -> bool:
    return any(row.get(flag) for flag in HIDDEN_FLAGS)


def is_ordinary(row: dict[str, Any]) -> bool:
    """Nothing suggests the edit was vandalism, so its text may be kept for review."""
    tags = set(row.get("tags", []))
    flagged = any(tag.startswith(FLAGGED_TAG_PREFIXES) for tag in tags)
    return not (tags & REVERT_TAGS or flagged or has_hidden_flag(row))


def diff_body(response: dict[str, Any]) -> str:
    return str(response.get("compare", {}).get("body", ""))


DIFF_SHAPES = {
    "added_only",
    "removed_only",
    "small_changed",
    "large",
    "multi_section",
    "moved_paragraph",
    "unicode",
    "wikitext_markup",
}


def diff_shapes(body: str, row: dict[str, Any]) -> set[str]:
    """The structures a parser has to handle that this diff shows."""
    added = "diff-addedline" in body
    removed = "diff-deletedline" in body
    shapes = set()
    if added and not removed:
        shapes.add("added_only")
    if removed and not added:
        shapes.add("removed_only")
    if added and removed and len(body) < 3000:
        shapes.add("small_changed")
    if abs(row.get("newlen", 0) - row.get("oldlen", 0)) > 5000 and len(body) < MAX_LARGE_DIFF_CHARS:
        shapes.add("large")
    if body.count("diff-lineno") >= 4:
        shapes.add("multi_section")
    if "mw-diff-movedpara" in body:
        shapes.add("moved_paragraph")
    # Greek and later scripts. Dashes and curly quotes are too common to test anything.
    if any(ord(char) >= 0x370 and char.isalpha() for char in body):
        shapes.add("unicode")
    if "&lt;ref" in body:
        shapes.add("wikitext_markup")
    return shapes


def compare_params(row: dict[str, Any]) -> dict[str, str]:
    return {
        "action": "compare",
        "fromrev": str(row["old_revid"]),
        "torev": str(row["revid"]),
        "prop": COMPARE_PROPS,
    }


def capture_ordinary_diffs(
    api: Wikimedia, rows: list[dict[str, Any]], wanted: set[str] | None = None
) -> list[str]:
    """Fetch a few diffs per byte-change bucket until each wanted shape has one example."""
    wanted = set(wanted or DIFF_SHAPES)
    ordinary = [row for row in rows if is_ordinary(row) and row.get("old_revid")]

    def delta(row: dict[str, Any]) -> int:
        return int(row["newlen"]) - int(row["oldlen"])

    buckets = [
        [row for row in ordinary if not row["title"].isascii()],
        [row for row in ordinary if not row["comment"].isascii()],
        sorted((row for row in ordinary if 5000 < delta(row) < 40000), key=delta),
        [row for row in ordinary if 200 < delta(row) < 2000],
        [row for row in ordinary if -2000 < delta(row) < -100],
        [row for row in ordinary if abs(delta(row)) < 60],
    ]
    used: set[int] = set()
    for _ in range(6):
        for bucket in buckets:
            row = next((candidate for candidate in bucket if candidate["revid"] not in used), None)
            if row is None or not wanted:
                continue
            used.add(row["revid"])
            params = compare_params(row)
            response = api.query(**params)
            found = diff_shapes(diff_body(response), row) & wanted
            if found:
                case = f"compare_{sorted(found)[0]}"
                what = f"A settled, unflagged edit. Shows: {', '.join(sorted(found))}."
                save_case(case, "ordinary", what, params, response)
                wanted -= found
    return sorted(wanted)


def first_row(rows: list[dict[str, Any]], *conditions: Any) -> dict[str, Any] | None:
    return next((row for row in rows if all(check(row) for check in conditions)), None)


def capture_neutral_diffs(api: Wikimedia, rows: list[dict[str, Any]]) -> None:
    """Reverted edits and the reverts themselves. Only their structure will reach the fixtures."""
    reverted = first_row(
        rows,
        lambda row: "mw-reverted" in row.get("tags", []),
        lambda row: row.get("temp"),
        lambda row: row.get("old_revid"),
        lambda row: not has_hidden_flag(row),
    )
    if reverted:
        params = compare_params(reverted)
        what = "A reverted edit by a temporary account. Words replaced, structure kept."
        fetch_and_save(api, "compare_reverted", "neutral", what, **params)

    revert = first_row(
        rows,
        lambda row: {"mw-undo", "mw-rollback"} & set(row.get("tags", [])),
        lambda row: row.get("old_revid"),
    )
    if revert:
        params = compare_params(revert)
        what = "An edit that undid another. Words replaced, structure kept."
        fetch_and_save(api, "compare_revert_action", "neutral", what, **params)


def save_recentchanges_sample(rows: list[dict[str, Any]], params: dict[str, str]) -> None:
    """A short list covering each kind of row. Neutral, because some are reverted."""
    checks = {
        "registered": lambda row: not row.get("temp"),
        "temporary": lambda row: row.get("temp"),
        "minor": lambda row: row.get("minor"),
        "reverted": lambda row: "mw-reverted" in row.get("tags", []),
        "undo": lambda row: "mw-undo" in row.get("tags", []),
        "rollback": lambda row: "mw-rollback" in row.get("tags", []),
        "hidden": has_hidden_flag,
    }
    chosen: dict[int, dict[str, Any]] = {}
    for check in checks.values():
        for row in [row for row in rows if check(row)][:3]:
            chosen[row["revid"]] = row
    save_case(
        "recentchanges_sample",
        "neutral",
        "Settled list=recentchanges rows, covering each kind. Words replaced.",
        params,
        {"query": {"recentchanges": list(chosen.values())}},
    )


def capture_hidden(api: Wikimedia) -> dict[str, Any]:
    """Find revisions the community hid, through the public revision-deletion log."""
    log = fetch_and_save(
        api,
        "logevents_revision_delete",
        "neutral",
        "Revision-deletion log entries in the main namespace.",
        action="query",
        list="logevents",
        letype="delete",
        leaction="delete/revision",
        lenamespace="0",
        lelimit="50",
        leprop="ids|title|type|user|userid|timestamp|comment|details",
    )
    entries = log.get("query", {}).get("logevents", [])
    revids: list[str] = []
    variety: Counter[str] = Counter()
    for entry in entries:
        details = entry.get("params", {})
        new_state = details.get("new", {})
        for flag in ("content", "comment", "user"):
            if new_state.get(flag):
                variety[flag] += 1
        ids = details.get("ids", [])
        if details.get("type") == "revision" and ids and len(revids) < 20:
            revids.extend(str(rev) for rev in ids[:2])
    summary: dict[str, Any] = {"log_entries": len(entries), "hidden_kinds": dict(variety)}
    if not revids:
        return summary

    response = fetch_and_save(
        api,
        "revisions_hidden",
        "neutral",
        "Revisions named in the deletion log, showing the API's hidden flags.",
        action="query",
        prop="revisions",
        revids="|".join(revids[:20]),
        rvprop=REV_PROPS,
        rvslots="main",
    )
    flags: Counter[str] = Counter()
    hidden_row = None
    for page in response.get("query", {}).get("pages", []):
        for revision in page.get("revisions", []):
            for key, value in revision.items():
                if "hidden" in key or key == "suppressed":
                    flags[f"{key}={value}"] += 1
                    hidden_row = hidden_row or revision
    summary["revision_flags"] = dict(flags)
    summary["badrevids"] = len(response.get("query", {}).get("badrevids", {}))

    if hidden_row and hidden_row.get("parentid"):
        fetch_and_save(
            api,
            "compare_hidden",
            "neutral",
            "Asking for the diff of a hidden revision.",
            action="compare",
            fromrev=str(hidden_row["parentid"]),
            torev=str(hidden_row["revid"]),
            prop=COMPARE_PROPS,
        )
    return summary


def capture_missing(api: Wikimedia, stream_revids: list[int]) -> dict[str, Any]:
    """Pages can be deleted within minutes, so re-check the stream's revisions for real examples."""
    chunk = [str(revid) for revid in stream_revids[:50]]
    found_missing: list[str] = []
    if chunk:
        body = api.query(
            action="query", prop="revisions", revids="|".join(chunk), rvprop="ids|timestamp"
        )
        found_missing = [str(rev) for rev in body.get("query", {}).get("badrevids", {})]
    if found_missing:
        fetch_and_save(
            api,
            "revisions_deleted_page",
            "neutral",
            "Revisions seen on the stream whose page was then deleted.",
            action="query",
            prop="revisions",
            revids="|".join(found_missing[:5]),
            rvprop=REV_PROPS,
        )
    fetch_and_save(
        api,
        "revisions_nonexistent",
        "neutral",
        "A revision id that was never used.",
        action="query",
        prop="revisions",
        revids=str(NONEXISTENT_REVID),
        rvprop=REV_PROPS,
    )
    fetch_and_save(
        api,
        "compare_nonexistent",
        "neutral",
        "Asking for the diff of a revision id that was never used.",
        action="compare",
        fromrev=str(NONEXISTENT_REVID),
        torev=str(NONEXISTENT_REVID + 1),
        prop=COMPARE_PROPS,
    )
    return {"checked": len(chunk), "now_missing": len(found_missing)}


def capture_users(api: Wikimedia, rows: list[dict[str, Any]]) -> None:
    registered = first_row(rows, lambda row: not row.get("temp"), lambda row: row.get("userid"))
    temporary = first_row(rows, lambda row: row.get("temp"))
    for kind, row in (("registered", registered), ("temporary", temporary)):
        if row is None:
            continue
        fetch_and_save(
            api,
            f"users_{kind}",
            "neutral",
            f"list=users for a {kind} account.",
            action="query",
            list="users",
            ususers=row["user"],
            usprop="editcount|registration|groups",
        )
        fetch_and_save(
            api,
            f"usercontribs_{kind}",
            "neutral",
            f"Contributions by a {kind} account before an edit, for point-in-time counts.",
            action="query",
            list="usercontribs",
            ucuser=row["user"],
            ucstart=row["timestamp"],
            ucdir="older",
            uclimit="10",
            ucprop="ids|timestamp|tags",
        )


def capture_tag_batch(api: Wikimedia, rows: list[dict[str, Any]]) -> None:
    """The labeler's request: tags for many revisions at once."""
    reverted = [row for row in rows if "mw-reverted" in row.get("tags", [])][:5]
    others = [row for row in rows if "mw-reverted" not in row.get("tags", [])][:5]
    revids = "|".join(str(row["revid"]) for row in reverted + others)
    fetch_and_save(
        api,
        "revisions_tag_batch",
        "neutral",
        "The labeler's batched tag lookup, mixing reverted and unreverted edits.",
        action="query",
        prop="revisions",
        revids=revids,
        rvprop="ids|tags|timestamp",
    )


def compare_stream_with_api(api: Wikimedia) -> dict[str, Any]:
    """Do the stream and the API describe the same edit the same way?"""
    rows = [
        json.loads(line)
        for line in (RAW_DIR / "stream_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stream = {
        row["event"]["revision"]["new"]: row["event"]
        for row in rows
        if row["category"].startswith("in_scope")
    }
    api_rows: dict[int, dict[str, Any]] = {}
    extra: dict[str, str] = {}
    for _ in range(3):
        body = api.query(
            action="query",
            list="recentchanges",
            rctype="edit",
            rcnamespace="0",
            rcshow="!bot",
            rclimit=str(PAGE_SIZE),
            rcprop=RC_PROPS,
            **extra,
        )
        api_rows.update({row["revid"]: row for row in body["query"]["recentchanges"]})
        extra = body.get("continue", {})
        if not extra:
            break

    shared = [revid for revid in stream if revid in api_rows]
    mismatches: Counter[str] = Counter()
    stream_tilde = api_temp = 0
    for revid in shared:
        event, row = stream[revid], api_rows[revid]
        tilde = event["user"].startswith("~")
        stream_tilde += tilde
        api_temp += bool(row.get("temp"))
        if tilde != bool(row.get("temp")):
            mismatches["tilde_vs_temp_flag"] += 1
        if event["user"] != row["user"]:
            mismatches["user"] += 1
        if event["title"] != row["title"]:
            mismatches["title"] += 1
        if event["comment"] != row["comment"]:
            mismatches["comment"] += 1
        if event["minor"] != bool(row.get("minor")):
            mismatches["minor"] += 1
        if event["length"].get("old") != row.get("oldlen"):
            mismatches["old_length"] += 1
        if event["length"].get("new") != row.get("newlen"):
            mismatches["new_length"] += 1
        if event["revision"].get("old") != row.get("old_revid"):
            mismatches["old_revid"] += 1
    return {
        "in_scope_stream_edits": len(stream),
        "found_in_api": len(shared),
        "stream_tilde_names": stream_tilde,
        "api_temp_flags": api_temp,
        "mismatches": dict(mismatches),
    }


def summarise_backfill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags: Counter[str] = Counter(tag for row in rows for tag in row.get("tags", []))
    names = [row.get("user", "") for row in rows]
    return {
        "rows": len(rows),
        "oldest": min(row["timestamp"] for row in rows),
        "newest": max(row["timestamp"] for row in rows),
        "temp_flag": sum(1 for row in rows if row.get("temp")),
        "anon_flag": sum(1 for row in rows if row.get("anon")),
        "tilde_names": sum(1 for name in names if name.startswith("~")),
        "tilde_without_temp_flag": sum(
            1 for row in rows if row.get("user", "").startswith("~") and not row.get("temp")
        ),
        "temp_flag_without_tilde": sum(
            1 for row in rows if row.get("temp") and not row.get("user", "").startswith("~")
        ),
        "temp_without_userid": sum(1 for row in rows if row.get("temp") and not row.get("userid")),
        "mw_reverted": tags["mw-reverted"],
        "hidden": {flag: sum(1 for row in rows if row.get(flag)) for flag in HIDDEN_FLAGS},
        "no_user_field": sum(1 for row in rows if "user" not in row),
        "fields": field_types(rows),
        "top_tags": dict(tags.most_common(30)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stream-seconds", type=int, default=600)
    parser.add_argument("--only-diff", choices=sorted(DIFF_SHAPES), help="skip everything else")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        settings = load_settings(WikimediaSettings)
    except ConfigError as error:
        print(error)
        return 2

    (RAW_DIR / "cases").mkdir(parents=True, exist_ok=True)
    api = Wikimedia(settings)
    if args.only_diff:
        rows, _ = backfill_rows(api)
        missing = capture_ordinary_diffs(api, rows, {args.only_diff})
        print(f"Not found: {missing}. API requests: {api.request_count}")
        return 0

    summary: dict[str, Any] = {"captured_at": datetime.now(UTC).isoformat(timespec="seconds")}

    print(f"Reading the stream for {args.stream_seconds} seconds...", flush=True)
    summary["stream"] = capture_stream(settings, args.stream_seconds)
    new_page_revids = summary["stream"].pop("new_page_revids")
    print("Checking resume...", flush=True)
    summary["resume"] = check_resume(settings, RAW_DIR / "stream_events.jsonl")
    print("Comparing the stream with the API...", flush=True)
    summary["stream_vs_api"] = compare_stream_with_api(api)

    print("Paging settled edits...", flush=True)
    rows, rc_params = backfill_rows(api)
    summary["backfill_sample"] = summarise_backfill(rows)
    save_recentchanges_sample(rows, rc_params)

    print("Fetching diffs...", flush=True)
    summary["diff_shapes_not_found"] = capture_ordinary_diffs(api, rows)
    capture_neutral_diffs(api, rows)

    print("Fetching hidden, missing, user, and tag responses...", flush=True)
    summary["hidden"] = capture_hidden(api)
    summary["missing"] = capture_missing(api, new_page_revids)
    capture_users(api, rows)
    capture_tag_batch(api, rows)

    if api.maxlag_response:
        save_case("maxlag", "neutral", "A real maxlag response.", {}, api.maxlag_response)
    summary["maxlag_seen"] = api.maxlag_response is not None
    summary["api_requests"] = api.request_count

    (RAW_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
