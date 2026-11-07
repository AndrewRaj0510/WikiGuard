"""Build the starting dataset: settled edits from the last 30 days, with diffs and final labels.

Enwiki keeps 30 days of recent changes. Reading all of it would mean millions of rows, so this
samples a block of edits from every hour in the window. Only edits over 48 hours old are read,
so their tags are final. The steps:

1. Listing: one request per hour of the window, saved as it goes.
2. Diffs: one request per edit, in random order, so stopping early still leaves an even spread.
   Batches go to parquet. A run can stop at any time and pick up where it left off. A power
   cut can lose the batch in progress, at most 500 diffs.
3. Labels: one request per reverted edit, to find when it was reverted. A tag can come from a
   revert long after the edit, while the live label means "within 48 hours", and 12.1% of tagged
   rows in the pilot were later than that (F-058). Use labels_48h.parquet for training, and never
   the raw tagged_reverted column.

Raw usernames never reach the disk: rows hold the HMAC and is_anonymous. user_edit_count stays
missing, because the API only knows today's count, which would leak the future (F-007, DEC-036).

    python scripts/backfill.py --target 2000 --out data/backfill_pilot   # about 13 minutes
    python scripts/backfill.py                                           # about 10 hours
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from wikiguard_core.config import (
    ConfigError,
    PrivacySettings,
    WikimediaSettings,
    load_settings,
)
from wikiguard_core.diff import parse_diff
from wikiguard_core.labels import LABEL_WAIT, REVERTED_TAG
from wikiguard_core.schemas import SCHEMA_VERSION, EditEvent, edit_from_recentchange
from wikiguard_core.wikimedia import DiffStatus, WikimediaClient

DEFAULT_TARGET = 100_000
RETENTION = timedelta(days=30)
# Enwiki trims the oldest rows continually, so stay clear of the very edge.
EDGE_MARGIN = timedelta(hours=6)
MAX_SLICES = 24 * 28
MIN_ROWS_PER_SLICE = 50
BATCH_ROWS = 500
PARQUET_COMPRESSION = "zstd"
SHUFFLE_SEED = 20261104

PARQUET_SCHEMA = pa.schema(
    [
        ("rev_id", pa.int64()),
        ("parent_rev_id", pa.int64()),
        ("schema_version", pa.int32()),
        ("title", pa.string()),
        ("timestamp", pa.timestamp("s", tz="UTC")),
        ("is_minor", pa.bool_()),
        ("is_anonymous", pa.bool_()),
        ("user_hash", pa.string()),
        ("comment", pa.string()),
        ("old_length", pa.int64()),
        ("new_length", pa.int64()),
        ("added_text", pa.string()),
        ("removed_text", pa.string()),
        ("user_edit_count", pa.int64()),
        ("user_hidden", pa.bool_()),
        ("comment_hidden", pa.bool_()),
        ("content_hidden", pa.bool_()),
        ("tagged_reverted", pa.int8()),
        # The tag at capture time, which isn't the label. Never an input to features.
        ("tags_at_capture", pa.list_(pa.string())),
        ("diff_status", pa.string()),
    ]
)


LABELS_SCHEMA = pa.schema(
    [
        ("rev_id", pa.int64()),
        ("label", pa.int8()),
        ("no_label_reason", pa.string()),
        ("revert_hours", pa.float32()),
    ]
)


def plan_slices(now: datetime, target: int) -> tuple[list[datetime], int]:
    """Where each sample starts, newest first, and how many rows to take from each."""
    newest = now - LABEL_WAIT
    oldest = now - RETENTION + EDGE_MARGIN
    hours = int((newest - oldest).total_seconds() // 3600)
    count = min(MAX_SLICES, hours, math.ceil(target / MIN_ROWS_PER_SLICE))
    step = (newest - oldest) / count
    return [newest - step * index for index in range(count)], math.ceil(target / count)


def to_row(raw: dict[str, Any], hash_key: str) -> dict[str, Any]:
    """Keep what the dataset needs, and drop the raw username here."""
    edit = edit_from_recentchange(raw, hash_key)
    return {"edit": edit.model_dump(mode="json"), "tags": raw.get("tags", [])}


def list_edits(
    client: WikimediaClient, out: Path, target: int, hash_key: str, now: datetime
) -> list[dict[str, Any]]:
    """Step 1. The plan is saved first, so a resumed run samples the same moments."""
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        starts = [datetime.fromisoformat(moment) for moment in plan["starts"]]
        per_slice = plan["per_slice"]
    else:
        starts, per_slice = plan_slices(now, target)
        plan_path.write_text(
            json.dumps(
                {
                    "created": now.isoformat(),
                    "target": target,
                    "per_slice": per_slice,
                    "starts": [moment.isoformat() for moment in starts],
                }
            ),
            encoding="utf-8",
        )

    listing = out / "listing.jsonl"
    done, rows = read_listing(listing)
    with listing.open("a", encoding="utf-8", newline="\n") as file:
        for index, start in enumerate(starts):
            if index in done:
                continue
            raw_rows = client.recent_changes(start, per_slice)
            span = 0.0
            if len(raw_rows) > 1:
                newest = datetime.fromisoformat(raw_rows[0]["timestamp"])
                oldest = datetime.fromisoformat(raw_rows[-1]["timestamp"])
                span = (newest - oldest).total_seconds()
            for raw in raw_rows:
                row = to_row(raw, hash_key)
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Written last: a slice without this marker is fetched again.
            marker = {"slice_done": index, "rows": len(raw_rows), "span_seconds": span}
            file.write(json.dumps(marker) + "\n")
            file.flush()
            if (index + 1) % 50 == 0:
                print(f"  listed {index + 1} of {len(starts)} hours", flush=True)
    return unique_rows(rows)


def read_listing(path: Path) -> tuple[set[int], list[dict[str, Any]]]:
    """Rows from finished slices only. A crash can leave a partial slice, which gets refetched."""
    done: set[int] = set()
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    if not path.exists():
        return done, rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "slice_done" in item:
            done.add(item["slice_done"])
            rows.extend(pending)
            pending = []
        else:
            pending.append(item)
    return done, rows


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rev: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_rev[row["edit"]["rev_id"]] = row
    return list(by_rev.values())


def row_for_parquet(row: dict[str, Any], status: DiffStatus, diff_html: str) -> dict[str, Any]:
    edit = EditEvent.model_validate(row["edit"])
    if status is DiffStatus.OK:
        edit = edit.with_diff(parse_diff(diff_html))
    tagged = 1 if REVERTED_TAG in row["tags"] else 0
    return {
        **edit.model_dump(),
        "tagged_reverted": tagged,
        "tags_at_capture": row["tags"],
        "diff_status": status.value,
    }


def done_rev_ids(parts: Path) -> set[int]:
    done: set[int] = set()
    for path in sorted(parts.glob("part-*.parquet")):
        done.update(pq.read_table(path, columns=["rev_id"]).column("rev_id").to_pylist())
    return done


def write_part(parts: Path, batch: list[dict[str, Any]]) -> None:
    number = len(list(parts.glob("part-*.parquet")))
    final = parts / f"part-{number:05d}.parquet"
    # Rename at the end, so a crash never leaves a half-written part that looks finished.
    temporary = final.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(batch, schema=PARQUET_SCHEMA),
        temporary,
        compression=PARQUET_COMPRESSION,
    )
    temporary.replace(final)


def fetch_diffs(client: WikimediaClient, out: Path, rows: list[dict[str, Any]]) -> None:
    """Step 2."""
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    done = done_rev_ids(parts)
    todo = [row for row in rows if row["edit"]["rev_id"] not in done]
    random.Random(SHUFFLE_SEED).shuffle(todo)  # noqa: S311 - sampling order, not security
    print(f"  {len(done)} done, {len(todo)} to fetch", flush=True)

    started = time.monotonic()
    batch: list[dict[str, Any]] = []
    # In a finally, so stopping or losing the network keeps the rows already fetched.
    try:
        for index, row in enumerate(todo, start=1):
            edit = row["edit"]
            if edit["content_hidden"]:
                # Nothing to fetch, but the row is still counted (the plan's censored-edit rule).
                batch.append(row_for_parquet(row, DiffStatus.HIDDEN, ""))
            elif not edit["parent_rev_id"]:
                batch.append(row_for_parquet(row, DiffStatus.MISSING, ""))
            else:
                result = client.compare(edit["parent_rev_id"], edit["rev_id"])
                batch.append(row_for_parquet(row, result.status, result.html))
            if len(batch) >= BATCH_ROWS:
                write_part(parts, batch)
                batch = []
                # A floor, because a fast batch can finish within one tick of the clock.
                rate = index / max(time.monotonic() - started, 0.001)
                hours_left = (len(todo) - index) / rate / 3600
                print(f"  {index} of {len(todo)}; about {hours_left:.1f} hours left", flush=True)
    finally:
        if batch:
            write_part(parts, batch)


def label_for(row: dict[str, Any], hours: float | None) -> tuple[int | None, str | None]:
    """The live rule applied to a backfilled row. Returns the label and any reason for none."""
    if row["diff_status"] != DiffStatus.OK.value:
        return None, row["diff_status"]
    if not row["tagged_reverted"]:
        return 0, None
    if hours is None:
        # Tagged, but the revert is too far along the page history to date.
        return None, "revert_not_dated"
    return (1 if hours <= LABEL_WAIT.total_seconds() / 3600 else 0), None


def build_labels(client: WikimediaClient, out: Path) -> None:
    """Step 3. Each answer is saved as it arrives, so this resumes too."""
    rows = pq.read_table(
        out / "parts",
        columns=["rev_id", "title", "timestamp", "tagged_reverted", "diff_status"],
    ).to_pylist()
    delays_path = out / "revert_hours.jsonl"
    delays: dict[int, float | None] = {}
    if delays_path.exists():
        for line in delays_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            delays[item["rev_id"]] = item["hours"]

    todo = [
        row
        for row in rows
        if row["tagged_reverted"]
        and row["diff_status"] == DiffStatus.OK.value
        and row["rev_id"] not in delays
    ]
    print(f"  {len(delays)} reverts dated, {len(todo)} to go", flush=True)
    with delays_path.open("a", encoding="utf-8", newline="\n") as file:
        for row in todo:
            hours = client.hours_until_next_revert(row["title"], row["rev_id"], row["timestamp"])
            delays[row["rev_id"]] = hours
            file.write(json.dumps({"rev_id": row["rev_id"], "hours": hours}) + "\n")
            file.flush()

    output = []
    for row in rows:
        hours = delays.get(row["rev_id"])
        label, reason = label_for(row, hours)
        output.append(
            {
                "rev_id": row["rev_id"],
                "label": label,
                "no_label_reason": reason,
                "revert_hours": hours,
            }
        )
    temporary = out / "labels_48h.tmp"
    pq.write_table(
        pa.Table.from_pylist(output, schema=LABELS_SCHEMA),
        temporary,
        compression=PARQUET_COMPRESSION,
    )
    temporary.replace(out / "labels_48h.parquet")


def edits_per_second(slices: list[dict[str, Any]]) -> float:
    """The average of each hour's own rate.

    Every hour gives the same number of rows, and a quiet hour takes longer to give them. Dividing
    total rows by total time would count quiet hours for more, and came out 7% too low (F-064).
    """
    rates = [
        (item["rows"] - 1) / item["span_seconds"] for item in slices if item["span_seconds"] > 0
    ]
    return sum(rates) / len(rates)


def summarize(out: Path) -> dict[str, Any]:
    table = pq.read_table(
        out / "parts",
        columns=[
            "rev_id",
            "timestamp",
            "is_anonymous",
            "tagged_reverted",
            "diff_status",
            "user_hash",
            "comment_hidden",
            "content_hidden",
            "user_hidden",
            "added_text",
            "removed_text",
            "comment",
        ],
    )
    rows = table.to_pylist()
    total = len(rows)
    statuses = Counter(row["diff_status"] for row in rows)
    usable = [row for row in rows if row["diff_status"] == DiffStatus.OK.value]
    days = Counter(row["timestamp"].date().isoformat() for row in rows)
    tagged = sum(row["tagged_reverted"] for row in rows)
    labels = pq.read_table(out / "labels_48h.parquet").to_pylist()
    labeled = [row for row in labels if row["label"] is not None]
    positives = sum(row["label"] for row in labeled)
    dated = [row["revert_hours"] for row in labels if row["revert_hours"] is not None]
    no_text = sum(1 for row in usable if not row["added_text"] and not row["removed_text"])
    text_chars = [
        len(row["added_text"] or "") + len(row["removed_text"] or "") + len(row["comment"] or "")
        for row in usable
    ]
    slices = [
        json.loads(line)
        for line in (out / "listing.jsonl").read_text(encoding="utf-8").splitlines()
        if '"slice_done"' in line
    ]
    rate = edits_per_second(slices)
    summary = {
        "rows": total,
        "distinct_days": len(days),
        "first": min(rows, key=lambda row: row["timestamp"])["timestamp"].isoformat(),
        "last": max(rows, key=lambda row: row["timestamp"])["timestamp"].isoformat(),
        "labeled_rows": len(labeled),
        "positives_48h": positives,
        "positive_rate_48h": round(positives / len(labeled), 4),
        "tagged_reverted": tagged,
        "tagged_rate": round(tagged / total, 4),
        "tagged_but_reverted_after_48h": tagged
        - positives
        - sum(1 for row in labels if row["no_label_reason"] == "revert_not_dated")
        - sum(r["tagged_reverted"] for r in rows if r["diff_status"] != DiffStatus.OK.value),
        "no_label": dict(Counter(row["no_label_reason"] for row in labels if row["label"] is None)),
        "median_revert_hours": round(sorted(dated)[len(dated) // 2], 3) if dated else None,
        "usable_rows": len(usable),
        "diff_status": dict(statuses),
        "excluded_rows": total - len(usable),
        "excluded_rate": round((total - len(usable)) / total, 4),
        "excluded_tagged_rate": round(
            sum(r["tagged_reverted"] for r in rows if r["diff_status"] != DiffStatus.OK.value)
            / max(1, total - len(usable)),
            4,
        ),
        "comment_hidden": sum(r["comment_hidden"] for r in rows),
        "content_hidden": sum(r["content_hidden"] for r in rows),
        "user_hidden": sum(r["user_hidden"] for r in rows),
        "temporary_or_ip_share": round(sum(r["is_anonymous"] for r in rows) / total, 4),
        "usable_with_no_changed_text": no_text,
        "mean_text_chars": round(sum(text_chars) / max(1, len(text_chars))),
        "sampled_edits_per_second": round(rate, 3),
        "estimated_edits_per_day": round(rate * 86400),
        "parquet_bytes": sum(path.stat().st_size for path in (out / "parts").glob("*.parquet")),
        "schema_version": SCHEMA_VERSION,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=Path, default=Path("data/backfill"))
    args = parser.parse_args()

    class Settings(WikimediaSettings, PrivacySettings):
        pass

    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        print(error)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    hash_key = settings.user_hash_key.get_secret_value()
    print("Listing edits...", flush=True)
    rows = list_edits(client, args.out, args.target, hash_key, datetime.now(UTC))
    print(f"  {len(rows)} unique edits listed; fetching diffs...", flush=True)
    fetch_diffs(client, args.out, rows)
    print("Dating reverts...", flush=True)
    build_labels(client, args.out)
    summary = summarize(args.out)
    print(json.dumps(summary, indent=2))
    print(f"API requests this run: {client.request_count}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
