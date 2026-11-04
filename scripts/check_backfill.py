"""Check that a backfill doesn't flatter the metrics later. It writes correctness.json beside it.

Three checks from section 2 of the plan:

1. Revert delay. The live label means "reverted within 48 hours", but a backfilled tag can come
   from a later revert. Samples reverted rows and finds the first revert after each. Over 2% late
   means the 48-hour labels from step 3 of scripts/backfill.py are needed, not the raw tag.
2. Point-in-time edit counts. Shows how far today's count is from the count at edit time, which is
   the size of the leak that leaving the feature missing on backfilled rows avoids (DEC-036).
3. Excluded revisions, counted so they aren't silently dropped.

Usernames are fetched for the point-in-time check, used, and never written anywhere.

    python scripts/check_backfill.py data/backfill_pilot
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from wikiguard_core.config import ConfigError, WikimediaSettings, load_settings
from wikiguard_core.features import edit_count_bucket
from wikiguard_core.identity import is_temporary_account
from wikiguard_core.labels import LABEL_WAIT
from wikiguard_core.wikimedia import WikimediaClient, chunks

REVERT_SAMPLE = 500
COUNT_SAMPLE = 200
# Point-in-time counts stop here, which also stops one very active editor taking many requests.
COUNT_CAP = 1000
MAX_LATE_SHARE = 0.02
SEED = 20261104


def check_revert_delay(client: WikimediaClient, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reverted = [row for row in rows if row["tagged_reverted"] == 1 and row["diff_status"] == "ok"]
    sample = random.Random(SEED).sample(reverted, min(REVERT_SAMPLE, len(reverted)))  # noqa: S311
    delays = [
        client.hours_until_next_revert(row["title"], row["rev_id"], row["timestamp"])
        for row in sample
    ]
    found = [delay for delay in delays if delay is not None]
    late = sum(1 for delay in found if delay > LABEL_WAIT.total_seconds() / 3600)
    return {
        "reverted_rows_available": len(reverted),
        "sampled": len(sample),
        "revert_not_found": len(sample) - len(found),
        "median_hours": round(statistics.median(found), 3) if found else None,
        "within_1_hour": sum(1 for delay in found if delay <= 1),
        "later_than_48_hours": late,
        "late_share": round(late / len(found), 4) if found else None,
        "over_threshold": bool(found) and late / len(found) > MAX_LATE_SHARE,
        "threshold": MAX_LATE_SHARE,
    }


def usernames_for(client: WikimediaClient, rev_ids: list[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for batch in chunks(rev_ids):
        body = client.query(
            action="query",
            prop="revisions",
            revids="|".join(str(rev_id) for rev_id in batch),
            rvprop="ids|user",
        )
        for page in body.get("query", {}).get("pages", []):
            for revision in page.get("revisions", []):
                if revision.get("user"):
                    names[revision["revid"]] = revision["user"]
    return names


def check_edit_counts(client: WikimediaClient, rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if not row["user_hidden"]]
    sample = random.Random(SEED).sample(usable, min(COUNT_SAMPLE, len(usable)))  # noqa: S311
    names = usernames_for(client, [row["rev_id"] for row in sample])
    today = client.user_edit_counts(sorted(set(names.values())))

    bucket_changed = 0
    capped = 0
    compared = 0
    temporary = 0
    inflation: list[float] = []
    for row in sample:
        name = names.get(row["rev_id"])
        if name is None or name not in today:
            continue
        before = client.edits_before(name, row["timestamp"], COUNT_CAP)
        compared += 1
        temporary += is_temporary_account(name)
        if before >= COUNT_CAP:
            capped += 1
            continue
        now = today[name]
        bucket_changed += edit_count_bucket(before) != edit_count_bucket(now)
        inflation.append(max(0, now - before))
    return {
        "sampled": len(sample),
        "compared": compared,
        "temporary_accounts": temporary,
        "over_the_cap_not_compared": capped,
        "bucket_would_change": bucket_changed,
        "bucket_change_share": round(bucket_changed / max(1, compared - capped), 4),
        "median_extra_edits_since": statistics.median(inflation) if inflation else None,
        "cap": COUNT_CAP,
    }


def check_exclusions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row["diff_status"] for row in rows)
    excluded = [row for row in rows if row["diff_status"] != "ok"]
    return {
        "rows": len(rows),
        "by_status": dict(statuses),
        "excluded": len(excluded),
        "excluded_share": round(len(excluded) / len(rows), 4),
        "excluded_that_were_reverted": sum(row["tagged_reverted"] for row in excluded),
        "comment_hidden_but_usable": sum(
            1 for row in rows if row["comment_hidden"] and row["diff_status"] == "ok"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    try:
        settings = load_settings(WikimediaSettings)
    except ConfigError as error:
        print(error)
        return 2

    rows = pq.read_table(
        args.folder / "parts",
        columns=[
            "rev_id",
            "title",
            "timestamp",
            "tagged_reverted",
            "diff_status",
            "user_hidden",
            "comment_hidden",
        ],
    ).to_pylist()
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    report = {
        "exclusions": check_exclusions(rows),
        "revert_delay": check_revert_delay(client, rows),
        "edit_counts": check_edit_counts(client, rows),
        "api_requests": client.request_count,
    }
    client.close()
    (args.folder / "correctness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
