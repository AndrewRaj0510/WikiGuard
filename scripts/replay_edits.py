"""Publish real edits to edits.raw at a fixed rate, for the streaming load test (Phase 8, task 3).

The plan asks for the stream at 5x its real rate. Measured on kind against the live stream, English
Wikipedia sent 74.5 article edits a minute (docs/metrics.md, section 19), so 5x is 6.2 a second.

The edits are real ones from the backfill's parquet parts, with their diffs, sampled at random
with a fixed seed. Two things change: each gets the current time as its timestamp, so the scorer's
edit-to-score histogram measures the pipeline and not the edit's age, and its revision id is moved
up by OFFSET, so replayed edits never collide with ones the database already holds.

    python scripts/replay_edits.py --rate 6.2 --minutes 10 --bootstrap 127.0.0.1:19095
"""

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from wikiguard_core.kafka import EDITS_RAW, KafkaPublisher
from wikiguard_core.schemas import EditEvent

PARTS = Path("data/backfill/parts")
OFFSET = 50_000_000_000
REAL_RATE_PER_MINUTE = 74.5
FIELDS = list(EditEvent.model_fields)


def sample(count: int, seed: int = 8) -> list[dict[str, Any]]:
    table = ds.dataset(PARTS, format="parquet").to_table(
        columns=[name for name in FIELDS if name != "schema_version"],
        filter=ds.field("content_hidden") == False,  # noqa: E712 - a pyarrow expression
    )
    rows = table.to_pylist()
    random.Random(seed).shuffle(rows)  # noqa: S311 - a sample, not a secret
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rate", type=float, default=5 * REAL_RATE_PER_MINUTE / 60)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--bootstrap", default="127.0.0.1:19095")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    total = int(args.rate * args.minutes * 60)
    rows = sample(total)
    publisher = KafkaPublisher(args.bootstrap)
    started = time.monotonic()
    sent = 0
    for index, row in enumerate(rows):
        # Hold to the schedule: edit N goes out at N / rate seconds, whatever the last one took.
        delay = started + index / args.rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        row = row | {"rev_id": row["rev_id"] + OFFSET, "timestamp": datetime.now(UTC)}
        if row["parent_rev_id"] is not None:
            row["parent_rev_id"] += OFFSET
        edit = EditEvent(**row)
        publisher.publish(EDITS_RAW, str(edit.rev_id), edit.model_dump_json().encode())
        sent += 1
        if sent % 200 == 0:
            publisher.flush()
    publisher.flush()
    elapsed = time.monotonic() - started
    result = {
        "edits": sent,
        "seconds": round(elapsed, 1),
        "rate_per_second": round(sent / elapsed, 2),
        "times_real_rate": round(sent / elapsed / (REAL_RATE_PER_MINUTE / 60), 2),
        "first_rev_id": rows[0]["rev_id"] + OFFSET if rows else None,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
