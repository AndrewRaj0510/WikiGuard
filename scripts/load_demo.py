"""Load the public demo's data: the snapshot's test window, scored by the champion (DEC-091).

No free server can run the streaming pipeline around the clock, so the demo serves a fixed set of
real edits instead: the snapshot's test window, which no model trained on, with labels that are
already final. Each edit gets the champion's stored test score, the number the evaluation measured
(`docs/metrics.md`), so nothing is rescored and the demo can't drift from the reported results.

Where each part comes from:
- **Which edits, and their labels:** the snapshot's test rows. Only rows still in the snapshot, so
  revisions purged from it since the backfill stay out.
- **The edits' fields** (title, comment, sizes, text, editor hash): the backfill's parquet parts,
  which have exactly the `edits` table's columns.
- **The score:** `data/runs/MODEL/scores.npz`, the served artifact's test logits, turned into the
  served probability the same way serving does: sigmoid(logit / temperature).

Run the migrations and `scripts/sync_model_versions.py` first, then run the API with
`DATA_MODE=snapshot`. Loading twice changes nothing. It refuses a database that already holds edits
from outside the dataset, such as the live pipeline's. Uses DATABASE_MIGRATIONS_URL, the owner role:

    python scripts/load_demo.py snapshots/snapshot-20261104-28c825af4b19 --dry-run
    python scripts/load_demo.py snapshots/snapshot-20261104-28c825af4b19

Before the demo goes public, and now and then after, run the purge job against the same database:
nothing runs it on a schedule any more (F-151).
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pyarrow.dataset as ds
from wikiguard_core.config import ConfigError, MigrationSettings, load_settings
from wikiguard_core.evaluation import sigmoid
from wikiguard_core.snapshot import read_snapshot

TEST_SPLIT = "test"
CHAMPION = "champion"
# Postgres `real` holds about 7 digits, and refuses values too small to hold (F-104).
SCORE_DECIMALS = 7
# The live labeler's rule: a label is final 48 hours after the edit.
LABEL_DELAY = timedelta(hours=48)
EDIT_COLUMNS = (
    "rev_id",
    "parent_rev_id",
    "schema_version",
    "title",
    "edited_at",
    "is_minor",
    "is_anonymous",
    "user_hash",
    "comment",
    "old_length",
    "new_length",
    "added_text",
    "removed_text",
    "user_edit_count",
    "user_hidden",
    "comment_hidden",
    "content_hidden",
)
TEXT_COLUMNS = ("title", "comment", "added_text", "removed_text")
SCORE_COLUMNS = ("rev_id", "model_name", "model_version", "score", "alias", "threshold")
LABEL_COLUMNS = ("rev_id", "label", "decided_at")


@dataclass(frozen=True)
class Champion:
    model_name: str
    version: str
    threshold: float | None
    scores: dict[int, float]


@dataclass
class Dataset:
    edits: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[int, int] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def snapshot_test_labels(folder: Path) -> dict[int, int]:
    """read_snapshot checks the hash and feature version first."""
    return {
        int(row["rev_id"]): int(row["label"])
        for row in read_snapshot(folder)
        if row["split"] == TEST_SPLIT
    }


def read_backfill(parts: Path, wanted: set[int]) -> list[dict[str, Any]]:
    """The backfill's rows for these revisions. Its `timestamp` is the table's `edited_at`."""
    dataset = ds.dataset(sorted(parts.glob("*.parquet")), format="parquet")
    names = [name if name != "edited_at" else "timestamp" for name in EDIT_COLUMNS]
    table = dataset.to_table(columns=names, filter=ds.field("rev_id").isin(sorted(wanted)))
    rows: list[dict[str, Any]] = table.to_pylist()
    for row in rows:
        row["edited_at"] = row.pop("timestamp")
    return rows


def champion_scores(runs: Path) -> Champion:
    """The champion's served probability for each test revision, as serving computes it."""
    aliases = json.loads((runs / "aliases.json").read_text("utf-8"))
    entry = aliases[CHAMPION]
    name = entry["model"]
    config = json.loads((runs / name / "model" / "config.json").read_text("utf-8"))
    stored = np.load(runs / name / "scores.npz", allow_pickle=False)
    probabilities = sigmoid(stored["test_logit"] / float(config.get("temperature", 1.0)))
    return Champion(
        model_name=name,
        version=str(entry["version"]),
        threshold=config.get("threshold_probability"),
        scores={
            int(rev): round(float(value), SCORE_DECIMALS)
            for rev, value in zip(stored["test_rev_id"], probabilities, strict=True)
        },
    )


def build_dataset(
    labels: dict[int, int], backfill: list[dict[str, Any]], champion: Champion
) -> Dataset:
    """Edits with a label, a score, and text Postgres will take. Everything else is counted out."""
    dataset = Dataset()
    found = {int(row["rev_id"]): row for row in backfill}
    for rev_id in sorted(labels):
        row = found.get(rev_id)
        if row is None:
            dataset.skip("not_in_backfill")
        elif rev_id not in champion.scores:
            dataset.skip("no_stored_score")
        elif row["content_hidden"] or row["user_hidden"] or row["comment_hidden"]:
            # The snapshot already left these out; this is a second check, not the first.
            dataset.skip("hidden")
        elif any("\x00" in (row[column] or "") for column in TEXT_COLUMNS):
            # Postgres text can't hold a NUL byte; the live scorer dead-letters these too.
            dataset.skip("nul_byte")
        else:
            dataset.edits.append(row)
            dataset.labels[rev_id] = labels[rev_id]
            dataset.scores[rev_id] = champion.scores[rev_id]
    return dataset


def check_target(connection: psycopg.Connection[Any], rev_ids: set[int]) -> None:
    """Refuse a database holding other edits: the demo must never mix with live data."""
    found = connection.execute("SELECT rev_id FROM edits").fetchall()
    connection.commit()
    others = sum(1 for (rev_id,) in found if rev_id not in rev_ids)
    if others:
        raise SystemExit(
            f"Refusing: this database holds {others} edits that aren't in the demo's dataset. "
            "Load the demo into its own database."
        )


def _copy(cursor: psycopg.Cursor[Any], table: str, columns: tuple[str, ...], rows: Any) -> None:
    with cursor.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def load(
    connection: psycopg.Connection[Any],
    dataset: Dataset,
    champion: Champion,
    scored_at: datetime,
) -> dict[str, int]:
    """One transaction. COPY into temporary tables, then insert what isn't there yet."""
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE demo_edits (LIKE edits INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        cursor.execute(
            """
            CREATE TEMP TABLE demo_scores (
                rev_id bigint, model_name text, model_version text, score real, alias text,
                threshold real
            ) ON COMMIT DROP
            """
        )
        cursor.execute(
            "CREATE TEMP TABLE demo_labels (rev_id bigint, label smallint, decided_at timestamptz)"
            " ON COMMIT DROP"
        )
        _copy(
            cursor,
            "demo_edits",
            EDIT_COLUMNS,
            ([row[column] for column in EDIT_COLUMNS] for row in dataset.edits),
        )
        _copy(
            cursor,
            "demo_scores",
            SCORE_COLUMNS,
            (
                (rev, champion.model_name, champion.version, score, CHAMPION, champion.threshold)
                for rev, score in dataset.scores.items()
            ),
        )
        edited = {int(row["rev_id"]): row["edited_at"] for row in dataset.edits}
        _copy(
            cursor,
            "demo_labels",
            LABEL_COLUMNS,
            ((rev, label, edited[rev] + LABEL_DELAY) for rev, label in dataset.labels.items()),
        )
        columns = ", ".join(EDIT_COLUMNS)
        cursor.execute(
            # The column names are this file's constants, never input.
            f"INSERT INTO edits ({columns}) SELECT {columns} FROM demo_edits "  # noqa: S608
            "ON CONFLICT (rev_id) DO NOTHING"
        )
        edits = cursor.rowcount
        cursor.execute(
            """
            INSERT INTO scores (rev_id, model_name, model_version, score, used_fallback, scored_at,
                alias, threshold, load_shed, forced)
            SELECT rev_id, model_name, model_version, score, false, %s, alias, threshold, false,
                false
            FROM demo_scores
            ON CONFLICT (rev_id, model_version) DO NOTHING
            """,
            (scored_at,),
        )
        scores = cursor.rowcount
        cursor.execute(
            """
            UPDATE edits e SET latest_score = s.score, latest_model_version = s.model_version,
                latest_scored_at = s.scored_at
            FROM scores s
            WHERE s.rev_id = e.rev_id AND s.model_version = %s AND e.latest_score IS NULL
            """,
            (champion.version,),
        )
        cursor.execute(
            """
            INSERT INTO labels (rev_id, label, decided_at)
            SELECT rev_id, label, decided_at FROM demo_labels
            ON CONFLICT (rev_id) DO NOTHING
            """
        )
        labels = cursor.rowcount
    return {"edits": edits, "scores": scores, "labels": labels}


def table_bytes(connection: psycopg.Connection[Any]) -> dict[str, int]:
    rows = connection.execute(
        "SELECT name, pg_total_relation_size(name::regclass) FROM unnest(%s::text[]) AS name",
        (["edits", "scores", "labels"],),
    ).fetchall()
    connection.commit()
    return {name: int(size) for name, size in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--backfill", type=Path, default=Path("data/backfill/parts"))
    parser.add_argument("--runs", type=Path, default=Path("data/runs"))
    parser.add_argument("--dry-run", action="store_true", help="count what would load, and stop")
    args = parser.parse_args()

    labels = snapshot_test_labels(args.snapshot)
    champion = champion_scores(args.runs)
    dataset = build_dataset(labels, read_backfill(args.backfill, set(labels)), champion)
    positives = sum(dataset.labels.values())
    first = min((row["edited_at"] for row in dataset.edits), default=None)
    last = max((row["edited_at"] for row in dataset.edits), default=None)
    print(  # noqa: T201
        f"{len(dataset.edits)} edits ({positives} reverted), {first} to {last}, scored by "
        f"{champion.model_name} v{champion.version}. Left out: {dataset.skipped or 'none'}"
    )
    if args.dry_run:
        return 0
    try:
        settings = load_settings(MigrationSettings)
    except ConfigError as error:
        print(error, file=sys.stderr)  # noqa: T201
        return 2
    with psycopg.connect(
        settings.database_migrations_url.get_secret_value(),
        prepare_threshold=None,
        connect_timeout=10,
    ) as connection:
        check_target(connection, {int(row["rev_id"]) for row in dataset.edits})
        added = load(connection, dataset, champion, datetime.now(UTC))
        print(f"Added {added}. Table sizes in bytes: {table_bytes(connection)}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
