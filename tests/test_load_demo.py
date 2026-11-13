"""The demo loader (DEC-091): the right edits, the served score, real labels, and no mixing."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import load_demo
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EDITED = datetime(2026, 10, 27, 12, tzinfo=UTC)


def backfill_row(rev_id: int, **changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "schema_version": 1,
        "title": "Moon",
        "edited_at": EDITED + timedelta(minutes=rev_id),
        "is_minor": False,
        "is_anonymous": rev_id % 2 == 0,
        "user_hash": "hash",
        "comment": "a comment",
        "old_length": 100,
        "new_length": 120,
        "added_text": "added words",
        "removed_text": "removed words",
        "user_edit_count": None,
        "user_hidden": False,
        "comment_hidden": False,
        "content_hidden": False,
    }
    return row | changes


def champion(*rev_ids: int) -> load_demo.Champion:
    return load_demo.Champion("roberta", "1", 0.9, {rev: 0.1 * rev for rev in rev_ids})


def test_only_labelled_scored_clean_rows_are_kept() -> None:
    labels = {1: 1, 2: 0, 3: 0, 4: 1, 5: 0}
    backfill = [
        backfill_row(1),
        backfill_row(2),
        backfill_row(3, content_hidden=True),
        backfill_row(4, added_text="bad\x00byte"),
        backfill_row(6),  # Not in the test window: never loaded.
    ]
    dataset = load_demo.build_dataset(labels, backfill, champion(1, 2, 3, 4))
    assert [row["rev_id"] for row in dataset.edits] == [1, 2]
    assert dataset.labels == {1: 1, 2: 0}
    assert dataset.scores == {1: 0.1, 2: 0.2}
    assert dataset.skipped == {"hidden": 1, "nul_byte": 1, "not_in_backfill": 1}


def test_a_row_with_no_stored_score_is_left_out() -> None:
    dataset = load_demo.build_dataset({1: 0}, [backfill_row(1)], champion())
    assert dataset.edits == []
    assert dataset.skipped == {"no_stored_score": 1}


def test_the_champion_score_is_the_served_probability(tmp_path: Path) -> None:
    """Serving divides the logit by the temperature before the sigmoid; so must the demo."""
    (tmp_path / "aliases.json").write_text(
        json.dumps({"champion": {"model": "roberta", "version": 1}}), "utf-8"
    )
    model = tmp_path / "roberta" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"temperature": 2.0, "threshold_probability": 0.97}), "utf-8"
    )
    np.savez(
        tmp_path / "roberta" / "scores.npz",
        test_rev_id=np.array([10, 11]),
        test_logit=np.array([0.0, 4.0]),
    )
    loaded = load_demo.champion_scores(tmp_path)
    assert (loaded.model_name, loaded.version, loaded.threshold) == ("roberta", "1", 0.97)
    assert loaded.scores[10] == 0.5
    assert loaded.scores[11] == round(1 / (1 + np.exp(-2.0)), 7)


def test_the_backfill_is_read_by_rev_id_with_the_tables_column_names(tmp_path: Path) -> None:
    rows = [backfill_row(rev) for rev in (1, 2, 3)]
    for row in rows:
        row["timestamp"] = row.pop("edited_at")
        row["tagged_reverted"] = 0  # A backfill column the table doesn't have.
    pq.write_table(pa.Table.from_pylist(rows[:2]), tmp_path / "part-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), tmp_path / "part-00001.parquet")
    found = load_demo.read_backfill(tmp_path, {1, 3})
    assert sorted(row["rev_id"] for row in found) == [1, 3]
    assert set(found[0]) == set(load_demo.EDIT_COLUMNS)
    assert found[0]["edited_at"] == EDITED + timedelta(minutes=found[0]["rev_id"])


@pytest.mark.integration
def test_loading_writes_edits_scores_and_labels_once(db: psycopg.Connection[Any]) -> None:
    backfill = [backfill_row(rev) for rev in (1, 2, 3)]
    dataset = load_demo.build_dataset({1: 1, 2: 0, 3: 0}, backfill, champion(1, 2, 3))
    scored_at = datetime(2026, 11, 13, tzinfo=UTC)
    added = load_demo.load(db, dataset, champion(1, 2, 3), scored_at)
    assert added == {"edits": 3, "scores": 3, "labels": 3}

    rows = db.execute(
        """
        SELECT e.rev_id, e.latest_score, e.latest_model_version, s.alias, s.threshold,
               s.used_fallback, s.scored_at, l.label, l.decided_at - e.edited_at
        FROM edits e JOIN scores s USING (rev_id) JOIN labels l USING (rev_id)
        ORDER BY e.rev_id
        """
    ).fetchall()
    db.commit()
    assert [(row[0], round(row[1], 4), row[2], row[3]) for row in rows] == [
        (1, 0.1, "1", "champion"),
        (2, 0.2, "1", "champion"),
        (3, 0.3, "1", "champion"),
    ]
    assert all(row[5] is False and row[6] == scored_at for row in rows)
    assert [row[7] for row in rows] == [1, 0, 0]
    assert all(row[8] == timedelta(hours=48) for row in rows)

    # A second load adds nothing and changes nothing.
    assert load_demo.load(db, dataset, champion(1, 2, 3), datetime.now(UTC)) == {
        "edits": 0,
        "scores": 0,
        "labels": 0,
    }
    load_demo.check_target(db, {1, 2, 3})


@pytest.mark.integration
def test_a_database_with_other_edits_is_refused(db: psycopg.Connection[Any]) -> None:
    dataset = load_demo.build_dataset({1: 0}, [backfill_row(1)], champion(1))
    load_demo.load(db, dataset, champion(1), datetime.now(UTC))
    with pytest.raises(SystemExit, match="1 edits that aren't in the demo"):
        load_demo.check_target(db, {2})
