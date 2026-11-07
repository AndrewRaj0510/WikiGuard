"""The backfill reader and the pre-publish recheck, on small files written here."""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from backfill import LABELS_SCHEMA, PARQUET_SCHEMA
from build_snapshot import backfill_as_of, read_candidates
from recheck_snapshot import recheck
from wikiguard_core.snapshot import (
    TEST,
    TRAIN,
    VALIDATION,
    build_snapshot,
    read_snapshot,
    verify_snapshot,
    write_snapshot,
)
from wikiguard_core.wikimedia import RevisionState

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
END = AS_OF - timedelta(hours=48)
TEXT_IP = "8.8.8.8"
AT = datetime(2026, 11, 21, tzinfo=UTC)


def source_row(rev_id: int, at: datetime, **extra: Any) -> dict[str, Any]:
    return {
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "schema_version": 1,
        "title": f"Page {rev_id}",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": f"hash-{rev_id}",
        "comment": "fix",
        "old_length": 100,
        "new_length": 110,
        "added_text": f"added {rev_id}",
        "removed_text": "",
        "user_edit_count": None,
        "user_hidden": False,
        "comment_hidden": False,
        "content_hidden": False,
        "tagged_reverted": 0,
        "tags_at_capture": [],
        "diff_status": "ok",
        **extra,
    }


def label_row(rev_id: int, label: int | None, reason: str | None = None) -> dict[str, Any]:
    return {"rev_id": rev_id, "label": label, "no_label_reason": reason, "revert_hours": None}


def write_backfill(folder: Path, rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> None:
    (folder / "parts").mkdir(parents=True)
    parts = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(parts, folder / "parts" / "part-00000.parquet")
    pq.write_table(
        pa.Table.from_pylist(labels, schema=LABELS_SCHEMA), folder / "labels_48h.parquet"
    )
    (folder / "plan.json").write_text(f'{{"created": "{AS_OF.isoformat()}"}}', "utf-8")


@pytest.fixture
def backfill(tmp_path: Path) -> Path:
    folder = tmp_path / "backfill"
    write_backfill(
        folder,
        [
            source_row(1, END - timedelta(days=20)),
            source_row(2, END - timedelta(days=10)),
            source_row(3, END - timedelta(days=1)),
            source_row(4, END - timedelta(days=1), diff_status="hidden", added_text=""),
            source_row(5, END - timedelta(days=1), diff_status="missing", added_text=""),
            source_row(6, END - timedelta(days=1)),
        ],
        [
            label_row(1, 1),
            label_row(2, 0),
            label_row(3, 1),
            label_row(4, None, "hidden"),
            label_row(5, None, "missing"),
            label_row(6, None, "revert_not_dated"),
        ],
    )
    return folder


def test_as_of_comes_from_when_the_backfill_started(backfill: Path) -> None:
    assert backfill_as_of(backfill) == AS_OF


def test_backfill_becomes_a_snapshot(backfill: Path) -> None:
    snapshot = build_snapshot(read_candidates(backfill), AS_OF, "backfill")

    assert {row["rev_id"]: row["split"] for row in snapshot.rows} == {
        1: TRAIN,
        2: VALIDATION,
        3: TEST,
    }
    assert snapshot.manifest["excluded"] == {"hidden": 1, "missing": 1, "revert_not_dated": 1}


def test_label_comes_from_the_dated_file_and_never_the_raw_tag(tmp_path: Path) -> None:
    """A late revert: the tag says reverted, and the 48-hour label says it wasn't (F-058)."""
    folder = tmp_path / "backfill"
    write_backfill(
        folder,
        [
            source_row(
                1, END - timedelta(days=20), tagged_reverted=1, tags_at_capture=["mw-reverted"]
            ),
            source_row(2, END - timedelta(days=10), tagged_reverted=1),
            source_row(3, END - timedelta(days=1)),
        ],
        [label_row(1, 0), label_row(2, 0), label_row(3, 1)],
    )

    labels = {item.rev_id: item.label for item in read_candidates(folder)}

    assert labels == {1: 0, 2: 0, 3: 1}


def test_a_row_with_no_label_record_is_counted(tmp_path: Path) -> None:
    folder = tmp_path / "backfill"
    write_backfill(folder, [source_row(1, END)], [])

    candidate = next(read_candidates(folder))

    assert candidate.edit is None
    assert candidate.excluded_reason == "no_label_record"


def test_addresses_the_old_redactor_missed_are_removed(tmp_path: Path) -> None:
    folder = tmp_path / "backfill"
    write_backfill(
        folder,
        [
            source_row(
                1,
                END - timedelta(days=20),
                added_text=f"from {TEXT_IP}.",
                removed_text=f"was {TEXT_IP}.",
                comment="Revert 2001:db8:85a3:0:0:8a2e:370:7334: spam",
            ),
            source_row(2, END - timedelta(days=10)),
            source_row(3, END - timedelta(days=1)),
        ],
        [label_row(1, 1), label_row(2, 0), label_row(3, 0)],
    )

    rows = build_snapshot(read_candidates(folder), AS_OF, "backfill").rows
    row = next(item for item in rows if item["rev_id"] == 1)

    for field in ("added_text", "removed_text", "comment"):
        assert TEXT_IP not in row[field]
        assert "7334" not in row[field]
        assert "[ip address]" in row[field]


class Client:
    """Answers for every revision unless told not to, and everything is visible by default."""

    def __init__(
        self, hidden: dict[int, RevisionState] | None = None, unanswered: Iterable[int] = ()
    ) -> None:
        self.hidden = hidden or {}
        self.unanswered = set(unanswered)
        self.asked: list[int] = []

    def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]:
        self.asked = list(rev_ids)
        return {
            rev_id: self.hidden.get(rev_id, RevisionState(rev_id))
            for rev_id in self.asked
            if rev_id not in self.unanswered
        }


@pytest.fixture
def folder(backfill: Path, tmp_path: Path) -> Path:
    snapshot = build_snapshot(read_candidates(backfill), AS_OF, "backfill")
    return write_snapshot(snapshot, tmp_path / "snapshots")


@pytest.mark.parametrize(
    "state",
    [
        RevisionState(3, missing=True),
        RevisionState(3, content_hidden=True),
        RevisionState(3, comment_hidden=True),
        RevisionState(3, user_hidden=True),
    ],
)
def test_recheck_removes_a_revision_hidden_in_any_way(folder: Path, state: RevisionState) -> None:
    client = Client({3: state})

    results = recheck(folder, client, AT)

    assert client.asked == [1, 2, 3]
    assert results["checked"] == 3
    assert results["rows_removed"] == 1
    assert {row["rev_id"] for row in read_snapshot(folder)} == {1, 2}
    assert verify_snapshot(folder)["purges"][0]["rows_removed"] == 1


def test_recheck_of_a_clean_snapshot_changes_nothing(folder: Path) -> None:
    before = verify_snapshot(folder)["content_sha256"]

    results = recheck(folder, Client(), AT)

    assert results["rows_removed"] == 0
    assert verify_snapshot(folder)["content_sha256"] == before


def test_recheck_stops_if_a_revision_gets_no_answer(folder: Path) -> None:
    before = verify_snapshot(folder)["content_sha256"]

    with pytest.raises(RuntimeError, match="1 revisions"):
        recheck(folder, Client({3: RevisionState(3, missing=True)}, unanswered=[1]), AT)

    assert verify_snapshot(folder)["content_sha256"] == before
