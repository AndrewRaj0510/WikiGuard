from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from archiver import Archiver, clean_up
from pydantic import ValidationError
from wikiguard_core.archive import (
    EDITS,
    LABELS,
    SCORES,
    delete_older_than,
    purge_revisions,
    read_rows,
    size_in_bytes,
    write_rows,
)
from wikiguard_core.config import ArchiveSettings, load_settings
from wikiguard_core.kafka import EDITS_RAW, EDITS_SCORED, LABELS_FINAL
from wikiguard_core.schemas import EditEvent, LabelRecord, ScoreRecord

DAY = datetime(2026, 11, 3, 12, tzinfo=UTC)


def edit(rev_id: int, when: datetime = DAY, text: str = "new words") -> EditEvent:
    return EditEvent(
        rev_id=rev_id,
        parent_rev_id=rev_id - 1,
        title="Moon",
        timestamp=when,
        is_minor=False,
        is_anonymous=True,
        user_hash="hash",
        comment="a comment",
        old_length=1,
        new_length=2,
        added_text=text,
        removed_text="old words",
    )


class FakeMessage:
    def __init__(self, topic: str, value: bytes, error: str | None = None) -> None:
        self._topic, self._value, self._error = topic, value, error

    def error(self) -> str | None:
        return self._error

    def topic(self) -> str:
        return self._topic

    def value(self) -> bytes:
        return self._value


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage] | None = None) -> None:
        self.messages = messages or []
        self.commits = 0

    def poll(self, timeout: float) -> FakeMessage | None:
        return self.messages.pop(0) if self.messages else None

    def commit(self, asynchronous: bool = False) -> None:
        self.commits += 1


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def messages_for(*edits: EditEvent) -> list[FakeMessage]:
    return [FakeMessage(EDITS_RAW, item.model_dump_json().encode()) for item in edits]


def run_all(archiver: Archiver, consumer: FakeConsumer) -> None:
    while consumer.messages:
        archiver.poll_once()


def test_all_three_topics_reach_their_datasets(tmp_path: Path) -> None:
    score = ScoreRecord(rev_id=1, model_name="mlp", model_version="1", score=0.5, scored_at=DAY)
    label = LabelRecord(rev_id=1, label=1, decided_at=DAY)
    consumer = FakeConsumer(
        [
            *messages_for(edit(1)),
            FakeMessage(EDITS_SCORED, score.model_dump_json().encode()),
            FakeMessage(LABELS_FINAL, label.model_dump_json().encode()),
        ]
    )
    archiver = Archiver(consumer, tmp_path)

    run_all(archiver, consumer)
    archiver.flush()

    assert read_rows(tmp_path, EDITS)[0]["added_text"] == "new words"
    assert read_rows(tmp_path, SCORES)[0]["model_name"] == "mlp"
    assert read_rows(tmp_path, LABELS)[0]["label"] == 1


def test_nothing_is_committed_before_it_is_written(tmp_path: Path) -> None:
    consumer = FakeConsumer(messages_for(edit(1), edit(2)))
    archiver = Archiver(consumer, tmp_path, flush_rows=10)

    run_all(archiver, consumer)

    assert consumer.commits == 0
    assert read_rows(tmp_path, EDITS) == []


def test_flushes_at_the_row_limit_and_commits(tmp_path: Path) -> None:
    consumer = FakeConsumer(messages_for(edit(1), edit(2)))
    archiver = Archiver(consumer, tmp_path, flush_rows=2)

    run_all(archiver, consumer)

    assert consumer.commits == 1
    assert len(read_rows(tmp_path, EDITS)) == 2


def test_flushes_after_the_time_limit(tmp_path: Path) -> None:
    clock = Clock()
    consumer = FakeConsumer(messages_for(edit(1)))
    archiver = Archiver(consumer, tmp_path, flush_rows=100, flush_seconds=60, clock=clock)
    archiver.poll_once()
    assert consumer.commits == 0

    clock.now = 61
    archiver.poll_once()

    assert consumer.commits == 1


def test_a_crash_before_commit_leaves_duplicates_that_readers_remove(tmp_path: Path) -> None:
    # After a crash, the consumer gets the same messages again.
    write_rows(tmp_path, EDITS, [edit(1).model_dump()])
    consumer = FakeConsumer(messages_for(edit(1), edit(2)))
    archiver = Archiver(consumer, tmp_path, flush_rows=2)

    run_all(archiver, consumer)

    assert sorted(row["rev_id"] for row in read_rows(tmp_path, EDITS)) == [1, 2]


def test_poison_message_is_skipped_and_the_rest_continue(tmp_path: Path) -> None:
    consumer = FakeConsumer(
        [FakeMessage(EDITS_RAW, b"not json"), *messages_for(edit(1))],
    )
    archiver = Archiver(consumer, tmp_path, flush_rows=1)

    run_all(archiver, consumer)

    assert [row["rev_id"] for row in read_rows(tmp_path, EDITS)] == [1]
    assert archiver.registry.get_sample_value("archiver_poison_messages_total") == 1


def test_consumer_error_does_not_stop_the_loop(tmp_path: Path) -> None:
    consumer = FakeConsumer([FakeMessage(EDITS_RAW, b"", error="broker down")])
    archiver = Archiver(consumer, tmp_path)

    archiver.poll_once()

    assert archiver.buffers == {}


def test_rows_land_in_the_folder_for_their_own_day(tmp_path: Path) -> None:
    rows = [
        edit(1, datetime(2026, 11, 1, 23, 59, tzinfo=UTC)).model_dump(),
        edit(2, datetime(2026, 11, 2, 0, 1, tzinfo=UTC)).model_dump(),
    ]

    write_rows(tmp_path, EDITS, rows)

    days = sorted(path.name for path in (tmp_path / EDITS).iterdir())
    assert days == ["date=2026-11-01", "date=2026-11-02"]


def test_no_unfinished_files_are_left_behind(tmp_path: Path) -> None:
    write_rows(tmp_path, EDITS, [edit(1).model_dump()])

    assert not list(tmp_path.rglob("*.tmp"))


def test_cleanup_deletes_only_what_is_past_the_window(tmp_path: Path) -> None:
    now = datetime(2026, 11, 30, tzinfo=UTC)
    write_rows(
        tmp_path,
        EDITS,
        [
            edit(1, now - timedelta(days=31)).model_dump(),
            edit(2, now - timedelta(days=29)).model_dump(),
            edit(3, now - timedelta(hours=1)).model_dump(),
        ],
    )

    assert clean_up(tmp_path, 30, now) == 1
    assert sorted(row["rev_id"] for row in read_rows(tmp_path, EDITS)) == [2, 3]


def test_cleanup_ignores_folders_it_does_not_recognize(tmp_path: Path) -> None:
    folder = tmp_path / EDITS / "date=not-a-date"
    folder.mkdir(parents=True)

    assert delete_older_than(tmp_path, date(2026, 11, 1)) == 0
    assert folder.exists()


def test_retention_window_can_not_reach_edits_waiting_for_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARCHIVE_RETENTION_DAYS", "2")

    with pytest.raises(Exception, match="ARCHIVE_RETENTION_DAYS"):
        load_settings(ArchiveSettings, env_file=None)
    with pytest.raises(ValidationError):
        ArchiveSettings(archive_retention_days=2, _env_file=None)  # type: ignore[call-arg]


def test_rebuild_from_the_archive_restores_every_row(tmp_path: Path) -> None:
    originals = [edit(rev, DAY - timedelta(days=rev % 3)) for rev in range(1, 40)]
    consumer = FakeConsumer(messages_for(*originals))
    archiver = Archiver(consumer, tmp_path, flush_rows=7)
    run_all(archiver, consumer)
    archiver.flush()

    rebuilt = sorted(read_rows(tmp_path, EDITS), key=lambda row: row["rev_id"])

    assert len(rebuilt) == len(originals)
    for row, original in zip(rebuilt, originals, strict=True):
        assert EditEvent.model_validate(row) == original


def test_purge_blanks_text_and_keeps_the_row(tmp_path: Path) -> None:
    write_rows(tmp_path, EDITS, [edit(1).model_dump(), edit(2).model_dump()])

    assert purge_revisions(tmp_path, {1, 99}) == 1

    by_rev: dict[int, dict[str, Any]] = {r["rev_id"]: r for r in read_rows(tmp_path, EDITS)}
    assert by_rev[1]["added_text"] is None
    assert by_rev[1]["removed_text"] is None
    assert by_rev[1]["comment"] is None
    assert by_rev[1]["user_hash"] is None
    assert by_rev[2]["added_text"] == "new words"


def test_size_counts_parquet_files(tmp_path: Path) -> None:
    write_rows(tmp_path, EDITS, [edit(1).model_dump()])

    assert size_in_bytes(tmp_path) > 0
