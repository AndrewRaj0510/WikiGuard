"""The backup's encryption, retention, and a real round trip: back up, restore, compare."""

import importlib.util
import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pyarrow.fs as pafs
import pytest
from alembic import command
from alembic.config import Config
from backup import (
    BACKUP_TABLES,
    BackupError,
    Settings,
    decrypt,
    encrypt,
    export_tables,
    keygen,
    private_key_from_text,
    prune,
    public_key_from_text,
    read_bundle,
    run,
    store_for,
    stored,
)
from restore import Settings as RestoreSettings
from restore import restore
from wikiguard_core.archive import EDITS, LABELS, SCORES, Archive, write_rows

URL = os.environ.get("TEST_DATABASE_URL", "")
INI = Path(__file__).parents[2] / "migrate" / "alembic.ini"
MIGRATION = (
    Path(__file__).parents[2] / "migrate/migrations/versions/0005_backup_and_observer_roles.py"
)
NOW = datetime(2026, 11, 15, 3, tzinfo=UTC)  # a Sunday


# --- Encryption ---


def test_a_round_trip_returns_the_same_bytes() -> None:
    public, private = keygen()
    blob = encrypt(b"reviews and audit log", public_key_from_text(public))
    assert b"reviews" not in blob
    assert decrypt(blob, private_key_from_text(private)) == b"reviews and audit log"


def test_the_wrong_key_is_refused() -> None:
    public, _ = keygen()
    _, other = keygen()
    blob = encrypt(b"secret", public_key_from_text(public))
    with pytest.raises(BackupError, match="wrong key"):
        decrypt(blob, private_key_from_text(other))


@pytest.mark.parametrize("position", [3, 10, 40, -1])
def test_any_changed_byte_is_detected(position: int) -> None:
    public, private = keygen()
    blob = bytearray(encrypt(b"x" * 100, public_key_from_text(public)))
    blob[position] ^= 0x01
    with pytest.raises(BackupError):
        decrypt(bytes(blob), private_key_from_text(private))


def test_each_file_uses_a_new_one_time_key() -> None:
    public, _ = keygen()
    key = public_key_from_text(public)
    assert encrypt(b"same", key)[:37] != encrypt(b"same", key)[:37]


def test_the_two_table_lists_match() -> None:
    """The backup reads what the role may read, and the migration grants what the backup reads."""
    text = MIGRATION.read_text("utf-8")
    listed = re.search(r"TABLES = \((.*?)\)", text, re.DOTALL)
    assert listed is not None
    assert tuple(re.findall(r'"(\w+)"', listed.group(1))) == BACKUP_TABLES


def test_a_tar_with_a_path_in_it_is_refused() -> None:
    import gzip
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        info = tarfile.TarInfo("../escape.csv")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(BackupError, match="unexpected entry"):
        read_bundle(gzip.compress(buffer.getvalue()))


# --- Retention ---


def test_seven_daily_and_four_weekly_copies_are_kept() -> None:
    store = Archive(pafs._MockFileSystem(), "bucket/backups", local=False)
    start = datetime(2026, 9, 1, tzinfo=UTC)
    for day in range(60):
        name = (start + timedelta(days=day)).date().isoformat()
        store.write_bytes(store.join("daily", f"wikiguard-{name}.wgbk"), b"x")
        if day % 7 == 0:
            store.write_bytes(store.join("weekly", f"wikiguard-{name}.wgbk"), b"x")
    prune(store)
    daily = [item.day for item in stored(store, "daily")]
    weekly = [item.day for item in stored(store, "weekly")]
    assert len(daily) == 7 and daily[-1] == "2026-10-30"
    assert len(weekly) == 4 and weekly[-1] == "2026-10-27"


def test_files_it_didnt_write_are_left_alone() -> None:
    store = Archive(pafs._MockFileSystem(), "bucket/backups", local=False)
    store.write_bytes(store.join("daily", "notes.txt"), b"x")
    assert prune(store) == []


def test_no_public_key_fails_clearly(tmp_path: Path) -> None:
    settings = Settings(
        database_url="postgresql://unused", backup_path=str(tmp_path), _env_file=None
    )
    with pytest.raises(BackupError, match="BACKUP_PUBLIC_KEY"):
        run(settings, None, NOW)  # type: ignore[arg-type]


# --- Against real Postgres ---

_jobs_conftest = importlib.util.spec_from_file_location(
    "jobs_conftest", Path(__file__).parents[2] / "conftest.py"
)


def migrate(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_MIGRATIONS_URL", url)
    command.upgrade(Config(str(INI)), "head")


def reset(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")


def second_database(url: str) -> str:
    """A sibling database on the same server, so a restore has somewhere fresh to go."""
    name = "wikiguard_restore_test"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        connection.execute(f"CREATE DATABASE {name}")
    return re.sub(r"/[^/?]+(\?|$)", rf"/{name}\1", url, count=1)


def fill(connection: psycopg.Connection[Any]) -> None:
    connection.execute(
        "INSERT INTO edits (rev_id, schema_version, title, edited_at, is_minor, is_anonymous) "
        "VALUES (1, 1, 'Moon', now(), false, false)"
    )
    connection.execute("INSERT INTO reviews (rev_id, decision) VALUES (1, 'vandalism')")
    connection.execute("INSERT INTO reviews (rev_id, decision) VALUES (2, 'ok')")
    connection.execute(
        "INSERT INTO audit_log (actor, action, target, details) "
        "VALUES ('user', 'review', '1', '{\"note\": \"a, \\\"quoted\\\" value\\nover lines\"}')"
    )
    connection.execute(
        "INSERT INTO retrain_requests (kind, reason, requested_by) "
        "VALUES ('retrain', 'weekly', 'monitor')"
    )


def rows(url: str, table: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(url) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()  # noqa: S608


@pytest.fixture
def databases(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, str]]:
    if not URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    reset(URL)
    migrate(URL, monkeypatch)
    target = second_database(URL)
    migrate(target, monkeypatch)
    yield URL, target
    reset(URL)
    with psycopg.connect(URL, autocommit=True) as connection:
        connection.execute("DROP DATABASE IF EXISTS wikiguard_restore_test WITH (FORCE)")


@pytest.mark.integration
def test_backup_then_restore_brings_back_every_row(
    databases: tuple[str, str], tmp_path: Path
) -> None:
    source, target = databases
    public, private = keygen()
    with psycopg.connect(source) as connection:
        fill(connection)
        connection.commit()
        settings = Settings(
            database_url=source,
            backup_public_key=public,
            backup_path=str(tmp_path / "backups"),
            archive_path=str(tmp_path / "archive"),
            _env_file=None,
        )
        result = run(settings, connection, NOW)
    assert result["written"] == 2  # Sunday: a daily and a weekly copy
    assert result["rows"]["reviews"] == 2

    edit_time = NOW - timedelta(hours=5)
    archive = tmp_path / "archive"
    write_rows(archive, EDITS, [edit_row(7, edit_time), edit_row(8, edit_time)])
    write_rows(archive, SCORES, [score_row(7, edit_time), score_row(99, edit_time)])
    write_rows(archive, LABELS, [label_row(8, edit_time)])

    restore_settings = RestoreSettings(
        database_migrations_url=target,
        backup_path=str(tmp_path / "backups"),
        archive_path=str(archive),
        _env_file=None,
    )
    stored_name = "daily/wikiguard-2026-11-15.wgbk"
    report = restore(restore_settings, stored_name, private, 3, NOW)

    for table in ("reviews", "audit_log", "retrain_requests"):
        assert rows(target, table) == rows(source, table), table
    assert report["archive_rows"] == {"edits": 2, "scores": 2, "labels": 1}
    # A score whose edit isn't in the window is skipped, not an error.
    assert [row[0] for row in rows(target, "scores")] == [7]
    with psycopg.connect(target) as connection:
        latest = connection.execute("SELECT latest_score FROM edits WHERE rev_id = 7").fetchone()
        # New rows continue after the restored ids.
        connection.execute("INSERT INTO reviews (rev_id, decision) VALUES (3, 'skip')")
        new_id = connection.execute("SELECT max(id) FROM reviews").fetchone()
    assert latest is not None and latest[0] == pytest.approx(0.9)
    assert new_id is not None and new_id[0] == 3
    assert report["total_seconds"] >= 0


@pytest.mark.integration
def test_a_restore_over_live_data_is_refused(databases: tuple[str, str], tmp_path: Path) -> None:
    source, target = databases
    public, private = keygen()
    with psycopg.connect(source) as connection:
        fill(connection)
        connection.commit()
        bundle, _ = export_tables(connection)
    blob = tmp_path / "backup.wgbk"
    blob.write_bytes(encrypt(bundle, public_key_from_text(public)))
    with psycopg.connect(target) as connection:
        connection.execute("INSERT INTO reviews (rev_id, decision) VALUES (5, 'skip')")
        connection.commit()
    settings = RestoreSettings(
        database_migrations_url=target, archive_path=str(tmp_path / "a"), _env_file=None
    )
    with pytest.raises(BackupError, match="reviews already has rows"):
        restore(settings, str(blob), private, 3, NOW)
    assert len(rows(target, "reviews")) == 1


@pytest.mark.integration
def test_the_backup_login_can_run_it(databases: tuple[str, str], tmp_path: Path) -> None:
    """As the real read-only role, not the owner: a missing grant fails here first."""
    source, _ = databases
    public, _ = keygen()
    with psycopg.connect(source, autocommit=True) as connection:
        fill(connection)
        connection.execute("SET ROLE wikiguard_backup")
        settings = Settings(
            database_url=source,
            backup_public_key=public,
            backup_path=str(tmp_path),
            _env_file=None,
        )
        result = run(settings, connection, NOW - timedelta(days=1))
    assert result["rows"]["reviews"] == 2 and result["written"] == 1


def test_backups_go_next_to_the_archive_in_a_bucket(tmp_path: Path) -> None:
    settings = Settings(
        database_url="postgresql://unused",
        archive_path="archive",
        archive_s3_bucket="wikiguard",
        archive_s3_endpoint="http://127.0.0.1:9",
        archive_s3_region="eu-frankfurt-1",
        archive_s3_access_key="a",
        archive_s3_secret_key="b",
        backup_path="backups",
        _env_file=None,
    )
    assert store_for(settings).base == "wikiguard/backups"
    local = Settings(database_url="postgresql://unused", backup_path=str(tmp_path), _env_file=None)
    assert store_for(local).local


def edit_row(rev_id: int, at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "parent_rev_id": rev_id - 1,
        "title": "Moon",
        "timestamp": at,
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": "h",
        "comment": "c",
        "old_length": 1,
        "new_length": 2,
        "added_text": "a",
        "removed_text": "",
        "user_edit_count": 3,
        "user_hidden": False,
        "comment_hidden": False,
        "content_hidden": False,
    }


def score_row(rev_id: int, at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "model_name": "mlp",
        "model_version": "1",
        "score": 0.9,
        "used_fallback": False,
        "scored_at": at,
        "threshold": 0.5,
        "alias": "champion",
    }


def label_row(rev_id: int, at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rev_id": rev_id,
        "label": 0,
        "excluded_reason": None,
        "decided_at": at,
        "decided_late": False,
    }
