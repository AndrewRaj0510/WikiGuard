"""Restore a database from a backup and the parquet archive, and time each step.

    python restore.py BACKUP --private-key-file KEY [--archive-days 3]

BACKUP is a local file, or a name in the backup store such as daily/wikiguard-2026-11-12.wgbk.
KEY is a file holding the private key from `backup.py keygen`, kept in the password manager. It's
read from a file, never the command line, where it would land in shell history.

It writes with DATABASE_MIGRATIONS_URL, the migration login, into a database that the migrations
already brought up to date. Run `alembic upgrade head` first, so the schema comes from the code and
never from a backup. It refuses a database whose tables already hold rows: a restore goes into a
fresh project (the second Supabase project, in the rehearsal), never over live data.

Two parts, as the plan says (section 14): the backup brings back the tables that can't be rebuilt,
and the archive rebuilds the last few days of edits, scores, and labels, which is all Supabase
keeps (DEC-038).
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from backup import (
    BACKUP_TABLES,
    BackupError,
    decrypt,
    private_key_from_text,
    read_bundle,
    store_for,
)
from psycopg import sql
from wikiguard_core.archive import EDITS, LABELS, SCORES, Root, open_archive, read_table
from wikiguard_core.config import (
    ArchiveSettings,
    BackupSettings,
    CommonSettings,
    ConfigError,
    MigrationSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter

logger = logging.getLogger("restore")
ARCHIVE_DAYS = 3
BATCH = 1_000

INSERT_EDIT = """
    INSERT INTO edits (
        rev_id, parent_rev_id, schema_version, title, edited_at, is_minor, is_anonymous, user_hash,
        comment, old_length, new_length, added_text, removed_text, user_edit_count, user_hidden,
        comment_hidden, content_hidden
    ) VALUES (
        %(rev_id)s, %(parent_rev_id)s, %(schema_version)s, %(title)s, %(timestamp)s, %(is_minor)s,
        %(is_anonymous)s, %(user_hash)s, %(comment)s, %(old_length)s, %(new_length)s,
        %(added_text)s, %(removed_text)s, %(user_edit_count)s, %(user_hidden)s, %(comment_hidden)s,
        %(content_hidden)s
    ) ON CONFLICT (rev_id) DO NOTHING
"""
# Latency and the load-shed flags aren't in the archive. They stay at their defaults.
INSERT_SCORE = """
    INSERT INTO scores (
        rev_id, model_name, model_version, score, used_fallback, scored_at, alias, threshold
    ) SELECT %(rev_id)s, %(model_name)s, %(model_version)s, %(score)s, %(used_fallback)s,
        %(scored_at)s, %(alias)s, %(threshold)s
    WHERE EXISTS (SELECT 1 FROM edits WHERE rev_id = %(rev_id)s)
    ON CONFLICT (rev_id, model_version) DO NOTHING
"""
INSERT_LABEL = """
    INSERT INTO labels (rev_id, label, excluded_reason, decided_at, decided_late)
    SELECT %(rev_id)s, %(label)s, %(excluded_reason)s, %(decided_at)s, %(decided_late)s
    WHERE EXISTS (SELECT 1 FROM edits WHERE rev_id = %(rev_id)s)
    ON CONFLICT (rev_id) DO NOTHING
"""
# The queue reads each edit's newest score from the edit itself, as the scorer leaves it.
SET_LATEST = """
    UPDATE edits e SET latest_score = s.score, latest_model_version = s.model_version,
        latest_scored_at = s.scored_at
    FROM (
        SELECT DISTINCT ON (rev_id) rev_id, score, model_version, scored_at
        FROM scores ORDER BY rev_id, scored_at DESC
    ) s
    WHERE e.rev_id = s.rev_id
"""


class Settings(CommonSettings, MigrationSettings, ArchiveSettings, BackupSettings):
    pass


def revision_number(value: str | None) -> int:
    return int(value) if value and value.isdigit() else -1


def check_target(connection: psycopg.Connection[Any], manifest: dict[str, Any]) -> None:
    row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    target = row[0] if row else None
    # Migrations only add things (DEC-079), so a newer schema takes an older backup's columns.
    if revision_number(target) < revision_number(manifest.get("schema_revision")):
        raise BackupError(
            f"the database is at schema {target}, older than the backup's "
            f"{manifest.get('schema_revision')}; run the migrations first"
        )
    for table in (*BACKUP_TABLES, "edits", "scores", "labels"):
        found = connection.execute(
            sql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(sql.Identifier(table))
        ).fetchone()
        if found and found[0]:
            raise BackupError(f"{table} already has rows; restore only into a fresh database")


def load_tables(connection: psycopg.Connection[Any], files: dict[str, bytes]) -> dict[str, int]:
    loaded = {}
    for table in BACKUP_TABLES:
        data = files.get(f"{table}.csv", b"")
        if not data.strip():
            loaded[table] = 0
            continue
        header = data.split(b"\n", 1)[0].decode("utf-8").strip().split(",")
        # COPY writes the ids from the file, as INSERT ... OVERRIDING SYSTEM VALUE would.
        query = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER)").format(
            sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(name) for name in header)
        )
        with connection.cursor().copy(query) as copy:
            copy.write(data)
        count = connection.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        ).fetchone()
        loaded[table] = count[0] if count else 0
        reset_identity(connection, table)
    return loaded


def reset_identity(connection: psycopg.Connection[Any], table: str) -> None:
    """New rows must continue after the restored ids, not collide with them."""
    columns = connection.execute(
        "SELECT attname FROM pg_attribute WHERE attrelid = %s::regclass AND attidentity <> ''",
        (table,),
    ).fetchall()
    for (column,) in columns:
        connection.execute(
            sql.SQL(
                "SELECT setval(pg_get_serial_sequence({table_text}, {column_text}), "
                "COALESCE((SELECT max({column}) FROM {table}), 0) + 1, false)"
            ).format(
                table_text=sql.Literal(table),
                column_text=sql.Literal(column),
                column=sql.Identifier(column),
                table=sql.Identifier(table),
            )
        )


def load_archive(
    connection: psycopg.Connection[Any], archive: Root, days: int, now: datetime
) -> dict[str, int]:
    since = (now - timedelta(days=days)).date()
    counts = {}
    steps = ((EDITS, INSERT_EDIT), (SCORES, INSERT_SCORE), (LABELS, INSERT_LABEL))
    for dataset, statement in steps:
        rows = read_table(archive, dataset, since=since).to_pylist()
        with connection.cursor() as cursor:
            for start in range(0, len(rows), BATCH):
                cursor.executemany(statement, rows[start : start + BATCH])
        counts[dataset] = len(rows)
    connection.execute(SET_LATEST)
    return counts


def timed(steps: dict[str, float], name: str, action: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    result = action()
    steps[name] = round(time.perf_counter() - start, 2)
    return result


def restore(
    settings: Settings, backup: str, private_key: str, archive_days: int, now: datetime
) -> dict[str, Any]:
    steps: dict[str, float] = {}
    local = Path(backup)
    if local.is_file():
        blob = timed(steps, "read_backup", local.read_bytes)
    else:
        store = store_for(settings)
        blob = timed(steps, "read_backup", lambda: store.read_bytes(store.join(backup)))
    bundle = timed(steps, "decrypt", lambda: decrypt(blob, private_key_from_text(private_key)))
    manifest, files = read_bundle(bundle)
    with psycopg.connect(
        settings.database_migrations_url.get_secret_value(),
        prepare_threshold=None,
        connect_timeout=10,
    ) as connection:
        # One transaction: a restore that fails halfway leaves nothing behind.
        with connection.transaction():
            check_target(connection, manifest)
            tables = timed(steps, "load_backup", lambda: load_tables(connection, files))
            archive = open_archive(settings)
            rebuilt = timed(
                steps,
                "load_archive",
                lambda: load_archive(connection, archive, archive_days, now),
            )
    for table, info in manifest["tables"].items():
        if tables.get(table) != info["rows"]:
            found, expected = tables.get(table), info["rows"]
            raise BackupError(f"{table}: {found} rows restored, {expected} expected")
    return {
        "backup_created_at": manifest["created_at"],
        "schema_revision": manifest["schema_revision"],
        "tables": tables,
        "archive_rows": rebuilt,
        "seconds": steps,
        "total_seconds": round(sum(steps.values()), 2),
    }


def main(argv: list[str] | None = None) -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("backup", help="a local file, or a name in the backup store")
    parser.add_argument("--private-key-file", type=Path, required=True)
    parser.add_argument("--archive-days", type=int, default=ARCHIVE_DAYS)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    try:
        result = restore(
            settings,
            args.backup,
            args.private_key_file.read_text("utf-8"),
            args.archive_days,
            datetime.now(UTC),
        )
    except BackupError as error:
        logger.error("Restore refused: %s", error)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
