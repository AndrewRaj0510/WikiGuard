"""Rehearse rebuilding the demo's database from nothing, as the runbook says, and time each step
(PLAN.md, Phase 8; docs/runbooks/backup-restore.md, "The demo's database").

    python scripts/load_test.py up              the "production" database: the demo's data
    python scripts/rehearse_restore.py --admin-url postgresql://USER:PASSWORD@127.0.0.1:5437/postgres
    python scripts/load_test.py down

--admin-url is a throwaway Postgres, never Supabase: the script makes a fresh database there, as
the second Supabase project would be, and refuses any host but this machine.

The steps, each run as its own command exactly as the runbook gives it:
1. Add what a running demo collects and can't rebuild (reviews and their audit entries) to the
   load test's database, then back it up with backup.py as the backup login, twice: once with an
   old key pair, then, after a key rotation, with a new one. Each backup must open with its own
   private key and not the other's.
2. Into the fresh database: the migrations, then restore.py with the new private key (the backed-up
   tables; the demo has no archive), then load_demo.py (edits, scores, labels).
3. Compare every table with the source: row counts, and a hash of every row's contents. Only the
   load time the demo loader stamps on scores is left out.

Every job runs from a scratch folder with its settings passed in, so none of them can read the
repo's .env (pydantic reads one from the working directory). The private keys are written owner-
only to that folder and deleted at the end. Writes data/restore_rehearsal.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs" / "backup"))
import backup  # noqa: E402
import compose_e2e as e2e  # noqa: E402
import load_test as lt  # noqa: E402

ROOT = lt.ROOT
SCRATCH = lt.WORK / "restore"
OUT = ROOT / "data" / "restore_rehearsal.json"
TARGET_DB = "wikiguard_restore"
DEMO_TABLES = ("edits", "scores", "labels")
# Stamped with the time of loading by load_demo.py, so they differ between two loads by design.
LOAD_TIME_COLUMNS = {"scored_at", "latest_scored_at", "ingested_at"}
REVIEWS = 200
# Each job's own output, searched afterwards for the passwords and keys it was handed.
OUTPUTS: dict[str, str] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


def run(step: str, command: list[str], env: dict[str, str], timings: dict[str, float]) -> str:
    """One job, from the scratch folder, with only the settings it's given and the system's PATH."""
    started = time.perf_counter()
    done = subprocess.run(  # noqa: S603 - fixed commands
        command,
        cwd=SCRATCH,
        env={
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
            if key in os.environ
        }
        | {"PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("jobs/backup", "scripts"))}
        | env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    timings[step] = round(time.perf_counter() - started, 2)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:], file=sys.stderr)  # noqa: T201
        raise RuntimeError(f"{step} failed")
    log(f"{step}: {timings[step]} s")
    OUTPUTS[step] = done.stdout + done.stderr
    return done.stdout + done.stderr


def seed_source(source: str) -> None:
    """What a running demo collects and can't rebuild: the owner's reviews and their audit trail."""
    with psycopg.connect(source) as connection:
        connection.execute(
            """
            INSERT INTO reviews (rev_id, decision, note)
            SELECT rev_id, (ARRAY['vandalism','good_faith_mistake','ok','skip'])[1 + rev_id %% 4],
                   'restore rehearsal'
            FROM edits WHERE rev_id NOT IN (SELECT rev_id FROM reviews)
            ORDER BY latest_score DESC NULLS LAST LIMIT %s
            """,
            (REVIEWS,),
        )
        connection.execute(
            """
            INSERT INTO audit_log (actor, action, target, details)
            SELECT 'user', 'review.saved', rev_id::text, jsonb_build_object('decision', decision)
            FROM reviews WHERE note = 'restore rehearsal'
            """
        )


def fingerprint(url: str) -> dict[str, dict[str, Any]]:
    """Per table: rows, and an md5 over every row's columns in a fixed order."""
    found = {}
    with psycopg.connect(url) as connection:
        for table in (*backup.BACKUP_TABLES, *DEMO_TABLES):
            columns = [
                row[0]
                for row in connection.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
                    (table,),
                ).fetchall()
                if not (table in DEMO_TABLES and row[0] in LOAD_TIME_COLUMNS)
            ]
            row_text = sql.SQL("concat_ws('|', {})").format(
                sql.SQL(", ").join(sql.SQL("{}::text").format(sql.Identifier(c)) for c in columns)
            )
            query = sql.SQL(
                "SELECT count(*), md5(coalesce(string_agg(r, E'\\n' ORDER BY r), '')) "
                "FROM (SELECT {} AS r FROM {}) rows"
            ).format(row_text, sql.Identifier(table))
            count, digest = connection.execute(query).fetchone() or (0, "")
            found[table] = {"rows": count, "md5": digest}
    return found


def key_checks(old: tuple[str, str], new: tuple[str, str]) -> dict[str, bool]:
    """Each backup opens with its own private key only."""
    store = backup.Archive.at(str(SCRATCH / "backups"))
    files = {item.day: store.read_bytes(item.path) for item in backup.stored(store, "daily")}
    older, newer = sorted(files)
    results = {}
    for label, blob, (_, own), (_, other) in (
        ("older_backup", files[older], old, new),
        ("newer_backup", files[newer], new, old),
    ):
        backup.decrypt(blob, backup.private_key_from_text(own))
        results[f"{label}_opens_with_its_key"] = True
        try:
            backup.decrypt(blob, backup.private_key_from_text(other))
            results[f"{label}_opens_with_the_other_key"] = True
        except Exception:  # noqa: BLE001 - any failure to open is the point
            results[f"{label}_opens_with_the_other_key"] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--admin-url", required=True, help="a throwaway Postgres, never Supabase")
    args = parser.parse_args()
    admin = urlparse(args.admin_url)
    if admin.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("--admin-url must be a Postgres on this machine")
    state = lt.load_state()
    values = lt.env_values(Path(state["env_file"]))
    host = f"127.0.0.1:{values['POSTGRES_PORT']}/{values['POSTGRES_DB']}"
    source_owner = f"postgresql://{values['POSTGRES_USER']}:{values['POSTGRES_PASSWORD']}@{host}"
    source_backup = f"postgresql://backup_login:{values['COMPOSE_LOGIN_PASSWORD']}@{host}"
    target = admin._replace(path=f"/{TARGET_DB}").geturl()

    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    (SCRATCH / "archive").mkdir()
    timings: dict[str, float] = {}
    python = sys.executable
    try:
        seed_source(source_owner)
        old, new = backup.keygen(), backup.keygen()
        backup_env = {"DATABASE_URL": source_backup, "BACKUP_PATH": str(SCRATCH / "backups")}
        # Yesterday's backup, under the old key: what the store holds before the rotation.
        yesterday = datetime.now(UTC) - timedelta(days=1)
        with psycopg.connect(source_backup) as connection:
            # _env_file=None: this process runs from the repo root, where .env must not be read.
            settings = backup.Settings(
                _env_file=None,
                database_url=source_backup,
                backup_path=str(SCRATCH / "backups"),
                backup_public_key=old[0],
            )
            backup.run(settings, connection, yesterday)
        # The rotation: the job gets the new public key, and tonight's backup uses it.
        run(
            "backup",
            [python, str(ROOT / "jobs/backup/backup.py")],
            backup_env | {"BACKUP_PUBLIC_KEY": new[0]},
            timings,
        )
        keys = key_checks(old, new)
        newest = backup.stored(backup.Archive.at(str(SCRATCH / "backups")), "daily")[-1]
        key_file = SCRATCH / "restore.key"
        e2e.write_private(key_file, new[1])

        with psycopg.connect(args.admin_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(TARGET_DB))
            )
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TARGET_DB)))
        started = time.perf_counter()
        migrate_env = {"DATABASE_MIGRATIONS_URL": target}
        run(
            "migrations",
            [
                python,
                "-m",
                "alembic",
                "-c",
                str(ROOT / "jobs/migrate/alembic.ini"),
                "upgrade",
                "head",
            ],
            migrate_env,
            timings,
        )
        restored = run(
            "restore_backup",
            [
                python,
                str(ROOT / "jobs/backup/restore.py"),
                f"daily/wikiguard-{newest.day}{backup.SUFFIX}",
                "--private-key-file",
                str(key_file),
                "--archive-days",
                "0",
            ],
            migrate_env
            | {"ARCHIVE_PATH": str(SCRATCH / "archive"), "BACKUP_PATH": str(SCRATCH / "backups")},
            timings,
        )
        run(
            "load_demo",
            [
                python,
                str(ROOT / "scripts/load_demo.py"),
                str(ROOT / lt.SNAPSHOT),
                "--backfill",
                str(ROOT / "data/backfill/parts"),
                "--runs",
                str(ROOT / "data/runs"),
            ],
            migrate_env,
            timings,
        )
        total = round(time.perf_counter() - started, 1)
        before, after = fingerprint(source_owner), fingerprint(target)
        differing = [table for table in before if before[table] != after[table]]
        handed = {
            "source owner password": values["POSTGRES_PASSWORD"],
            "backup_login password": values["COMPOSE_LOGIN_PASSWORD"],
            "target password": admin.password or "",
            "old private key": old[1],
            "new private key": new[1],
        }
        leaked = sorted(
            f"{name} in {step}"
            for step, text in OUTPUTS.items()
            for name, value in handed.items()
            if value and value in text
        )
        result = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "keys": keys,
            "seconds": timings,
            "restore_total_seconds": total,
            "tables": {
                table: {"source": before[table], "restored": after[table]} for table in before
            },
            "differing_tables": differing,
            "secrets_in_job_output": leaked,
            "restore_log": restored.strip().splitlines()[-3:],
            "held": not differing
            and not leaked
            and keys["older_backup_opens_with_its_key"]
            and keys["newer_backup_opens_with_its_key"]
            and not keys["older_backup_opens_with_the_other_key"]
            and not keys["newer_backup_opens_with_the_other_key"],
        }
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n", "utf-8")
    log(f"{'held' if result['held'] else 'FAILED'}: restored in {total} s; {OUT}")
    return 0 if result["held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
