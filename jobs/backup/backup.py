"""Nightly encrypted backup of the tables that can't be rebuilt (Part 1, section 14 of the plan).

    python backup.py            back up now, then keep 7 daily and 4 weekly copies
    python backup.py keygen     make a key pair: the public half for the cluster, the private half
                                for the owner's password manager

What's backed up: reviews, the model history, retraining, the audit log, usage counters, and the
daily rollups (BACKUP_TABLES). Edits, scores, and labels aren't: the parquet archive rebuilds the
last 30 days (restore.py), and anything older is deleted on purpose. A full dump every night would
use about 15 GB of Supabase's 5 GB monthly egress; these tables are a few megabytes.

Encryption is to a public key, so the cluster can write backups but can't read them. A stolen
cluster secret then can't decrypt old backups. Each file gets a fresh X25519 key, combined with
the recipient's key through HKDF into an AES-256-GCM key, which also detects tampering. This is
the same construction as `age`, written out with the `cryptography` package already in the locks.

The file is the magic bytes, the one-time public key, the nonce, and the ciphertext. Inside is a
gzipped tar: one CSV per table, plus a manifest with row counts and the schema revision.
"""

import argparse
import base64
import gzip
import io
import json
import logging
import os
import sys
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from psycopg import sql
from wikiguard_core.archive import Archive, open_bucket
from wikiguard_core.config import (
    ArchiveSettings,
    BackupSettings,
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter

logger = logging.getLogger("backup")

# Must match TABLES in migration 0005 (jobs/migrate/migrations/versions), which a test checks.
BACKUP_TABLES = (
    "reviews",
    "model_versions",
    "retrain_runs",
    "retrain_requests",
    "audit_log",
    "llm_usage",
    "live_metrics",
    "drift_reports",
)
MAGIC = b"WGBK1"
KEY_INFO = b"wikiguard-backup-v1"
NONCE_BYTES = 12
DAILY_KEEP = 7
WEEKLY_KEEP = 4
# Sunday: that night's copy is also kept as a weekly one.
WEEKLY_DAY = 6
SUFFIX = ".wgbk"


class BackupError(RuntimeError):
    pass


class Settings(CommonSettings, DatabaseSettings, ArchiveSettings, BackupSettings):
    pass


# --- Encryption ---


def public_key_from_text(text: str) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(base64.b64decode(text.strip(), validate=True))


def private_key_from_text(text: str) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(base64.b64decode(text.strip(), validate=True))


def _raw(key: X25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _derive(shared: bytes, ephemeral: bytes, recipient: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=ephemeral + recipient, info=KEY_INFO
    ).derive(shared)


def encrypt(data: bytes, recipient: X25519PublicKey) -> bytes:
    ephemeral = X25519PrivateKey.generate()
    ephemeral_public = _raw(ephemeral.public_key())
    key = _derive(ephemeral.exchange(recipient), ephemeral_public, _raw(recipient))
    nonce = os.urandom(NONCE_BYTES)
    header = MAGIC + ephemeral_public + nonce
    # The header is authenticated too, so swapping the one-time key or nonce is detected.
    return header + AESGCM(key).encrypt(nonce, data, header)


def decrypt(blob: bytes, private: X25519PrivateKey) -> bytes:
    if not blob.startswith(MAGIC):
        raise BackupError("not a WikiGuard backup")
    ephemeral_public = blob[len(MAGIC) : len(MAGIC) + 32]
    nonce = blob[len(MAGIC) + 32 : len(MAGIC) + 32 + NONCE_BYTES]
    header = blob[: len(MAGIC) + 32 + NONCE_BYTES]
    recipient = _raw(private.public_key())
    shared = private.exchange(X25519PublicKey.from_public_bytes(ephemeral_public))
    key = _derive(shared, ephemeral_public, recipient)
    try:
        return AESGCM(key).decrypt(nonce, blob[len(header) :], header)
    except Exception as error:
        raise BackupError("wrong key, or the file was changed") from error


def keygen() -> tuple[str, str]:
    private = X25519PrivateKey.generate()
    private_text = base64.b64encode(
        private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    ).decode()
    public_text = base64.b64encode(_raw(private.public_key())).decode()
    return public_text, private_text


# --- Export ---


def export_tables(connection: psycopg.Connection[Any]) -> tuple[bytes, dict[str, Any]]:
    """Every table as CSV, read in one snapshot so the files agree with each other."""
    revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    manifest: dict[str, Any] = {
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "schema_revision": revision[0] if revision else None,
        "tables": {},
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for table in BACKUP_TABLES:
            data = bytearray()
            query = sql.SQL("COPY (SELECT * FROM {}) TO STDOUT WITH (FORMAT csv, HEADER)").format(
                sql.Identifier(table)
            )
            with connection.cursor().copy(query) as copy:
                for chunk in copy:
                    data += chunk
            count = connection.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
            ).fetchone()
            manifest["tables"][table] = {"rows": count[0] if count else 0, "bytes": len(data)}
            _add(bundle, f"{table}.csv", bytes(data))
        _add(bundle, "manifest.json", json.dumps(manifest, indent=2).encode())
    return gzip.compress(buffer.getvalue(), mtime=0), manifest


def _add(bundle: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    bundle.addfile(info, io.BytesIO(data))


def read_bundle(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data)), mode="r") as bundle:
        for member in bundle.getmembers():
            # Only plain files with plain names: a crafted tar can't write anywhere.
            if not member.isfile() or "/" in member.name or member.name.startswith("."):
                raise BackupError(f"unexpected entry in the backup: {member.name!r}")
            extracted = bundle.extractfile(member)
            files[member.name] = extracted.read() if extracted else b""
    manifest: dict[str, Any] = json.loads(files.pop("manifest.json"))
    return manifest, files


# --- Storage and retention ---


@dataclass(frozen=True)
class Stored:
    kind: str
    path: str
    day: str


def store_for(settings: Any) -> Archive:
    """The bucket the archive mirrors to, under BACKUP_PATH. A local folder with no bucket set.

    `settings` holds both the archive's and the backup's settings, as both jobs' Settings do.
    """
    return open_bucket(settings, settings.backup_path) or Archive.at(settings.backup_path)


def stored(store: Archive, kind: str) -> list[Stored]:
    found = []
    for path in store.files(store.join(kind)):
        name = path.rsplit("/", 1)[-1]
        if name.startswith("wikiguard-") and name.endswith(SUFFIX):
            found.append(Stored(kind, path, name.removeprefix("wikiguard-").removesuffix(SUFFIX)))
    return sorted(found, key=lambda item: item.day)


def prune(store: Archive) -> list[str]:
    """Keep the newest 7 daily and 4 weekly copies. Never the newest of either."""
    removed = []
    for kind, keep in (("daily", DAILY_KEEP), ("weekly", WEEKLY_KEEP)):
        for item in stored(store, kind)[:-keep]:
            store.delete_file(item.path)
            removed.append(item.path)
    return removed


def run(settings: Settings, connection: psycopg.Connection[Any], now: datetime) -> dict[str, Any]:
    if not settings.backup_public_key:
        raise BackupError("BACKUP_PUBLIC_KEY isn't set; make one with `python backup.py keygen`")
    recipient = public_key_from_text(settings.backup_public_key)
    with connection.transaction():
        # One consistent view of every table, and nothing written.
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        bundle, manifest = export_tables(connection)
    blob = encrypt(bundle, recipient)
    store = store_for(settings)
    day = now.date().isoformat()
    written = [store.join("daily", f"wikiguard-{day}{SUFFIX}")]
    if now.weekday() == WEEKLY_DAY:
        written.append(store.join("weekly", f"wikiguard-{day}{SUFFIX}"))
    for path in written:
        store.write_bytes(path, blob)
    removed = prune(store)
    rows = {table: info["rows"] for table, info in manifest["tables"].items()}
    return {"written": len(written), "bytes": len(blob), "removed": len(removed), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=["backup", "keygen"], default="backup")
    args = parser.parse_args(argv)
    if args.command == "keygen":
        public, private = keygen()
        # Printed for the owner to copy: the public key to the cluster's Secret, the private key
        # to the password manager only. Nothing is written to disk.
        print(f"BACKUP_PUBLIC_KEY={public}")  # noqa: T201
        print(f"Private key, for the password manager only: {private}")  # noqa: T201
        return 0
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    try:
        with psycopg.connect(
            settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
        ) as connection:
            result = run(settings, connection, datetime.now(UTC))
    except BackupError as error:
        logger.error("%s", error)
        return 1
    logger.info("Backup done: %s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
