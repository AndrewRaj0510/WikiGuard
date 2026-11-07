"""The safe model format: readable files, a checksum for each, and a refusal to load anything else.

Loading a pickle can run code, so a model folder holds only ONNX, safetensors, and JSON.
checksums.json lists the SHA-256 of every file, and its own digest is what registration records.
A changed, added, or removed file then fails verification before anything is loaded.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from wikiguard_core.features import FEATURE_VERSION

CHECKSUMS_FILE = "checksums.json"
CONFIG_FILE = "config.json"
MODEL_FILE = "model.onnx"
ALLOWED_SUFFIXES = {".onnx", ".json", ".safetensors"}
CHUNK_BYTES = 1024 * 1024


class ArtifactError(ValueError):
    """The model folder can't be trusted, so it must not be loaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.name != CHECKSUMS_FILE
    )


def _relative(path: Path, folder: Path) -> str:
    return path.relative_to(folder).as_posix()


def write_checksums(folder: Path) -> str:
    """Record every file, and return the digest to store at registration."""
    checksums = {}
    for path in _files(folder):
        if path.suffix not in ALLOWED_SUFFIXES:
            raise ArtifactError(f"{path.name} isn't an allowed artifact format")
        checksums[_relative(path, folder)] = sha256_file(path)
    text = json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    with (folder / CHECKSUMS_FILE).open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
    return sha256_file(folder / CHECKSUMS_FILE)


def verify_artifact(folder: Path, expected_digest: str | None = None) -> dict[str, Any]:
    """Check the folder, and return its config. Raises before any model code runs."""
    listing = folder / CHECKSUMS_FILE
    if not listing.is_file():
        raise ArtifactError(f"{folder.name} has no {CHECKSUMS_FILE}")
    if expected_digest is not None and sha256_file(listing) != expected_digest:
        raise ArtifactError(f"{folder.name}: {CHECKSUMS_FILE} doesn't match the registered digest")
    recorded: dict[str, str] = json.loads(listing.read_text("utf-8"))
    found = {_relative(path, folder): path for path in _files(folder)}
    for name in sorted(found):
        if found[name].suffix not in ALLOWED_SUFFIXES:
            raise ArtifactError(f"{name} isn't an allowed artifact format")
    if set(found) != set(recorded):
        different = sorted(set(found) ^ set(recorded))
        raise ArtifactError(f"{folder.name}: files don't match the list: {different}")
    for name, path in found.items():
        if sha256_file(path) != recorded[name]:
            raise ArtifactError(f"{name} failed its checksum")
    config: dict[str, Any] = json.loads((folder / CONFIG_FILE).read_text("utf-8"))
    if config["feature_version"] != FEATURE_VERSION:
        raise ArtifactError(
            f"{folder.name} needs feature_version {config['feature_version']}, "
            f"this code has {FEATURE_VERSION}"
        )
    return config


def write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
