"""Registering versions and moving aliases: the only code that changes what serves.

Two backends with one interface:
- MlflowAliasRegistry: the real registry on DagsHub, through the MLflow client.
- LocalRegistry: a folder holding versions.json and aliases.json, for Compose, tests, and the
  simulations. Serving reads it with chain.LocalRuns, so a promotion here is a real promotion
  for a scorer pointed at it.

Registering never changes an alias (PLAN.md, section 8). Only promotion and rollback, in
trainer.lifecycle, call set_chain.
"""

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from wikiguard_core.artifacts import CHECKSUMS_FILE, sha256_file
from wikiguard_core.chain import (
    ALIASES_FILE,
    ARTIFACT_FOLDER,
    CANDIDATE_ALIAS,
    CHAIN_ALIASES,
    LocalRuns,
    MlflowRegistry,
    ModelRef,
    ModelSource,
)

VERSIONS_FILE = "versions.json"
WARNINGS_TAG = "training_warnings"
SNAPSHOT_TAG = "snapshot_content_sha256"


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class VersionInfo:
    version: str
    model: str
    artifact_sha256: str
    run_id: str = ""
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    # The content hash of the snapshot it trained on, so retention knows which snapshots to keep.
    snapshot_hash: str = ""


class Registry(Protocol):
    def aliases(self) -> dict[str, str]:
        """Alias to version, for every alias the registry holds."""
        ...

    def info(self, version: str) -> VersionInfo: ...

    def versions(self) -> list[VersionInfo]: ...

    def register(
        self,
        served_folder: Path,
        model: str,
        run_id: str,
        digest: str,
        metrics: Mapping[str, float],
        warnings: list[str],
        snapshot_hash: str = "",
    ) -> str: ...

    def artifact_bytes(self, version: str) -> int:
        """What one version's files take up in the registry's storage."""
        ...

    def set_aliases(self, wanted: Mapping[str, str | None]) -> None:
        """Set each alias to a version, or remove it with None. Others are left alone."""
        ...

    def delete_version(self, version: str) -> None: ...

    def source(self) -> ModelSource:
        """The read side, for downloading a version's served files."""
        ...

    def ref_for(self, version: str) -> ModelRef:
        """A reference source().fetch can download, for a version with or without an alias."""
        ...


def chain_mapping(order: list[str]) -> dict[str, str | None]:
    """champion, fallback-1... for an ordered list of versions, clearing any fallback past it."""
    wanted: dict[str, str | None] = dict.fromkeys(CHAIN_ALIASES)
    for alias, version in zip(CHAIN_ALIASES, order, strict=False):
        wanted[alias] = version
    return wanted


def chain_of(aliases: Mapping[str, str]) -> list[tuple[str, str]]:
    """The serving chain from a registry's aliases, in order, stopping at the first gap."""
    found = []
    for alias in CHAIN_ALIASES:
        if alias not in aliases:
            break
        found.append((alias, aliases[alias]))
    return found


class LocalRegistry:
    """A folder registry. It writes only where versions.json exists, so pointing it at the
    trainer's own data/runs by mistake can't change the models it holds."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, root: Path) -> "LocalRegistry":
        root.mkdir(parents=True, exist_ok=True)
        if not (root / VERSIONS_FILE).exists():
            _write_json(root / VERSIONS_FILE, {})
            _write_json(root / ALIASES_FILE, {})
        return cls(root)

    def _versions(self) -> dict[str, dict[str, Any]]:
        path = self.root / VERSIONS_FILE
        if not path.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(path.read_text("utf-8"))
        return data

    def _require_writable(self) -> None:
        if not (self.root / VERSIONS_FILE).exists():
            raise RegistryError(f"{self.root} isn't a registry (no {VERSIONS_FILE}); refusing")

    def aliases(self) -> dict[str, str]:
        path = self.root / ALIASES_FILE
        if not path.exists():
            return {}
        raw = json.loads(path.read_text("utf-8"))
        return {alias: str(entry["version"]) for alias, entry in raw.items()}

    def info(self, version: str) -> VersionInfo:
        entry = self._versions().get(version)
        if entry is None:
            raise RegistryError(f"no version {version}")
        return VersionInfo(
            version,
            entry["model"],
            entry["artifact_sha256"],
            entry.get("run_id", ""),
            tuple(entry.get("warnings", [])),
            entry.get("metrics", {}),
            entry.get("snapshot_hash", ""),
        )

    def versions(self) -> list[VersionInfo]:
        return [self.info(version) for version in self._versions()]

    def register(
        self,
        served_folder: Path,
        model: str,
        run_id: str,
        digest: str,
        metrics: Mapping[str, float],
        warnings: list[str],
        snapshot_hash: str = "",
    ) -> str:
        """Copy the served files in, so the registry owns what it serves."""
        self._require_writable()
        versions = self._versions()
        version = str(max((int(key) for key in versions), default=0) + 1)
        folder = f"v{version}"
        target = self.root / folder / ARTIFACT_FOLDER
        shutil.copytree(served_folder, target)
        if sha256_file(target / CHECKSUMS_FILE) != digest:
            shutil.rmtree(self.root / folder)
            raise RegistryError(f"{model}: the copied files don't match {digest[:12]}")
        versions[version] = {
            "model": model,
            "folder": folder,
            "artifact_sha256": digest,
            "run_id": run_id,
            "warnings": warnings,
            "metrics": dict(metrics),
            "snapshot_hash": snapshot_hash,
        }
        _write_json(self.root / VERSIONS_FILE, versions)
        return version

    def artifact_bytes(self, version: str) -> int:
        entry = self._versions().get(version)
        if entry is None:
            return 0
        folder = self.root / entry["folder"]
        return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())

    def set_aliases(self, wanted: Mapping[str, str | None]) -> None:
        self._require_writable()
        versions = self._versions()
        raw: dict[str, Any] = json.loads((self.root / ALIASES_FILE).read_text("utf-8"))
        for alias, version in wanted.items():
            if version is None:
                raw.pop(alias, None)
                continue
            entry = versions.get(version)
            if entry is None:
                raise RegistryError(f"{alias}: no version {version}")
            raw[alias] = {
                "version": int(version),
                "model": entry["model"],
                "folder": entry["folder"],
            }
        _write_json(self.root / ALIASES_FILE, raw)

    def delete_version(self, version: str) -> None:
        self._require_writable()
        if version in self.aliases().values():
            raise RegistryError(f"version {version} still has an alias")
        versions = self._versions()
        entry = versions.pop(version, None)
        if entry is None:
            return
        shutil.rmtree(self.root / entry["folder"], ignore_errors=True)
        _write_json(self.root / VERSIONS_FILE, versions)

    def source(self) -> ModelSource:
        return LocalRuns(self.root)

    def ref_for(self, version: str) -> ModelRef:
        entry = self._versions().get(version)
        if entry is None:
            raise RegistryError(f"no version {version}")
        # LocalRuns fetches from the folder named where a registry keeps the run id.
        return ModelRef(
            "version", entry["model"], version, entry["artifact_sha256"], entry["folder"]
        )

    def adopt(
        self,
        folder: Path,
        model: str,
        metrics: Mapping[str, Any],
        warnings: list[str],
        run_id: str = "",
        snapshot_hash: str = "",
    ) -> str:
        """Register an existing served folder (a copy), for seeding a sandbox from data/runs."""
        digest = sha256_file(folder / CHECKSUMS_FILE)
        return self.register(folder, model, run_id, digest, dict(metrics), warnings, snapshot_hash)


class MlflowAliasRegistry:
    """The real registry. Needs a token that can write, which only the trainer holds (DEC-071)."""

    def __init__(self, tracking_uri: str, username: str | None, password: str | None) -> None:
        from mlflow import MlflowClient
        from wikiguard_core.tracking import configure

        configure(tracking_uri)
        self._client = MlflowClient()
        self._read = MlflowRegistry(tracking_uri, username, password)

    def aliases(self) -> dict[str, str]:
        from wikiguard_core.tracking import REGISTERED_MODEL

        model = self._client.get_registered_model(REGISTERED_MODEL)
        return {alias: str(version) for alias, version in (model.aliases or {}).items()}

    def info(self, version: str) -> VersionInfo:
        from wikiguard_core.tracking import ARTIFACT_DIGEST_TAG, MODEL_KIND_TAG, REGISTERED_MODEL

        found = self._client.get_model_version(REGISTERED_MODEL, version)
        tags = dict(found.tags or {})
        warnings = json.loads(tags.get(WARNINGS_TAG, "[]"))
        metrics = {
            key: float(value)
            for key, value in tags.items()
            if key not in (ARTIFACT_DIGEST_TAG, MODEL_KIND_TAG, WARNINGS_TAG, SNAPSHOT_TAG)
            and _is_number(value)
        }
        snapshot = tags.get(SNAPSHOT_TAG, "")
        if not snapshot and found.run_id:
            # Versions from before Phase 5 carry it on their run (tracking.run_tags).
            run = self._client.get_run(found.run_id)
            snapshot = run.data.tags.get(SNAPSHOT_TAG, "")
        return VersionInfo(
            str(found.version),
            tags[MODEL_KIND_TAG],
            tags[ARTIFACT_DIGEST_TAG],
            found.run_id or "",
            tuple(warnings),
            metrics,
            snapshot,
        )

    def versions(self) -> list[VersionInfo]:
        from wikiguard_core.tracking import REGISTERED_MODEL

        found = self._client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        return [self.info(str(version.version)) for version in found]

    def register(
        self,
        served_folder: Path,
        model: str,
        run_id: str,
        digest: str,
        metrics: Mapping[str, float],
        warnings: list[str],
        snapshot_hash: str = "",
    ) -> str:
        from wikiguard_core.tracking import REGISTERED_MODEL, register_version

        # The run already holds the served files under "model" (trainer.train).
        version = register_version(run_id, ARTIFACT_FOLDER, model, digest, metrics)
        self._client.set_model_version_tag(
            REGISTERED_MODEL, version.version, WARNINGS_TAG, json.dumps(warnings)
        )
        self._client.set_model_version_tag(
            REGISTERED_MODEL, version.version, SNAPSHOT_TAG, snapshot_hash
        )
        return str(version.version)

    def artifact_bytes(self, version: str) -> int:
        """Everything the version's run stored: served files and PyTorch weights (F-091)."""
        run_id = self.info(version).run_id

        def walk(path: str | None) -> int:
            total = 0
            for item in self._client.list_artifacts(run_id, path):
                total += walk(item.path) if item.is_dir else int(item.file_size or 0)
            return total

        return walk(None) if run_id else 0

    def set_aliases(self, wanted: Mapping[str, str | None]) -> None:
        from wikiguard_core.tracking import REGISTERED_MODEL

        current = self.aliases()
        for alias, version in wanted.items():
            if version is None:
                if alias in current:
                    self._client.delete_registered_model_alias(REGISTERED_MODEL, alias)
            elif current.get(alias) != version:
                self._client.set_registered_model_alias(REGISTERED_MODEL, alias, version)

    def delete_version(self, version: str) -> None:
        from wikiguard_core.tracking import REGISTERED_MODEL

        if version in self.aliases().values():
            raise RegistryError(f"version {version} still has an alias")
        self._client.delete_model_version(REGISTERED_MODEL, version)

    def source(self) -> ModelSource:
        return self._read

    def ref_for(self, version: str) -> ModelRef:
        info = self.info(version)
        return ModelRef("version", info.model, version, info.artifact_sha256, info.run_id)


def registry_from_settings(settings: Any) -> Registry:
    if settings.model_source == "local":
        return LocalRegistry(Path(settings.local_runs_path))
    if not settings.mlflow_tracking_uri:
        raise RegistryError("MLFLOW_TRACKING_URI is required when MODEL_SOURCE is mlflow")
    password = settings.mlflow_tracking_password
    return MlflowAliasRegistry(
        settings.mlflow_tracking_uri,
        settings.mlflow_tracking_username,
        password.get_secret_value() if password else None,
    )


def snapshot_of_aliases(registry: Registry) -> dict[str, str]:
    """Every chain alias and the candidate, to restore exactly on rollback."""
    aliases = registry.aliases()
    keep = (*CHAIN_ALIASES, CANDIDATE_ALIAS)
    return {alias: version for alias, version in aliases.items() if alias in keep}


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _write_json(path: Path, data: Any) -> None:
    """Write then rename, so a reader never sees half a file."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
    os.replace(temporary, path)


__all__ = [
    "LocalRegistry",
    "MlflowAliasRegistry",
    "Registry",
    "RegistryError",
    "VersionInfo",
    "chain_mapping",
    "chain_of",
    "registry_from_settings",
    "snapshot_of_aliases",
]
