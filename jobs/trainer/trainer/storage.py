"""After each retrain: keep what live models need, delete the rest, and warn before storage fills.

PLAN.md, section 14: "MLflow retention keeps the champion, fallbacks, and the last 3 versions of
each model". Section 2: snapshots are deleted once no live model uses them, and `dvc gc` runs so
the space is really freed. DagsHub's storage limit isn't confirmed (F-071), so the alert fires at
STORAGE_ALERT_BYTES, well below either figure DagsHub shows.

    python -m trainer.storage            # what would go, changing nothing
    python -m trainer.storage --apply

Deleting is permanent. Nothing an alias points at is ever deleted, and neither is the snapshot
behind it (DEC-073).
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wikiguard_core.alerts import Alerter
from wikiguard_core.archive import Root, delete_reference, open_archive, reference_versions
from wikiguard_core.dvcops import DvcError, DvcRepo
from wikiguard_core.snapshot import read_manifest

from trainer.records import Recorder
from trainer.registry import Registry, VersionInfo, snapshot_of_aliases

KEEP_PER_MODEL = 3
GOLDEN_PREFIX = "golden-set"


def versions_to_delete(
    versions: Iterable[VersionInfo], aliased: set[str], keep_per_model: int = KEEP_PER_MODEL
) -> list[str]:
    """Every version with no alias that isn't among the newest few of its model."""
    by_model: dict[str, list[str]] = defaultdict(list)
    for info in versions:
        by_model[info.model].append(info.version)
    doomed = []
    for found in by_model.values():
        newest = sorted(found, key=int, reverse=True)
        keep = set(newest[:keep_per_model]) | aliased
        doomed += [version for version in newest if version not in keep]
    return sorted(doomed, key=int)


def snapshot_hashes(folder: Path) -> set[str]:
    """Every hash a snapshot has had: purges change it, and models log the one they trained on."""
    manifest = read_manifest(folder)
    hashes = {manifest["content_sha256"], manifest["original_content_sha256"]}
    hashes |= {purge["previous_content_sha256"] for purge in manifest.get("purges", [])}
    return hashes


def snapshots_to_delete(root: Path, in_use: set[str]) -> list[Path]:
    doomed = []
    for folder in sorted(root.glob("snapshot-*")):
        if not folder.is_dir() or folder.name.endswith(".tmp"):
            continue
        if not snapshot_hashes(folder) & in_use:
            doomed.append(folder)
    return doomed


def storage_bytes(registry: Registry, snapshots: Path) -> dict[str, int]:
    model_bytes = sum(registry.artifact_bytes(info.version) for info in registry.versions())
    snapshot_bytes = (
        sum(
            path.stat().st_size
            for folder in snapshots.iterdir()
            if folder.is_dir() and not folder.name.endswith(".tmp")
            for path in folder.rglob("*")
            if path.is_file()
        )
        if snapshots.is_dir()
        else 0
    )
    return {
        "models": model_bytes,
        "snapshots": snapshot_bytes,
        "total": model_bytes + snapshot_bytes,
    }


def retain(
    registry: Registry,
    repo: DvcRepo,
    archive_root: Root,
    gc_enabled: bool,
    alert_bytes: int,
    alerter: Alerter,
    apply: bool = True,
) -> dict[str, Any]:
    aliases = snapshot_of_aliases(registry)
    aliased = set(aliases.values())
    versions = registry.versions()
    live = [info for info in versions if info.version in aliased]
    in_use = {info.snapshot_hash for info in live if info.snapshot_hash}
    doomed_versions = versions_to_delete(versions, aliased)
    snapshots = repo.root / "snapshots"
    # With no hash known for a live model, nothing can be shown unused, so nothing goes.
    unknown = [info.version for info in live if not info.snapshot_hash]
    doomed_snapshots = [] if unknown else snapshots_to_delete(snapshots, in_use)
    doomed_references = [
        version for version in reference_versions(archive_root) if version not in aliased
    ]
    result: dict[str, Any] = {
        "applied": apply,
        "versions_deleted": doomed_versions,
        "snapshots_deleted": [folder.name for folder in doomed_snapshots],
        "references_deleted": doomed_references,
        "live_versions_without_snapshot_hash": unknown,
    }
    if apply:
        for version in doomed_versions:
            registry.delete_version(version)
        for folder in doomed_snapshots:
            pointer = folder.with_name(f"{folder.name}.dvc")
            if pointer.exists():
                repo.remove(pointer)
            shutil.rmtree(folder, ignore_errors=True)
        for version in doomed_references:
            delete_reference(archive_root, version)
        try:
            result["gc"] = repo.gc_cloud(gc_enabled)
        except DvcError as error:
            result["gc"] = f"failed: {error}"
            alerter.send("dvc gc failed", str(error)[:300])
    used = storage_bytes(registry, snapshots)
    result["storage_bytes"] = used
    if used["total"] >= alert_bytes:
        alerter.send(
            "Storage nearly full",
            f"{used['total'] / 1e9:.2f} GB of models and snapshots, alert set at "
            f"{alert_bytes / 1e9:.2f} GB (DagsHub's limit is unconfirmed, F-071)",
        )
    return result


def retention_after_retrain(
    settings: Any, registry: Registry, recorder: Recorder, alerter: Alerter
) -> dict[str, Any]:
    result = retain(
        registry,
        DvcRepo(Path(settings.snapshot_repo_path)),
        open_archive(settings),
        settings.snapshot_gc_enabled,
        settings.storage_alert_bytes,
        alerter,
    )
    recorder.audit("storage.retention", None, result)
    return result


def main() -> int:
    from wikiguard_core.config import load_settings

    from trainer.lifecycle import Settings
    from trainer.registry import registry_from_settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete; otherwise only report")
    args = parser.parse_args()
    settings = load_settings(Settings)
    webhook = settings.discord_webhook_url
    alerter = Alerter("storage", webhook.get_secret_value() if webhook else None)
    result = retain(
        registry_from_settings(settings),
        DvcRepo(Path(settings.snapshot_repo_path)),
        open_archive(settings),
        settings.snapshot_gc_enabled,
        settings.storage_alert_bytes,
        alerter,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
