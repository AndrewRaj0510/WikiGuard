"""MLflow helpers: every run logs the same identity, wherever it's tracked.

MLFLOW_TRACKING_URI picks the server, and MLflow reads the username and password variables itself.
Without a URI, runs go to a local SQLite file, so training works with no account and no network.
"""

import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

EXPERIMENT = "wikiguard"
REGISTERED_MODEL = "wikiguard"
LOCAL_FOLDER = Path("data/mlflow")
URI_VARIABLE = "MLFLOW_TRACKING_URI"
ARTIFACT_DIGEST_TAG = "artifact_sha256"
MODEL_KIND_TAG = "model_name"
MAX_VALUE_LENGTH = 500


def configure(uri: str | None = None) -> str:
    """Point MLflow at the server, and return its address."""
    # MLflow prints emoji, which crash a piped Windows console (F-049).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    target = uri or os.environ.get(URI_VARIABLE)
    local = target is None
    if target is None:
        LOCAL_FOLDER.mkdir(parents=True, exist_ok=True)
        target = f"sqlite:///{LOCAL_FOLDER.as_posix()}/mlflow.db"
    mlflow.set_tracking_uri(target)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        location = (LOCAL_FOLDER / "artifacts").resolve().as_uri() if local else None
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)
    return target


def git_sha() -> str:
    """The commit a run came from, marked when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status.strip() else sha


def run_tags(manifest: Mapping[str, Any], model: str, stage: str) -> dict[str, str]:
    """What makes a run comparable: results only compare if both logged one content hash."""
    return {
        MODEL_KIND_TAG: model,
        "stage": stage,
        "snapshot_name": manifest["name"],
        "snapshot_content_sha256": manifest["content_sha256"],
        "snapshot_original_sha256": manifest["original_content_sha256"],
        "feature_version": str(manifest["feature_version"]),
        "git_sha": git_sha(),
    }


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Nested numbers as flat metric names. Text and missing values aren't metrics, so they're left
    out of this and stay in the JSON report."""
    flat: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}{key}".replace("@", "_at_")
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, int | float):
            flat[name] = float(value)
        elif isinstance(value, Mapping):
            flat |= flatten(value, f"{name}.")
    return flat


def clean_params(params: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(value)[:MAX_VALUE_LENGTH] for key, value in params.items()}


def log_files(paths: Iterable[Path], artifact_path: str) -> None:
    for path in paths:
        mlflow.log_artifact(str(path), artifact_path)


def register_version(
    run_id: str, artifact_path: str, model: str, digest: str, metrics: Mapping[str, float]
) -> ModelVersion:
    """Add a version under one registered name, tagged with its checksum and model type.

    One name holds all five, because aliases such as champion belong to a registered model and the
    champion can be any of the five. Registering never changes an alias.
    """
    client = MlflowClient()
    try:
        client.get_registered_model(REGISTERED_MODEL)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(REGISTERED_MODEL)
    run = client.get_run(run_id)
    version = client.create_model_version(
        REGISTERED_MODEL,
        f"{run.info.artifact_uri}/{artifact_path}",
        run_id=run_id,
        tags={MODEL_KIND_TAG: model, ARTIFACT_DIGEST_TAG: digest}
        | {key: f"{value:.6f}" for key, value in metrics.items()},
    )
    return version


def set_aliases(order: list[tuple[str, str]]) -> None:
    """(alias, version) pairs. Only promotion calls this, never training."""
    client = MlflowClient()
    for alias, version in order:
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
