"""Check that the MLflow server supports registered model aliases, which the serving chain needs.

DagsHub's docs don't say whether it supports them (F-044), so this tries them for real and then
deletes what it created. It reads the MLFLOW_* settings from .env and never prints their values.

Run it from the repo root in its own environment, because mlflow-skinny 3.5.1 (the version
DagsHub's server reported on 2026-11-04) needs an older `packaging` than pre-commit does:

    py -3.11 -m venv <folder outside the repo>
    <folder>\\Scripts\\python -m pip install mlflow-skinny==3.5.1 -e packages/wikiguard_core
    <folder>\\Scripts\\python scripts/check_mlflow_aliases.py

Exit code: 0 aliases work, 1 aliases failed, 2 the check couldn't get as far as testing them.
"""

import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from wikiguard_core.config import ConfigError, MlflowSettings, load_settings

EXPERIMENT = "wikiguard-alias-check"
MODEL = "wikiguard-alias-check"
ALIAS = "champion"


def prepare_version(client: MlflowClient) -> str:
    """Create a run and a registered model version for the alias to point at."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as folder:
        marker = Path(folder) / "marker.txt"
        marker.write_text("WikiGuard alias check\n")
        mlflow.log_artifact(str(marker))
    client.create_registered_model(MODEL)
    version = client.create_model_version(
        MODEL, f"{run.info.artifact_uri}/marker.txt", run.info.run_id
    )
    return version.version


def check_alias(client: MlflowClient, version: str) -> None:
    client.set_registered_model_alias(MODEL, ALIAS, version)
    found = client.get_model_version_by_alias(MODEL, ALIAS)
    if found.version != version:
        raise RuntimeError(f"alias returned version {found.version}, expected {version}")
    client.delete_registered_model_alias(MODEL, ALIAS)


def clean_up(client: MlflowClient) -> None:
    """Remove the test model and every run in the test experiment, including from failed checks."""
    try:
        client.delete_registered_model(MODEL)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            print(f"Couldn't delete test model {MODEL}: {error.error_code}")
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment:
            for run in client.search_runs([experiment.experiment_id]):
                client.delete_run(run.info.run_id)
    except MlflowException as error:
        print(f"Couldn't delete test runs in {EXPERIMENT}: {error.error_code}")


def main() -> int:
    # MLflow prints emoji that the default Windows console encoding can't represent.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    try:
        settings = load_settings(MlflowSettings)
    except ConfigError as error:
        print(error)
        return 2

    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password.get_secret_value()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        client = MlflowClient()
    except Exception as error:  # Any failure here means the check could not run.
        print(f"INCONCLUSIVE: couldn't connect ({type(error).__name__}: {error})")
        return 2

    try:
        version = prepare_version(client)
    except Exception as error:  # Any failure here means the check could not run.
        print(
            f"INCONCLUSIVE: couldn't create a test model version ({type(error).__name__}: {error})"
        )
        clean_up(client)
        return 2

    try:
        check_alias(client, version)
    except Exception as error:  # Any failure here means aliases do not work.
        print(f"ALIASES FAILED: {type(error).__name__}: {error}")
        print("Use model version tags for the serving chain instead, and update the plan.")
        return 1
    finally:
        clean_up(client)

    print(f"ALIASES WORK: set, read, and deleted on MLflow client {mlflow.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
