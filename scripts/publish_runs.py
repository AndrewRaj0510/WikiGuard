"""Send finished runs to the MLflow server named in the settings, and set the same aliases there.

Training logs to a local file, so it works with no token. The owner runs this once to publish the
results to DagsHub. It reads the MLflow settings from .env itself, so no token is typed or printed.
Artifacts are checked first, so a damaged folder is never published.

    python scripts/publish_runs.py
    python scripts/publish_runs.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
from trainer.finish import alias_order, load_reports
from wikiguard_core.artifacts import verify_artifact
from wikiguard_core.config import ConfigError, MlflowSettings, load_settings
from wikiguard_core.tracking import (
    clean_params,
    configure,
    flatten,
    register_version,
    set_aliases,
)

RUNS_FOLDER = Path("data/runs")
IMAGES_FOLDER = Path("docs/images")


def publish_one(name: str, report: dict[str, Any]) -> str:
    folder = RUNS_FOLDER / name
    tags = {
        "model_name": name,
        "stage": report["label"],
        "snapshot_name": report["snapshot"],
        "snapshot_content_sha256": report["snapshot_content_sha256"],
        "local_run_id": report["run_id"],
    }
    with mlflow.start_run(run_name=f"{name}-final", tags=tags) as run:
        mlflow.log_params(clean_params(report["params"]))
        for entry in report["fit"]["history"]:
            metrics = {key: value for key, value in entry.items() if key != "epoch"}
            mlflow.log_metrics(metrics, step=int(entry["epoch"]))
        mlflow.log_metrics(flatten(report, "test."))
        mlflow.log_artifact(str(folder / "report.json"))
        mlflow.log_artifacts(str(folder / "model"), "model")
        mlflow.log_artifact(str(folder / "weights.safetensors"), "pytorch")
    return run.info.run_id


def publish(reports: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    versions: dict[str, str] = {}
    order = alias_order(reports)
    for name in order:
        report = reports[name]
        run_id = publish_one(name, report)
        version = register_version(
            run_id,
            "model",
            name,
            report["artifact_sha256"],
            {
                "pr_auc": report["pr_auc"],
                "recall_at_90_precision": report["recall_at_90_precision"],
            },
        )
        versions[name] = version.version
    aliases = [("champion", versions[order[0]])]
    aliases += [(f"fallback-{index}", versions[name]) for index, name in enumerate(order[1:], 1)]
    set_aliases(aliases)
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="check the folders and stop")
    args = parser.parse_args()

    reports = load_reports()
    for name, report in reports.items():
        verify_artifact(RUNS_FOLDER / name / "model", report["artifact_sha256"])
        print(f"{name}: artifact verified, PR-AUC {report['pr_auc']:.4f}")
    if args.dry_run:
        return 0
    try:
        settings = load_settings(MlflowSettings)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password.get_secret_value()
    configure(settings.mlflow_tracking_uri)
    print(json.dumps(publish(reports)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
