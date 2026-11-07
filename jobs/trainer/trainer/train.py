"""Search and train one model, and leave a checked, served artifact behind.

    python -m trainer.train snapshots/NAME --model mlp --search
    python -m trainer.train snapshots/NAME --model distilbert --device cpu --cap 30000 --epochs 1

A snapshot that fails validation stops here. Every trial and final run logs to MLflow with the
snapshot hash and git commit, and test scores are only computed for the final run.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from wikiguard_core.models import MODEL_NAMES, TRANSFORMERS, count_parameters
from wikiguard_core.tracking import clean_params, configure, flatten, run_tags

from trainer.data import Dataset, load_dataset
from trainer.export import save_weights, write_served_folder
from trainer.loop import FitResult, fit, predict_logits
from trainer.recipes import SEARCH_CAP, SEARCH_SPACES, Prepared, encode_like, prepare
from trainer.report import evaluate_served, save_report, training_warnings

RUNS_FOLDER = Path("data/runs")
PREDICT_BATCH = 64


def _metadata_only_score() -> float | None:
    path = RUNS_FOLDER / "sanity.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text("utf-8"))["metadata_only_validation_pr_auc"])


def _log_history(result: FitResult) -> None:
    for entry in result.history:
        step = int(entry["epoch"])
        mlflow.log_metrics(
            {key: value for key, value in entry.items() if key != "epoch"}, step=step
        )


def _summary_metrics(result: FitResult, prepared: Prepared) -> dict[str, float]:
    first = result.history[0]
    return {
        "epoch0_validation_pr_auc": first["validation_pr_auc"],
        "best_validation_pr_auc": result.best_validation_pr_auc,
        "best_epoch": float(result.best_epoch),
        "fit_seconds": result.seconds,
        "train_rows": float(prepared.train_rows),
    }


def run_trial(
    name: str,
    data: Dataset,
    params: dict[str, Any],
    device: torch.device,
    cap: int | None,
    seed: int,
) -> dict[str, Any]:
    """One search trial. It scores validation only, and keeps nothing but the numbers."""
    prepared = prepare(name, data, params, device, seed, cap)
    with mlflow.start_run(run_name=f"{name}-search", tags=run_tags(data.manifest, name, "search")):
        mlflow.log_params(clean_params(params | {"cap": cap, "device": device.type, "seed": seed}))
        result = fit(
            prepared.model,
            prepared.train,
            prepared.train_sample,
            prepared.validation,
            prepared.fit_config,
            device,
        )
        _log_history(result)
        mlflow.log_metrics(_summary_metrics(result, prepared))
    prepared.model.to("cpu")
    del prepared
    torch.cuda.empty_cache()
    return {
        "params": params,
        "cap": cap,
        "validation_pr_auc": result.best_validation_pr_auc,
        "best_epoch": result.best_epoch,
        "seconds": result.seconds,
        "history": result.history,
    }


def search(
    name: str, data: Dataset, device: torch.device, seed: int, out: Path, recipe: dict[str, Any]
) -> dict[str, Any]:
    """Try each candidate on validation, and return the best parameters.

    Transformers search on a sample for one epoch, since each full run takes far longer.
    """
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    is_transformer = name in TRANSFORMERS
    cap = SEARCH_CAP if is_transformer else None
    for candidate in SEARCH_SPACES[name]:
        params = candidate | recipe | ({"epochs": 1} if is_transformer else {})
        trials.append(run_trial(name, data, params, device, cap, seed))
        print(json.dumps(trials[-1]), flush=True)
    (out / "search.json").write_text(json.dumps(trials, indent=2), "utf-8")
    best = max(trials, key=lambda trial: trial["validation_pr_auc"])
    return {key: value for key, value in best["params"].items() if key != "epochs"}


def train_final(
    name: str,
    data: Dataset,
    params: dict[str, Any],
    device: torch.device,
    cap: int | None,
    seed: int,
    out: Path,
    label: str,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    prepared = prepare(name, data, params, device, seed, cap)
    parameters = count_parameters(prepared.model)
    with mlflow.start_run(
        run_name=f"{name}-{label}", tags=run_tags(data.manifest, name, label)
    ) as run:
        mlflow.log_params(clean_params(params | {"cap": cap, "device": device.type, "seed": seed}))
        result = fit(
            prepared.model,
            prepared.train,
            prepared.train_sample,
            prepared.validation,
            prepared.fit_config,
            device,
        )
        _log_history(result)
        mlflow.log_metrics(_summary_metrics(result, prepared))
        weights = save_weights(prepared.model, out)
        summary = result.summary()

        batch = PREDICT_BATCH
        amp = prepared.fit_config.mixed_precision
        torch_validation = predict_logits(prepared.model, prepared.validation, batch, device, amp)
        test_encoded = encode_like(prepared, data.test, "test", data)
        torch_test = predict_logits(prepared.model, test_encoded, batch, device, amp)

        export_started = time.perf_counter()
        served_folder = out / "model"
        write_served_folder(prepared, served_folder, data.manifest, summary)
        export_seconds = time.perf_counter() - export_started

        report, digest = evaluate_served(
            served_folder, data, torch_validation, torch_test, out, seed
        )
        report |= {
            "model": name,
            "label": label,
            "run_id": run.info.run_id,
            "artifact_sha256": digest,
            "snapshot": data.manifest["name"],
            "snapshot_content_sha256": data.manifest["content_sha256"],
            "params": params,
            "cap": cap,
            "train_rows": prepared.train_rows,
            "device": device.type,
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "parameters": parameters,
            "fit": summary,
            "export_seconds": export_seconds,
            "weights_bytes": weights.stat().st_size,
            "weights_sha256_file": weights.name,
            "train_sample_pr_auc_at_best": next(
                entry["train_sample_pr_auc"]
                for entry in result.history
                if int(entry["epoch"]) == result.best_epoch
            ),
            "served_bytes": sum(
                path.stat().st_size for path in served_folder.rglob("*") if path.is_file()
            ),
        }
        report["training_warnings"] = training_warnings(
            result.history,
            report["validation_pr_auc"],
            _metadata_only_score(),
            report["parity"]["validation"]["passed"] and report["parity"]["test"]["passed"],
        )
        if name not in ("mlp", "bilstm"):
            report["frozen_blocks"] = prepared.extras["frozen_blocks"]
        if name == "bilstm":
            report["glove_tokens_found"] = prepared.extras["glove_tokens_found"]
            report["vocabulary_size"] = prepared.extras["vocabulary"].size
        if name == "mlp":
            report["vocabulary_size"] = prepared.extras["tfidf"].size
        report_path = save_report(out, report)
        mlflow.log_metrics(flatten(report, "test."))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifacts(str(served_folder), "model")
        mlflow.log_artifact(str(weights), "pytorch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search", action="store_true", help="choose parameters on validation")
    parser.add_argument("--search-device", help="defaults to --device")
    parser.add_argument("--params", type=json.loads, help="fixed parameters as JSON")
    parser.add_argument("--recipe", type=json.loads, default={}, help="applied to every trial")
    parser.add_argument("--cap", type=int, help="largest training sample")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--label", default="final")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure()
    data = load_dataset(args.snapshot)
    device = torch.device(args.device)
    out = RUNS_FOLDER / (args.model if args.label == "final" else f"{args.model}-{args.label}")
    params = args.params or {}
    if args.search:
        search_device = torch.device(args.search_device or args.device)
        params = search(args.model, data, search_device, args.seed, out, args.recipe)
    params = params | args.recipe
    if args.epochs:
        params["epochs"] = args.epochs
    if not params:
        params = SEARCH_SPACES[args.model][0]
    report = train_final(args.model, data, params, device, args.cap, args.seed, out, args.label)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model",
                    "pr_auc",
                    "pr_auc_interval",
                    "roc_auc",
                    "recall_at_90_precision",
                    "precision_at_50",
                    "validation_pr_auc",
                    "train_sample_pr_auc_at_best",
                    "temperature",
                    "ece_before",
                    "ece_after",
                    "threshold_rule",
                    "served_bytes",
                    "training_warnings",
                )
            }
            | {
                "parity_gap": report["parity"]["test"]["gap"],
                "fit_seconds": report["fit"]["seconds"],
            },
            indent=2,
        )
    )
    return 0 if np.isfinite(report["pr_auc"]) else 1


if __name__ == "__main__":
    sys.exit(main())
