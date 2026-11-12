"""Train the smallest real model on synthetic edits, laid out like data/runs, for CI.

CI can't use the real models: they're large and live in DagsHub, and CI holds no token for it.
So this trains the MLP on 2,300 invented edits (300 of them for training), through the same code
path as a real run (train, export to ONNX, calibrate, check parity, write checksums), and names
it champion. The Compose end-to-end test then serves it like any other model.

    python scripts/make_tiny_runs.py data/runs

It doubles as the training smoke test: it fails if the model didn't learn the planted signal or
if the exported model's scores drifted from PyTorch's. Needs the trainer's environment. The model
is useless on real edits, so this refuses to write into a folder that already has runs in it.
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlp"
PARAMS = {"hidden": 32, "dropout": 0.1, "learning_rate": 3e-3, "use_comment": True}
# A third of the test edits are bad, so a model that learned nothing scores about 0.33. The same
# bar as the trainer's own test of this path (test_report_writes_calibration_into_the_artifact).
MINIMUM_PR_AUC = 0.5


def synthetic_candidates() -> tuple[Any, Any]:
    """The trainer tests' dataset, so the smoke test and the unit tests can't drift apart."""
    spec = importlib.util.spec_from_file_location(
        "trainer_fixtures", ROOT / "jobs" / "trainer" / "tests" / "conftest.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("jobs/trainer/tests/conftest.py is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tiny_candidates(), module.AS_OF


def build(out: Path) -> dict[str, Any]:
    # Imported here so --help works without PyTorch installed.
    import mlflow
    import torch
    from trainer.data import load_dataset
    from trainer.train import train_final
    from wikiguard_core.snapshot import build_snapshot, write_snapshot

    candidates, as_of = synthetic_candidates()
    with tempfile.TemporaryDirectory(prefix="wikiguard-tiny-") as scratch:
        folder = write_snapshot(
            build_snapshot(candidates, as_of, "synthetic"), Path(scratch) / "snapshots"
        )
        # A throwaway local MLflow, set directly: configure() would read the real one from .env.
        mlflow.set_tracking_uri((Path(scratch) / "mlruns").as_uri())
        data = load_dataset(folder)
        # metadata_only=0.0 skips the "no better than metadata" warning, which needs the sanity
        # run's score. The synthetic set isn't built to pass that check.
        report = train_final(
            MODEL, data, PARAMS, torch.device("cpu"), None, 0, out / MODEL, "ci", metadata_only=0.0
        )
    aliases = {"champion": {"version": 1, "model": MODEL}}
    (out / "aliases.json").write_text(json.dumps(aliases, indent=2) + "\n", "utf-8")
    return report


def failures(report: dict[str, Any]) -> list[str]:
    found = []
    if report["pr_auc"] < MINIMUM_PR_AUC:
        found.append(f"test PR-AUC {report['pr_auc']:.3f} is under {MINIMUM_PR_AUC}")
    for split in ("validation", "test"):
        if not report["parity"][split]["passed"]:
            found.append(f"ONNX and PyTorch scores differ on {split}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="an empty or new folder, such as data/runs in CI")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} isn't empty; refusing to mix a toy model into it", file=sys.stderr)  # noqa: T201
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    report = build(args.out)
    summary = {
        "model": MODEL,
        "train_rows": report["train_rows"],
        "pr_auc": round(report["pr_auc"], 4),
        "parity_gap_test": report["parity"]["test"]["gap"],
        "served_bytes": report["served_bytes"],
        "fit_seconds": round(report["fit"]["seconds"], 1),
    }
    print(json.dumps(summary, indent=2))  # noqa: T201
    found = failures(report)
    for problem in found:
        print(problem, file=sys.stderr)  # noqa: T201
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
