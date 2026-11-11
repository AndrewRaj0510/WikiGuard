"""The drift reference sample saved with each promoted model (PLAN.md, sections 2 and 5).

2,000 edits from the data the model learned from, with the model's own score on each. The monitor
compares live edits with the features and text here, and live scores with these scores. It's kept
in the archive, past the 30 days, for as long as the model serves, and the purge job covers it
like any other copy of edit text.

    python -m trainer.reference snapshots/NAME --version 1 --model-folder data/runs/roberta/model
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from wikiguard_core.archive import write_reference
from wikiguard_core.config import ArchiveSettings, load_settings
from wikiguard_core.inference import ServedModel
from wikiguard_core.snapshot import TRAIN, VALIDATION, read_manifest, read_snapshot

REFERENCE_ROWS = 2_000


def sample(
    rows: list[dict[str, Any]], count: int = REFERENCE_ROWS, seed: int = 0
) -> list[dict[str, Any]]:
    """What the model learned from: train and validation, never the test window."""
    pool = [row for row in rows if row["split"] in (TRAIN, VALIDATION)]
    order = np.random.default_rng(seed).permutation(len(pool))[:count]
    return [pool[index] for index in sorted(order.tolist())]


def save_for_model(
    archive_root: Path,
    version: str,
    model_name: str,
    served_folder: Path,
    rows: list[dict[str, Any]],
    snapshot_name: str,
    threads: int = 2,
) -> Path:
    chosen = sample(rows)
    model = ServedModel(served_folder, threads=threads)
    # Calibrated probabilities, scored one at a time, as serving scores them (DEC-056).
    scores = np.array([model.probabilities([row], batch_size=1)[0] for row in chosen])
    meta = {
        "version": version,
        "model": model_name,
        "snapshot": snapshot_name,
        "rows": len(chosen),
        "created": datetime.now(UTC).isoformat(),
    }
    scored = [row | {"score": float(score)} for row, score in zip(chosen, scores, strict=True)]
    return write_reference(archive_root, version, scored, meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(ArchiveSettings)
    config = json.loads((args.model_folder / "config.json").read_text("utf-8"))
    folder = save_for_model(
        Path(settings.archive_path),
        args.version,
        config["kind"],
        args.model_folder,
        read_snapshot(args.snapshot),
        read_manifest(args.snapshot)["name"],
    )
    print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
