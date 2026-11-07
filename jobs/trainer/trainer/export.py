"""Turn a trained model into the folder that gets served: int8 ONNX, JSON, and checksums."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import save_file
from torch import nn
from wikiguard_core.artifacts import CONFIG_FILE, MODEL_FILE, write_checksums, write_json
from wikiguard_core.features import FEATURE_VERSION
from wikiguard_core.inference import BILSTM, MLP, TFIDF_FILE, TOKENIZER_FILE, TRANSFORMER
from wikiguard_core.inference import VOCABULARY_FILE as VOCABULARY_JSON
from wikiguard_core.models import TRANSFORMERS
from wikiguard_core.schemas import SCHEMA_VERSION

from trainer.recipes import Prepared

WEIGHTS_FILE = "weights.safetensors"
OPSET = 18


def kind_of(name: str) -> str:
    return TRANSFORMER if name in TRANSFORMERS else name


def save_weights(model: nn.Module, folder: Path) -> Path:
    """The PyTorch weights as safetensors, which holds numbers only and can't run code."""
    folder.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path = folder / WEIGHTS_FILE
    save_file(state, path)
    return path


def _example_inputs(
    prepared: Prepared,
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, Any]]:
    """What the exporter traces with. Two rows and a longer sequence, so no size gets frozen in."""
    metadata_size = len(prepared.metadata.columns)
    metadata = torch.zeros(2, metadata_size)
    if prepared.name == MLP:
        size = prepared.extras["tfidf"].size + metadata_size
        return (
            {"features": torch.zeros(2, size)},
            ["features"],
            {"features": {0: "batch"}, "logit": {0: "batch"}},
        )
    ids = torch.ones(2, 12, dtype=torch.int64)
    if prepared.name == BILSTM:
        return (
            {"input_ids": ids, "metadata": metadata},
            ["input_ids", "metadata"],
            {
                "input_ids": {0: "batch", 1: "steps"},
                "metadata": {0: "batch"},
                "logit": {0: "batch"},
            },
        )
    mask = torch.ones(2, 12, dtype=torch.int64)
    return (
        {"input_ids": ids, "attention_mask": mask, "metadata": metadata},
        ["input_ids", "attention_mask", "metadata"],
        {
            "input_ids": {0: "batch", 1: "steps"},
            "attention_mask": {0: "batch", 1: "steps"},
            "metadata": {0: "batch"},
            "logit": {0: "batch"},
        },
    )


def export_float(prepared: Prepared, path: Path) -> None:
    model = prepared.model.to("cpu").eval()
    inputs, names, axes = _example_inputs(prepared)
    # The TorchScript path: it needs no extra dependency at serving time, and it handles the LSTM.
    torch.onnx.export(
        model,
        (),
        str(path),
        kwargs=inputs,
        input_names=names,
        output_names=["logit"],
        dynamic_axes=axes,
        opset_version=OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(str(path), full_check=False)


def quantize(source: Path, target: Path) -> None:
    """Int8 weights, activations computed at run time. Needs no calibration data."""
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def write_served_folder(
    prepared: Prepared, folder: Path, manifest: dict[str, Any], fit_summary: dict[str, Any]
) -> dict[str, Any]:
    """Export, quantize, and write the model folder. Calibration is added later by the report."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        float_path = Path(scratch) / "float.onnx"
        export_float(prepared, float_path)
        quantize(float_path, folder / MODEL_FILE)
        float_bytes = sum(path.stat().st_size for path in Path(scratch).iterdir())

    config: dict[str, Any] = {
        "name": prepared.name,
        "kind": kind_of(prepared.name),
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "metadata": prepared.metadata.to_dict(),
        "params": prepared.params,
        "train_rows": prepared.train_rows,
        "snapshot": manifest["name"],
        "snapshot_content_sha256": manifest["content_sha256"],
        "snapshot_original_sha256": manifest["original_content_sha256"],
        "best_epoch": fit_summary["best_epoch"],
        "float_onnx_bytes": float_bytes,
    }
    if prepared.name == MLP:
        write_json(folder / TFIDF_FILE, prepared.extras["tfidf"].to_dict())
    elif prepared.name == BILSTM:
        write_json(folder / VOCABULARY_JSON, prepared.extras["vocabulary"].to_dict())
    else:
        shutil.copy(prepared.extras["tokenizer"], folder / TOKENIZER_FILE)
        config |= {
            "pad_token_id": prepared.extras["pad_token_id"],
            "pretrained_repo": prepared.extras["repo"],
            "pretrained_revision": prepared.extras["revision"],
            "frozen_blocks": prepared.extras["frozen_blocks"],
        }
    write_json(folder / CONFIG_FILE, config)
    write_checksums(folder)
    return config


def update_config(folder: Path, values: dict[str, Any]) -> str:
    """Add calibration and the threshold, then record the checksums again. Returns the digest."""
    config = json.loads((folder / CONFIG_FILE).read_text("utf-8"))
    config |= values
    write_json(folder / CONFIG_FILE, config)
    return write_checksums(folder)
