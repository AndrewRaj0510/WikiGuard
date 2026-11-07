"""How each of the five models is built and trained, and the small search space for each.

A recipe takes rows and hyperparameters and returns everything the loop and the exporter need, so
the search, the final run, and the sanity checks can't build a model differently.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from torch import nn
from wikiguard_core.inference import load_tokenizer
from wikiguard_core.inputs import MetadataSpec, TfidfEncoder, TokenVocabulary, transformer_text
from wikiguard_core.models import (
    BILSTM,
    MLP,
    PRETRAINED,
    TRANSFORMERS,
    BiLSTMClassifier,
    MLPClassifier,
    TransformerClassifier,
    freeze_lower_layers,
    load_encoder,
)

from trainer.data import (
    Dataset,
    Encoded,
    SequenceEncoded,
    encode_tfidf,
    encode_tokens,
    labels_of,
    sample_rows,
)
from trainer.glove import load_embeddings
from trainer.loop import GRADIENT_CLIP, FitConfig

CACHE_FOLDER = Path("data/cache")
TRAIN_SAMPLE_ROWS = 5000
# DistilBERT keeps a CPU recipe with a capped sample, matching what retraining can do (DEC-044).
CPU_TRANSFORMER_CAP = 30_000
SEARCH_CAP = 20_000
# A new head trained at the encoder's rate barely moves, and the model ends up no better than its
# metadata (F-075). Warmup and decay apply to both rates.
HEAD_LEARNING_RATE = 1e-3


def _transformer_trials(first: float, second: float, third: float) -> list[dict[str, Any]]:
    """Three encoder rates, then one change each from the first: a slower head, and class weight."""
    base = {"dropout": 0.2, "head_learning_rate": HEAD_LEARNING_RATE}
    return [
        base | {"learning_rate": first},
        base | {"learning_rate": second},
        base | {"learning_rate": third},
        base | {"learning_rate": first, "head_learning_rate": HEAD_LEARNING_RATE / 3},
        base | {"learning_rate": first, "weight_power": 1.0},
    ]


# At most 5 trials per model, scored on validation only (PLAN.md, section 3). The first of each is
# the plan's starting point.
SEARCH_SPACES: dict[str, list[dict[str, Any]]] = {
    "mlp": [
        {"hidden": 128, "dropout": 0.5, "learning_rate": 1e-3, "use_comment": False},
        {"hidden": 128, "dropout": 0.5, "learning_rate": 1e-3, "use_comment": True},
        {"hidden": 64, "dropout": 0.5, "learning_rate": 1e-3, "use_comment": True},
        {"hidden": 128, "dropout": 0.3, "learning_rate": 3e-4, "use_comment": True},
        {"hidden": 256, "dropout": 0.5, "learning_rate": 3e-4, "use_comment": True},
    ],
    "bilstm": [
        {"hidden": 128, "dropout": 0.4, "learning_rate": 1e-3, "use_comment": False},
        {"hidden": 128, "dropout": 0.4, "learning_rate": 1e-3, "use_comment": True},
        {"hidden": 64, "dropout": 0.4, "learning_rate": 1e-3, "use_comment": True},
        {"hidden": 128, "dropout": 0.25, "learning_rate": 5e-4, "use_comment": True},
        {"hidden": 192, "dropout": 0.4, "learning_rate": 5e-4, "use_comment": True},
    ],
    "distilbert": _transformer_trials(5e-5, 2e-5, 1e-4),
    "bert": _transformer_trials(2e-5, 3e-5, 5e-5),
    "roberta": _transformer_trials(2e-5, 3e-5, 5e-5),
}


@dataclass
class Prepared:
    name: str
    params: dict[str, Any]
    model: nn.Module
    metadata: MetadataSpec
    fit_config: FitConfig
    train: Encoded
    train_sample: Encoded
    validation: Encoded
    # Filled by each recipe with what the exporter must save beside the model.
    extras: dict[str, Any] = field(default_factory=dict)
    train_rows: int = 0


def _mlp(data: Dataset, params: dict[str, Any], seed: int, cap: int | None) -> Prepared:
    train_rows = sample_rows(data.train, cap, seed) if cap else data.train
    encoder = TfidfEncoder.fit(
        train_rows,
        params.get("max_features", 20_000),
        params.get("min_df", 3),
        params["use_comment"],
    )
    metadata = MetadataSpec.fit(train_rows)
    torch.manual_seed(seed)
    model = MLPClassifier(encoder.size + len(metadata.columns), params["hidden"], params["dropout"])
    sample = sample_rows(train_rows, TRAIN_SAMPLE_ROWS, seed)
    return Prepared(
        MLP,
        params,
        model,
        metadata,
        FitConfig(
            epochs=params.get("epochs", 12),
            batch_size=128,
            learning_rate=params["learning_rate"],
            patience=3,
            seed=seed,
            weight_power=params.get("weight_power", FitConfig.weight_power),
        ),
        encode_tfidf(train_rows, encoder, metadata),
        encode_tfidf(sample, encoder, metadata),
        encode_tfidf(data.validation, encoder, metadata),
        {"tfidf": encoder},
        len(train_rows),
    )


def _bilstm(
    data: Dataset, params: dict[str, Any], seed: int, cap: int | None, pretrained: bool = True
) -> Prepared:
    train_rows = sample_rows(data.train, cap, seed) if cap else data.train
    vocabulary = TokenVocabulary.fit(
        train_rows,
        params.get("max_tokens", 30_000),
        params.get("min_count", 3),
        params["use_comment"],
    )
    metadata = MetadataSpec.fit(train_rows)
    torch.manual_seed(seed)
    weights, found = None, 0
    if pretrained:
        weights, found = load_embeddings(vocabulary, seed)
    model = BiLSTMClassifier(
        vocabulary.size,
        params.get("embedding_size", 100),
        params["hidden"],
        len(metadata.columns),
        params["dropout"],
        weights,
    )
    sample = sample_rows(train_rows, TRAIN_SAMPLE_ROWS, seed)
    return Prepared(
        BILSTM,
        params,
        model,
        metadata,
        FitConfig(
            epochs=params.get("epochs", 8),
            batch_size=64,
            learning_rate=params["learning_rate"],
            patience=2,
            mixed_precision=False,
            seed=seed,
            weight_power=params.get("weight_power", FitConfig.weight_power),
        ),
        encode_tokens(train_rows, vocabulary, metadata),
        encode_tokens(sample, vocabulary, metadata),
        encode_tokens(data.validation, vocabulary, metadata),
        {"vocabulary": vocabulary, "glove_tokens_found": found},
        len(train_rows),
    )


def tokenizer_path(name: str) -> Path:
    repo, revision = PRETRAINED[name]
    return Path(hf_hub_download(repo, "tokenizer.json", revision=revision))


def _tokenized(
    rows: list[dict[str, Any]], name: str, snapshot_hash: str, label: str
) -> list[NDArray[np.int64]]:
    """Token ids, cached by snapshot and tokenizer revision (PLAN.md, section 12).

    Saved as flat integer arrays with pickling off, so the cache can't carry code. The stored row
    ids must match, so a different sample under the same label is rebuilt, not reused.
    """
    revision = PRETRAINED[name][1]
    rev_ids = np.array([row["rev_id"] for row in rows], dtype=np.int64)
    path = CACHE_FOLDER / "tokenized" / f"{name}-{revision[:12]}-{snapshot_hash[:12]}-{label}.npz"
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        if np.array_equal(stored["rev_ids"], rev_ids):
            offsets = stored["offsets"]
            flat = stored["ids"]
            return [flat[offsets[i] : offsets[i + 1]] for i in range(offsets.size - 1)]
    tokenizer = load_tokenizer(tokenizer_path(name))
    encodings = tokenizer.encode_batch_fast([transformer_text(row) for row in rows])
    arrays = [np.array(encoding.ids, dtype=np.int64) for encoding in encodings]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.concatenate(arrays),
        offsets=np.cumsum([0, *[array.size for array in arrays]]),
        rev_ids=rev_ids,
    )
    return arrays


def _sequence(
    rows: list[dict[str, Any]],
    name: str,
    snapshot_hash: str,
    label: str,
    metadata: MetadataSpec,
    pad_id: int,
) -> SequenceEncoded:
    ids = _tokenized(rows, name, snapshot_hash, label)
    return SequenceEncoded(ids, metadata.transform(rows), labels_of(rows), pad_id, True)


def _transformer(
    name: str,
    data: Dataset,
    params: dict[str, Any],
    seed: int,
    cap: int | None,
    device: torch.device,
    pretrained: bool = True,
) -> Prepared:
    train_rows = sample_rows(data.train, cap, seed) if cap else data.train
    # Text-only exists to measure what metadata adds (loss_probe), not as a served recipe.
    use_metadata = params.get("use_metadata", True)
    metadata = MetadataSpec.fit(train_rows) if use_metadata else MetadataSpec((), (), ())
    torch.manual_seed(seed)
    encoder = load_encoder(name, pretrained)
    model = TransformerClassifier(encoder, len(metadata.columns), params["dropout"])
    frozen = 0
    trainable_blocks = params.get("trainable_blocks")
    if trainable_blocks is not None:
        frozen = freeze_lower_layers(model, trainable_blocks)
    pad_id = int(encoder.config.pad_token_id or 0)
    snapshot_hash = data.manifest["content_sha256"]
    sample = sample_rows(train_rows, TRAIN_SAMPLE_ROWS, seed)
    gpu = device.type == "cuda"
    return Prepared(
        name,
        params,
        model,
        metadata,
        FitConfig(
            epochs=params.get("epochs", 2),
            batch_size=params.get("batch_size", 16),
            learning_rate=params["learning_rate"],
            patience=1,
            warmup_share=0.1,
            mixed_precision=gpu,
            seed=seed,
            accumulate=params.get("accumulate", 1),
            weight_power=params.get("weight_power", FitConfig.weight_power),
            gradient_clip=params.get("gradient_clip", GRADIENT_CLIP),
            head_learning_rate=params.get("head_learning_rate"),
        ),
        _sequence(train_rows, name, snapshot_hash, f"train{len(train_rows)}", metadata, pad_id),
        _sequence(sample, name, snapshot_hash, f"sample{len(sample)}", metadata, pad_id),
        _sequence(data.validation, name, snapshot_hash, "validation", metadata, pad_id),
        {
            "tokenizer": tokenizer_path(name),
            "pad_token_id": pad_id,
            "frozen_blocks": frozen,
            "repo": PRETRAINED[name][0],
            "revision": PRETRAINED[name][1],
        },
        len(train_rows),
    )


def prepare(
    name: str,
    data: Dataset,
    params: dict[str, Any],
    device: torch.device,
    seed: int = 0,
    cap: int | None = None,
    pretrained: bool = True,
) -> Prepared:
    if name == MLP:
        return _mlp(data, params, seed, cap)
    if name == BILSTM:
        return _bilstm(data, params, seed, cap, pretrained)
    if name in TRANSFORMERS:
        return _transformer(name, data, params, seed, cap, device, pretrained)
    raise ValueError(f"unknown model {name}")


def encode_like(
    prepared: Prepared, rows: list[dict[str, Any]], label: str, data: Dataset
) -> Encoded:
    """Encode other rows the way this model's training rows were."""
    if prepared.name == MLP:
        return encode_tfidf(rows, prepared.extras["tfidf"], prepared.metadata)
    if prepared.name == BILSTM:
        return encode_tokens(rows, prepared.extras["vocabulary"], prepared.metadata)
    return _sequence(
        rows,
        prepared.name,
        data.manifest["content_sha256"],
        label,
        prepared.metadata,
        prepared.extras["pad_token_id"],
    )
