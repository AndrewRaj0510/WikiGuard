"""Score edits with an exported model, using only ONNX Runtime and no PyTorch.

Evaluation goes through this same class, so the numbers in the reports describe exactly what users
get. Phase 3 builds the fallback chain on top of it.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from tokenizers import Tokenizer

from wikiguard_core.artifacts import CONFIG_FILE, MODEL_FILE, verify_artifact
from wikiguard_core.evaluation import sigmoid
from wikiguard_core.inputs import (
    MAX_SEQUENCE_LENGTH,
    MetadataSpec,
    Row,
    TfidfEncoder,
    TokenVocabulary,
    transformer_text,
)

MLP = "mlp"
BILSTM = "bilstm"
TRANSFORMER = "transformer"
TFIDF_FILE = "tfidf.json"
VOCABULARY_FILE = "vocabulary.json"
TOKENIZER_FILE = "tokenizer.json"
DEFAULT_BATCH = 32


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(MAX_SEQUENCE_LENGTH)
    tokenizer.no_padding()
    return tokenizer


class ServedModel:
    def __init__(self, folder: Path, threads: int = 1, expected_digest: str | None = None) -> None:
        # Verified first, so a tampered file is never handed to ONNX Runtime.
        self.config = verify_artifact(folder, expected_digest)
        self.folder = folder
        self.kind: str = self.config["kind"]
        self.temperature: float = self.config.get("temperature", 1.0)
        self.threshold: float | None = self.config.get("threshold")
        self.metadata = MetadataSpec.from_dict(self.config["metadata"])
        self.tfidf: TfidfEncoder | None = None
        self.vocabulary: TokenVocabulary | None = None
        self.tokenizer: Tokenizer | None = None
        if self.kind == MLP:
            self.tfidf = TfidfEncoder.from_dict(
                json.loads((folder / TFIDF_FILE).read_text("utf-8"))
            )
        elif self.kind == BILSTM:
            self.vocabulary = TokenVocabulary.from_dict(
                json.loads((folder / VOCABULARY_FILE).read_text("utf-8"))
            )
        else:
            self.tokenizer = load_tokenizer(folder / TOKENIZER_FILE)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )

    def _mlp_batches(
        self, rows: Sequence[Row], batch_size: int
    ) -> Iterator[tuple[list[int], dict[str, NDArray[Any]]]]:
        assert self.tfidf is not None  # noqa: S101 - narrows the type; set in __init__ for this kind
        for start in range(0, len(rows), batch_size):
            members = list(range(start, min(start + batch_size, len(rows))))
            chosen = [rows[index] for index in members]
            features = np.hstack([self.tfidf.dense(chosen), self.metadata.transform(chosen)])
            yield members, {"features": features}

    def _bilstm_batches(
        self, rows: Sequence[Row], batch_size: int
    ) -> Iterator[tuple[list[int], dict[str, NDArray[Any]]]]:
        assert self.vocabulary is not None  # noqa: S101
        encoded = [self.vocabulary.encode(row) for row in rows]
        # An LSTM reads padding as input, so only sequences of one length share a batch. That makes
        # a batched score identical to a single-edit score.
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(encoded):
            by_length.setdefault(ids.size, []).append(index)
        for indices in by_length.values():
            for start in range(0, len(indices), batch_size):
                members = indices[start : start + batch_size]
                yield (
                    members,
                    {
                        "input_ids": np.stack([encoded[index] for index in members]),
                        "metadata": self.metadata.transform([rows[index] for index in members]),
                    },
                )

    def _transformer_batches(
        self, rows: Sequence[Row], batch_size: int
    ) -> Iterator[tuple[list[int], dict[str, NDArray[Any]]]]:
        assert self.tokenizer is not None  # noqa: S101
        encodings = self.tokenizer.encode_batch_fast([transformer_text(row) for row in rows])
        ids = [np.array(encoding.ids, dtype=np.int64) for encoding in encodings]
        order = sorted(range(len(ids)), key=lambda index: ids[index].size)
        pad = int(self.config["pad_token_id"])
        for start in range(0, len(order), batch_size):
            members = order[start : start + batch_size]
            longest = max(ids[index].size for index in members)
            input_ids = np.full((len(members), longest), pad, dtype=np.int64)
            mask = np.zeros((len(members), longest), dtype=np.int64)
            for position, index in enumerate(members):
                input_ids[position, : ids[index].size] = ids[index]
                mask[position, : ids[index].size] = 1
            yield (
                members,
                {
                    "input_ids": input_ids,
                    "attention_mask": mask,
                    "metadata": self.metadata.transform([rows[index] for index in members]),
                },
            )

    def logits(self, rows: Sequence[Row], batch_size: int = DEFAULT_BATCH) -> NDArray[np.float64]:
        """Raw scores in the order of the rows, before calibration."""
        makers = {
            MLP: self._mlp_batches,
            BILSTM: self._bilstm_batches,
            TRANSFORMER: self._transformer_batches,
        }
        result = np.zeros(len(rows), dtype=np.float64)
        for members, inputs in makers[self.kind](rows, batch_size):
            output = self.session.run(None, inputs)[0]
            result[members] = np.asarray(output, dtype=np.float64).reshape(-1)
        return result

    def probabilities(
        self, rows: Sequence[Row], batch_size: int = DEFAULT_BATCH
    ) -> NDArray[np.float64]:
        return sigmoid(self.logits(rows, batch_size) / self.temperature)


def read_config(folder: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads((folder / CONFIG_FILE).read_text("utf-8"))
    return config
