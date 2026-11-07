"""Load a validated snapshot and encode it for each kind of model."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from wikiguard_core.evaluation import slice_masks
from wikiguard_core.inputs import PAD, MetadataSpec, Row, TfidfEncoder, TokenVocabulary
from wikiguard_core.snapshot import SPLITS, read_manifest, read_snapshot
from wikiguard_core.validation import validate_snapshot

TRAIN, VALIDATION, TEST = SPLITS
# Sorting within this many batches groups similar lengths while keeping the order mostly random.
BUCKET_BATCHES = 50


@dataclass(frozen=True)
class Dataset:
    manifest: dict[str, Any]
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]

    def split(self, name: str) -> list[dict[str, Any]]:
        return {TRAIN: self.train, VALIDATION: self.validation, TEST: self.test}[name]


def load_dataset(folder: Path) -> Dataset:
    """Refuses to return anything if the snapshot fails its checks."""
    validate_snapshot(folder).require_passed()
    rows = read_snapshot(folder)
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in sorted(rows, key=lambda item: item["timestamp"]):
        by_split[row["split"]].append(row)
    return Dataset(read_manifest(folder), by_split[TRAIN], by_split[VALIDATION], by_split[TEST])


def labels_of(rows: Sequence[Row]) -> NDArray[np.int64]:
    return np.array([row["label"] for row in rows], dtype=np.int64)


def masks_of(rows: Sequence[Row]) -> dict[str, NDArray[np.bool_]]:
    return slice_masks(list(rows))


def sample_rows(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """A random sample that keeps the positive rate, returned in time order.

    Stratified so a cap can't change the class balance, and the loss weights mean the same thing.
    """
    if len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    labels = labels_of(rows)
    chosen: list[int] = []
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        count = round(cap * members.size / len(rows))
        chosen += rng.choice(members, size=count, replace=False).tolist()
    return [rows[index] for index in sorted(chosen)]


Batch = dict[str, Tensor]


class Encoded(Protocol):
    labels: NDArray[np.int64]

    def batches(
        self, batch_size: int, rng: np.random.Generator | None
    ) -> Iterator[tuple[Batch, NDArray[np.int64]]]:
        """Yields (inputs, row positions). A generator shuffles, and None keeps the order."""
        ...


def _positions(count: int, rng: np.random.Generator | None) -> NDArray[np.int64]:
    return np.arange(count) if rng is None else rng.permutation(count)


@dataclass
class TfidfEncoded:
    encoder: TfidfEncoder
    rows: Sequence[Row]
    metadata: NDArray[np.float32]
    labels: NDArray[np.int64]

    def batches(
        self, batch_size: int, rng: np.random.Generator | None
    ) -> Iterator[tuple[Batch, NDArray[np.int64]]]:
        order = _positions(len(self.rows), rng)
        for start in range(0, order.size, batch_size):
            members = order[start : start + batch_size]
            text = self.encoder.dense([self.rows[index] for index in members])
            features = np.hstack([text, self.metadata[members]])
            yield {"features": torch.from_numpy(features)}, members


@dataclass
class SequenceEncoded:
    """Token ids of varied length. The BiLSTM needs lengths and the transformers need a mask."""

    ids: list[NDArray[np.int64]]
    metadata: NDArray[np.float32]
    labels: NDArray[np.int64]
    pad_id: int = PAD
    use_mask: bool = False

    def _ordered_batches(
        self, batch_size: int, rng: np.random.Generator | None
    ) -> Iterator[NDArray[np.int64]]:
        sizes = np.array([ids.size for ids in self.ids])
        if rng is None:
            # Longest first, so a batch too big for the GPU fails right away, not an hour in.
            order = np.argsort(-sizes, kind="stable")
            for start in range(0, order.size, batch_size):
                yield order[start : start + batch_size]
            return
        order = rng.permutation(sizes.size)
        groups = []
        for start in range(0, order.size, batch_size * BUCKET_BATCHES):
            chunk = order[start : start + batch_size * BUCKET_BATCHES]
            chunk = chunk[np.argsort(sizes[chunk], kind="stable")]
            groups += [chunk[at : at + batch_size] for at in range(0, chunk.size, batch_size)]
        for index in rng.permutation(len(groups)):
            yield groups[index]

    def batches(
        self, batch_size: int, rng: np.random.Generator | None
    ) -> Iterator[tuple[Batch, NDArray[np.int64]]]:
        for members in self._ordered_batches(batch_size, rng):
            lengths = np.array([self.ids[index].size for index in members])
            input_ids = np.full((members.size, lengths.max()), self.pad_id, dtype=np.int64)
            for position, index in enumerate(members):
                input_ids[position, : lengths[position]] = self.ids[index]
            batch: Batch = {
                "input_ids": torch.from_numpy(input_ids),
                "metadata": torch.from_numpy(self.metadata[members]),
            }
            if self.use_mask:
                steps = np.arange(input_ids.shape[1])[None, :]
                batch["attention_mask"] = torch.from_numpy(
                    (steps < lengths[:, None]).astype(np.int64)
                )
            else:
                batch["lengths"] = torch.from_numpy(lengths)
            yield batch, members


def encode_tfidf(
    rows: Sequence[Row], encoder: TfidfEncoder, metadata: MetadataSpec
) -> TfidfEncoded:
    return TfidfEncoded(encoder, rows, metadata.transform(rows), labels_of(rows))


def encode_tokens(
    rows: Sequence[Row], vocabulary: TokenVocabulary, metadata: MetadataSpec
) -> SequenceEncoded:
    ids = [vocabulary.encode(row) for row in rows]
    return SequenceEncoded(ids, metadata.transform(rows), labels_of(rows))
