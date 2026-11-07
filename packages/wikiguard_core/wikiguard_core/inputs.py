"""Turn build_features() output into the numbers each model reads, for training and for serving.

A snapshot row and EditFeatures.as_dict() have the same keys, so one code path covers both, which
is what keeps training and serving from drifting apart. Nothing here needs PyTorch, and everything
fitted on training data is plain JSON, because a pickled preprocessor can run code when loaded.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Row = Mapping[str, Any]
FloatMatrix = NDArray[np.float32]

MAX_SEQUENCE_LENGTH = 256
# Hostile edits can be 200k characters, so text is cut before any tokenizer sees it.
MAX_FIELD_CHARS = 20_000
# Share of the token budget per field, so a huge addition can't crowd out the comment or removal.
COMMENT_CHARS = 200
ADDED_CHARS = 640
REMOVED_CHARS = 320
LSTM_ADDED_TOKENS = 170
LSTM_REMOVED_TOKENS = 85
# Standardized values are clipped, so an extreme edit can't produce an extreme input.
METADATA_CLIP = 10.0
# Below this a column is rounding noise: float32 inputs can't hold differences this small.
MIN_METADATA_STD = 1e-6
WORD = re.compile(r"\w+")
# Punctuation is kept for the BiLSTM because vandalism is often "!!!!!" as much as words.
TOKEN = re.compile(r"\w+|[^\w\s]")
HOURS_IN_DAY = 24


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _metadata_values(row: Row) -> dict[str, float]:
    # Sizes span many orders of magnitude, so they're logged. Hours are a circle: 23 is next to 0.
    angle = 2 * math.pi * row["hour_of_day"] / HOURS_IN_DAY
    return {
        "is_anonymous": float(row["is_anonymous"]),
        "user_edit_count_bucket": float(row["user_edit_count_bucket"]),
        "bytes_delta": _signed_log(row["bytes_delta"]),
        "is_minor": float(row["is_minor"]),
        "chars_added": math.log1p(row["chars_added"]),
        "uppercase_ratio": float(row["uppercase_ratio"]),
        "badword_count": math.log1p(row["badword_count"]),
        "has_external_link": float(row["has_external_link"]),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "page_length": math.log1p(row["page_length"]),
    }


@dataclass(frozen=True)
class MetadataSpec:
    """Which metadata a model reads, and how to scale it. Stored with the model.

    A feature that never varies in training is left out, so the model can't react to a value it
    has never seen. That's what happens to user_edit_count on the backfill, where it's always
    missing (F-070). Once live rows hold real counts, a retrain picks the feature up on its own.
    """

    columns: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Iterable[Row]) -> "MetadataSpec":
        values = [_metadata_values(row) for row in rows]
        kept, means, stds = [], [], []
        for column in values[0]:
            series = np.array([value[column] for value in values], dtype=np.float64)
            std = float(series.std())
            if std > MIN_METADATA_STD:
                kept.append(column)
                means.append(float(series.mean()))
                stds.append(std)
        return cls(tuple(kept), tuple(means), tuple(stds))

    def transform(self, rows: Sequence[Row]) -> FloatMatrix:
        matrix = np.zeros((len(rows), len(self.columns)), dtype=np.float32)
        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        for index, row in enumerate(rows):
            values = _metadata_values(row)
            matrix[index] = [values[column] for column in self.columns]
        scaled: FloatMatrix = np.clip((matrix - means) / stds, -METADATA_CLIP, METADATA_CLIP)
        return scaled

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetadataSpec":
        return cls(tuple(data["columns"]), tuple(data["means"]), tuple(data["stds"]))


def _prefixed_words(row: Row, use_comment: bool) -> list[str]:
    """Added and removed words are separate features: adding a word isn't removing it."""
    words = [f"a:{word}" for word in WORD.findall(row["added_text"][:MAX_FIELD_CHARS].lower())]
    words += [f"r:{word}" for word in WORD.findall(row["removed_text"][:MAX_FIELD_CHARS].lower())]
    if use_comment:
        words += [f"c:{word}" for word in WORD.findall(row["comment"][:MAX_FIELD_CHARS].lower())]
    return words


@dataclass(frozen=True)
class TfidfEncoder:
    """TF-IDF written out rather than taken from scikit-learn, so serving needs neither sklearn nor
    a pickle, and the vocabulary is a JSON file anyone can read."""

    vocabulary: Mapping[str, int]
    idf: tuple[float, ...]
    use_comment: bool

    @classmethod
    def fit(
        cls, rows: Sequence[Row], max_features: int, min_df: int, use_comment: bool
    ) -> "TfidfEncoder":
        document_frequency: Counter[str] = Counter()
        for row in rows:
            document_frequency.update(set(_prefixed_words(row, use_comment)))
        # Sorting by count then term keeps the order the same on every run.
        ranked = sorted(document_frequency.items(), key=lambda item: (-item[1], item[0]))
        kept = [(term, count) for term, count in ranked if count >= min_df][:max_features]
        total = len(rows)
        return cls(
            {term: index for index, (term, _) in enumerate(kept)},
            tuple(math.log((1 + total) / (1 + count)) + 1 for _, count in kept),
            use_comment,
        )

    @property
    def size(self) -> int:
        return len(self.vocabulary)

    def encode(self, row: Row) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        counts = Counter(
            self.vocabulary[word]
            for word in _prefixed_words(row, self.use_comment)
            if word in self.vocabulary
        )
        indices = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        # 1 + log(count), so a word repeated 500 times doesn't count 500 times as much.
        values = np.array(
            [(1 + math.log(count)) * self.idf[index] for index, count in counts.items()],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(values))
        return indices, (values / norm if norm else values)

    def dense(self, rows: Sequence[Row]) -> FloatMatrix:
        matrix = np.zeros((len(rows), self.size), dtype=np.float32)
        for index, row in enumerate(rows):
            columns, values = self.encode(row)
            matrix[index, columns] = values
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabulary": dict(self.vocabulary),
            "idf": list(self.idf),
            "use_comment": self.use_comment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TfidfEncoder":
        return cls(data["vocabulary"], tuple(data["idf"]), data["use_comment"])


PAD = 0
UNKNOWN = 1
SEPARATOR = 2
SPECIAL_TOKENS = ("<pad>", "<unk>", "<sep>")


@dataclass(frozen=True)
class TokenVocabulary:
    """Word-to-id table for the BiLSTM. Saved as JSON beside the model."""

    tokens: Mapping[str, int]
    use_comment: bool

    @classmethod
    def fit(
        cls, rows: Iterable[Row], max_tokens: int, min_count: int, use_comment: bool
    ) -> "TokenVocabulary":
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(TOKEN.findall(row["added_text"][:MAX_FIELD_CHARS].lower()))
            counts.update(TOKEN.findall(row["removed_text"][:MAX_FIELD_CHARS].lower()))
            if use_comment:
                counts.update(TOKEN.findall(row["comment"][:MAX_FIELD_CHARS].lower()))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        room = max_tokens - len(SPECIAL_TOKENS)
        kept = [token for token, count in ranked if count >= min_count][:room]
        table = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        table |= {token: index + len(SPECIAL_TOKENS) for index, token in enumerate(kept)}
        return cls(table, use_comment)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def _ids(self, text: str, limit: int) -> list[int]:
        words = TOKEN.findall(text[:MAX_FIELD_CHARS].lower())[:limit]
        return [self.tokens.get(word, UNKNOWN) for word in words]

    def encode(self, row: Row) -> NDArray[np.int64]:
        """Added tokens, a separator, removed tokens. Never empty, so there's always a step."""
        ids = self._ids(row["added_text"], LSTM_ADDED_TOKENS)
        ids += [SEPARATOR, *self._ids(row["removed_text"], LSTM_REMOVED_TOKENS)]
        if self.use_comment:
            room = MAX_SEQUENCE_LENGTH - len(ids) - 1
            ids += [SEPARATOR, *self._ids(row["comment"], max(room, 0))]
        return np.array(ids[:MAX_SEQUENCE_LENGTH], dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": dict(self.tokens), "use_comment": self.use_comment}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenVocabulary":
        return cls(data["tokens"], data["use_comment"])


def transformer_text(row: Row) -> str:
    """One string for the three transformers. The comment comes first: it's short and informative,
    and it would otherwise be the first thing truncation removes."""
    comment = row["comment"][:COMMENT_CHARS]
    added = row["added_text"][:ADDED_CHARS]
    removed = row["removed_text"][:REMOVED_CHARS]
    return f"comment: {comment} added: {added} removed: {removed}"
