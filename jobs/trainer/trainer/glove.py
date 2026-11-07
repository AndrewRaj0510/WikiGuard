"""GloVe word vectors for the BiLSTM.

Only vectors for words in the vocabulary are used, and the trained embedding is saved with the
model, so serving never needs this download.
"""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from wikiguard_core.inputs import SPECIAL_TOKENS, TokenVocabulary

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
GLOVE_FOLDER = Path("data/glove")
EMBEDDING_SIZE = 100
# Unknown words start small and random, close to the spread of real GloVe values.
RANDOM_SCALE = 0.25


def ensure_glove(folder: Path = GLOVE_FOLDER) -> Path:
    target = folder / GLOVE_FILE
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "glove.6B.zip"
    if not archive.exists():
        request = urllib.request.Request(  # noqa: S310 - fixed https address
            GLOVE_URL, headers={"User-Agent": "WikiGuard/0.1"}
        )
        with urllib.request.urlopen(request) as response, archive.open("wb") as file:  # noqa: S310
            shutil.copyfileobj(response, file)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract(GLOVE_FILE, folder)
    return target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_vectors(path: Path, wanted: set[str], size: int = EMBEDDING_SIZE) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            word, _, values = line.rstrip().partition(" ")
            if word in wanted:
                vector = np.array(values.split(" "), dtype=np.float32)
                if vector.size == size:
                    vectors[word] = vector
    return vectors


def load_embeddings(vocabulary: TokenVocabulary, seed: int) -> tuple[Tensor, int]:
    """A matrix for the vocabulary, and how many of its words GloVe knew."""
    vectors = read_vectors(ensure_glove(), set(vocabulary.tokens))
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, RANDOM_SCALE, (vocabulary.size, EMBEDDING_SIZE)).astype(np.float32)
    # Padding must stay all zeros.
    matrix[0] = 0.0
    found = 0
    for token, index in vocabulary.tokens.items():
        if token in vectors and token not in SPECIAL_TOKENS:
            matrix[index] = vectors[token]
            found += 1
    return torch.from_numpy(matrix), found
