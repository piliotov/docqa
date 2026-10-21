import hashlib
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from docqa.tokenize import tokenize

Vector = NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> Vector: ...


def _normalize(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Feature hashing over word unigrams, bigrams and character trigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenize(text)
        feats = [(w, 1.0) for w in words]
        feats += [(f"{a}_{b}", 0.5) for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"#{w}#"
            feats += [(padded[i : i + 3], 0.25) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> Vector:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                col, sign = self._bucket(feature)
                matrix[row, col] += sign * weight
        return _normalize(matrix)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> Vector:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))
