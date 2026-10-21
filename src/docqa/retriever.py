from dataclasses import dataclass

import numpy as np

from docqa.chunking import Chunk
from docqa.embeddings import Embedder
from docqa.store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense: float
    keyword: float


def _ranks(scores: list[float]) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {idx: rank for rank, idx in enumerate(order) if scores[idx] > 0}


class HybridRetriever:
    """Combines dense and BM25 rankings with reciprocal rank fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.store = store
        self.embedder = embedder
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if len(self.store) == 0 or not query.strip():
            return []
        query_vec = self.embedder.embed([query])[0]
        dense = self.store.dense_scores(query_vec).tolist()
        keyword = self.store.bm25.scores(query)

        dense_ranks = _ranks(dense)
        keyword_ranks = _ranks(keyword)
        fused = np.zeros(len(self.store), dtype=np.float64)
        for idx, rank in dense_ranks.items():
            fused[idx] += self.alpha / (self.rrf_k + rank + 1)
        for idx, rank in keyword_ranks.items():
            fused[idx] += (1 - self.alpha) / (self.rrf_k + rank + 1)

        top = [int(i) for i in np.argsort(-fused)[:k] if fused[i] > 0]
        return [
            Hit(self.store.chunks[i], float(fused[i]), float(dense[i]), float(keyword[i]))
            for i in top
        ]
