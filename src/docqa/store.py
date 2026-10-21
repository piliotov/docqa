import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from docqa.bm25 import BM25
from docqa.chunking import Chunk
from docqa.embeddings import Vector


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vectors: Vector = np.zeros((0, dim), dtype=np.float32)
        self.bm25 = BM25()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: Vector) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.bm25.add([c.text for c in chunks])

    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    def delete(self, doc_id: str) -> int:
        drop = {i for i, c in enumerate(self.chunks) if c.doc_id == doc_id}
        if not drop:
            return 0
        keep = [i for i in range(len(self.chunks)) if i not in drop]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep]
        self.bm25.remove(drop)
        return len(drop)

    def dense_scores(self, query: Vector) -> Vector:
        if len(self.chunks) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ query

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        payload = {"dim": self.dim, "chunks": [asdict(c) for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        payload = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        store = cls(int(payload["dim"]))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        store.add(chunks, np.load(folder / "vectors.npy"))
        return store
