import numpy as np
import pytest

from docqa.chunking import Chunk
from docqa.embeddings import HashingEmbedder
from docqa.store import VectorStore


def make_store() -> VectorStore:
    embedder = HashingEmbedder(dim=64)
    chunks = [Chunk("a.md", 0, "first text"), Chunk("a.md", 1, "more"), Chunk("b.md", 0, "other")]
    store = VectorStore(64)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_add_and_list_documents():
    store = make_store()
    assert len(store) == 3
    assert store.doc_ids() == ["a.md", "b.md"]


def test_delete_removes_all_chunks_of_document():
    store = make_store()
    assert store.delete("a.md") == 2
    assert store.doc_ids() == ["b.md"]
    assert store.vectors.shape == (1, 64)
    assert len(store.bm25) == 1
    assert store.delete("missing") == 0


def test_rejects_mismatched_input():
    store = VectorStore(8)
    with pytest.raises(ValueError):
        store.add([Chunk("a", 0, "x")], np.zeros((2, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        store.add([Chunk("a", 0, "x")], np.zeros((1, 4), dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path):
    store = make_store()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.chunks == store.chunks
    np.testing.assert_array_equal(loaded.vectors, store.vectors)
    assert loaded.bm25.scores("other") == store.bm25.scores("other")


def test_empty_store_scores():
    assert VectorStore(4).dense_scores(np.ones(4, dtype=np.float32)).size == 0
