import pytest

from docqa.embeddings import HashingEmbedder
from docqa.retriever import HybridRetriever
from docqa.store import VectorStore


def test_finds_relevant_document(assistant):
    hits = assistant.search("what happens when the liveness probe fails", k=3)
    assert hits[0].chunk.doc_id == "kubernetes-probes.md"
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_single_signal_modes(assistant, alpha):
    assistant.retriever.alpha = alpha
    hits = assistant.search("multiprocessing interpreter processes", k=2)
    assert hits[0].chunk.doc_id == "python-gil.md"


def test_respects_k(assistant):
    assert len(assistant.search("the", k=2)) <= 2


def test_empty_store_and_blank_query():
    retriever = HybridRetriever(VectorStore(32), HashingEmbedder(32))
    assert retriever.search("anything") == []


def test_blank_query(assistant):
    assert assistant.search("   ") == []


def test_invalid_alpha():
    with pytest.raises(ValueError):
        HybridRetriever(VectorStore(8), HashingEmbedder(8), alpha=1.5)
