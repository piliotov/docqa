import numpy as np

from docqa.embeddings import HashingEmbedder


def test_vectors_are_normalized_and_deterministic():
    embedder = HashingEmbedder(dim=128)
    first = embedder.embed(["kubernetes restarts the container", "coffee brewing"])
    second = embedder.embed(["kubernetes restarts the container", "coffee brewing"])
    assert first.shape == (2, 128)
    assert first.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(first, second)


def test_similar_texts_score_higher():
    embedder = HashingEmbedder()
    query, close, far = embedder.embed(
        ["when does the container restart", "the kubelet restarts containers", "brew coffee"]
    )
    assert query @ close > query @ far


def test_empty_text_gives_zero_vector():
    [vec] = HashingEmbedder(dim=16).embed([""])
    assert not vec.any()
