from docqa.bm25 import BM25


def test_matching_document_ranks_first():
    index = BM25()
    index.add(["the cat sat on the mat", "dogs chase cats", "stock market news"])
    scores = index.scores("market")
    assert scores.index(max(scores)) == 2
    assert scores[0] == 0


def test_rare_terms_weigh_more():
    index = BM25()
    index.add(["python python python", "python rust", "python go"])
    scores = index.scores("rust python")
    assert scores[1] > scores[0]


def test_remove_rebuilds_statistics():
    index = BM25()
    index.add(["alpha", "beta", "gamma"])
    index.remove({1})
    assert len(index) == 2
    assert index.scores("beta") == [0.0, 0.0]


def test_empty_index():
    assert BM25().scores("anything") == []
