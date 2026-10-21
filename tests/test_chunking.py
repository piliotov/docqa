import pytest

from docqa.chunking import chunk_text, split_sentences


def test_split_sentences_handles_punctuation_and_paragraphs():
    text = "First one. Second one!  Third?\n\nNew   paragraph"
    assert split_sentences(text) == ["First one.", "Second one!", "Third?", "New paragraph"]


def test_chunks_respect_word_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_text(text, "doc", max_words=20, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 20 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_last_sentence():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = chunk_text(text, "doc", max_words=6, overlap=1)
    assert chunks[0].text.endswith("Delta epsilon zeta.")
    assert chunks[1].text.startswith("Delta epsilon zeta.")


def test_long_sentence_becomes_its_own_chunk():
    long = " ".join(["word"] * 50) + "."
    chunks = chunk_text(f"Short. {long} Tail.", "doc", max_words=10, overlap=0)
    assert any(c.text == long for c in chunks)


def test_empty_text_has_no_chunks():
    assert chunk_text("   ", "doc") == []


def test_chunk_id_and_metadata():
    [chunk] = chunk_text("Hello there.", "notes.md", metadata={"path": "/tmp/notes.md"})
    assert chunk.id == "notes.md:0"
    assert chunk.metadata == {"path": "/tmp/notes.md"}


@pytest.mark.parametrize("kwargs", [{"max_words": 0}, {"overlap": -1}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        chunk_text("Text.", "doc", **kwargs)
