from pathlib import Path

import pytest

from docqa import Assistant
from docqa.config import Settings
from docqa.embeddings import HashingEmbedder
from docqa.llm import ExtractiveLLM, OpenAICompatibleLLM
from docqa.loaders import UnsupportedFileError, load_text
from docqa.store import VectorStore


def test_ask_returns_answer_with_sources(assistant):
    answer = assistant.ask("Which directive should be used for sensitive data?")
    assert answer.found
    assert "no-store" in answer.text
    assert answer.sources[0].doc_id == "http-caching.md"


def test_ask_without_documents():
    answer = Assistant().ask("anything?")
    assert not answer.found
    assert answer.sources == []


def test_reingesting_replaces_document():
    bot = Assistant()
    bot.add_text("notes", "Old content about apples.")
    bot.add_text("notes", "New content about pears.")
    assert bot.documents() == ["notes"]
    assert all("pears" in c.text for c in bot.store.chunks)


def test_delete(assistant):
    assert assistant.delete("coffee-brewing.md")
    assert "coffee-brewing.md" not in assistant.documents()
    assert not assistant.delete("coffee-brewing.md")


def test_add_folder_lists_files(docs_dir):
    added = Assistant().add_folder(docs_dir)
    assert set(added) == {p.name for p in docs_dir.glob("*.md")}
    assert all(n > 0 for n in added.values())


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        Assistant(embedder=HashingEmbedder(32), store=VectorStore(64))


def test_from_settings_loads_saved_index(tmp_path, assistant):
    assistant.save(tmp_path)
    bot = Assistant.from_settings(Settings(index_dir=tmp_path))
    assert bot.documents() == assistant.documents()
    assert isinstance(bot.llm, ExtractiveLLM)


def test_from_settings_openai():
    bot = Assistant.from_settings(Settings(index_dir=Path("missing"), llm="openai"))
    assert isinstance(bot.llm, OpenAICompatibleLLM)
    assert len(bot.store) == 0


@pytest.mark.parametrize("field", ["embedder", "llm"])
def test_from_settings_rejects_unknown(field):
    with pytest.raises(ValueError):
        Assistant.from_settings(Settings(**{field: "nope"}))


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCQA_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("DOCQA_TOP_K", "7")
    monkeypatch.setenv("DOCQA_LLM_API_KEY", "")
    settings = Settings.from_env()
    assert settings.index_dir == tmp_path
    assert settings.top_k == 7
    assert settings.llm_api_key is None


def test_loader_rejects_unknown_type(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedFileError):
        load_text(path)
