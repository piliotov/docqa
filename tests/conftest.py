from pathlib import Path

import pytest

from docqa import Assistant

DOCS = Path(__file__).resolve().parent.parent / "eval" / "docs"


@pytest.fixture
def docs_dir() -> Path:
    return DOCS


@pytest.fixture
def assistant(docs_dir: Path) -> Assistant:
    bot = Assistant(chunk_words=60)
    bot.add_folder(docs_dir)
    return bot
