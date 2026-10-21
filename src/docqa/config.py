import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path("data/index")
    embedder: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm: str = "extractive"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str | None = None
    chunk_words: int = 120
    top_k: int = 4
    alpha: float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            index_dir=Path(env.get("DOCQA_INDEX_DIR", "data/index")),
            embedder=env.get("DOCQA_EMBEDDER", "hashing"),
            embedding_model=env.get(
                "DOCQA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm=env.get("DOCQA_LLM", "extractive"),
            llm_model=env.get("DOCQA_LLM_MODEL", "llama3.2"),
            llm_base_url=env.get("DOCQA_LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_api_key=env.get("DOCQA_LLM_API_KEY") or None,
            chunk_words=int(env.get("DOCQA_CHUNK_WORDS", "120")),
            top_k=int(env.get("DOCQA_TOP_K", "4")),
            alpha=float(env.get("DOCQA_ALPHA", "0.5")),
        )
