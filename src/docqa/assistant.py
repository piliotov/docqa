from dataclasses import dataclass, field
from pathlib import Path

from docqa.chunking import chunk_text
from docqa.config import Settings
from docqa.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from docqa.llm import LLM, NO_ANSWER, ExtractiveLLM, OpenAICompatibleLLM
from docqa.loaders import iter_documents, load_text
from docqa.retriever import Hit, HybridRetriever
from docqa.store import VectorStore


@dataclass(frozen=True)
class Source:
    doc_id: str
    chunk: int
    score: float
    text: str


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.text != NO_ANSWER


class Assistant:
    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        store: VectorStore | None = None,
        chunk_words: int = 120,
        top_k: int = 4,
        alpha: float = 0.5,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm or ExtractiveLLM()
        self.store = store if store is not None else VectorStore(self.embedder.dim)
        if self.store.dim != self.embedder.dim:
            raise ValueError("store and embedder dimensions differ")
        self.chunk_words = chunk_words
        self.top_k = top_k
        self.retriever = HybridRetriever(self.store, self.embedder, alpha=alpha)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Assistant":
        embedder: Embedder
        if settings.embedder == "sentence-transformers":
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
        elif settings.embedder == "hashing":
            embedder = HashingEmbedder()
        else:
            raise ValueError(f"unknown embedder: {settings.embedder}")

        llm: LLM
        if settings.llm == "openai":
            llm = OpenAICompatibleLLM(
                settings.llm_model, settings.llm_base_url, settings.llm_api_key
            )
        elif settings.llm == "extractive":
            llm = ExtractiveLLM()
        else:
            raise ValueError(f"unknown llm: {settings.llm}")

        index = settings.index_dir
        store = VectorStore.load(index) if (index / "chunks.json").exists() else None
        return cls(
            embedder,
            llm,
            store,
            chunk_words=settings.chunk_words,
            top_k=settings.top_k,
            alpha=settings.alpha,
        )

    def add_text(self, doc_id: str, text: str, metadata: dict[str, str] | None = None) -> int:
        self.store.delete(doc_id)
        chunks = chunk_text(text, doc_id, max_words=self.chunk_words, metadata=metadata)
        if not chunks:
            return 0
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(chunks)

    def add_file(self, path: Path, doc_id: str | None = None) -> int:
        return self.add_text(doc_id or path.name, load_text(path), {"path": str(path)})

    def add_folder(self, folder: Path) -> dict[str, int]:
        return {
            str(p.relative_to(folder)): self.add_file(p, str(p.relative_to(folder)))
            for p in iter_documents(folder)
        }

    def delete(self, doc_id: str) -> bool:
        return self.store.delete(doc_id) > 0

    def documents(self) -> list[str]:
        return self.store.doc_ids()

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(query, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.search(question, k)
        text = self.llm.generate(question, hits) if hits else NO_ANSWER
        sources = [
            Source(h.chunk.doc_id, h.chunk.index, round(h.score, 6), h.chunk.text) for h in hits
        ]
        return Answer(question, text, sources)

    def save(self, folder: Path) -> None:
        self.store.save(folder)
