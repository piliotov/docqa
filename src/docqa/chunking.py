import re
from dataclasses import dataclass, field

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.index}"


def split_sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENTENCE_END.split(text))
    return [" ".join(p.split()) for p in parts if p]


def chunk_text(
    text: str,
    doc_id: str,
    max_words: int = 120,
    overlap: int = 1,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")

    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        if current:
            chunks.append(Chunk(doc_id, len(chunks), " ".join(current), dict(metadata or {})))

    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            flush()
            current = current[-overlap:] if overlap else []
            count = sum(len(s.split()) for s in current)
        current.append(sentence)
        count += words
    flush()
    return chunks
