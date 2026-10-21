from typing import Protocol

import httpx

from docqa.chunking import split_sentences
from docqa.retriever import Hit
from docqa.tokenize import tokenize

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Cite sources as [n] using the context numbers. "
    "If the context does not contain the answer, say you do not know."
)

NO_ANSWER = "I could not find an answer in the provided documents."


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> str: ...


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(
        f"[{n}] ({hit.chunk.doc_id})\n{hit.chunk.text}" for n, hit in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class ExtractiveLLM:
    """Offline fallback that returns the context sentences that best match the question."""

    def __init__(self, max_sentences: int = 2, min_overlap: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, hits: list[Hit]) -> str:
        terms = set(tokenize(question))
        candidates: list[tuple[int, int, int, str]] = []
        for n, hit in enumerate(hits, start=1):
            for position, sentence in enumerate(split_sentences(hit.chunk.text)):
                overlap = len(terms & set(tokenize(sentence)))
                if overlap >= self.min_overlap:
                    candidates.append((overlap, n, position, sentence))
        if not candidates:
            return NO_ANSWER
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        best = sorted(candidates[: self.max_sentences], key=lambda c: (c[1], c[2]))
        return " ".join(f"{sentence} [{n}]" for _, n, _, sentence in best)


class OpenAICompatibleLLM:
    """Chat completions client for OpenAI, Groq, Ollama or any compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return NO_ANSWER
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
        }
        try:
            response = self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"generation failed: {exc}") from exc
        return str(content).strip()
