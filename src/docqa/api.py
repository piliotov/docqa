import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docqa import __version__
from docqa.assistant import Assistant
from docqa.config import Settings
from docqa.llm import LLMError
from docqa.loaders import SUPPORTED, UnsupportedFileError


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class TextDocument(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class SourceOut(BaseModel):
    doc_id: str
    chunk: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[SourceOut]


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


def create_app(assistant: Assistant | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    bot = assistant or Assistant.from_settings(settings)
    app = FastAPI(title="docqa", version=__version__)

    def persist() -> None:
        if assistant is None:
            bot.save(settings.index_dir)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "documents": len(bot.documents()), "chunks": len(bot.store)}

    @app.get("/documents")
    def list_documents() -> list[str]:
        return bot.documents()

    @app.post("/documents/text", response_model=IngestResponse, status_code=201)
    def add_text(doc: TextDocument) -> IngestResponse:
        chunks = bot.add_text(doc.doc_id, doc.text)
        persist()
        return IngestResponse(doc_id=doc.doc_id, chunks=chunks)

    @app.post("/documents", response_model=IngestResponse, status_code=201)
    async def upload(file: UploadFile) -> IngestResponse:
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(415, f"supported types: {', '.join(sorted(SUPPORTED))}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(await file.read())
            try:
                chunks = bot.add_file(path, doc_id=name)
            except UnsupportedFileError as exc:
                raise HTTPException(415, str(exc)) from exc
        persist()
        return IngestResponse(doc_id=name, chunks=chunks)

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete(doc_id: str) -> None:
        if not bot.delete(doc_id):
            raise HTTPException(404, "document not found")
        persist()

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            answer = bot.ask(req.question, req.k)
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc
        return AskResponse(
            question=answer.question,
            answer=answer.text,
            found=answer.found,
            sources=[SourceOut(**s.__dict__) for s in answer.sources],
        )

    return app
