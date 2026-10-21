import io

import pytest
from fastapi.testclient import TestClient

from docqa import Assistant
from docqa.api import create_app
from docqa.config import Settings
from docqa.llm import LLMError


@pytest.fixture
def client(assistant) -> TestClient:
    return TestClient(create_app(assistant))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 4


def test_ask(client):
    response = client.post("/ask", json={"question": "How long does cold brew steep?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["found"]
    assert "12 to 24 hours" in body["answer"]
    assert body["sources"][0]["doc_id"] == "coffee-brewing.md"
    assert len(body["sources"]) <= 2


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "k": 0}, {}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_add_text_and_delete(client):
    response = client.post(
        "/documents/text", json={"doc_id": "faq", "text": "Refunds take 5 days."}
    )
    assert response.status_code == 201
    assert response.json() == {"doc_id": "faq", "chunks": 1}
    assert "faq" in client.get("/documents").json()

    answer = client.post("/ask", json={"question": "How long do refunds take?"}).json()
    assert answer["sources"][0]["doc_id"] == "faq"

    assert client.delete("/documents/faq").status_code == 204
    assert client.delete("/documents/faq").status_code == 404


def test_upload_file(client):
    files = {"file": ("guide.txt", io.BytesIO(b"The office opens at nine."), "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == 201
    assert response.json()["doc_id"] == "guide.txt"


def test_upload_rejects_unsupported_type(client):
    files = {"file": ("photo.jpg", io.BytesIO(b"\xff\xd8"), "image/jpeg")}
    assert client.post("/documents", files=files).status_code == 415


def test_llm_failure_maps_to_bad_gateway(docs_dir):
    class Broken:
        def generate(self, question, hits):
            raise LLMError("down")

    bot = Assistant(llm=Broken())
    bot.add_folder(docs_dir)
    response = TestClient(create_app(bot)).post("/ask", json={"question": "liveness probe"})
    assert response.status_code == 502


def test_app_persists_index(tmp_path):
    client = TestClient(create_app(settings=Settings(index_dir=tmp_path)))
    client.post("/documents/text", json={"doc_id": "a", "text": "Saved to disk."})
    assert (tmp_path / "chunks.json").exists()
    reloaded = TestClient(create_app(settings=Settings(index_dir=tmp_path)))
    assert reloaded.get("/documents").json() == ["a"]
