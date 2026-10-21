import json

import httpx
import pytest

from docqa.chunking import Chunk
from docqa.llm import NO_ANSWER, ExtractiveLLM, LLMError, OpenAICompatibleLLM, build_prompt
from docqa.retriever import Hit


def hit(doc_id: str, text: str) -> Hit:
    return Hit(Chunk(doc_id, 0, text), 1.0, 1.0, 1.0)


HITS = [
    hit("probes.md", "The kubelet restarts the container. Readiness controls traffic."),
    hit("coffee.md", "Coffee needs hot water."),
]


def test_build_prompt_numbers_sources():
    prompt = build_prompt("why?", HITS)
    assert "[1] (probes.md)" in prompt
    assert "[2] (coffee.md)" in prompt
    assert prompt.endswith("Question: why?\nAnswer:")


def test_extractive_picks_best_sentence_with_citation():
    answer = ExtractiveLLM(max_sentences=1).generate("who restarts the container", HITS)
    assert answer == "The kubelet restarts the container. [1]"


def test_extractive_keeps_document_order():
    answer = ExtractiveLLM(max_sentences=2).generate("container water", HITS)
    assert answer.index("[1]") < answer.index("[2]")


def test_extractive_without_overlap():
    assert ExtractiveLLM().generate("quantum entanglement", HITS) == NO_ANSWER


def client_returning(handler) -> httpx.Client:
    return httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))


def test_openai_client_sends_chat_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " It restarts. "}}]})

    llm = OpenAICompatibleLLM("test-model", client=client_returning(handler))
    assert llm.generate("what happens?", HITS) == "It restarts."
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "[1] (probes.md)" in seen["body"]["messages"][1]["content"]


def test_openai_client_sets_auth_header():
    llm = OpenAICompatibleLLM("m", base_url="https://api.example.com/v1/", api_key="secret")
    assert llm._client.headers["Authorization"] == "Bearer secret"
    assert str(llm._client.base_url) == "https://api.example.com/v1/"


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, text="boom"), httpx.Response(200, json={"choices": []})],
)
def test_openai_client_errors(response):
    llm = OpenAICompatibleLLM("m", client=client_returning(lambda request: response))
    with pytest.raises(LLMError):
        llm.generate("q", HITS)


def test_openai_client_skips_call_without_context():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    llm = OpenAICompatibleLLM("m", client=client_returning(handler))
    assert llm.generate("q", []) == NO_ANSWER
