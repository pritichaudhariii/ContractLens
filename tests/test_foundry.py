"""Azure AI Foundry clients against a local mock server: URL shape for both endpoint
styles, auth header, retries on 429, error handling, embedding batching and ordering."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from contractlens.embeddings import FoundryEmbedder, build_embedder
from contractlens.foundry import FoundryClient, FoundryConfig, FoundryError
from contractlens.llm import FoundryLLM, build_llm


class MockFoundry(BaseHTTPRequestHandler):
    calls: list[dict] = []
    fail_first = 0
    dimension = 4

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockFoundry.calls.append({"path": self.path, "api_key": self.headers.get("api-key"), "body": body})
        if MockFoundry.fail_first > 0:
            MockFoundry.fail_first -= 1
            return self._send(429, {"error": "rate limited"}, {"Retry-After": "0"})
        if self.path.split("?")[0].endswith("/embeddings"):
            rows = [{"index": i, "embedding": [float(i)] * MockFoundry.dimension} for i in range(len(body["input"]))]
            return self._send(200, {"data": rows[::-1]})  # out of order on purpose
        if body["messages"][-1]["content"] == "bad":
            return self._send(400, {"error": "bad request"})
        return self._send(200, {"choices": [{"message": {"content": "  The cap is $2,000,000 [1].  "}}]})


@pytest.fixture()
def server():
    MockFoundry.calls, MockFoundry.fail_first, MockFoundry.dimension = [], 0, 4
    httpd = HTTPServer(("127.0.0.1", 0), MockFoundry)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def client(url: str, **kw) -> FoundryClient:
    return FoundryClient(FoundryConfig(endpoint=url, api_key="test-key", max_retries=3, timeout_s=5, **kw))


def test_azure_openai_style_chat(server):
    llm = FoundryLLM(client(server), deployment="gpt-4o-mini")
    assert llm.complete("system prompt", "question") == "The cap is $2,000,000 [1]."
    call = MockFoundry.calls[0]
    assert call["path"] == "/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21"
    assert call["api_key"] == "test-key"
    assert [m["role"] for m in call["body"]["messages"]] == ["system", "user"]
    assert call["body"]["temperature"] == 0.0 and "model" not in call["body"]


def test_inference_style_sends_model_in_body(server):
    llm = FoundryLLM(client(server + "/models", api_style="inference"), deployment="Llama-3.3-70B-Instruct")
    llm.complete("s", "q")
    call = MockFoundry.calls[0]
    assert call["path"].startswith("/models/chat/completions?api-version=")
    assert call["body"]["model"] == "Llama-3.3-70B-Instruct"


def test_retries_rate_limits(server):
    MockFoundry.fail_first = 2
    FoundryLLM(client(server), deployment="gpt-4o-mini").complete("s", "q")
    assert len(MockFoundry.calls) == 3


def test_non_retryable_error_is_raised(server):
    with pytest.raises(FoundryError) as err:
        FoundryLLM(client(server), deployment="gpt-4o-mini").complete("s", "bad")
    assert err.value.status == 400 and len(MockFoundry.calls) == 1


def test_embeddings_batched_and_reordered(server):
    emb = FoundryEmbedder(client(server), deployment="text-embedding-3-small", dimension=4, batch_size=2)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert vectors == [[0.0] * 4, [1.0] * 4, [0.0] * 4]
    assert len(MockFoundry.calls) == 2
    assert MockFoundry.calls[0]["body"]["dimensions"] == 4
    assert emb.embed_query("q") == [0.0] * 4


def test_embedding_dimension_mismatch_is_explained(server):
    MockFoundry.dimension = 8
    emb = FoundryEmbedder(client(server), deployment="text-embedding-3-small", dimension=4)
    with pytest.raises(RuntimeError, match="AZURE_FOUNDRY_EMBEDDING_DIMENSION=8"):
        emb.embed_query("q")


def test_builders_require_configuration():
    with pytest.raises(RuntimeError):
        FoundryClient(FoundryConfig(endpoint="", api_key=""))
    with pytest.raises(RuntimeError):
        build_llm("foundry", model="gpt-4o-mini", foundry=None)
    with pytest.raises(RuntimeError):
        build_embedder("foundry", model="text-embedding-3-small", dimension=1536, foundry=None)
