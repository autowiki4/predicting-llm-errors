"""Exercise the real Anthropic and OpenAI SDKs against tiny local fake servers.

No network and no API key are needed, but the actual SDK code paths run: if a
request parameter were invalid for the installed SDK, or the response parsing
were wrong, these tests would fail.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_reliability.config import LLMConfig
from llm_reliability.llm import make_client
from llm_reliability.llm.anthropic_client import FALLBACK_BETA, AnthropicClient
from llm_reliability.llm.openai_compat import OpenAICompatibleClient

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")


class FakeServer:
    def __init__(self, response_body: dict):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                outer.requests.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(self.rfile.read(length) or b"{}"),
                })
                payload = json.dumps(response_body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


def _claude_message(text="Step 1...\nAnswer: 18", stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 345},
    }


def _sdk_anthropic(url):
    return anthropic.Anthropic(api_key="test-key", base_url=url, max_retries=0)


def test_anthropic_request_shape_and_parsing():
    server = FakeServer(_claude_message())
    try:
        client = AnthropicClient(model="claude-opus-5-5", effort="low", client=_sdk_anthropic(server.url))
        r = client.complete("What is 9*2?")
        client.client.close()
    finally:
        server.close()
    assert r.text.endswith("Answer: 18")
    assert r.output_tokens == 345 and r.input_tokens == 12
    req = server.requests[0]
    assert req["path"].startswith("/v1/messages")
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body  # not accepted by current Claude models
    assert FALLBACK_BETA in req["headers"].get("anthropic-beta", "")


def test_anthropic_without_fallbacks_uses_plain_endpoint():
    server = FakeServer(_claude_message())
    try:
        client = AnthropicClient(model="claude-haiku-5-5", client=_sdk_anthropic(server.url))
        assert client.use_fallbacks is False  # "auto" only enables it where supported
        client.complete("hi")
        client.client.close()
    finally:
        server.close()
    req = server.requests[0]
    assert "fallbacks" not in req["body"]
    assert "anthropic-beta" not in req["headers"]


def test_anthropic_refusal_gives_empty_text():
    server = FakeServer(_claude_message(text="", stop_reason="refusal"))
    try:
        client = AnthropicClient(client=_sdk_anthropic(server.url))
        r = client.complete("hi")
        client.client.close()
    finally:
        server.close()
    assert r.text == "" and r.stop_reason == "refusal"


def test_openai_compatible_request_shape_and_parsing():
    server = FakeServer({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "llama3.2:3b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer: 3"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    })
    try:
        sdk = openai.OpenAI(api_key="x", base_url=server.url + "/v1", max_retries=0)
        client = OpenAICompatibleClient(model="llama3.2:3b", client=sdk,
                                        answer_temperature=0.0, sample_temperature=0.9)
        r1 = client.complete("q", purpose="answer")
        client.complete("q", purpose="sample")
        sdk.close()
    finally:
        server.close()
    assert r1.text == "Answer: 3" and r1.output_tokens == 7
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["body"]["temperature"] == 0.0
    assert server.requests[1]["body"]["temperature"] == 0.9


def test_make_client_factory(questions, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert make_client(LLMConfig(provider="simulated"), questions).provider == "simulated"
    assert make_client(LLMConfig(provider="anthropic", model="claude-opus-5-5")).provider == "anthropic"
    c = make_client(LLMConfig(provider="openai_compatible", model="m", base_url="http://localhost:1/v1"))
    assert c.provider == "openai_compatible"
    with pytest.raises(ValueError):
        make_client(LLMConfig(provider="nope"))
    with pytest.raises(ValueError):
        make_client(LLMConfig(provider="simulated"))  # needs questions
