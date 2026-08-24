from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.main import app as fastapi_app
from llm.types import ChatResponse, ProviderClientError, ProviderError

BODY = {"model": "mock/demo", "messages": [{"role": "user", "content": "hello"}]}


def _gateway(client: TestClient):
    """The gateway assembled in create_app. Dispatch is patched through it rather
    than through a module singleton, so two tests can carry two different clients."""
    return client.app.state.gateway


def test_returns_an_openai_shaped_completion(client):
    resp = client.post("/v1/chat/completions", json=BODY)
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "hello" in body["choices"][0]["message"]["content"]


def test_carries_the_three_tollgate_headers(client):
    resp = client.post("/v1/chat/completions", json=BODY)
    assert resp.headers["x-tollgate-provider"] == "mock"
    assert resp.headers["x-tollgate-cache"] == "miss"
    assert float(resp.headers["x-tollgate-overhead-ms"]) >= 0


def test_overhead_excludes_the_upstream_call(client, monkeypatch):
    """The header measures Tollgate, so a slow provider must not inflate it."""

    async def _slow(*args: object, **kwargs: object) -> ChatResponse:
        import asyncio

        await asyncio.sleep(0.25)
        return ChatResponse(text="late", provider="mock", model="demo")

    monkeypatch.setattr(_gateway(client).client, "complete", _slow)
    resp = client.post("/v1/chat/completions", json=BODY)

    assert resp.status_code == 200
    assert float(resp.headers["x-tollgate-overhead-ms"]) < 250


def test_usage_is_null_rather_than_zero_when_tokens_are_unknown(client, monkeypatch):
    async def _no_usage(*args: object, **kwargs: object) -> ChatResponse:
        return ChatResponse(text="hi", provider="mock", model="demo")

    monkeypatch.setattr(_gateway(client).client, "complete", _no_usage)
    resp = client.post("/v1/chat/completions", json=BODY)

    assert resp.json()["usage"] is None


def test_bare_model_name_falls_back_to_the_configured_provider(client):
    resp = client.post(
        "/v1/chat/completions",
        json={**BODY, "model": "gpt-4o"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-tollgate-provider"] == "mock"


def test_stream_flag_returns_an_sse_response(client):
    resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in resp.text


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"model": "mock/demo"},
        {"model": "mock/demo", "messages": []},
        {"model": "mock/demo", "messages": [{"role": "user"}]},
    ],
    ids=["no-model", "no-messages", "empty-messages", "message-without-content"],
)
def test_malformed_bodies_are_rejected(client, payload):
    assert client.post("/v1/chat/completions", json=payload).status_code == 422


def test_unknown_provider_prefix_is_treated_as_a_model_name(client):
    """`meta-llama/llama-3.1` is a model, not a typo, so it routes to the default."""
    resp = client.post(
        "/v1/chat/completions",
        json={**BODY, "model": "meta-llama/llama-3.1-8b-instruct"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-tollgate-provider"] == "mock"


def test_provider_error_maps_to_502(client, monkeypatch):
    async def _boom(*args: object, **kwargs: object):
        raise ProviderError("upstream is down")

    monkeypatch.setattr(_gateway(client).client, "complete", _boom)
    assert client.post("/v1/chat/completions", json=BODY).status_code == 502


def test_client_error_maps_to_400(client, monkeypatch):
    async def _bad(*args: object, **kwargs: object):
        raise ProviderClientError("unknown model upstream")

    monkeypatch.setattr(_gateway(client).client, "complete", _bad)
    assert client.post("/v1/chat/completions", json=BODY).status_code == 400


async def test_the_real_openai_client_parses_the_response():
    """The acceptance criterion for M0.3.

    Hand-checking JSON shape proves nothing about whether a caller's SDK accepts it.
    This drives the actual `openai` package against the ASGI app, so a missing field
    or a wrong type fails here rather than in a downstream project.
    """
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tollgate") as http_client:
        sdk = AsyncOpenAI(
            base_url="http://tollgate/v1",
            api_key="not-used-by-the-mock-provider",
            http_client=http_client,
        )
        completion = await sdk.chat.completions.create(
            model="mock/demo",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert completion.choices[0].message.content
    assert completion.object == "chat.completion"
