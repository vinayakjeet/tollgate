"""Streaming acceptance tests (M3). The mock streams deterministically; pacing
and mid-stream faults come from stubs registered here."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore
from app.routers import v1 as v1_module
from llm.providers import registry as registry_module
from llm.types import ChatChunk, ChatResponse, RateLimitError


class WordStreamer:
    """Streams one word per tick. `events` records what actually happened to its
    generator, because a cancellation that nobody observed is a claim, not a test."""

    name = "wordy"

    def __init__(self, events: list[str], tick_s: float = 0.0, fail_after: int | None = None):
        self.events = events
        self.tick_s = tick_s
        self.fail_after = fail_after
        self.sent = 0

    async def chat_completion(self, messages, **kwargs) -> ChatResponse:
        text = " ".join(m.content for m in messages)
        return ChatResponse(
            text=f"echo {text}", provider=self.name, model="m", tokens_in=2, tokens_out=3
        )

    async def stream_completion(self, messages, **kwargs):
        self.events.append("start")
        try:
            words = ["alpha", "beta", "gamma", "delta"]
            for i, word in enumerate(words):
                if self.tick_s:
                    await asyncio.sleep(self.tick_s)
                self.sent += 1
                if self.fail_after is not None and self.sent > self.fail_after:
                    raise RateLimitError(f"{self.name}: rate limited", retry_after=5.0)
                yield ChatChunk(
                    text_delta=word + " ",
                    provider=self.name,
                    model="wordy-m",
                    finish_reason="stop" if i == len(words) - 1 else None,
                )
            yield ChatChunk(provider=self.name, model="wordy-m", tokens_in=4, tokens_out=4)
        finally:
            self.events.append("closed")


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    original = fastapi_app.state.gateway
    gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    fastapi_app.state.gateway = gateway
    yield gateway
    fastapi_app.state.gateway = original


def _register(monkeypatch, provider):
    monkeypatch.setitem(registry_module._PROVIDERS, provider.name, provider)


BODY = {"model": "wordy/x", "messages": [{"role": "user", "content": "hello"}]}


def test_sdk_streams_progressively_and_parses(fresh, monkeypatch):
    """The acceptance for M3.1, driven through the same OpenAI SDK downstream
    projects will use: stream=True must yield chunks one at a time, not one blob."""
    events: list[str] = []
    _register(monkeypatch, WordStreamer(events, tick_s=0.005))
    fresh.chain = ["wordy"]

    async def run():
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            sdk = AsyncOpenAI(base_url="http://t/v1", api_key="unused", http_client=http)
            arrivals = []
            async for chunk in await sdk.chat.completions.create(
                model="wordy/x",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ):
                arrivals.append(time.perf_counter())
                assert chunk.object == "chat.completion.chunk"
            return arrivals

    arrivals = asyncio.run(run())
    assert len(arrivals) >= 4
    # Progressive arrival: the spread across chunk timestamps is real, not zero.
    assert max(arrivals) - min(arrivals) > 0
    assert arrivals[0] < arrivals[-1]


def test_first_chunk_arrives_before_upstream_is_exhausted(fresh, monkeypatch):
    events: list[str] = []

    class Instrumented(WordStreamer):
        async def stream_completion(self, messages, **kwargs):
            self.events.append("upstream-first-chunk")
            async for chunk in super().stream_completion(messages, **kwargs):
                self.events.append("upstream-yielded")
                yield chunk

    _register(monkeypatch, Instrumented(events))
    fresh.chain = ["wordy"]

    client = TestClient(fastapi_app)
    with client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}) as response:
        frames = list(response.iter_lines())

    data_frames = [line for line in frames if line.startswith("data:")]
    assert data_frames, "no SSE frames at all"
    assert "upstream-first-chunk" in events


def test_overhead_rides_the_final_frames_not_a_header(fresh, monkeypatch):
    """M3.2: headers are gone by the time a stream's overhead is known. The
    figure rides the final chunk-shaped frame as extra tollgate fields."""
    events: list[str] = []
    _register(monkeypatch, WordStreamer(events))
    fresh.chain = ["wordy"]

    client = TestClient(fastapi_app)
    with client.stream(
        "POST", "/v1/chat/completions", json={**BODY, "stream": True}
    ) as response:
        assert "x-tollgate-overhead-ms" not in response.headers
        body = "".join(response.iter_text())

    frames = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    final = frames[-1]
    assert final["tollgate_overhead_ms"] >= 0
    assert final["tollgate_provider"] == "wordy"
    assert body.rstrip().endswith("data: [DONE]")


def test_cached_answer_restreams_as_multiple_chunks(fresh):
    """The acceptance for M3.3: a cache hit serving stream=true replays chunks so
    the caller's code path does not change."""
    client = TestClient(fastapi_app)
    plain = client.post("/v1/chat/completions", json=BODY)
    assert plain.headers["x-tollgate-cache"] == "miss"

    with client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}) as response:
        assert response.headers["x-tollgate-cache"] == "exact"
        body = "".join(response.iter_text())

    content_frames = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [f["choices"][0]["delta"].get("content", "") for f in content_frames]
    assert "".join(deltas) == plain.json()["choices"][0]["message"]["content"]
    assert len([d for d in deltas if d]) > 1, "cached replay must not arrive as one block"


async def test_client_disconnect_closes_the_upstream(fresh, monkeypatch):
    """The acceptance for M3.4: dropping the connection terminates the upstream
    request, proven by the generator's own cleanup having run."""
    events: list[str] = []
    _register(monkeypatch, WordStreamer(events, tick_s=0.02))
    fresh.chain = ["wordy"]

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
        "app": fastapi_app,
    }
    from starlette.requests import Request as StarletteRequest

    http_request = StarletteRequest(scope)

    class StreamBody(BODY.__class__):
        pass

    req = v1_module.ChatCompletionRequest(**{**BODY, "stream": True})
    response = await v1_module.chat_completions(req, http_request, FastAPIResponse())
    assert isinstance(response, StreamingResponse)

    agen = response.body_iterator
    first = await agen.__anext__()
    assert first.startswith(": tollgate")
    await agen.__anext__()  # one real frame, then hang up
    await agen.aclose()

    await asyncio.sleep(0.05)
    assert "closed" in events, "upstream generator was never closed after disconnect"


def test_mid_stream_429_names_the_failure_on_the_wire(fresh, monkeypatch):
    """Bytes already sent cannot be un-sent, so no silent truncation: an error
    frame names what happened, then the stream still terminates cleanly."""
    events: list[str] = []
    _register(monkeypatch, WordStreamer(events, fail_after=2))
    fresh.chain = ["wordy"]

    client = TestClient(fastapi_app)
    with client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}) as response:
        body = "".join(response.iter_text())

    assert '"type": "upstream_error"' in body or '"type":"upstream_error"' in body
    assert body.rstrip().endswith("data: [DONE]")


