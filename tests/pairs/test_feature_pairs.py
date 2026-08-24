"""The four feature pairs from M4.4.

Spanlight shipped a sampler whose crash no suite could see because sampling and
detectors had no test in common. The equivalent pairs here are exercised in one
file, together, so a fixture that makes one convenient cannot quietly hide the
other: streaming during a fallback, a cache hit during an exhaustion event, a
429 mid-stream, and cancellation during a semantic lookup."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import Response as FastAPIResponse
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore
from llm.providers import registry as registry_module
from llm.types import ChatChunk, ChatResponse, RateLimitError


class Streamer:
    name = "streamy"

    def __init__(self, fail_every: int | None = None):
        self.fail_every = fail_every
        self.calls = 0
        self.closed_early = False

    async def chat_completion(self, messages, **kwargs) -> ChatResponse:
        self.calls += 1
        last = messages[-1].content
        return ChatResponse(
            text=f"echo {last}", provider=self.name, model="m", tokens_in=2, tokens_out=3
        )

    async def stream_completion(self, messages, **kwargs):
        self.calls += 1
        try:
            words = ["the", "answer", "is", "forty", "two"]
            for i, word in enumerate(words):
                if self.fail_every is not None and i >= self.fail_every:
                    raise RateLimitError(f"{self.name} limited", retry_after=1.0)
                yield ChatChunk(
                    text_delta=word + " ",
                    provider=self.name,
                    model="m",
                    finish_reason="stop" if i == len(words) - 1 else None,
                )
            yield ChatChunk(provider=self.name, model="m")
        except GeneratorExit:
            self.closed_early = True
            raise


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    original = fastapi_app.state.gateway
    gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    fastapi_app.state.gateway = gateway
    yield gateway
    fastapi_app.state.gateway = original


def _rpm_key(provider: str) -> str:
    minute = int(time.time()) // 60 * 60
    return f"tollgate:budget:{provider}:rpm:{minute}"


async def test_pair_one_streaming_during_a_fallback(fresh, monkeypatch):
    """Head provider exhausted pre-flight; the stream still answers from the next
    link. Selection and streaming have to compose; each alone is easy."""
    blocked = Streamer()
    healthy = Streamer()
    healthy.name = "healthy"
    monkeypatch.setitem(registry_module._PROVIDERS, "streamy", blocked)
    monkeypatch.setitem(registry_module._PROVIDERS, "healthy", healthy)
    # Known limits are what make a skip possible at all; without them no
    # provider is ever "nearly gone" and the fallback never fires.
    monkeypatch.setattr(registry_module, "quota_limits", lambda name: (10, 1000, None))
    fresh.chain = ["streamy", "healthy"]
    await fresh.tracker._store.incr(_rpm_key("streamy"), 10, ttl_s=90)

    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "streamy/q",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.headers["x-tollgate-provider"] == "healthy"
        body = "".join(response.iter_text())

    assert blocked.calls == 0
    assert healthy.calls == 1
    assert body.rstrip().endswith("data: [DONE]")


async def test_pair_two_cache_hit_during_an_exhaustion_event(fresh, monkeypatch):
    """Every provider's counter forced over the line, yet an identical request
    already cached is served. The probe runs before the budget stage, and this
    proves that ordering carries weight."""
    provider = Streamer()
    monkeypatch.setitem(registry_module._PROVIDERS, "streamy", provider)
    fresh.chain = ["streamy"]

    client = TestClient(fastapi_app)
    body = {"model": "streamy/q", "messages": [{"role": "user", "content": "hi"}]}
    warm = client.post("/v1/chat/completions", json=body)
    assert warm.headers["x-tollgate-cache"] == "miss"

    await fresh.tracker._store.incr(_rpm_key("streamy"), 10, ttl_s=90)

    cached = client.post("/v1/chat/completions", json=body)
    assert cached.status_code == 200, "exhaustion must not evict the cache path"
    assert cached.headers["x-tollgate-cache"] == "exact"
    assert provider.calls == 1


def test_pair_three_429_mid_stream(fresh, monkeypatch):
    """A trip after chunks were forwarded: no retry can unsend bytes, so the
    caller gets a named error frame and a clean end instead of a spliced answer."""
    flaky = Streamer(fail_every=2)
    monkeypatch.setitem(registry_module._PROVIDERS, "streamy", flaky)
    fresh.chain = ["streamy"]

    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "streamy/q",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert '"upstream_error"' in body
    assert body.rstrip().endswith("data: [DONE]")
    assert flaky.calls == 1, "a mid-stream failure must not be retried"


class HangingIndex:
    """Stands in for pgvector mid-query when the client vanishes."""

    def __init__(self) -> None:
        self.cancelled = False

    async def query(self, vector):
        try:
            await asyncio.sleep(30)
            return None, 0.0
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def upsert(self, key, vector, payload):
        return None

    async def payload_for(self, key):
        return None


async def test_pair_four_cancellation_during_a_semantic_lookup(fresh):
    """The client vanishes while the semantic layer is still querying. The
    lookup must observe the cancellation rather than keep running orphaned.

    Note the honest shape of this test: embedding is synchronous here, so the
    awaitable seam where cancellation can actually land is the vector query. A
    disconnect during `embed` would be delivered at the next await instead;
    either way nothing outlives the request."""
    from app.caches import LocalKVBackend, ResponseCache

    index = HangingIndex()
    from app.embeddings import HashingStubEmbedder

    fresh.cache = ResponseCache(
        kv=LocalKVBackend(),
        salt="pair4",
        embedder=HashingStubEmbedder(dims=8),
        index=index,
        threshold=0.9,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
        "app": fastapi_app,
    }
    from app.routers.v1 import ChatCompletionRequest, chat_completions

    req = ChatCompletionRequest(model="mock/demo", messages=[{"role": "user", "content": "hi"}])
    task = asyncio.ensure_future(
        chat_completions(req, StarletteRequest(scope), FastAPIResponse())
    )
    await asyncio.sleep(0.05)  # enter the hanging query
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert index.cancelled, "the semantic lookup kept running after the client left"
