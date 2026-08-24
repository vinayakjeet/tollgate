"""Fault injection per external dependency (M4.1).

One parameterised suite over the provider surface, plus the stores. The
assertion is always the same shape: the failure is absorbed, its degradation is
observable, and nothing reaches the caller as a 500."""

from __future__ import annotations

import httpx
import pytest
from structlog.testing import capture_logs

from llm.providers.base import OpenAICompatibleProvider
from llm.types import (
    ProviderError,
    RateLimitError,
)

KEY_ENV = "FAULTPROV_API_KEY"


def make_provider(handler) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleProvider(
        name="faultprov",
        base_url="http://fault.test/v1",
        api_key_env=KEY_ENV,
        default_model="m",
        # The base_url lives on this injected client: when a caller supplies one,
        # the provider uses it as-is rather than building its own.
        client=httpx.AsyncClient(base_url="http://fault.test/v1", transport=transport),
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "test-key")


async def _complete(provider):
    from llm.types import ChatMessage

    return await provider.chat_completion([ChatMessage(role="user", content="hi")])


async def test_provider_500_is_a_retryable_provider_error():
    provider = make_provider(lambda req: httpx.Response(500))
    with pytest.raises(ProviderError):
        await _complete(provider)


async def test_429_with_retry_after_header_is_parsed():
    provider = make_provider(
        lambda req: httpx.Response(429, headers={"Retry-After": "37"})
    )
    with pytest.raises(RateLimitError) as exc_info:
        await _complete(provider)
    assert exc_info.value.retry_after == 37.0


async def test_429_without_any_parseable_delay_still_trips():
    """The dangerous case: no header, no details, no message hint. The throttle
    falls back to a short cooldown rather than spinning hot against a wall."""
    provider = make_provider(lambda req: httpx.Response(429, text="slow down"))
    with pytest.raises(RateLimitError) as exc_info:
        await _complete(provider)
    assert exc_info.value.retry_after is None


async def test_network_timeout_is_retryable_not_fatal():
    def handler(req):
        raise httpx.ConnectTimeout("timed out")

    provider = make_provider(handler)
    with pytest.raises(ProviderError):
        await _complete(provider)


async def test_malformed_success_body_is_a_provider_fault():
    """A 200 whose body is not JSON used to explode in our own decoder and reach
    callers as a 500. It is the provider's fault and must look like one."""
    provider = make_provider(lambda req: httpx.Response(200, content=b"{not json at all"))
    with pytest.raises(ProviderError, match="malformed"):
        await _complete(provider)


class ExplodingIndex:
    async def query(self, vector):
        raise ConnectionError("pgvector is down")

    async def upsert(self, key, vector, payload):
        raise ConnectionError("pgvector is down")

    async def payload_for(self, key):
        raise ConnectionError("pgvector is down")


async def test_postgres_down_degrades_the_semantic_layer_to_misses():
    """Semantic lookup failing must cost hits, never requests: fail open to a
    miss, log it once, keep serving."""
    from app.caches import LocalKVBackend, ResponseCache

    cache = ResponseCache(
        kv=LocalKVBackend(),
        salt="s",
        embedder=_StubEmbedder(),
        index=ExplodingIndex(),
        threshold=0.5,
    )

    with capture_logs() as logs:
        lookup = await cache.lookup(_fields())

    assert lookup.outcome == "miss"
    assert cache.degraded is True
    assert any(e.get("event") == "cache.semantic_query_failed" for e in logs)
    # And a store attempt through the same broken index also degrades quietly.
    with capture_logs() as logs2:
        await cache.store(lookup, _entry())
    assert any(e.get("event") == "cache.semantic_store_failed" for e in logs2)


def _fields(**overrides) -> dict:
    base = {
        "model": "mock/demo",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
    }
    base.update(overrides)
    return base


def _entry(text: str = "answer"):
    from app.caches import CacheEntry

    return CacheEntry(
        text=text, provider="mock", model="m", tokens_in=1, tokens_out=2, cost_usd=None
    )


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 8


async def test_metering_write_failure_never_breaks_a_response(tmp_path, monkeypatch):
    """The metering row is written after success; raising there would convert
    bookkeeping into an outage for exactly the requests worth metering."""

    from app.metering import JsonlMeteringStore, MeterRow, ResilientMeteringStore, now_row

    class BrokenStore(JsonlMeteringStore):
        async def append(self, row: MeterRow) -> None:
            raise OSError("disk full")

    store = ResilientMeteringStore(BrokenStore(tmp_path / "m.jsonl"))
    with capture_logs() as logs:
        await store.append(now_row("request", "mock", status=200))

    assert any(e.get("event") == "metering.append_failed" for e in logs)
