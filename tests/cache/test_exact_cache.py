from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from app.caches import CacheEntry, LocalKVBackend, ResponseCache


def _entry(text: str = "cached answer") -> CacheEntry:
    return CacheEntry(
        text=text,
        provider="mock",
        model="mock-echo",
        tokens_in=3,
        tokens_out=5,
        cost_usd=0.0,
    )


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


async def test_second_identical_lookup_is_an_exact_hit():
    cache = ResponseCache(kv=LocalKVBackend(), salt="s")
    miss = await cache.lookup(_fields())
    assert miss.outcome == "miss"

    await cache.store(miss, _entry())
    hit = await cache.lookup(_fields())
    assert hit.outcome == "exact"
    assert hit.entry is not None and hit.entry.text == "cached answer"


async def test_a_corrupt_entry_costs_a_miss_not_a_crash():
    """A half-written payload (killed process mid-set) must not take the request
    down: the cache's worst legitimate outcome is 'no money saved'."""
    kv = LocalKVBackend()
    await kv.set(await _key(kv), b"{not json", ttl_s=60)
    cache = ResponseCache(kv=kv, salt="s")
    lookup = await cache.lookup(_fields())
    assert lookup.outcome == "miss"


async def _key(kv) -> str:
    return ResponseCache(kv=kv, salt="s").key_for(_fields())


class _ExplodingKV(LocalKVBackend):
    async def get(self, key: str) -> bytes | None:
        raise OSError("connection refused")

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        raise OSError("connection refused")


async def test_redis_down_degrades_to_miss_and_announces_it():
    """The acceptance for degradation visibility on the cache path: requests keep
    working, every probe misses, and the log says why."""
    cache = ResponseCache(kv=_ExplodingKV(), salt="s")

    with capture_logs() as logs:
        lookup = await cache.lookup(_fields())

    assert lookup.outcome == "miss"
    assert cache.degraded is True
    assert any(entry.get("event") == "cache.redis_unreachable" for entry in logs)


async def test_store_failure_does_not_raise():
    """The response has already been served by the time store() runs; failing it
    would be an error with no one left to receive it."""
    cache = ResponseCache(kv=_ExplodingKV(), salt="s")
    miss = await cache.lookup(_fields())

    with capture_logs() as logs:
        await cache.store(miss, _entry())

    assert any(entry.get("event") == "cache.store_failed" for entry in logs)


def test_entry_json_round_trip_preserves_null_token_counts():
    raw = _entry().to_json()
    loaded = CacheEntry.from_json(raw)
    assert loaded.tokens_in == 3


@pytest.mark.parametrize("outcome", ["miss", "exact"])
async def test_probe_records_its_outcome_on_the_span(outcome, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app import spans

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(spans.spanlight, "get_tracer", lambda: provider.get_tracer("t"))

    cache = ResponseCache(kv=LocalKVBackend(), salt="s")
    if outcome == "exact":
        first = await cache.lookup(_fields())
        await cache.store(first, _entry())
    await cache.lookup(_fields())

    probes = [s for s in memory.get_finished_spans() if s.name == spans.CACHE_PROBE]
    assert probes[-1].attributes["tollgate.cache.outcome"] == outcome
