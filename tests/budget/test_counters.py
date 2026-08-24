from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from app.budget import LocalCounterStore, RedisCounterStore


class _ExplodingStore:
    """Stands in for a Redis that cannot be reached, which is how redis-py fails:
    an OSError on the first command, not at construction."""

    def __init__(self) -> None:
        self.calls = 0

    async def incr(self, key: str, amount: int, ttl_s: int) -> int:
        self.calls += 1
        raise OSError("connection refused")

    async def get(self, key: str) -> int:
        self.calls += 1
        raise OSError("connection refused")


async def test_local_counters_increment_and_expire():
    store = LocalCounterStore()
    assert await store.incr("k", 3, ttl_s=60) == 3
    assert await store.incr("k", 2, ttl_s=60) == 5
    assert await store.get("k") == 5


async def test_local_counters_disappear_after_the_ttl():
    store = LocalCounterStore()
    await store.incr("k", 1, ttl_s=60)
    # Reach into the expiry directly rather than sleeping a minute.
    count, _ = store._counts["k"]
    store._counts["k"] = (count, 0.0)
    assert await store.get("k") == 0


async def test_local_counters_survive_concurrent_increments():
    """Two coroutines incrementing the same key must both land. A lost increment
    here is under-counted quota, and under-counting is only acceptable when it is
    predictable, not when it depends on scheduling luck."""
    store = LocalCounterStore()
    results = await asyncio.gather(*[store.incr("k", 1, ttl_s=60) for _ in range(50)])
    assert sorted(results)[-1] == 50
    assert await store.get("k") == 50


def test_redis_store_builds_a_pipeline_of_incr_then_expire():
    """The TTL arming is NX, so a later request never extends a window the
    provider already reset. This checks the command shape against a fake pipeline,
    because that shape is the whole of the correctness argument."""
    recorded: list[tuple[str, tuple, dict]] = []


    class _FakePipeline:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def incrby(self, key, amount):
            recorded.append(("incrby", (key, amount), {}))
            return self

        def expire(self, key, ttl, nx=False):
            recorded.append(("expire", (key, ttl), {"nx": nx}))
            return self

        async def execute(self):
            return [7]

    class _FakeRedis:
        def pipeline(self, transaction=True):
            return _FakePipeline()

    store = RedisCounterStore.__new__(RedisCounterStore)
    store._redis = _FakeRedis()
    assert asyncio.run(store.incr("k", 1, ttl_s=65)) == 7
    assert [name for name, _, _ in recorded] == ["incrby", "expire"]
    assert recorded[1][2]["nx"] is True


def test_redis_store_rejects_an_empty_url_by_not_existing():
    with pytest.raises(ValueError):
        RedisCounterStore("")


async def test_tracker_degrades_to_local_and_announces_it(monkeypatch):
    from app.budget import BudgetTracker

    exploding = _ExplodingStore()
    tracker = BudgetTracker(exploding)

    monkeypatch.setattr(
        "llm.providers.registry.quota_limits", lambda name: (10, None, None)
    )

    with capture_logs() as logs:
        estimate = await tracker.estimate("groq")

    assert exploding.calls >= 1
    assert tracker.degraded is True
    assert estimate.degraded is True
    assert any(entry.get("event") == "budget.redis_unreachable" for entry in logs)
    # The read retried locally after the swap, so the request still got an answer.
    assert estimate.requests_per_minute.limit == 10
    assert estimate.requests_per_minute.used == 0


async def test_recording_after_degradation_lands_in_the_local_store(monkeypatch):
    from app.budget import BudgetTracker

    tracker = BudgetTracker(_ExplodingStore())
    monkeypatch.setattr(
        "llm.providers.registry.quota_limits", lambda name: (None, None, None)
    )
    await tracker.record("groq", tokens=42)

    estimate = await tracker.estimate("groq")
    assert estimate.requests_per_minute.used == 1
    assert estimate.tokens_per_minute.used == 42
