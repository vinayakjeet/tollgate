"""Fixed-window budget accounting, per provider, with a process-local fallback.

The windows are fixed, not rolling, for the reason recorded in SPEC.md: a rolling
counter that disagrees with the provider's own accounting is harder to debug than a
fixed one that under-counts predictably, and the only arbiter of a disagreement is
the provider's response headers. Predictably wrong beats subtly wrong when the
number decides where traffic goes.

Counters live in Redis when REDIS_URL is set and degrade to an in-process dict when
Redis cannot be reached, because a gateway that dies when its cache dies is worse
than one that under-counts. The degradation is never silent: it emits one log line
and sets `degraded` on every estimate it produces afterwards, which is what the
`tollgate.budget.degraded` span attribute reads.

A `None` limit means "unknown", never zero. No provider here publishes all three
limits, so treating unknown as exhausted would idle healthy providers forever.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

# Quiet the redis client's own connection-error spam; one line from us is enough.
logging.getLogger("redis").setLevel(logging.CRITICAL)

MINUTE_SECONDS = 60
DAY_SECONDS = 86_400


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def reset_in(self) -> float:
        return max(self.end - time.time(), 0.0)


def minute_window(now: float | None = None) -> Window:
    """The UTC minute containing `now`, aligned to the epoch."""
    ts = time.time() if now is None else now
    start = int(ts) // MINUTE_SECONDS * MINUTE_SECONDS
    return Window(start=start, end=start + MINUTE_SECONDS)


def day_window(now: float | None = None) -> Window:
    """The UTC day containing `now`. Providers reset on their own clocks; UTC is
    the honest approximation and is what /budget reports beside every figure."""
    ts = time.time() if now is None else now
    start = int(ts) // DAY_SECONDS * DAY_SECONDS
    return Window(start=start, end=start + DAY_SECONDS)


class CounterStore(Protocol):
    """The one thing a counter backend has to do: add atomically and read."""

    async def incr(self, key: str, amount: int, ttl_s: int) -> int: ...

    async def get(self, key: str) -> int: ...


class LocalCounterStore:
    """In-process counters, correct for exactly one replica.

    Loses counts on restart, which is the honest failure mode: after a restart the
    gateway knows less than it did, not more. TTLs are monotonic-clock based so a
    system clock jump cannot strand a counter at its limit forever.
    """

    def __init__(self) -> None:
        self._counts: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def incr(self, key: str, amount: int, ttl_s: int) -> int:
        async with self._lock:
            count, expires_at = self._counts.get(key, (0, 0.0))
            now = time.monotonic()
            if expires_at <= now:
                count, expires_at = 0, now + ttl_s
            count += amount
            self._counts[key] = (count, expires_at)
            return count

    async def get(self, key: str) -> int:
        async with self._lock:
            count, expires_at = self._counts.get(key, (0, 0.0))
            if expires_at <= time.monotonic():
                return 0
            return count


class RedisCounterStore:
    """Counters in Redis, shared across replicas and restarts.

    INCRBY then EXPIRE NX keeps a key alive for slightly longer than its window;
    a second request re-arming the TTL would extend a window the provider had
    already reset, which is the one direction a fixed window must never drift.
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(url, decode_responses=True)

    async def incr(self, key: str, amount: int, ttl_s: int) -> int:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(key, amount)
            pipe.expire(key, ttl_s, nx=True)
            results = await pipe.execute()
        return int(results[0])

    async def get(self, key: str) -> int:
        value = await self._redis.get(key)
        return int(value) if value is not None else 0


def build_counter_store(redis_url: str | None) -> CounterStore:
    """Redis when configured, local otherwise. A bad URL fails here rather than on
    the first request, and the caller decides whether that is fatal."""
    if not redis_url:
        return LocalCounterStore()
    return RedisCounterStore(redis_url)


@dataclass(frozen=True)
class LimitState:
    """One limit's position in its window. `remaining` is None when the limit
    itself is unknown, which is a different answer from zero and stays different
    all the way to the /budget response body."""

    kind: str
    limit: int | None
    used: int
    window: Window

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(self.limit - self.used, 0)


@dataclass(frozen=True)
class ProviderEstimate:
    provider: str
    requests_per_minute: LimitState
    tokens_per_minute: LimitState
    requests_per_day: LimitState
    degraded: bool

    @property
    def binding_reset(self) -> float | None:
        """When the nearest known window that could still block this provider
        resets. Unknown limits contribute nothing: there is no basis for a wait."""
        resets = [
            state.window.end
            for state in (
                self.requests_per_minute,
                self.tokens_per_minute,
                self.requests_per_day,
            )
            if state.limit is not None and state.remaining == 0
        ]
        return min(resets) if resets else None


class BudgetTracker:
    """Reads and writes the per-provider counters behind one estimate."""

    def __init__(self, store: CounterStore) -> None:
        self._store = store
        self.degraded = False

    def _switched_to_local(self, exc: Exception) -> LocalCounterStore:
        if isinstance(self._store, LocalCounterStore):
            raise exc
        logger.error("budget.redis_unreachable", error=str(exc))
        self.degraded = True
        self._store = LocalCounterStore()
        return self._store

    @staticmethod
    def _keys(provider: str) -> dict[str, str]:
        minute = minute_window()
        day = day_window()
        return {
            "rpm": f"tollgate:budget:{provider}:rpm:{minute.start}",
            "tpm": f"tollgate:budget:{provider}:tpm:{minute.start}",
            "rpd": f"tollgate:budget:{provider}:rpd:{day.start}",
        }

    @staticmethod
    def _limits(provider: str) -> tuple[int | None, int | None, int | None]:
        from llm.providers.registry import quota_limits

        rpm, tpm, rpd = quota_limits(provider)
        return rpm, tpm, rpd

    async def _state(
        self, provider: str, kind: str, limit: int | None, window: Window
    ) -> LimitState:
        try:
            used = await self._store.get(self._keys(provider)[kind])
        except Exception as exc:
            store = self._switched_to_local(exc)
            used = await store.get(self._keys(provider)[kind])
        return LimitState(kind=kind, limit=limit, used=used, window=window)

    async def estimate(self, provider: str) -> ProviderEstimate:
        rpm_limit, tpm_limit, rpd_limit = self._limits(provider)
        minute = minute_window()
        day = day_window()
        return ProviderEstimate(
            provider=provider,
            requests_per_minute=await self._state(provider, "rpm", rpm_limit, minute),
            tokens_per_minute=await self._state(provider, "tpm", tpm_limit, minute),
            requests_per_day=await self._state(provider, "rpd", rpd_limit, day),
            degraded=self.degraded,
        )

    async def record(self, provider: str, tokens: int) -> None:
        """Add one request and its tokens to every known-shape counter.

        Tokens are recorded even when tpm_limit is None: the cost of keeping the
        number is one increment, and filling the schema field later starts from a
        real count instead of from nothing.
        """
        keys = self._keys(provider)
        try:
            await self._store.incr(keys["rpm"], 1, ttl_s=MINUTE_SECONDS + 5)
            await self._store.incr(keys["tpm"], tokens, ttl_s=MINUTE_SECONDS + 5)
            await self._store.incr(keys["rpd"], 1, ttl_s=DAY_SECONDS + 5)
        except Exception as exc:
            store = self._switched_to_local(exc)
            await store.incr(keys["rpm"], 1, ttl_s=MINUTE_SECONDS + 5)
            await store.incr(keys["tpm"], tokens, ttl_s=MINUTE_SECONDS + 5)
            await store.incr(keys["rpd"], 1, ttl_s=DAY_SECONDS + 5)
