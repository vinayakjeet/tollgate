"""The two-layer cache: exact first, semantic second, probe cost published.

Ordering is the architecture in SPEC: L1 is a hash lookup and answers in
microseconds; L2 pays for an embedding plus a nearest-neighbour scan and only
runs on an L1 miss. A threshold of None disables L2 entirely, which is the
honest default until M6 measures what a threshold costs: GPTCache ships 0.75,
practitioners say 0.92 to 0.97, and nobody in that gap has published what it
does to wrong answers.

Both layers degrade to a miss rather than an error. A cache that turns a Redis
outage into user-facing failures has the polarity backwards: the worst thing a
cache can do to a request is fail to save it money.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from app import spans
from app.cache_key import canonical_payload, exact_cache_key
from app.embeddings import EmbeddingBackend, cosine
from app.spans import stage_span

logger = structlog.get_logger(__name__)

DEFAULT_TTL_S = 3600


@dataclass(frozen=True)
class CacheEntry:
    text: str
    provider: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, separators=(",", ":")).encode()

    @staticmethod
    def from_json(raw: bytes) -> CacheEntry:
        return CacheEntry(**json.loads(raw))


@dataclass(frozen=True)
class Lookup:
    """The outcome of one probe. `similarity` rides along on a semantic hit so a
    caller can audit a suspicious answer (M2.6)."""

    outcome: str  # "miss" | "exact" | "semantic"
    entry: CacheEntry | None = None
    key: str | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class MissWithVector(Lookup):
    vector: list[float] | None = None


class KVBackend(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_s: int) -> None: ...


class LocalKVBackend(KVBackend):
    """Process-local stand-in for Redis. Same degradation philosophy as the
    budget counters: monotonic expiry, one replica, counts nothing shared."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[bytes, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            item = self._entries.get(key)
            if item is None or item[1] <= time.monotonic():
                return None
            return item[0]

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        async with self._lock:
            self._entries[key] = (value, time.monotonic() + ttl_s)


class RedisKVBackend(KVBackend):
    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(url, decode_responses=False)

    async def get(self, key: str) -> bytes | None:
        return await self._redis.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._redis.set(key, value, ex=ttl_s)


def build_kv_backend(redis_url: str | None) -> KVBackend:
    if not redis_url:
        return LocalKVBackend()
    try:
        return RedisKVBackend(redis_url)
    except Exception as exc:
        logger.error("cache.redis_unusable", error=str(exc))
        return LocalKVBackend()


class VectorIndex(Protocol):
    """The smallest surface a nearest-neighbour cache needs."""

    async def query(self, vector: list[float]) -> tuple[str | None, float]: ...

    async def upsert(self, key: str, vector: list[float], payload: bytes) -> None: ...

    async def payload_for(self, key: str) -> bytes | None: ...


class InMemoryVectorIndex(VectorIndex):
    """Cosine scan over everything stored. O(n) per query is fine at study scale
    (thousands), and honest: pgvector's index structures buy speed, not recall,
    so behaviour measured here transfers except for latency."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], bytes]] = {}
        self._lock = asyncio.Lock()

    async def query(self, vector: list[float]) -> tuple[str | None, float]:
        best_key, best_score = None, -1.0
        async with self._lock:
            for key, (candidate, _) in self._vectors.items():
                score = cosine(vector, candidate)
                if score > best_score:
                    best_key, best_score = key, score
        return best_key, max(best_score, 0.0)

    async def upsert(self, key: str, vector: list[float], payload: bytes) -> None:
        async with self._lock:
            self._vectors[key] = (vector, payload)

    async def payload_for(self, key: str) -> bytes | None:
        async with self._lock:
            item = self._vectors.get(key)
            return item[1] if item else None


class PgVectorIndex(VectorIndex):
    """pgvector-backed index for the Neon deployment.

    Live-verified when DATABASE_URL exists; every failure degrades to "no match"
    by returning None from query, which callers treat as a miss.
    """

    def __init__(self, dsn: str, dims: int) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "the pgvector index needs the 'pg' dependency group: uv sync --group pg"
            ) from exc
        self._dims = dims
        self._psycopg = psycopg
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("create extension if not exists vector")
            cur.execute(
                f"""
                create table if not exists semantic_cache (
                    key text primary key,
                    embedding vector({dims}),
                    payload bytea
                )
                """
            )

    async def query(self, vector: list[float]) -> tuple[str | None, float]:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    select key, 1 - (embedding <=> %s::vector) as similarity
                    from semantic_cache
                    order by embedding <=> %s::vector
                    limit 1
                    """,
                    ("[" + ",".join(map(str, vector)) + "]",
                     "[" + ",".join(map(str, vector)) + "]"),
                )
                row = cur.fetchone()
        except Exception as exc:
            logger.warning("cache.semantic_query_failed", error=str(exc))
            return None, 0.0
        if row is None:
            return None, 0.0
        return row[0], float(row[1])

    async def upsert(self, key: str, vector: list[float], payload: bytes) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into semantic_cache (key, embedding, payload)
                values (%s, %s::vector, %s)
                on conflict (key) do update
                set embedding = excluded.embedding, payload = excluded.payload
                """,
                (key, "[" + ",".join(map(str, vector)) + "]", self._psycopg.Binary(payload)),
            )

    async def payload_for(self, key: str) -> bytes | None:
        with self._conn.cursor() as cur:
            cur.execute("select payload from semantic_cache where key = %s", (key,))
            row = cur.fetchone()
        return bytes(row[0]) if row else None


class ResponseCache:
    """L1 and L2 behind one probe, opening the span subtree bench/stages.md hashes."""

    def __init__(
        self,
        kv: KVBackend,
        salt: str,
        *,
        embedder: EmbeddingBackend | None = None,
        index: VectorIndex | None = None,
        threshold: float | None = None,
        ttl_s: int = DEFAULT_TTL_S,
    ) -> None:
        self._kv = kv
        self._salt = salt
        self._embedder = embedder
        self._index = index
        self._threshold = threshold
        self._ttl_s = ttl_s
        self.degraded = False

    @property
    def semantic_active(self) -> bool:
        return (
            self._threshold is not None
            and self._embedder is not None
            and self._index is not None
        )

    def key_for(self, request_fields: dict) -> str:
        return exact_cache_key(self._salt, canonical_payload(request_fields))

    async def lookup(self, request_fields: dict) -> Lookup:
        with stage_span(spans.CACHE_PROBE) as probe_span:
            key = self.key_for(request_fields)
            text = embeddable_text(request_fields.get("messages") or [])
            outcome, entry, vector, similarity = await self._probe(key, text)
            probe_span.record(**{"tollgate.cache.outcome": outcome})
        if entry is not None:
            return Lookup(outcome=outcome, entry=entry, key=key, similarity=similarity)
        return MissWithVector(
            outcome=outcome, entry=None, key=key, vector=vector, similarity=similarity
        )

    async def _probe(
        self, key: str, text: str
    ) -> tuple[str, CacheEntry | None, list[float] | None, float | None]:
        with stage_span(spans.CACHE_EXACT) as exact_span:
            raw = await self._kv_get(key)
            exact_span.record(**{"tollgate.cache.hit": raw is not None})
            if raw is not None:
                try:
                    return "exact", CacheEntry.from_json(raw), None, None
                except (ValueError, TypeError):
                    logger.warning("cache.exact_corrupt", key=key[:16])
                    return "miss", None, None, None

        if not self.semantic_active:
            return "miss", None, None, None

        with stage_span(spans.CACHE_SEMANTIC) as semantic_span:
            # Embedding time sits inside this span deliberately: it is unavoidable
            # cost of a semantic lookup, and hiding it would flatter layer two.
            vector = self._embedder.embed(text)
            found_key, similarity = await self._semantic_query(vector)
            hit = found_key is not None and similarity >= (self._threshold or 0.0)
            semantic_span.record(
                **{
                    "tollgate.cache.hit": hit,
                    "tollgate.cache.similarity": round(similarity, 4),
                    "tollgate.cache.threshold": self._threshold,
                }
            )
            if hit and (payload := await self._payload(found_key)) is not None:
                try:
                    return "semantic", CacheEntry.from_json(payload), vector, similarity
                except (ValueError, TypeError):
                    logger.warning("cache.semantic_corrupt", key=(found_key or "")[:16])

        return "miss", None, vector, similarity

    async def store(self, miss: MissWithVector, entry: CacheEntry) -> None:
        assert miss.key is not None
        payload = entry.to_json()
        try:
            await self._kv.set(miss.key, payload, self._ttl_s)
        except Exception as exc:
            self._degrade("cache.store_failed", exc)
        if self.semantic_active and miss.vector is not None:
            try:
                await self._index.upsert(miss.key, miss.vector, payload)
            except Exception as exc:
                self._degrade("cache.semantic_store_failed", exc)

    def _degrade(self, event: str, exc: Exception) -> None:
        self.degraded = True
        logger.error(event, error=str(exc))

    async def _kv_get(self, key: str):
        try:
            return await self._kv.get(key)
        except Exception as exc:
            self._degrade("cache.redis_unreachable", exc)
            return None

    async def _semantic_query(self, vector):
        try:
            return await self._index.query(vector)
        except Exception as exc:
            self._degrade("cache.semantic_query_failed", exc)
            return None, 0.0

    async def _payload(self, key: str | None):
        try:
            return await self._index.payload_for(key)
        except Exception as exc:
            self._degrade("cache.semantic_payload_failed", exc)
            return None


def embeddable_text(messages: list[dict[str, Any]]) -> str:
    """What the semantic layer matches on: the conversation as one string.

    System prompt included, because "answer in JSON" versus prose changes whether
    an answer serves. Truncated hard; embeddings of ten-thousand-token prompts
    mostly encode their beginnings anyway.
    """
    joined = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)
    return joined[-4000:]
