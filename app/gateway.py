"""The one object that holds the gateway's moving parts together.

Assembled once in `create_app` and reachable as `app.state.gateway`, so tests can
build a Gateway out of fakes and point the app at it instead of patching module
singletons. The singletons were the chassis inheritance: convenient until the
first test needed two different clients in one process, which is also the shape
of every fallback-chain scenario.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import structlog

from app.budget import BudgetTracker, build_counter_store
from app.caches import InMemoryVectorIndex, LocalKVBackend, ResponseCache, build_kv_backend
from app.config import Settings
from app.embeddings import HashingStubEmbedder, LocalEmbedder
from app.metering import MeteringStore, NullMeteringStore, ResilientMeteringStore
from app.routing import Selector
from llm import ChatClient

logger = structlog.get_logger(__name__)


def _disabled_cache() -> ResponseCache:
    """A cache whose semantic layer is off and whose store is process-local.
    Used when a caller (usually a test) builds a Gateway without one."""
    return ResponseCache(kv=LocalKVBackend(), salt="unset-gateway-cache")


def default_chain(settings: Settings) -> list[str]:
    """Who answers when the caller does not name a provider.

    Local development defaults to the mock alone so nothing requires an API key.
    A configured deployment walks the free tiers in price order, cheapest first,
    and never includes the mock: a gateway that can silently answer from a fake
    provider is not a gateway, it is a demo of one.
    """
    if settings.tollgate_chain:
        chain = [p.strip() for p in settings.tollgate_chain.split(",") if p.strip()]
    elif settings.llm_provider == "mock":
        return ["mock"]
    else:
        chain = ["groq", "cerebras", "gemini", "openrouter", "sarvam"]
    return [p for p in chain if p != "mock"]


@dataclass
class Gateway:
    client: ChatClient
    tracker: BudgetTracker
    selector: Selector
    metering: MeteringStore
    chain: list[str]
    margin: float
    cache: ResponseCache = field(default_factory=_disabled_cache)

    @staticmethod
    def from_settings(
        settings: Settings,
        *,
        metering: MeteringStore | None = None,
        client: ChatClient | None = None,
        cache: ResponseCache | None = None,
    ) -> Gateway:
        store = build_counter_store(settings.redis_url)
        tracker = BudgetTracker(store)
        metering_store = metering or NullMeteringStore()
        # The null store needs no wrapper: there is nothing to fail.
        if not isinstance(metering_store, NullMeteringStore):
            metering_store = ResilientMeteringStore(metering_store)
        return Gateway(
            client=client or ChatClient(max_retry_attempts=settings.llm_max_retry_attempts),
            tracker=tracker,
            selector=Selector(tracker, margin=settings.skip_margin),
            metering=metering_store,
            chain=default_chain(settings),
            margin=settings.skip_margin,
            cache=cache or build_cache(settings),
        )


def build_cache(settings: Settings) -> ResponseCache:
    salt = settings.cache_salt
    if not salt:
        salt = secrets.token_hex(16)
        logger.warning(
            "cache.salt_generated",
            detail="set CACHE_SALT to keep cache entries across restarts",
        )

    embedder = index = None
    if settings.embedding_backend == "local":
        embedder = LocalEmbedder(settings.embedding_model_dir)
        index = InMemoryVectorIndex()
    elif settings.embedding_backend == "stub":
        # Tests and keyless demos only. The stub matches on shared trigrams, not
        # meaning; serving production traffic through it would be a quiet way to
        # serve wrong answers while calling it a semantic cache.
        logger.warning("cache.stub_embedder_active")
        embedder = HashingStubEmbedder()
        index = InMemoryVectorIndex()

    return ResponseCache(
        kv=build_kv_backend(settings.redis_url),
        salt=salt,
        embedder=embedder,
        index=index,
        threshold=settings.semantic_threshold,
        ttl_s=settings.cache_ttl_s,
    )
