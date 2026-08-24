"""The one object that holds the gateway's moving parts together.

Assembled once in `create_app` and reachable as `app.state.gateway`, so tests can
build a Gateway out of fakes and point the app at it instead of patching module
singletons. The singletons were the chassis inheritance: convenient until the
first test needed two different clients in one process, which is also the shape
of every fallback-chain scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.budget import BudgetTracker, build_counter_store
from app.config import Settings
from app.metering import MeteringStore, NullMeteringStore
from app.routing import Selector
from llm import ChatClient


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

    @staticmethod
    def from_settings(
        settings: Settings,
        *,
        metering: MeteringStore | None = None,
        client: ChatClient | None = None,
    ) -> Gateway:
        store = build_counter_store(settings.redis_url)
        tracker = BudgetTracker(store)
        return Gateway(
            client=client or ChatClient(max_retry_attempts=settings.llm_max_retry_attempts),
            tracker=tracker,
            selector=Selector(tracker, margin=settings.skip_margin),
            metering=metering or NullMeteringStore(),
            chain=default_chain(settings),
            margin=settings.skip_margin,
        )
