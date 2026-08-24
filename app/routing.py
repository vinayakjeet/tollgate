"""Provider selection: an ordered fallback chain that routes around exhaustion.

The chain is walked in order and a provider is skipped when its own estimate says
it is within a configured margin of exhaustion. The estimate is allowed to be
wrong in both directions, which is why the inherited 429 throttle stays in place
as the backstop rather than being replaced: a provider that trips mid-flight is
recorded and the walk continues from where it stopped. SPEC.md carries the full
argument for keeping both layers.

`Retry-After` on a total exhaustion is computed from the earliest window reset
across every provider considered, never quoted as a constant. When no known limit
is binding anywhere there is nothing to compute from, so the shortest window on
the table (one minute) is the floor; that case is a configuration smell rather
than a runtime state and the log says so.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import structlog

from app.budget import MINUTE_SECONDS, BudgetTracker, ProviderEstimate

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Skip:
    provider: str
    reason: str


@dataclass(frozen=True)
class Selection:
    chosen: str | None
    skipped: list[Skip]
    retry_after: int


class Selector:
    def __init__(self, tracker: BudgetTracker, margin: float = 0.1) -> None:
        self._tracker = tracker
        self._margin = margin

    @staticmethod
    def plan(
        requested: str | None,
        default_chain: list[str],
        allow_local: bool,
        local_provider: str = "ollama",
    ) -> list[str]:
        """The order to try.

        An explicit `provider/model` prefix pins its provider to the head of the
        chain but does not make the request exclusive to it. A caller who asked
        for groq by name still prefers being served via cerebras over seeing a
        429, and `x-tollgate-provider` tells them what happened. Ollama joins the
        tail only when the request opted in, because silently answering with a
        local model changes what sits behind the response.
        """
        if requested and requested not in default_chain:
            return [requested]
        if requested:
            chain = [requested]
            chain += [p for p in default_chain if p != requested]
        else:
            chain = list(default_chain)
        if allow_local and local_provider not in chain:
            chain.append(local_provider)
        return chain

    async def choose(self, chain: list[str]) -> Selection:
        skipped: list[Skip] = []
        resets: list[float] = []

        for provider in chain:
            estimate: ProviderEstimate = await self._tracker.estimate(provider)
            exhausted = _binding_limit(estimate, self._margin)
            if exhausted is None:
                if estimate.degraded:
                    logger.warning(
                        "budget.estimating_from_degraded_counters", provider=provider
                    )
                return Selection(
                    chosen=provider,
                    skipped=skipped,
                    retry_after=_earliest_wait(resets),
                )
            skipped.append(Skip(provider=provider, reason=f"{exhausted}_exhausted"))
            reset = estimate.binding_reset
            if reset is not None:
                resets.append(reset)

        if resets:
            retry_after = _earliest_wait(resets)
        else:
            logger.error("budget.no_known_limits_anywhere", chain=list(chain))
            retry_after = MINUTE_SECONDS
        return Selection(chosen=None, skipped=skipped, retry_after=retry_after)

    async def earliest_retry_after(self, chain: list[str]) -> int:
        """Retry-After after mid-flight failures: the earliest reset among the
        providers that were actually tried and refused. Recomputed from the
        counters rather than carried over from selection, because the trips just
        recorded may be the first evidence of where the windows really were."""
        resets: list[float] = []
        for provider in chain:
            estimate = await self._tracker.estimate(provider)
            reset = estimate.binding_reset
            if reset is not None:
                resets.append(reset)
        if resets:
            return _earliest_wait(resets)
        logger.error("budget.no_known_limits_anywhere", chain=list(chain))
        return MINUTE_SECONDS


def _earliest_wait(resets: list[float]) -> int:
    """Seconds until the earliest reset, rounded up, never below one."""
    if not resets:
        return 0
    return max(math.ceil(max(min(resets) - time.time(), 0.0)), 1)


def _binding_limit(estimate: ProviderEstimate, margin: float) -> str | None:
    """Which limit puts this provider inside the skip margin, if any.

    The margin exists because an estimate computed at check time is stale by the
    time dispatch finishes: near a limit, one in-flight batch can tip the real
    provider into refusing. Skipping early trades a little usable quota for fewer
    429 round trips, and M1.5 measures whether that trade was worth it.
    """
    for name, state in (
        ("rpm", estimate.requests_per_minute),
        ("tpm", estimate.tokens_per_minute),
        ("rpd", estimate.requests_per_day),
    ):
        if state.limit is None or state.limit == 0:
            continue
        if state.used >= state.limit * (1 - margin):
            return name
    return None
