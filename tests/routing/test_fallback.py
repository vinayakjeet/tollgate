from __future__ import annotations

import time

from app.budget import LimitState, ProviderEstimate, Window
from app.routing import Selector, _binding_limit

_NOW = time.time()


def _window(end_offset_s: float = 60.0) -> Window:
    return Window(start=int(_NOW), end=int(_NOW + end_offset_s))


def _estimate(
    provider: str,
    rpm: tuple[int, int] | None,
    tpm: tuple[int, int] | None = None,
    rpd: tuple[int, int] | None = None,
) -> ProviderEstimate:
    """(limit, used) pairs; None means the limit is unknown."""

    def state(kind: str, pair: tuple[int, int] | None, window: Window) -> LimitState:
        limit, used = pair if pair is not None else (None, 0)
        return LimitState(kind=kind, limit=limit, used=used, window=window)

    minute = _window()
    return ProviderEstimate(
        provider=provider,
        requests_per_minute=state("rpm", rpm, minute),
        tokens_per_minute=state("tpm", tpm, minute),
        requests_per_day=state("rpd", rpd, Window(start=0, end=int(_NOW) + 86_400)),
        degraded=False,
    )


class FakeTracker:
    def __init__(self, estimates: dict[str, ProviderEstimate]) -> None:
        self.estimates = estimates

    async def estimate(self, provider: str) -> ProviderEstimate:
        return self.estimates[provider]


async def test_unknown_limits_are_never_treated_as_exhausted():
    """The acceptance criterion for M1.1's nullable schema: a provider with no
    recorded limits stays in the chain forever instead of being skipped as
    exhausted-by-default."""
    estimate = _estimate("groq", None)
    assert estimate.requests_per_minute.remaining is None
    assert estimate.tokens_per_minute.remaining is None
    assert _binding_limit(estimate, margin=0.1) is None


def test_zero_remaining_is_different_from_unknown_and_does_skip():
    estimate = _estimate("gemini", rpm=(15, 15))
    assert estimate.requests_per_minute.remaining == 0
    assert _binding_limit(estimate, margin=0.1) == "rpm"


def test_margin_skips_before_the_wall_not_at_it():
    near = _estimate("cerebras", rpm=(30, 28))
    assert _binding_limit(near, margin=0.1) == "rpm"
    assert _binding_limit(near, margin=0.0) is None


async def test_selection_skips_exhausted_and_reports_the_reason():
    tracker = FakeTracker(
        {
            "groq": _estimate("groq", rpm=(10, 10)),
            "cerebras": _estimate("cerebras", rpm=(30, 1)),
        }
    )
    selection = await Selector(tracker).choose(["groq", "cerebras"])
    assert selection.chosen == "cerebras"
    assert [(s.provider, s.reason) for s in selection.skipped] == [("groq", "rpm_exhausted")]


async def test_all_exhausted_yields_computed_retry_after_from_earliest_reset():
    """Retry-After must be computed rather than constant, so two exhaustion
    shapes must produce different waits: a minute window that resets in seconds
    cannot quote the same number as a daily one resetting in hours."""
    soon = ProviderEstimate(
        provider="a",
        requests_per_minute=LimitState("rpm", 10, 10, _window(end_offset_s=10)),
        tokens_per_minute=_estimate("a", None).tokens_per_minute,
        requests_per_day=_estimate("a", None).requests_per_day,
        degraded=False,
    )
    late_reset = Window(start=0, end=int(time.time()) + 7_000)
    late = ProviderEstimate(
        provider="b",
        requests_per_minute=_estimate("b", None).requests_per_minute,
        tokens_per_minute=_estimate("b", None).tokens_per_minute,
        requests_per_day=LimitState("rpd", 100, 100, late_reset),
        degraded=False,
    )

    both = await Selector(FakeTracker({"a": soon, "b": late})).choose(["a", "b"])
    assert both.chosen is None
    # The earliest reset wins even though another provider holds a much later one.
    assert 1 <= both.retry_after <= 11

    only_late = await Selector(FakeTracker({"b": late})).choose(["b"])
    assert 6_900 <= only_late.retry_after <= 7_001


async def test_no_known_limit_anywhere_still_yields_a_retry_after():
    """Reachable mid-flight: a provider with no recorded limits cannot be skipped,
    but it can still refuse, and then Retry-After has nothing to be computed from.
    The floor is the shortest window on the table rather than a magic number."""
    selector = Selector(FakeTracker({"solo": _estimate("solo", None)}))
    assert await selector.earliest_retry_after(["solo"]) == 60


def test_plan_pins_requested_provider_to_the_head_without_exclusivity():
    plan = Selector.plan("groq", ["groq", "cerebras", "gemini"], allow_local=False)
    assert plan == ["groq", "cerebras", "gemini"]

    plan = Selector.plan(None, ["groq", "cerebras"], allow_local=False)
    assert plan == ["groq", "cerebras"]


def test_plan_appends_ollama_only_when_opted_in():
    without = Selector.plan("groq", ["groq", "cerebras"], allow_local=False)
    assert "ollama" not in without

    with_local = Selector.plan("groq", ["groq", "cerebras"], allow_local=True)
    assert with_local[-1] == "ollama"

    deduped = Selector.plan("ollama", ["ollama"], allow_local=True)
    assert deduped.count("ollama") == 1


def test_unknown_provider_prefix_travels_alone():
    """`mock/x` is not in the deployed chain and must not drag the whole chain to
    a provider nobody configured."""
    assert Selector.plan("mock", ["groq", "cerebras"], allow_local=False) == ["mock"]
