"""Every stage span name and `tollgate.*` attribute, in one place.

`bench/stages.md` says what each of these spans starts and ends at, and it is hashed
because this project publishes its own overhead as a headline number. This module is
the attribute half of that contract as code, and `stage_span` refuses an attribute
the table does not declare, so the contract is enforced where attributes are written
rather than only asserted afterwards. A name that drifts fails the first test that
touches it instead of quietly producing a panel that renders nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import spanlight
import structlog
from opentelemetry.trace import Span, Status, StatusCode

logger = structlog.get_logger(__name__)

REQUEST = "tollgate.request"
CACHE_PROBE = "tollgate.cache.probe"
CACHE_EXACT = "tollgate.cache.exact"
CACHE_SEMANTIC = "tollgate.cache.semantic"
ROUTE = "tollgate.route"
BUDGET = "tollgate.budget"
SELECT = "tollgate.select"
DISPATCH = "tollgate.dispatch"
DISPATCH_WAIT = "tollgate.dispatch.wait"
DISPATCH_UPSTREAM = "tollgate.dispatch.upstream"
METER = "tollgate.meter"

# Span name to the `tollgate.*` attributes it may carry. Attributes from a shared
# convention (`gen_ai.*`, `spanlight.*`) are set by Spanlight itself and are listed
# separately below: this table owns the names this project invented, which are the
# ones with nobody else to keep them honest.
CONTRACT: dict[str, frozenset[str]] = {
    REQUEST: frozenset(
        {
            "tollgate.provider",
            "tollgate.cache",
            "tollgate.overhead_ms",
            "tollgate.streaming",
            "tollgate.model_requested",
        }
    ),
    CACHE_PROBE: frozenset({"tollgate.cache.outcome"}),
    CACHE_EXACT: frozenset({"tollgate.cache.hit"}),
    CACHE_SEMANTIC: frozenset(
        {"tollgate.cache.hit", "tollgate.cache.similarity", "tollgate.cache.threshold"}
    ),
    ROUTE: frozenset({"tollgate.route.tier", "tollgate.route.category"}),
    BUDGET: frozenset(
        {
            "tollgate.budget.requests_remaining",
            "tollgate.budget.tokens_remaining",
            "tollgate.budget.window",
            "tollgate.budget.degraded",
        }
    ),
    SELECT: frozenset(
        {"tollgate.select.chosen", "tollgate.select.skipped", "tollgate.select.chain_length"}
    ),
    DISPATCH: frozenset({"tollgate.dispatch.attempts"}),
    DISPATCH_WAIT: frozenset({"tollgate.wait.reason", "tollgate.wait.ms"}),
    DISPATCH_UPSTREAM: frozenset({"tollgate.attempt.index"}),
    METER: frozenset({"tollgate.meter.persisted"}),
}

# Counts, names and durations only. This gateway sees every prompt and every
# response in the entire portfolio, which makes it the single worst place to leak
# content, and an attribute added in a hurry is how that happens. Span attributes are
# indexed by the backend and are the documented anti-pattern for prompt text: they
# carry size limits and they put the content somewhere it can be queried by anyone
# with dashboard access. `tollgate.model_requested` is a model name, not content, and
# is the reason the rule is a substring check rather than a blanket ban on strings.
FORBIDDEN_SUBSTRINGS = (
    "prompt",
    "message",
    "content",
    "text",
    "completion",
    "response_body",
    "api_key",
    "secret",
)

# Names from shared conventions, allowed on any stage. They are not in the table
# above because that table owns what this project invented; these belong to the
# GenAI semantic convention and to Spanlight, which is where their meaning is defined
# and kept.
SHARED_ATTRIBUTES = frozenset(
    {
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "spanlight.cost_usd",
        "error.type",
    }
)


class UndeclaredAttribute(KeyError):
    """An attribute this span is not allowed to carry, per the contract above."""


# Whether a contract breach raises or is logged and dropped.
#
# Lenient by default, inherited as a correction rather than a preference. In a
# sibling project raising took a live turn down: an attribute was recorded on a span
# where it was not declared, and the only call site no test could reach was the one
# that was wrong, because it needed a real API key.
#
# Instrumentation that throws turns a mistyped attribute into an outage. That matters
# more here than anywhere else in the portfolio, because every other project's model
# calls pass through this process. So in production the attribute is dropped and the
# mistake is logged at error level, which is loud without being fatal.
# `tests/conftest.py` turns strict on, so CI fails rather than warns.
strict = False


class StageSpan:
    """A stage span that checks attribute names on the way in.

    Most stage attributes are only known at the end: which provider was chosen, how
    many attempts it took, what the cache similarity was. Handing out the raw span
    would leave the contract enforced on the attributes set at entry and unenforced on
    exactly the ones a hurried change adds.
    """

    def __init__(self, span: Span, stage: str) -> None:
        self.span = span
        self._stage = stage

    def record(self, **attributes: object) -> None:
        allowed = _permitted(self._stage, set(attributes))
        for key, value in attributes.items():
            if key in allowed:
                self.span.set_attribute(key, value)

    def mark(self, event: str) -> None:
        """A point in time on this span, for a moment that is not a duration."""
        self.span.add_event(event)


def _permitted(stage: str, offered: set[str]) -> set[str]:
    """Which of these attributes the stage may carry, complaining about the rest."""
    allowed = CONTRACT[stage] | SHARED_ATTRIBUTES
    unknown = offered - allowed
    if unknown:
        message = f"{stage} may not carry {sorted(unknown)}"
        if strict:
            raise UndeclaredAttribute(message)
        logger.error("spans.undeclared_attribute", stage=stage, attributes=sorted(unknown))
    return allowed


@contextmanager
def stage_span(stage: str, **attributes: object) -> Iterator[StageSpan]:
    """Open one of the spans named in `bench/stages.md`.

    Nesting comes from the tracer's current context, so a stage opened inside another
    stage is its child without either naming the other. That is what makes the tree
    in `bench/stages.md` real rather than a diagram: a request produces one
    `tollgate.request` with its stages beneath it, not eleven parentless roots.
    """
    if stage not in CONTRACT:
        raise UndeclaredAttribute(f"unknown stage {stage!r}, not in bench/stages.md")

    tracer = spanlight.get_tracer()
    with tracer.start_as_current_span(stage) as span:
        handle = StageSpan(span, stage)
        if attributes:
            handle.record(**attributes)
        try:
            yield handle
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
