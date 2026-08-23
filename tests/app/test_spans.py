from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import spans
from app.spans import CONTRACT, FORBIDDEN_SUBSTRINGS, UndeclaredAttribute, stage_span


@pytest.fixture
def exporter(monkeypatch):
    """A real tracer, because the shape being asserted is the tracer's own.

    `spanlight.get_tracer` is monkeypatched rather than initialised, since
    `spanlight.init` is a no-op without an endpoint and would hand back spans that
    record nothing. The parent-child relationship is what this file exists to check,
    and a no-op span has no parent to check.
    """
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(spans.spanlight, "get_tracer", lambda: provider.get_tracer("test"))
    return memory


def _by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_a_request_produces_one_tree_not_eleven_roots(client, exporter):
    """The acceptance criterion for M0.4.

    Without an enclosing span every stage is a parentless root, and one request
    arrives as a handful of unrelated traces that happen to share an attribute. There
    is no waterfall to read and no way to decompose the overhead figure, which is the
    entire reason these spans exist.
    """
    assert client.post(
        "/v1/chat/completions",
        json={"model": "mock/demo", "messages": [{"role": "user", "content": "hello"}]},
    ).status_code == 200

    finished = exporter.get_finished_spans()
    names = {span.name for span in finished}
    assert spans.REQUEST in names

    request_span = _by_name(exporter)[spans.REQUEST]
    roots = [span for span in finished if span.parent is None]
    assert [span.name for span in roots] == [spans.REQUEST]

    stages = [span for span in finished if span.name != spans.REQUEST]
    assert stages, "no stage spans were recorded at all"
    for span in stages:
        assert span.parent is not None, f"{span.name} is a parentless root"
        assert span.parent.span_id == request_span.context.span_id


def test_every_stage_in_the_contract_has_a_boundary_written_down():
    """`bench/stages.md` is hashed, so a stage added here without a boundary there is
    a measurement whose extent nobody defined."""
    from pathlib import Path

    boundaries = Path("bench/stages.md").read_text(encoding="utf-8")
    missing = [stage for stage in CONTRACT if f"`{stage}`" not in boundaries]
    assert missing == []


def test_the_contract_declares_no_attribute_that_could_carry_content():
    """This gateway sees every prompt and every response in the whole portfolio.

    Span attributes are indexed by the backend and are the documented anti-pattern
    for prompt text: size limits, plus content sitting somewhere queryable by anyone
    with dashboard access. The check is on the contract rather than on a call site,
    because the contract is what a hurried change edits.
    """
    offenders = [
        attribute
        for allowed in CONTRACT.values()
        for attribute in allowed
        if any(word in attribute for word in FORBIDDEN_SUBSTRINGS)
    ]
    assert offenders == []


def test_an_undeclared_attribute_fails_in_the_suite(exporter):
    """Lenient in production, strict here. tests/conftest.py flips the switch."""
    with pytest.raises(UndeclaredAttribute), stage_span(spans.ROUTE) as span:
        span.record(**{"tollgate.route.invented": "nope"})


def test_an_unknown_stage_is_refused():
    with pytest.raises(UndeclaredAttribute), stage_span("tollgate.not.a.stage"):
        pass


def test_an_error_inside_a_stage_is_recorded_on_it(exporter):
    with pytest.raises(ValueError), stage_span(spans.DISPATCH):
        raise ValueError("upstream exploded")

    span = _by_name(exporter)[spans.DISPATCH]
    assert span.attributes["error.type"] == "ValueError"
    assert span.status.status_code is trace.StatusCode.ERROR
