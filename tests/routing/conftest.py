from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.main import app as fastapi_app


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Throttle cooldowns are up to 40 real seconds; the suite cannot wait them."""

    async def _no_sleep(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


@pytest.fixture
def exporter(monkeypatch):
    """A recording tracer, so span attributes can be asserted rather than trusted."""
    from app import spans

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(spans.spanlight, "get_tracer", lambda: provider.get_tracer("test"))
    return memory
