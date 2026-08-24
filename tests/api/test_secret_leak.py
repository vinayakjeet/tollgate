"""No configured secret may reach a log line or an exported span attribute, on
any code path (M4.2's acceptance).

The check runs the gateway through success, provider failure and auth failure
with a distinctive key value set, then sweeps everything that left the process.
A substring this long cannot appear by coincidence: finding it means a leak."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from structlog.testing import capture_logs

from app import spans
from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore
from llm.providers import registry as registry_module
from llm.types import ChatResponse, ProviderError

SECRET = "sk-tollgate-leak-canary-9f3a1c"


class _Exploding:
    name = "exploding"

    async def chat_completion(self, messages, **kwargs) -> ChatResponse:
        raise ProviderError(f"upstream exploded holding {SECRET}")

    async def stream_completion(self, messages, **kwargs):
        raise ProviderError(f"upstream exploded holding {SECRET}")
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture
def canary(tmp_path, monkeypatch):
    original = fastapi_app.state.gateway
    monkeypatch.setenv("EDGE_API_KEY", SECRET)
    monkeypatch.setitem(registry_module._PROVIDERS, "exploding", _Exploding())
    gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    gateway.chain = ["exploding"]
    fastapi_app.state.gateway = gateway

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(spans.spanlight, "get_tracer", lambda: provider.get_tracer("t"))

    yield memory, TestClient(fastapi_app, raise_server_exceptions=False)

    fastapi_app.state.gateway = original


def test_no_secret_in_logs_or_spans_across_paths(canary):
    memory, client = canary

    logs = []
    with capture_logs() as captured:
        logs.extend(captured)

        # Failure path (provider error mapped to 502), auth failure, success,
        # budget endpoint: every response shape the service produces.
        boom = client.post(
            "/v1/chat/completions",
            json={"model": "exploding/m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        assert boom.status_code == 502

        unauth = client.post(
            "/v1/chat/completions",
            json={"model": "mock/demo", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauth.status_code == 401

        ok = client.post(
            "/v1/chat/completions",
            json={"model": "mock/demo", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        assert ok.status_code == 200

        budget = client.get("/budget", headers={"Authorization": f"Bearer {SECRET}"})
        assert budget.status_code == 200

    for entry in logs:
        blob = json.dumps(entry, default=str)
        assert SECRET not in blob, f"secret leaked into log event: {entry.get('event')}"

    for span in memory.get_finished_spans():
        blob = json.dumps(dict(span.attributes or {}), default=str)
        assert SECRET not in blob, f"secret leaked onto span {span.name}"
