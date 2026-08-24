"""End-to-end fallback. The acceptance for M1.4 is demonstrated by forcing a real
counter to its limit and watching traffic move, never by asserting against a
fixture that says it moved (the ShipGate lesson, applied here)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.budget import BudgetTracker, LocalCounterStore, minute_window
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore
from app.routing import Selector
from llm.providers import registry as registry_module
from llm.types import ChatMessage, ChatResponse, RateLimitError


class StubProvider:
    """Registered under a real name so the chain walks it like any other."""

    def __init__(self, name: str, fail_first: int = 0) -> None:
        self.name = name
        self.calls = 0
        self.fail_first = fail_first

    async def chat_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse:
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RateLimitError(f"{self.name}: rate limited", retry_after=30.0)
        return ChatResponse(
            text=f"hello from {self.name}",
            provider=self.name,
            model="stub-model",
            tokens_in=3,
            tokens_out=4,
        )


def _rpm_key(provider: str) -> str:
    return f"tollgate:budget:{provider}:rpm:{minute_window().start}"


def _tpm_key(provider: str) -> str:
    return f"tollgate:budget:{provider}:tpm:{minute_window().start}"


@pytest.fixture
def two_provider_chain(monkeypatch, tmp_path):
    """Swap the shared app's gateway onto stub providers with clean local counters,
    then put everything back. The app object is module-level, so state that leaks
    from here leaks into every other suite."""
    groq = StubProvider("groq")
    cerebras = StubProvider("cerebras")
    monkeypatch.setitem(registry_module._PROVIDERS, "groq", groq)
    monkeypatch.setitem(registry_module._PROVIDERS, "cerebras", cerebras)
    monkeypatch.setattr(registry_module, "quota_limits", lambda name: (10, 1000, None))

    tracker = BudgetTracker(LocalCounterStore())
    original = fastapi_app.state.gateway
    gateway = Gateway(
        client=original.client,
        tracker=tracker,
        selector=Selector(tracker, margin=0.1),
        metering=JsonlMeteringStore(tmp_path / "metering.jsonl"),
        chain=["groq", "cerebras"],
        margin=0.1,
    )
    fastapi_app.state.gateway = gateway
    yield gateway, groq, cerebras
    fastapi_app.state.gateway = original


def _post(client: TestClient):
    return client.post(
        "/v1/chat/completions",
        json={"model": "groq/stub-model", "messages": [{"role": "user", "content": "hi"}]},
    )


async def _fill(store, key: str, amount: int):
    await store.incr(key, amount, ttl_s=90_000)


async def test_traffic_moves_off_a_forced_exhausted_provider(two_provider_chain, client):
    """Force groq to its rpm limit through the same store dispatch writes to, then
    ask for groq by name. Cerebras must answer."""
    gateway, groq, cerebras = two_provider_chain
    await _fill(gateway.tracker._store, _rpm_key("groq"), 10)

    resp = _post(client)

    assert resp.status_code == 200
    assert resp.headers["x-tollgate-provider"] == "cerebras"
    assert groq.calls == 0
    assert cerebras.calls == 1


async def test_the_skip_is_recorded_on_the_span(two_provider_chain, client, exporter):
    gateway, _, _ = two_provider_chain
    await _fill(gateway.tracker._store, _rpm_key("groq"), 10)

    assert _post(client).status_code == 200

    spans_by_name = {s.name: s for s in exporter.get_finished_spans()}
    select = spans_by_name["tollgate.select"]
    assert "groq:rpm_exhausted" in select.attributes["tollgate.select.skipped"]


async def test_a_mid_flight_trip_moves_down_the_chain_and_is_persisted(
    two_provider_chain, client
):
    """The estimate kept groq healthy, the provider refused anyway. That
    disagreement is exactly what M1.5 exists to count, so it lands in the store."""
    gateway, groq, cerebras = two_provider_chain
    groq.fail_first = 5  # more than ChatClient's retry budget, so it propagates

    resp = _post(client)

    assert resp.status_code == 200
    assert resp.headers["x-tollgate-provider"] == "cerebras"
    trips = [r for r in gateway.metering.read_all() if r.kind == "trip"]
    assert trips and trips[-1].provider == "groq"


async def test_every_provider_exhausted_fails_fast_with_computed_retry_after(
    two_provider_chain, client
):
    """M1.6: all-exhausted is a documented behaviour, not an accident. Fail fast,
    with the wait computed from the windows rather than quoted as a constant."""
    gateway, _, _ = two_provider_chain
    await _fill(gateway.tracker._store, _rpm_key("groq"), 10)
    await _fill(gateway.tracker._store, _rpm_key("cerebras"), 10)

    resp = _post(client)

    assert resp.status_code == 429
    assert 1 <= int(resp.headers["retry-after"]) <= 60
    assert resp.json()["error"]["code"] == "all_providers_exhausted"


async def test_retry_after_matches_the_live_window_not_a_constant(
    two_provider_chain, client
):
    """The header must equal the seconds left in the exhausted minute window,
    which is only true if something computed it. A constant would drift away from
    the window edge on every observation."""
    import math
    import time

    gateway, _, _ = two_provider_chain
    await _fill(gateway.tracker._store, _rpm_key("groq"), 10)
    await _fill(gateway.tracker._store, _rpm_key("cerebras"), 10)

    resp = _post(client)

    remaining = max(minute_window().end - time.time(), 1.0)
    assert int(resp.headers["retry-after"]) == math.ceil(remaining)


async def test_ollama_joins_only_when_opted_in(two_provider_chain, client, monkeypatch):
    """SPEC: degrading to local Ollama silently changes the model behind an
    answer, so it happens only when the request asks for it, by header."""
    gateway, _, _ = two_provider_chain
    ollama = StubProvider("ollama")
    monkeypatch.setitem(registry_module._PROVIDERS, "ollama", ollama)
    await _fill(gateway.tracker._store, _rpm_key("groq"), 10)
    await _fill(gateway.tracker._store, _rpm_key("cerebras"), 10)
    await _fill(gateway.tracker._store, _tpm_key("groq"), 1000)
    await _fill(gateway.tracker._store, _tpm_key("cerebras"), 1000)

    body = {"model": "groq/stub-model", "messages": [{"role": "user", "content": "hi"}]}

    without_opt_in = client.post("/v1/chat/completions", json=body)
    assert without_opt_in.status_code == 429
    assert ollama.calls == 0

    opted_in = client.post(
        "/v1/chat/completions",
        headers={"x-tollgate-allow-local": "true"},
        json=body,
    )
    assert opted_in.status_code == 200
    assert opted_in.headers["x-tollgate-provider"] == "ollama"


async def test_a_bad_request_stops_the_walk_instead_of_burning_the_chain(
    two_provider_chain, client
):
    """A 4xx is the caller's mistake; no downstream provider can fix it, so the
    remaining providers are not tried."""
    gateway, _, cerebras = two_provider_chain

    async def _bad(*args: object, **kwargs: object):
        from llm.types import ProviderClientError

        raise ProviderClientError("unknown model upstream")

    gateway.client.complete = _bad  # type: ignore[method-assign]

    resp = _post(client)

    assert resp.status_code == 400
    assert cerebras.calls == 0
