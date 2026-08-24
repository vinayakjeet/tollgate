"""M4.3's test: the cache is shared across every caller of this deployment, on
purpose, and the test pins that decision as implemented so the README section
and the code cannot drift apart.

In a multi-tenant deployment this design would serve tenant A's cached answer
to tenant B whenever their prompts collide, including prompts containing
private data. Single-tenant is the chosen scope (SPEC non-goals); the salt
protects prompt privacy against Redis readers, not against co-tenants, because
there are none by design."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore

BODY_A = {
    "model": "mock/demo",
    "messages": [{"role": "user", "content": "tenant A private question"}],
}


@pytest.fixture
def client(tmp_path):
    original = fastapi_app.state.gateway
    fastapi_app.state.gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    yield TestClient(fastapi_app)
    fastapi_app.state.gateway = original


def test_cache_is_deliberately_not_partitioned_by_caller(client):
    """Caller 1 warms the cache with a credential-free header; caller 2 arrives
    with different credentials and gets the same cached entry. That is the
    documented single-tenant behaviour, asserted here so a future change to it
    must change this test and the README section together."""
    first = client.post("/v1/chat/completions", json=BODY_A)
    assert first.headers["x-tollgate-cache"] == "miss"

    second = client.post(
        "/v1/chat/completions",
        json=BODY_A,
        headers={"X-Caller": "a-different-caller"},
    )
    assert second.status_code == 200
    assert second.headers["x-tollgate-cache"] == "exact"
    assert (
        first.json()["choices"][0]["message"]["content"]
        == second.json()["choices"][0]["message"]["content"]
    )
