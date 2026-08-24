"""Edge auth (M4.2): one key at the edge, loud when absent, silent when present."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.auth import maybe_warn_unauthenticated
from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore

BODY = {"model": "mock/demo", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def keyed(tmp_path, monkeypatch):
    """The shared app with EDGE_API_KEY pinned for the duration. Settings are
    read per-call, so the env var is enough."""
    original = fastapi_app.state.gateway
    fastapi_app.state.gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    monkeypatch.setenv("EDGE_API_KEY", "test-edge-key-123")
    yield
    monkeypatch.delenv("EDGE_API_KEY")
    fastapi_app.state.gateway = original


def _post(client: TestClient, headers=None):
    return client.post("/v1/chat/completions", json=BODY, headers=headers or {})


def test_open_by_default_locally(client):
    assert _post(client).status_code == 200


def test_missing_key_is_401_when_configured(keyed):
    client = TestClient(fastapi_app, raise_server_exceptions=False)
    resp = _post(client)
    assert resp.status_code == 401


def test_wrong_key_is_401_and_right_key_passes(keyed):
    client = TestClient(fastapi_app, raise_server_exceptions=False)

    wrong = _post(client, {"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    right = _post(client, {"Authorization": "Bearer test-edge-key-123"})
    assert right.status_code == 200


def test_budget_route_is_behind_the_same_key(keyed):
    client = TestClient(fastapi_app, raise_server_exceptions=False)
    assert client.get("/budget").status_code == 401
    ok = client.get("/budget", headers={"Authorization": "Bearer test-edge-key-123"})
    assert ok.status_code == 200


def test_an_open_edge_announces_itself():
    """An unauthenticated gateway must be a decision, not a default nobody
    noticed. The warning fires once and names the fix."""
    import app.auth as auth_module

    auth_module._warned = False
    with capture_logs() as logs:
        maybe_warn_unauthenticated()

    assert any(e.get("event") == "auth.edge_disabled" for e in logs)
    # And only once per process.
    with capture_logs() as again:
        maybe_warn_unauthenticated()
    assert not any(e.get("event") == "auth.edge_disabled" for e in again)
