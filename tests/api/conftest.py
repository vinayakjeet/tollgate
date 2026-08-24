from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore


@pytest.fixture
def client(fresh_gateway) -> TestClient:
    return TestClient(fastapi_app)


@pytest.fixture(autouse=True)
def fresh_gateway(tmp_path):
    """A clean Gateway per test: empty cache, empty counters, tmp metering file.

    The app object is module-level, so anything left inside it leaks into the
    next test through the cache, which serves stale answers instead of calling
    the provider the test just patched.
    """
    original = fastapi_app.state.gateway
    fastapi_app.state.gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "metering.jsonl")
    )
    yield fastapi_app.state.gateway
    fastapi_app.state.gateway = original
