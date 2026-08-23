from __future__ import annotations

import pytest
import structlog
from fastapi.testclient import TestClient

from app.main import app as fastapi_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


@pytest.fixture(autouse=True)
def restore_structlog_config():
    """`create_app()` reconfigures logging globally, so a test that calls it leaks
    that configuration into every test that runs afterwards.

    The victim is `structlog.testing.capture_logs`, which cannot see through a
    logger that was cached against a different processor chain. That surfaced as
    `tests/llm/test_client.py` failing only when this directory ran first, which is
    the worst shape a test failure can take: it points at the wrong file and it
    depends on collection order.
    """
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)
