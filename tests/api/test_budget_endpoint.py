from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


def test_budget_distinguishes_unknown_from_zero(client):
    """The acceptance criterion for M1.3.

    gemini carries a measured rpm limit and no tpm limit at all. The response has
    to show a real number for one and null for the other: reporting 0 remaining on
    the unknown one would tell a caller to back off a provider with quota left.
    """
    body = client.get("/budget").json()

    gemini = body["providers"]["gemini"]
    assert isinstance(gemini["rpm"]["limit"], int)
    assert isinstance(gemini["rpm"]["remaining"], int)

    assert gemini["tpm"]["limit"] is None
    assert gemini["tpm"]["remaining"] is None
    assert gemini["rpd"]["limit"] is None
    assert gemini["rpd"]["remaining"] is None


def test_budget_reports_the_window_each_figure_belongs_to(client):
    body = client.get("/budget").json()
    groq = body["providers"]["groq"]

    assert groq["rpm"]["window_end"] - groq["rpm"]["window_start"] == 60
    assert 0 <= groq["rpm"]["reset_in_s"] <= 60


async def test_recorded_requests_show_up_in_the_response(client):
    """`used` rising while `remaining` falls is the whole budget model in
    miniature, and the endpoint reads the same store dispatch writes to."""
    gateway = client.app.state.gateway
    await gateway.tracker.record("groq", tokens=500)

    groq = client.get("/budget").json()["providers"]["groq"]
    assert groq["rpm"]["used"] >= 1
    assert groq["tpm"]["used"] >= 500
    if groq["rpm"]["limit"] is not None:
        assert groq["rpm"]["remaining"] == groq["rpm"]["limit"] - groq["rpm"]["used"]


def test_every_provider_appears_even_without_limits(client):
    body = client.get("/budget").json()["providers"]
    for name in ("gemini", "groq", "cerebras", "openrouter", "sarvam", "ollama", "mock"):
        assert name in body, f"{name} missing from /budget"
        assert set(body[name]) >= {"rpm", "tpm", "rpd", "degraded"}


def test_verification_dates_travel_with_their_limits(client):
    """A limit without its measurement date invites the reader to assume it holds
    forever. The date is part of the datum."""
    body = client.get("/budget").json()
    groq = body["providers"]["groq"]
    assert groq["rpm"]["last_verified"]
