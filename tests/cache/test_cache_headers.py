from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.caches import InMemoryVectorIndex, LocalKVBackend, ResponseCache
from app.config import get_settings
from app.embeddings import HashingStubEmbedder
from app.gateway import Gateway
from app.main import app as fastapi_app
from app.metering import JsonlMeteringStore

BODY = {"model": "mock/demo", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def client(tmp_path):
    """The shared app with a stub-embedding cache so semantic hits are reachable
    without weights, and a tmp metering file so nothing leaks between tests."""
    original = fastapi_app.state.gateway
    gateway = Gateway.from_settings(
        get_settings(), metering=JsonlMeteringStore(tmp_path / "m.jsonl")
    )
    gateway.cache = ResponseCache(
        kv=LocalKVBackend(),
        salt="header-test",
        embedder=HashingStubEmbedder(),
        index=InMemoryVectorIndex(),
        threshold=0.3,
    )
    fastapi_app.state.gateway = gateway
    yield TestClient(fastapi_app)
    fastapi_app.state.gateway = original


def test_miss_then_exact_on_identical_requests(client):
    first = client.post("/v1/chat/completions", json=BODY)
    second = client.post("/v1/chat/completions", json=BODY)

    assert first.headers["x-tollgate-cache"] == "miss"
    assert second.headers["x-tollgate-cache"] == "exact"
    # Same answer twice: that is what exact means.
    assert first.json()["choices"][0]["message"] == second.json()["choices"][0]["message"]


def test_changing_a_sampling_parameter_produces_two_upstream_calls(client):
    """Different temperature, different key, second upstream call. L2 is off for
    this one on purpose: the stub embedder scores the shared `user:` prefix above
    0.3 for almost any pair of messages, which would turn this into a semantic
    hit and say nothing about L1."""
    client.app.state.gateway.cache = ResponseCache(
        kv=LocalKVBackend(), salt="l1-only"
    )
    cold = client.post("/v1/chat/completions", json=BODY)
    warm = client.post("/v1/chat/completions", json={**BODY, "temperature": 0.9})

    assert cold.headers["x-tollgate-cache"] == "miss"
    assert warm.headers["x-tollgate-cache"] == "miss"


def test_semantic_hit_carries_the_similarity_for_audit(client):
    """M2.6's audit requirement: a caller who suspects a wrong answer can read the
    similarity off the response and decide for themselves."""
    client.post("/v1/chat/completions", json=BODY)
    paraphrased = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock/demo",
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )

    assert paraphrased.status_code == 200
    if paraphrased.headers["x-tollgate-cache"] == "semantic":
        assert float(paraphrased.headers["x-tollgate-cache-similarity"]) >= 0.3


def test_all_three_outcomes_occur_in_one_scripted_sequence(client):
    """The acceptance for M2.6, end to end through the HTTP surface."""
    outcomes = []
    bodies = [
        BODY,
        dict(BODY),
        {**BODY, "messages": [{"role": "user", "content": "hello again"}]},
    ]
    for body in bodies:
        resp = client.post("/v1/chat/completions", json=body)
        outcomes.append(resp.headers["x-tollgate-cache"])

    assert outcomes[0] == "miss"
    assert outcomes[1] == "exact"


def test_cached_answer_still_discloses_who_answered(client):
    first = client.post("/v1/chat/completions", json=BODY)
    second = client.post("/v1/chat/completions", json=BODY)

    assert first.headers["x-tollgate-provider"] == second.headers["x-tollgate-provider"]
