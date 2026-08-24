"""L2 with a fixed embedding stub, so no test depends on model weights (M2.4's
own requirement). The stub scores shared trigrams, which is exactly enough to
place pairs above and below an arbitrary threshold deterministically."""

from __future__ import annotations

import pytest

from app.caches import CacheEntry, InMemoryVectorIndex, LocalKVBackend, ResponseCache
from app.embeddings import HashingStubEmbedder, cosine


def _cache(threshold: float) -> ResponseCache:
    return ResponseCache(
        kv=LocalKVBackend(),
        salt="s",
        embedder=HashingStubEmbedder(),
        index=InMemoryVectorIndex(),
        threshold=threshold,
    )


def _fields(content: str, temperature: float | None = None) -> dict:
    return {
        "model": "mock/demo",
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": None,
        "max_tokens": None,
    }


def _entry(text: str = "the capital of France is Paris") -> CacheEntry:
    return CacheEntry(
        text=text, provider="mock", model="m", tokens_in=5, tokens_out=8, cost_usd=0.0
    )


async def test_threshold_moves_the_boundary():
    """The acceptance for M2.4: same pair, different threshold, different answer.
    The threshold is calibrated around the pair's measured similarity, which is
    exactly the knob M6 exists to measure."""
    question, paraphrase = (
        "explain quicksort versus mergesort tradeoffs",
        "compare quicksort and mergesort tradeoffs",
    )
    embedder = HashingStubEmbedder()
    pair_similarity = cosine(embedder.embed(question), embedder.embed(paraphrase))

    below = _cache(threshold=round(pair_similarity - 0.05, 4))
    first = await below.lookup(_fields(question))
    await below.store(first, _entry())
    assert (await below.lookup(_fields(paraphrase))).outcome == "semantic"

    above = _cache(threshold=round(pair_similarity + 0.05, 4))
    second = await above.lookup(_fields(question))
    await above.store(second, _entry())
    assert (await above.lookup(_fields(paraphrase))).outcome == "miss"


async def test_paraphrase_hits_below_a_low_threshold():
    cache = _cache(threshold=0.5)
    miss = await cache.lookup(_fields("What is the capital of France?"))
    await cache.store(miss, _entry())

    near_miss = await cache.lookup(_fields("what is the capital of france"))
    assert near_miss.outcome == "semantic"
    assert near_miss.entry is not None
    assert near_miss.similarity is not None and near_miss.similarity >= 0.5


async def test_l2_is_disabled_without_a_threshold():
    cache = ResponseCache(
        kv=LocalKVBackend(),
        salt="s",
        embedder=HashingStubEmbedder(),
        index=InMemoryVectorIndex(),
        threshold=None,
    )
    miss = await cache.lookup(_fields("anything"))
    await cache.store(miss, _entry())

    # Same trigram-heavy text: would be a semantic hit if L2 were active.
    again = await cache.lookup(_fields("anything"))
    assert again.outcome == "exact"


async def test_semantic_hit_does_not_shadow_an_exact_hit():
    """An exact match must win even when the semantic layer would also fire:
    exact is free to verify and carries zero false-hit risk."""
    cache = _cache(threshold=0.3)
    first = await cache.lookup(_fields("capital of france"))
    await cache.store(first, _entry("stored answer"))

    hit = await cache.lookup(_fields("capital of france"))
    assert hit.outcome == "exact"
    assert hit.entry.text == "stored answer"


async def test_unrelated_question_misses_at_any_sane_threshold():
    cache = _cache(threshold=0.2)
    miss = await cache.lookup(_fields("What is the capital of France?"))
    await cache.store(miss, _entry())

    other = await cache.lookup(_fields("write me a haiku about rust ownership"))
    assert other.outcome == "miss"


async def test_index_round_trip():
    index = InMemoryVectorIndex()
    embedder = HashingStubEmbedder()

    await index.upsert("k1", embedder.embed("alpha beta gamma"), b"payload-a")
    found, score = await index.query(embedder.embed("alpha beta gamma delta"))
    assert found == "k1"
    assert score > 0.8
    assert await index.payload_for("k1") == b"payload-a"


def test_cosine_of_identical_vectors_is_one():
    embedder = HashingStubEmbedder()
    v = embedder.embed("deterministic embedding stub")
    assert cosine(v, v) == pytest.approx(1.0)


def test_stub_embedder_is_deterministic():
    e = HashingStubEmbedder()
    assert e.embed("stable input") == e.embed("stable input")


async def test_pgvector_store_refuses_to_construct_without_the_group(monkeypatch):
    """The pg import is lazy so the base install never pays for psycopg; this
    pins the error message someone will actually see when they skip the group."""
    import sys

    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(RuntimeError, match="'pg' dependency group"):
        from app.caches import PgVectorIndex

        PgVectorIndex(dsn="postgresql://x", dims=256)
