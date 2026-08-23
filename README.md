# Tollgate

One OpenAI-compatible door in front of every free-tier model provider, so a calling
service never learns which provider answered it.

**Status: building.** The sections below fill in as the work lands. Nothing here
claims a number that a script in `bench/` cannot regenerate, which is why most of
this page is currently empty rather than aspirational.

## Problem

Eleven services in this portfolio call a model. Each one needs the same five things:
somewhere to put an API key, a retry that honours the provider's own backoff, a way
to notice a provider is out of quota before sending it traffic, a cache, and a
record of what was spent. Built per service, that is the same bug fixed eleven
times.

The harder half is that free tiers do not fail cleanly. A provider under sustained
load can degrade for hours while a single isolated probe still answers in under a
second, so a healthcheck reports it as fine. Routing around that is the thing this
gateway exists to do, and measuring whether its budget model actually predicts
exhaustion is the thing it has to prove.

## Architecture

```
caller
  |
  |  POST /v1/chat/completions        OpenAI-compatible, streaming or not
  v
Tollgate
  |  1. cache probe        L1 exact (Redis)  ->  L2 semantic (pgvector)
  |  2. router             cheap or strong, per-category logged
  |  3. budget check       requests and tokens, per provider, fixed window
  |  4. selection          fallback chain, skipping exhausted providers
  |  5. dispatch           throttle gate, retry, 429 trip
  |  6. meter              tokens, cost, provider, cache outcome, overhead
  v
groq, gemini, cerebras, openrouter, sarvam, ollama
```

Each numbered stage is a span, which is what makes `x-tollgate-overhead-ms`
decomposable rather than one opaque figure.

## Benchmarks

Not yet measured. What will appear here, and the rules it has to satisfy:

- Gateway overhead, reported with five facts attached: mock or real upstream,
  streaming or not, callbacks on or off, warm or cold, steady-state or saturation.
  Streaming and non-streaming are reported separately and never averaged, because a
  non-streaming measurement folds time to first token into the total and flatters
  the gateway.
- Cache hit rate against a replayed workload, stated alongside the fact that the
  workload is authored here rather than sampled from anyone's production traffic.
- The semantic cache false-hit curve over 300 hand-adjudicated pairs, which is the
  number a hit rate hides.
- A cost-quality routing frontier broken down per category, because a router that
  learned topic rather than difficulty looks identical in aggregate.

## Technical Decisions

See [DECISIONS.md](DECISIONS.md).

## What Broke

Nothing yet worth reporting. Entries land here as they happen, not reconstructed
afterwards.

## Run It

Requires [uv](https://docs.astral.sh/uv/) and Docker.

Local, no Docker:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Docker Compose (app, postgres with pgvector, redis; no API keys required, defaults
to the mock provider):

```bash
docker compose up --build
```

Then:

```bash
curl localhost:8000/healthz
curl localhost:8000/version
```

Tests and lint:

```bash
uv run pytest
uv run ruff check .
```

Using a real provider: copy `.env.example` to `.env` and set the relevant
`*_API_KEY`. Free-tier limits per provider, with the date each was measured, are in
`llm/providers/quotas.yaml`.
