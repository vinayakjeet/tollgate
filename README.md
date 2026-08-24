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

## Threat model

Tollgate fronts free-tier quota that the whole portfolio shares, so the assets
are budget and privacy, in that order.

- **Budget theft.** An open gateway means anyone who finds the URL spends the
  requests-per-minute every downstream project depends on. One bearer key at
  the edge (`EDGE_API_KEY`) is the whole defense; it is compared in constant
  time. When the variable is unset the edge is open and the process says so once
  at startup: an open gateway must be a decision, not a default nobody noticed.
- **Prompt disclosure through the cache.** Cache keys are salted (SHA-256 over a
  per-process secret), so someone with read access to Redis cannot confirm a
  prompt was served by replaying a hash of it. The salt is not encryption; a
  cache entry's payload still holds answers in plaintext, which is acceptable
  only because this deployment has one tenant and no untrusted Redis readers.
- **Prompt disclosure through traces.** Span attributes are indexed by the
  backend and queryable by anyone with dashboard access, so `app/spans.py`
  refuses attribute names that could carry content and the contract is enforced
  where attributes are written. Prompts never enter spans.
- **What this threat model does not cover:** multi-tenancy (a non-goal per
  SPEC), per-caller authorization beyond the one key, and denial of service by
  request volume. The free tiers themselves rate-limit the latter better than
  this service could.

## The shared cache is a hazard, and single-tenant is the choice

Both cache layers are keyed on the request alone: caller identity plays no part,
and `tests/cache/test_shared_scope.py` pins that fact. In a deployment with two
organizations behind it, organization A's cached answer would be served to
organization B whenever their prompts collide closely enough, including answers
derived from prompts containing private data. That failure mode is the entire
risk surface of a shared semantic cache, and it is why SPEC names multi-tenancy
a non-goal rather than a roadmap item. This deployment has one tenant: the
portfolio itself. If that ever changes, the scoping decision changes with it,
and both the code and this paragraph have to move together.

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
