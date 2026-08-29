# Tollgate

One OpenAI-compatible door in front of every free-tier model provider, so a calling
service never learns which provider answered it.

**Status: core complete, studies pending.** Budget accounting, fallback routing,
the two-layer cache, streaming and the overhead harness all work from a fresh
clone with no API keys. What remains is the part that needs live credentials:
the false-hit curve over 300 hand-adjudicated pairs, the routing study, and a
72-hour trace against real traffic.

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

### Where the overhead figure lives on a stream

Non-streaming responses carry it in the `x-tollgate-overhead-ms` header. A
stream's headers are gone before the total is known, so there the figure rides
the final chunk frame as extra fields (`tollgate_overhead_ms`, provider, cache
outcome, token counts) just before `data: [DONE]`. A separately named SSE event
was tried first and rejected: the official OpenAI SDK parses every data line as
a chat chunk, named event or not.

## Benchmarks

Every number below regenerates from `make bench` (or `uv run python bench/overhead.py`),
which refuses to print a row whose five facts are unstated. Methodology, and what
these numbers do not measure, is stated under the table rather than in prose here.
The DeepInspect write-up on mock-upstream benchmarks is the reason this section
looks the way it does; the five traps it names are the five facts.

<!-- bench:begin -->
### Overhead against deterministic upstreams

| cell | RPS | overhead ms p50/p95/p99 | max in-flight | five facts |
|---|---|---|---|---|
| in-process mock, non-streaming |    362.4 | p50   0.208  p95   0.239  p99   0.320 |   4 | upstream=mock, streaming=no, callbacks=off, warmth=warm, load=steady c=4 |
| in-process mock, non-streaming, saturation |    383.1 | p50   0.201  p95   0.258  p99   0.369 |  32 | upstream=mock, streaming=no, callbacks=off, warmth=warm, load=saturation c=32 |
| in-process mock, streaming |    303.1 | p50   4.097  p95   6.543  p99   6.927 |   4 | upstream=mock, streaming=yes, callbacks=off, warmth=warm, load=steady c=4 |
| in-process mock, streaming, saturation |    332.7 | p50  32.006  p95  58.883  p99  62.513 |  32 | upstream=mock, streaming=yes, callbacks=off, warmth=warm, load=saturation c=32 |
| network_mock, non-streaming |     83.9 | p50   0.332  p95   0.528  p99   0.797 |   4 | upstream=network_mock, streaming=no, callbacks=off, warmth=warm, load=steady c=4 |
| network_mock, non-streaming, saturation |    151.0 | p50   0.227  p95   0.311  p99   0.453 |  32 | upstream=network_mock, streaming=no, callbacks=off, warmth=warm, load=saturation c=32 |
| network_mock, streaming |     32.9 | p50   4.800  p95   9.979  p99  10.528 |   4 | upstream=network_mock, streaming=yes, callbacks=off, warmth=warm, load=steady c=4 |
| network_mock, streaming, saturation |    141.0 | p50  19.153  p95  39.578  p99  54.690 |  32 | upstream=network_mock, streaming=yes, callbacks=off, warmth=warm, load=saturation c=32 |

TTFT network_mock, streaming: live median 121.261ms, cached-replay median 24.344ms. A cached hit skips dispatch entirely; the gap is disclosed rather than hidden.

Published p99 added-latency figures for scale (none measured here):
- LiteLLM Rust gateway, July 2026 post: ~0.7 ms
- Portkey, same source: ~2.3 ms
- Bifrost v1.6.4, same source: ~4.5 ms
- Legacy LiteLLM Python path: ~257.7 ms

Excluded from every figure: client-to-gateway network (same-process ASGI), provider behaviour beyond fixed delays, any hardware but this one. Streaming and non-streaming rows are separate on purpose; a non-streaming total folds TTFT into latency and flatters the gateway.
<!-- bench:end -->

## Cache hit rate

Measured by `bench/hit_rate.py` against the replay workload in
`bench/workloads/replay-v1.jsonl`, whose duplicate caps live in the artifact's
own meta line: per category, 100 unique bases, 60 byte-identical repeats (30
percent of traffic, which is also L1's ceiling), 40 paraphrases. The script
reports n>=3 runs with variance and refuses to print a semantic-cache row it
cannot stand behind:

- Without weights under `models/`, the L2 rows are refused outright. Fetching
  the pinned model is one command; printing a number from the trigram stub
  without saying so is not an option this repo ships.
- With `--allow-stub`, L2 rows print labelled as stub runs. They exercise the
  machinery; they do not measure similarity. The stub matches shared character
  trigrams, so its "semantic" hits are format matches wearing the name.

The first real semantic number this repo publishes will come from M6's
false-hit curve, where 300 hand-adjudicated pairs decide what the threshold
costs. Until then, off is the honest setting and this section says why at
length rather than quietly shipping GPTCache's default.

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

### The official SDK parses every SSE line as a chunk, named event or not

The streamed-overhead figure originally rode a separately named SSE event
(`event: tollgate.overhead`), which every raw client read fine. The test suite
then drove the actual `openai` package against it and caught it constructing an
empty `ChatCompletionChunk` out of that line: the SDK's parser does not route by
event name on this endpoint, so a "safely ignored" event was a broken frame for
every downstream project in the portfolio. The figure now rides extra fields on
a final chunk-shaped frame instead, which the SDK tolerates and tests assert
through both readers.

### An OpenTelemetry context manager cannot cross an async-generator boundary

Holding the REQUEST span open across the streamed body required entering its
context manager in the handler task and exiting it wherever the generator
finished. The exit detaches a contextvars token created in a different context,
which fails with "token was created in a different Context" and leaves logging
noise on every streamed request. The fix is `open_stage` in `app/spans.py`: the
span is opened manually, finished explicitly at each of the generator's exits,
and child stages anchor to it via an explicit context rather than ambient
propagation. The tree test from M0.4 is what caught the first attempt shipping
parentless roots; the noise complaint is what killed the second.

### The first bench run died inside Windows' Proactor event loop

Under the load run's connect churn, the server thread's Proactor loop failed to
allocate a socket transport buffer (MemoryError at `bytearray(buffer_size)`),
and per-request `httpx.AsyncClient` construction left hundreds of TIME_WAIT
sockets behind. Both fixed by being boring: selector event loops for the bench
process and the mock server thread alike, and one pooled client shared across
the whole run. The variance gate from M5.1 earned its keep here too: 0.03ms of
fixture spread is what makes the sub-millisecond rows meaningful at all.

## Run It

Requires [uv](https://docs.astral.sh/uv/). Docker is optional and only adds
local Redis and Postgres; without them every stateful piece runs its documented
in-process fallback.

Local, no Docker, no API keys (defaults to the mock provider):

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Docker Compose (app, Postgres with pgvector, Redis; no API keys required,
defaults to the mock provider):

```bash
docker compose up --build
```

Then:

```bash
curl localhost:8000/healthz
curl localhost:8000/version
curl localhost:8000/budget          # remaining budget per provider
make bench                          # regenerate the Benchmarks section above
make test                           # uv run pytest -q
make lint                           # uv run ruff check .
```

Using a real provider: copy `.env.example` to `.env` and set the relevant
`*_API_KEY`. Free-tier limits per provider, with the date each was measured,
are in `llm/providers/quotas.yaml`.

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `mock` | Provider a bare model name resolves to; `mock` also pins the dev chain |
| `TOLLGATE_CHAIN` | price-ordered free tiers | Comma-separated fallback order |
| `SKIP_MARGIN` | `0.1` | Skip a provider under this fraction of a known limit |
| `EDGE_API_KEY` | unset (open) | Bearer key on all gateway routes; startup warns when open |
| `REDIS_URL` | unset (local counters/cache) | Upstash or local Redis for shared state |
| `CACHE_SALT` | random per process | Set it to keep cache entries across restarts |
| `SEMANTIC_THRESHOLD` | unset (L2 off) | Similarity floor for the semantic layer |
| `EMBEDDING_BACKEND` | `none` | `local` (needs `uv sync --group semantic` + fetched model) or `stub` |
| `METERING_PATH` | `metering.jsonl` | Where request rows land until Postgres is configured |

### Degraded modes, and how you know

Nothing here dies quietly when a backing service does:

- Redis unreachable: budget counters and the exact cache swap to in-process
  stores. One error log (`budget.redis_unreachable` / `cache.redis_unreachable`),
  `degraded: true` in `/budget`, and counts that describe one replica only.
- Postgres unreachable (or never configured): metering lands in JSONL at
  `METERING_PATH`, readable by everything in `bench/`; semantic-cache lookups
  fail open to misses.
- Every provider exhausted: HTTP 429 with `Retry-After` computed from the
  earliest window reset across the providers tried. A dead upstream answers 502
  instead, because telling a caller to slow down will not fix a socket.
