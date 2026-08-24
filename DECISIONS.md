# Decisions

Every nontrivial choice gets an entry here at the time it's made - not
reconstructed later from memory. Newest entries at the top.

## Format
```
## YYYY-MM-DD: <short title>
**Context:** what problem/question forced a decision.
**Decision:** what was chosen.
**Alternatives considered:** what else was on the table, and why it lost.
**Consequences:** what this makes easier/harder later.
```

<!-- Add entries above this line. -->

## 2026-08-24: Edge auth is one key, optional locally, loud about it
**Context:** M4.2 requires the edge be secure by default, while BAR-1 requires a fresh clone to work in five minutes with no configuration at all.
**Decision:** a single bearer key (`EDGE_API_KEY`) guards every gateway route when set, compared via `hmac.compare_digest`. When unset, routes are open and startup emits exactly one warning naming the fix.
**Alternatives considered:** making the key mandatory would break works-from-zero; per-caller keys are multi-tenant machinery for a deployment with one tenant (SPEC non-goal).
**Consequences:** the deployed compose file must set the variable or ship an open edge with a warning nobody reads. The runbook says so.

## 2026-08-24: Streaming retries until first byte, then never
**Context:** the chassis retries whole requests; a stream is not whole once its first chunk has been forwarded, and retrying mid-stream would splice two answers into one.
**Decision:** `ChatClient.stream_complete` applies throttle gating, 429 trips and backoff only before the first chunk arrives. After it, any failure propagates and the API layer emits an error frame followed by a clean `[DONE]`, so the client sees an honest end rather than a hang or a spliced answer.
**Alternatives considered:** buffering the stream and retrying transparently would fold time-to-first-token into total latency and hand this project's headline number exactly the flattering bias SPEC forbids.
**Consequences:** callers of streamed requests must tolerate error frames mid-stream. The official SDK surfaces them as exceptions on iteration, which is the correct client behaviour.

## 2026-08-24: Overhead on streams rides a final chunk frame, not a named event
**Context:** headers are gone before a stream's overhead is known. The obvious SSE answer, a separately named event, turned out to break the official OpenAI SDK: it parses every data line as a chat chunk, named event or not.
**Decision:** the last frame before `[DONE]` is a chunk-shaped object whose extra top-level fields carry `tollgate_overhead_ms`, provider, cache outcome and token counts. The SDK tolerates the unknown fields; raw readers get their numbers; the README documents the shape.
**Alternatives considered:** omitting the figure from streams entirely would hide half the measurement; SSE comments carry it but no client can read them programmatically.
**Consequences:** one empty delta chunk appears at the end of every stream. Callers filtering on content see nothing; auditors see everything.

## 2026-08-24: Metering failures cost log lines, never responses
**Context:** metering writes happen after success; raising there converts bookkeeping into an outage for precisely the requests worth recording.
**Decision:** every store handed to the gateway is wrapped in `ResilientMeteringStore`, which swallows and logs append failures and returns empty (never partial) on read failures.
**Alternatives considered:** letting errors propagate keeps the store honest and the service fragile; dropping the metering layer silently keeps the service up and the disagreement study blind.
**Consequences:** `/budget`-style consumers of metering must treat an empty read as "unknown", which they already do.

## 2026-08-24: Cache keys are a whitelist, salted, keyed on the request not the answer
**Context:** M2.1's key derivation decides which wrong answers are possible. A key that ignores any sampling parameter serves a deterministic answer to a creative request forever; a key readable as a hash of a known prompt lets anyone with Redis access confirm what was served.
**Decision:** SHA-256 over a per-process salt and the canonical JSON of exactly five fields: model, messages, temperature, top_p, max_tokens. `stream` is excluded on purpose so one stored completion can serve both transports. Unknown request fields are dropped, visibly, by the whitelist.
**Alternatives considered:** hashing the whole request body would pick up client-side noise (extra fields vary across SDKs) and would re-key when a caller adds an unrelated flag. Blacklisting content fields is the GPTCache-era mistake: every new field silently participates.
**Consequences:** adding a sampling parameter to the whitelist later is a conscious edit with its own test. Until then it cannot influence keys at all.

## 2026-08-24: The semantic layer ships switched off
**Context:** GPTCache defaults to similarity 0.75, practitioner consensus sits at 0.92 to 0.97, and nobody publishing in that gap says what it costs in wrong answers. Shipping any number there unmeasured repeats ShipGate's intuited-threshold failure in a new costume.
**Decision:** L2 activates only when both a threshold (`SEMANTIC_THRESHOLD`) and an embedding backend are explicitly configured. The default gateway runs L1 only. Embeddings come from a local sentence-transformers snapshot fetched by script against a pinned revision (weights never committed); pgvector lives behind the `pg` dependency group until Neon credentials exist.
**Alternatives considered:** enabling L2 with a "reasonable" threshold would make demos look better and measurements meaningless. Enabling it behind the trigram stub would serve format matches as meaning matches; the stub exists for tests and is labelled wherever it appears.
**Consequences:** the honest first semantic number this repo publishes will come from M6's curve, and until then `/budget`-style honesty about what is off is the feature.

## 2026-08-24: Both cache layers fail open to a miss
**Context:** Redis unreachable, or a corrupt payload from a killed process. The question is whether a cache outage may become user-facing.
**Decision:** every failure inside a probe or a store degrades to a miss (or to skipping the write), logs once at error level, and sets a degraded flag. No exception from the caching layer escapes into a response.
**Alternatives considered:** failing closed keeps the numbers pure and takes the service down with the cache. Retrying inside the probe adds latency to exactly the path that must be faster than dispatch.
**Consequences:** the worst thing the cache can do to a request is fail to save money. The degradation counters are visible in logs; wiring them into metrics lands with M5's dashboard work.

## 2026-08-24: The replay workload's duplicate rate is part of the artifact
**Context:** a hit rate measured against an authored workload is trivially inflatable, and SPEC already commits to saying so wherever a hit rate appears. The defence has to live somewhere stronger than prose.
**Decision:** `bench/workloads/replay-v1.jsonl` carries its composition in its meta line: per category, 100 unique bases, 60 byte-identical repeats, 40 paraphrases. Regenerated deterministically by `bench/build_workload.py`; two runs produce identical bytes under the published sha256. Hit-rate rows print n>=3 with variance and refuse L2 rows without weights unless `--allow-stub` labels them.
**Alternatives considered:** sampling real traffic would be better and is impossible here: there is no production traffic to sample, and pretending otherwise is the inflation the meta line exists to prevent.
**Consequences:** every hit rate quoted from replays of this file can be read against its 30 percent exact-layer ceiling, and anyone can regenerate the file to check the quote.

## 2026-08-24: Fixed windows aligned to UTC, with the earliest reset as Retry-After
**Context:** M1 needed counters that predict provider exhaustion, and a documented answer to "all providers exhausted".
**Decision:** requests and tokens counted in fixed windows (UTC minute for rpm/tpm, UTC day for rpd), keyed by provider and window start, TTL armed once per window. On total exhaustion the 429's `Retry-After` is `ceil(earliest known window reset - now)` across the providers actually tried, floored at one second; when no known limit exists anywhere it floors at 60 because that is the shortest window on the table.
**Alternatives considered:** rolling windows model provider behaviour more faithfully but disagree with the provider in ways that are hard to debug, and only the provider's response headers can settle an argument about accounting. Queueing was rejected in SPEC (callers are interactive); degrading to Ollama by default was rejected there too and survives only as a per-request header opt-in.
**Consequences:** the estimate under-counts around boundary crossings, predictably. `/budget` publishes each figure beside its window so nobody mistakes an alignment artifact for drift.

## 2026-08-24: An explicit provider prefix pins the chain head, not the whole chain
**Context:** `groq/model` names a provider; the demo checkpoint demands that forcing groq's limit moves traffic to cerebras "without the caller seeing an error". What happens when the caller explicitly asked for groq?
**Decision:** the requested provider heads the chain and the rest follows in configured order. `x-tollgate-provider` always says who answered.
**Alternatives considered:** treating an explicit prefix as exclusive would turn every exhaustion into a user-facing 429 for exactly the callers most likely to be pinning (tests, repro scripts). Serving exclusively also removes the failure mode the gateway exists to absorb.
**Consequences:** a caller cannot use the prefix to demand exclusivity. That is a feature: the README says to treat `x-tollgate-provider`, not the request, as the source of truth for who answered.

## 2026-08-24: Exhaustion answers 429; breakage answers 502
**Context:** when the whole chain fails mid-flight the caller needs one status that says what happened. Rate limits and dead upstreams are different failures with different caller responses.
**Decision:** trips (provider refused while the estimate said healthy) and pre-flight exhaustion return 429 with computed `Retry-After`. Provider 5xx/network errors after the walk returns 502. Provider 4xx stops the walk immediately with 400, since re-sending the same bad request elsewhere cannot fix it. Misconfiguration surfaces as 502.
**Alternatives considered:** mapping everything onto 429 would tell callers to slow down into a provider outage; mapping everything onto 502 loses the retry guidance the free tiers make mandatory.
**Consequences:** callers get three distinguishable signals out of one endpoint, each with its own correct client-side behaviour.

## 2026-08-24: Counters and metering degrade rather than fail
**Context:** Redis unreachable, or never configured (no Docker on this machine, Upstash creds are an M0.1 affair). A gateway that dies when its cache dies is worse than one that under-counts.
**Decision:** `BudgetTracker` swaps to an in-process store on the first connection error, logs `budget.redis_unreachable` at error level, and stamps `degraded` on every estimate afterwards; the flag reaches `/budget` bodies and span attributes. Metering defaults to append-only JSONL until Postgres credentials exist; both stores implement one Protocol so M1.5's disagreement query and M8.1's trace regeneration do not care which is live.
**Alternatives considered:** failing closed would keep the numbers honest and the gateway down. Silently continuing would keep the gateway up and the numbers dishonest. Degrading loudly keeps both.
**Consequences:** multi-replica deployments must read the degraded flag as "counts are per-replica", not "counts are current". The README runbook says so.

## 2026-08-24: Gateway state lives in one assembled object, not module singletons
**Context:** the chassis wired `ChatClient` as an import-time singleton. Every fallback scenario needs two providers behaving differently in one process, and tests need to build that out of fakes.
**Decision:** `app/gateway.py` holds client, tracker, selector, metering, chain and margin; `create_app` builds it from Settings into `app.state`. Routers take what they need from the request.
**Alternatives considered:** patching module attributes worked until the first test needed the shared app object to carry different state than production wiring, which is also the point where leaks between suites started.
**Consequences:** one place owns lifecycle, and the fixture that swaps providers restores the original object afterwards.
