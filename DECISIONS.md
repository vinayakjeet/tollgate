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
