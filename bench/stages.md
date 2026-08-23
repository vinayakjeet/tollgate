# Stage boundaries

What every span in Tollgate starts and ends at, and exactly which of them
`x-tollgate-overhead-ms` sums.

This file is hashed into `stages.sha256` and the hash is checked by the bench
harness. The reason is not ceremony. This project publishes its own overhead as a
headline number, and a measurement whose boundaries can move after a result is seen
is not a measurement. A sibling project published a session latency of 1184ms
against a provider's own 559ms because its span wrapped a concurrency semaphore, so
half the figure was queueing reported as work. Writing the boundaries down first is
the cheapest defence against repeating that.

Changing this file is allowed. Changing it without re-running every published number
is not, which is what the hash check enforces.

## The span tree

```
tollgate.request                    one per HTTP request, parent of everything below
  tollgate.cache.probe
    tollgate.cache.exact
    tollgate.cache.semantic
  tollgate.route
  tollgate.budget
  tollgate.select
  tollgate.dispatch
    tollgate.dispatch.wait
    tollgate.dispatch.upstream
  tollgate.meter
```

## Boundaries

| Span | Starts at | Ends at |
|---|---|---|
| `tollgate.request` | The first line of the route handler, after FastAPI has parsed and validated the body | Immediately before the response object is handed back to FastAPI |
| `tollgate.cache.probe` | Before the L1 lookup is issued | After L2 returns a hit or both layers have missed |
| `tollgate.cache.exact` | Before the Redis GET | After the Redis GET returns or errors |
| `tollgate.cache.semantic` | Before the embedding is computed | After the pgvector query returns. Embedding time is inside this span, deliberately, because it is unavoidable cost of a semantic lookup |
| `tollgate.route` | Before the router is asked for a tier | After the tier is returned |
| `tollgate.budget` | Before the first counter read | After the last counter read for this request |
| `tollgate.select` | Before the fallback chain is walked | After a provider is chosen or the chain is exhausted |
| `tollgate.dispatch` | Before the first attempt is made | After the final attempt returns or the last retry is exhausted |
| `tollgate.dispatch.wait` | Before a throttle sleep or a retry backoff sleep begins | When that sleep ends. One span per sleep, so a request with three retries has three |
| `tollgate.dispatch.upstream` | Immediately before the HTTP request is written to the provider socket | Immediately after the provider's response body is fully read. One span per attempt |
| `tollgate.meter` | Before the metering row is constructed | After it is persisted or queued |

## What `x-tollgate-overhead-ms` sums

**Included:** `cache.probe`, `route`, `budget`, `select`, `meter`, and the handler
time not covered by any child span (argument marshalling, response assembly).

**Excluded:** `dispatch.upstream` and `dispatch.wait`.

Equivalently, and this is how the code computes it, overhead is the wall time inside
`tollgate.request` minus the wall time inside `tollgate.dispatch`.

### Why `dispatch.wait` is excluded, and why it is still published

A throttle sleep and a retry backoff are Tollgate deliberately not sending traffic,
because the provider asked it not to. It is not compute Tollgate spends, so counting
it as gateway overhead would make the gateway look slow for obeying a rate limit
correctly.

It is still wall time the caller waits, so burying it inside "upstream" would flatter
Tollgate in the other direction: a gateway that sleeps four seconds and reports
0.4ms of overhead has told the truth and communicated a lie.

So `dispatch.wait` is a published headline of its own, beside overhead and upstream,
and the three sum to the request duration. Any report that shows overhead without
showing wait beside it is incomplete.

### Why `cache.probe` is included

A cache probe is work Tollgate chose to do. When it hits, it replaces the upstream
call and the trade is obviously good. When it misses, it is pure added latency on the
critical path, and that is exactly the cost a reader needs to see. Excluding it would
hide the one number that makes a cache an engineering decision rather than a free
win.

## The five facts every published overhead figure carries

Named here so the harness can refuse to print a number that is missing one.

1. **Upstream:** mock, network_mock, or a named real provider.
2. **Streaming:** yes or no. Never averaged across both.
3. **Callbacks:** metering and tracing on or off.
4. **Warmth:** cold process or warm. The first request through a fresh process pays
   import and connection setup that no steady-state figure should carry.
5. **Load shape:** steady-state at a stated request rate, or saturation.

## Turn one is not turn two

The first request through a fresh process pays connection setup that later requests
do not. Every table reports p50 and p95 both including and excluding the first
request of each run, rather than picking whichever is kinder.
