"""Tollgate's status number: gateway overhead, measured the LiteLLM way.

Every overhead figure this repo publishes comes from this script, which is why
it enforces the five facts from bench/stages.md on every row and refuses to
print a number that cannot say what it measured:

    uv run python bench/overhead.py            # print the tables
    uv run python bench/overhead.py --write    # regenerate README Benchmarks

Method, and what it does not measure: client and gateway share one process over
an ASGI transport, so no row includes client-to-gateway network. The two
upstreams isolate different halves of the path. The in-process mock has no
network at all, so its numbers are Tollgate's code alone; network_mock adds a
real socket round trip to a fixed-delay server, so the gap between their rows is
transport, not gateway. Callbacks (metering writes, span export) are off for
headline rows because that is how the published references were measured.

A non-streaming total folds time-to-first-token into latency and flatters the
gateway, which is why streaming rows exist separately and carry their own TTFT,
live versus cached-replay, never averaged together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A load run that logs every request measures its own logging. Quieted before
# the app imports so configure_logging picks the level up.
import logging  # noqa: E402
import os  # noqa: E402

os.environ.setdefault("LOG_LEVEL", "ERROR")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.gateway import Gateway  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.metering import NullMeteringStore  # noqa: E402
from bench.network_mock import NetworkMock  # noqa: E402
from llm.providers import registry as registry_module  # noqa: E402
from llm.providers.base import OpenAICompatibleProvider  # noqa: E402

FACT_KEYS = ("upstream", "streaming", "callbacks", "warmth", "load")

STEADY_CONCURRENCY = 4
STEADY_REQUESTS = 200
SATURATION_CONCURRENCY = 32
SATURATION_REQUESTS = 250
WARMUP_REQUESTS = 20

# Published p99 added-latency figures these rows sit beside, for scale only.
# None was measured on this machine and none pretends otherwise.
PUBLISHED_REFERENCES = [
    ("LiteLLM Rust gateway, July 2026 post", 0.7),
    ("Portkey, same source", 2.3),
    ("Bifrost v1.6.4, same source", 4.5),
    ("Legacy LiteLLM Python path", 257.7),
]

VARIANCE_BUDGET_MS = 1.0


@dataclass
class Stats:
    total_ms: list[float] = field(default_factory=list)
    overhead_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    cache_outcomes: list[str] = field(default_factory=list)
    max_inflight: int = 0


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def fmt_pct(values: list[float]) -> str:
    return (
        f"p50 {percentile(values, 50):7.3f}  "
        f"p95 {percentile(values, 95):7.3f}  "
        f"p99 {percentile(values, 99):7.3f}"
    )


def variance_gate() -> None:
    """The in-process mock's own spread across 1,000 calls must sit under one
    millisecond before anything may be measured with it (M5.1's acceptance)."""
    from llm.providers.mock import MockProvider
    from llm.types import ChatMessage

    provider = MockProvider()
    messages = [ChatMessage(role="user", content="variance probe")]
    samples: list[float] = []
    loop = asyncio.new_event_loop()
    try:
        for _ in range(1000):
            t0 = time.perf_counter()
            loop.run_until_complete(provider.chat_completion(messages))
            samples.append((time.perf_counter() - t0) * 1000)
    finally:
        loop.close()

    spread = percentile(samples, 99) - statistics.median(samples)
    verdict = "PASS" if spread < VARIANCE_BUDGET_MS else "FAIL"
    print(
        f"variance gate: {len(samples)} in-process mock calls, "
        f"p99-median {spread:.4f}ms (budget {VARIANCE_BUDGET_MS}ms) ... {verdict}\n"
    )
    if spread >= VARIANCE_BUDGET_MS:
        print("fixture noise exceeds the effect under measurement; refusing to publish.")
        raise SystemExit(1)


def build_gateway(chain_head: str) -> Gateway:
    """A gateway wired for measurement: null metering (callbacks off), clean
    local counters, chain pinned to the upstream under test."""
    gateway = Gateway.from_settings(get_settings(), metering=NullMeteringStore())
    gateway.chain = [chain_head]
    fastapi_app.state.gateway = gateway
    return gateway


def register_network_mock(mock: NetworkMock) -> None:
    import os

    os.environ.setdefault("NETWORK_MOCK_KEY", "bench-only")
    # One shared, pooled client for the whole run: a fresh AsyncClient per
    # request leaves hundreds of TIME_WAIT sockets behind on Windows, which is
    # how the first bench attempt died.
    shared_client = httpx.AsyncClient(
        base_url=mock.base_url,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=60.0),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )
    registry_module._PROVIDERS["network_mock"] = OpenAICompatibleProvider(
        name="network_mock",
        base_url=mock.base_url,
        api_key_env="NETWORK_MOCK_KEY",
        default_model="network-mock",
        client=shared_client,
    )


async def drive_stream(
    client: httpx.AsyncClient, body: dict, stats: Stats, inflight: list[int]
) -> None:
    inflight[0] += 1
    stats.max_inflight = max(stats.max_inflight, inflight[0])
    t0 = time.perf_counter()
    ttft: float | None = None
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200, await resp.aread()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    frame = json.loads(line[6:])
                    if "tollgate_overhead_ms" in frame:
                        stats.overhead_ms.append(float(frame["tollgate_overhead_ms"]))
                        stats.cache_outcomes.append(str(frame.get("tollgate_cache", "")))
                elif line and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
    finally:
        inflight[0] -= 1
    stats.total_ms.append((time.perf_counter() - t0) * 1000)
    if ttft is not None:
        stats.ttft_ms.append(ttft)


async def drive_plain(
    client: httpx.AsyncClient, body: dict, stats: Stats, inflight: list[int]
) -> None:
    inflight[0] += 1
    stats.max_inflight = max(stats.max_inflight, inflight[0])
    t0 = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=body)
    elapsed = (time.perf_counter() - t0) * 1000
    inflight[0] -= 1
    assert resp.status_code == 200, resp.text
    stats.total_ms.append(elapsed)
    stats.overhead_ms.append(float(resp.headers["x-tollgate-overhead-ms"]))
    stats.cache_outcomes.append(resp.headers["x-tollgate-cache"])


async def run_load(
    client: httpx.AsyncClient, bodies: list[dict], stream: bool, concurrency: int
) -> tuple[Stats, float]:
    stats = Stats()
    inflight = [0]
    cursor = {"i": 0}
    wall_start = time.perf_counter()

    async def worker():
        while True:
            i = cursor["i"]
            if i >= len(bodies):
                return
            cursor["i"] = i + 1
            driver = drive_stream if stream else drive_plain
            await driver(client, bodies[i], stats, inflight)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - wall_start
    return stats, wall


def make_body(i: int, stream: bool) -> dict:
    """Half unique prompts, half byte-identical repeats, so streaming rows can
    report live-versus-cached-replay TTFT from the same run (M3.3's disclosure)."""
    unique = i % 2 == 0
    content = f"bench prompt {i}" if unique else f"bench prompt {i - 1}"
    return {
        "model": "bench/model",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


async def measure_cell(upstream: str, stream: bool, load: str):
    build_gateway(upstream)
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tollgate") as client:
        for k in range(WARMUP_REQUESTS):
            warm = make_body(10_000 + k, False)
            resp = await client.post("/v1/chat/completions", json=warm)
            assert resp.status_code == 200

        n = STEADY_REQUESTS if load == "steady" else SATURATION_REQUESTS
        conc = STEADY_CONCURRENCY if load == "steady" else SATURATION_CONCURRENCY
        bodies = [make_body(k, stream) for k in range(n)]
        stats, wall = await run_load(client, bodies, stream, conc)

    rps = len(stats.total_ms) / wall
    facts = {
        "upstream": upstream,
        "streaming": "yes" if stream else "no",
        "callbacks": "off",
        "warmth": "warm",
        "load": f"{load} c={conc}",
    }
    missing = [k for k in FACT_KEYS if not facts[k]]
    assert not missing, f"missing facts {missing}: refusing to publish a number"
    return facts, stats, rps


def render_row(name: str, facts: dict, stats: Stats, rps: float) -> str:
    facts_str = ", ".join(f"{k}={facts[k]}" for k in FACT_KEYS)
    return (
        f"| {name} | {rps:8.1f} | {fmt_pct(stats.overhead_ms)} | "
        f"{stats.max_inflight:>3} | {facts_str} |"
    )


def ttft_note(label: str, stats: Stats) -> str:
    live = [t for t, o in zip(stats.ttft_ms, stats.cache_outcomes, strict=True) if o != "exact"]
    cached = [t for t, o in zip(stats.ttft_ms, stats.cache_outcomes, strict=True) if o == "exact"]
    if not live or not cached:
        return ""
    return (
        f"\nTTFT {label}: live median {statistics.median(live):.3f}ms, "
        f"cached-replay median {statistics.median(cached):.3f}ms. A cached hit "
        "skips dispatch entirely; the gap is disclosed rather than hidden."
    )


CELLS = [
    ("in-process mock, non-streaming", "mock", False),
    ("in-process mock, streaming", "mock", True),
    ("network_mock, non-streaming", "network_mock", False),
    ("network_mock, streaming", "network_mock", True),
]


async def full_run(lines: list[str]) -> str:
    notes = ""
    mock = NetworkMock(delay_s=0.02)
    try:
        for phase_cells in (CELLS[:2], CELLS[2:]):
            if phase_cells[0][1] == "network_mock":
                mock.start()
                register_network_mock(mock)
            for name, upstream, stream in phase_cells:
                facts, stats, rps = await measure_cell(upstream, stream, "steady")
                lines.append(render_row(name, facts, stats, rps))
                # Saturation replays the same cell shape under pressure, so its
                # five facts say streaming=yes where the cell does.
                sat_facts, sat_stats, sat_rps = await measure_cell(
                    upstream, stream, "saturation"
                )
                lines.append(
                    render_row(f"{name}, saturation", sat_facts, sat_stats, sat_rps)
                )
                if stream:
                    notes += ttft_note(f"{name}", stats)
    finally:
        mock.stop()
    return notes


TABLE_HEADER = (
    "| cell | RPS | overhead ms p50/p95/p99 | max in-flight | five facts |\n"
    "|---|---|---|---|---|"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate README section")
    args = parser.parse_args()

    # Same reason as NetworkMock's own thread: the Proactor loop has failed
    # transport allocation under this run's connect churn before. Selector
    # everywhere, for the main loop and the server thread alike.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    variance_gate()
    assert not get_settings().otel_exporter_otlp_endpoint, (
        "tracing must be off for headline rows; unset OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    lines: list[str] = []
    notes = asyncio.run(full_run(lines))

    out = ["### Overhead against deterministic upstreams\n", TABLE_HEADER, *lines, ""]
    if notes:
        out.append(notes.strip("\n") + "\n")
    out.append("Published p99 added-latency figures for scale (none measured here):")
    for label, value in PUBLISHED_REFERENCES:
        out.append(f"- {label}: ~{value} ms")
    out.append(
        "\nExcluded from every figure: client-to-gateway network (same-process ASGI), "
        "provider behaviour beyond fixed delays, any hardware but this one. Streaming "
        "and non-streaming rows are separate on purpose; a non-streaming total folds "
        "TTFT into latency and flatters the gateway."
    )

    report = "\n".join(out) + "\n"
    print(report)

    if args.write:
        readme_path = REPO_ROOT / "README.md"
        text = readme_path.read_text(encoding="utf-8")
        begin, end = "<!-- bench:begin -->", "<!-- bench:end -->"
        if begin not in text or end not in text:
            raise SystemExit("README lacks bench markers; nothing was written.")
        pre, _, rest = text.partition(begin)
        _, _, post = rest.partition(end)
        readme_path.write_text(f"{pre}{begin}\n{report}{end}{post}", encoding="utf-8")
        print("README Benchmarks regenerated.")


if __name__ == "__main__":
    main()

