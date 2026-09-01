"""What fraction of the replay workload each cache layer catches, measured, with
variance across runs (BACKLOG M2.2 and M2.5).

Replays bench/workloads/replay-v1.jsonl through a real ResponseCache in three
configurations: L1 alone, L2 alone (the exact layer disabled), and both. The
published expectation for L1 on service traffic is 15 to 30 percent; this
workload's own composition caps it at 30. L2's published number is its marginal
gain over L1, never the combined rate, because the combined figure is the one
that flatters.

    uv run python bench/hit_rate.py --runs 3

L2 rows need an embedding model under models/. With --allow-stub they run on
the trigram stub and print labelled as such; without weights and without that
flag, the script prints the refusal rather than a number it cannot stand
behind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.caches import CacheEntry, InMemoryVectorIndex, LocalKVBackend, ResponseCache  # noqa: E402
from app.embeddings import HashingStubEmbedder, LocalEmbedder, snapshot_label  # noqa: E402

WORKLOAD = REPO_ROOT / "bench" / "workloads" / "replay-v1.jsonl"


class NullKV(LocalKVBackend):
    """L2-alone configuration: the exact layer is present but always misses."""

    async def get(self, key: str) -> bytes | None:
        return None


def load_workload() -> tuple[dict, list[dict]]:
    lines = WORKLOAD.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    rows = [json.loads(line) for line in lines[1:]]
    return meta, rows


def build_cache(mode: str, threshold: float, embedder) -> ResponseCache:
    kwargs: dict = dict(salt="hit-rate-bench", threshold=threshold)
    if mode in ("l2_alone", "both"):
        kwargs.update(embedder=embedder, index=InMemoryVectorIndex())
    kwargs["kv"] = NullKV() if mode == "l2_alone" else LocalKVBackend()
    return ResponseCache(**kwargs)


def entry_for(row: dict) -> CacheEntry:
    return CacheEntry(
        text=f"answer to: {row['text'][:40]}",
        provider="mock",
        model="mock-echo",
        tokens_in=8,
        tokens_out=12,
        cost_usd=0.0,
    )


async def run_pass(cache: ResponseCache, rows: list[dict]) -> dict[str, int]:
    counts = {"exact": 0, "semantic": 0, "miss": 0}
    for row in rows:
        fields = {
            "model": "mock/demo",
            "messages": [{"role": "user", "content": row["text"]}],
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
        }
        lookup = await cache.lookup(fields)
        if lookup.outcome == "miss":
            await cache.store(lookup, entry_for(row))
        counts[lookup.outcome] += 1
    return counts


def fmt_row(label: str, per_run: list[dict[str, int]], n: int) -> str:
    def cell(kind: str) -> str:
        rates = [c[kind] / n for c in per_run]
        if not any(rates):
            return "n/a"
        mean = statistics.mean(rates)
        stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
        return f"{mean * 100:5.1f}% +- {stdev * 100:.1f}"

    return f"| {label} | {cell('exact')} | {cell('semantic')} | {cell('miss')} | n={len(per_run)} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="run L2 rows on the trigram stub embedder; labelled as such",
    )
    args = parser.parse_args()

    meta, rows = load_workload()
    composition = meta["composition_per_category"]
    print(
        f"workload: {WORKLOAD.name}  hash={meta['content_hash_sha256'][:16]}...  "
        f"n={meta['n_requests']}"
    )
    print(
        f"duplicate caps per category: repeat={composition['repeat_byte_identical']} "
        f"(30% of traffic, the L1 ceiling), paraphrase={composition['paraphrase']} (20%)"
    )

    stub_labelled = False
    try:
        embedder = LocalEmbedder(REPO_ROOT / "models" / "all-MiniLM-L6-v2")
        # The label carries the revision the snapshot directory names, so a
        # published hit rate says which weights produced it.
        embedder_label = snapshot_label(embedder.path)
    except Exception:
        if not args.allow_stub:
            print(
                "\nno embedding model under models/; refusing to print an L2 row "
                "rather than guessing. fetch one with scripts/"
                "fetch_embedding_model.py --repo sentence-transformers/"
                "all-MiniLM-L6-v2 --revision <sha>, or pass --allow-stub."
            )
            embedder, embedder_label = None, "none"
        else:
            embedder = HashingStubEmbedder()
            embedder_label = "trigram-stub (not a similarity signal)"
            stub_labelled = True

    print(
        "\nfacts: upstream=in-process mock | streaming=no | metering=off | "
        f"tracing=off | warm | sequential replay | embedder={embedder_label}"
    )
    print(f"threshold: {args.threshold}\n")
    print("| config | exact hit | semantic hit | miss | runs |")
    print("|---|---|---|---|---|")

    configs = [("l1_only", "L1 alone"), ("both", "L1+L2")]
    if embedder is not None:
        configs.insert(1, ("l2_alone", "L2 alone"))

    results: dict[str, list[dict[str, int]]] = {}
    for mode, label in configs:
        per_run = [
            asyncio.run(run_pass(build_cache(mode, args.threshold, embedder), rows))
            for _ in range(args.runs)
        ]
        results[label] = per_run
        print(fmt_row(label, per_run, len(rows)))

    if "L1 alone" in results and "L1+L2" in results:
        # Both figures read straight off the measured configs. Deriving one from
        # the other assumes layers partition traffic cleanly; they do not, since
        # a semantic hit can shadow what L1 would have caught.
        l1_rate = statistics.mean(c["exact"] for c in results["L1 alone"]) / len(rows)
        both = results["L1+L2"]
        combined = statistics.mean(c["exact"] + c["semantic"] for c in both) / len(rows)
        marginal = combined - l1_rate
        print(
            f"\ncombined rate {combined * 100:.1f}% against L1 alone at "
            f"{l1_rate * 100:.1f}%: L2's marginal gain is {marginal * 100:+.1f} "
            "points. The marginal figure is the finding; the combined one is what "
            "a headline would quote."
        )
    if stub_labelled:
        print(
            "\nNOTE: those rows ran on the trigram stub. They exercise the "
            "machinery; they do not measure similarity."
        )


if __name__ == "__main__":
    main()
