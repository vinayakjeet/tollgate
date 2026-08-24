"""Author the replay workload: 1,000 requests, five categories, duplicates at a
rate that is written down before any hit rate is measured (BACKLOG M6.1).

A cache hit rate is trivially inflatable by choosing a repetitive workload, so
the duplicate rate lives inside the artifact itself and travels with every
number quoted from replays of it. Per category: 100 unique bases, 60
byte-identical repeats, 40 paraphrases. That is more repetitive than human chat
and typical of service callers, whose retries and pipelines produce
byte-identical requests all day; this portfolio's callers are such services.

    uv run python bench/build_workload.py            # verify determinism
    uv run python bench/build_workload.py --write    # regenerate the artifact

Determinism is load-bearing: two runs must produce byte-identical files or the
content hash means nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "bench" / "workloads" / "replay-v1.jsonl"
SEED = 20260824

PER_CATEGORY = 200
UNIQUE_PER_CATEGORY = 100
REPEAT_COUNT = 60
PARAPHRASE_COUNT = 40

CATEGORIES = ("code", "math", "extraction", "summarisation", "open-ended")

SYNONYMS = [
    ("explain", "walk through"),
    ("write", "draft"),
    ("give me", "produce"),
    ("summarise", "condense"),
    ("extract", "pull out"),
    ("what is", "what's"),
    ("describe", "characterise"),
    ("compare", "contrast"),
]

CODE_TEMPLATES = [
    "explain what {thing} does in python",
    "write a function that validates {thing}",
    "explain how to test {thing} with pytest",
    "compare approaches to {thing} in a web service",
    "sketch a small module that wraps {thing}",
]
THINGS_CODE = [
    "list comprehensions",
    "context managers",
    "async generators",
    "dataclasses",
    "decorators",
    "iterators",
    "type hints",
    "exceptions",
    "thread pools",
    "queues",
    "locks",
    "semaphores",
    "logging setup",
    "retry helpers",
    "connection pools",
    "background tasks",
    "circuit breakers",
    "rate limiters",
    "config objects",
    "health checks",
]

ORGS = ["RBI", "SEBI", "Ministry of Agriculture", "NIC", "UIDAI"]
SCHEMES = ["PM-Kisan", "Ayushman Bharat", "PMAY", "UPI", "Jan Dhan"]

EXTRACT_TEMPLATES = [
    "extract the dates from this text: {text}",
    "extract all amounts mentioned here: {text}",
    "pull out the organisation names in: {text}",
    "list every identifier that appears in: {text}",
]

TOPICS = [
    "engineering",
    "platform",
    "data",
    "infrastructure",
    "product",
    "research",
    "security",
    "reliability",
    "payments",
    "mobile",
    "search",
    "ml",
    "compliance",
    "developer-experience",
    "support",
    "growth",
    "ops",
    "design-systems",
    "integrations",
    "billing",
]

OPEN_TEMPLATES = [
    ("is it better to specialise deeply or broadly early in a career?", False),
    ("why do {topic} teams keep rebuilding the same internal tools?", True),
    ("what makes a benchmark result in {topic} trustworthy?", True),
    ("when does caching make a {topic} system harder to reason about?", True),
    ("should a gateway hide which provider answered a {topic} request?", True),
    ("why do free tiers shape how {topic} systems get designed?", True),
    ("is measured honesty about negative results worth the embarrassment in {topic}?", True),
]


def _circular_text(rng: random.Random) -> str:
    return (
        f"{rng.choice(ORGS)} circular {rng.randint(100, 999)}/{rng.randint(1, 12):02d}."
        f"{rng.randint(100, 999):03d}, dated {rng.randint(1, 28)} "
        f"{rng.choice(['January', 'March', 'July', 'October'])} {rng.choice([2023, 2024, 2025])}, "
        f"caps {rng.choice(SCHEMES)} transactions at Rs {rng.randint(1, 99)},"
        f"{rng.choice(['000', '500'])} per day."
    )


def _summary_text(rng: random.Random) -> str:
    sentences = [
        "The free tier allows fifteen requests per minute and one thousand per day.",
        ("Under sustained traffic, latency degrades for hours even though isolated "
         "probes still answer quickly."),
        "A healthcheck alone therefore cannot be trusted to route production traffic.",
        "Vector caches match on cosine similarity above a configurable threshold.",
        "Defaults shipped by vendors sit far below what practitioners actually deploy.",
        "Fixed windows under-count near boundaries compared with rolling ones.",
        ("The team chose them anyway because predictably wrong beats subtly wrong."),
    ]
    picked = rng.sample(sentences, k=3)
    return " ".join(picked)


def _bases_for_category(category: str) -> list[str]:
    rng = random.Random(f"{SEED}-{category}")
    bases: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(bases) < UNIQUE_PER_CATEGORY:
        attempts += 1
        if attempts > 50_000:
            sys.exit(f"could not build {UNIQUE_PER_CATEGORY} unique prompts for {category}")

        if category == "code":
            candidate = rng.choice(CODE_TEMPLATES).format(thing=rng.choice(THINGS_CODE))
        elif category == "math":
            a, b = rng.randint(3, 9999), rng.randint(2, 999)
            price, discount = rng.randint(100, 99999), rng.randint(5, 70)
            principal, rate = rng.randint(1000, 999999), rng.randint(2, 18)
            candidate = rng.choice(
                [
                    f"what is {a} times {b}?",
                    f"a train covers {rng.randint(20, 900)} km in "
                    f"{rng.randint(2, 40)} hours. average speed?",
                    f"a shop discounts a {price} rupee item by {discount} percent. final price?",
                    f"compound interest on {principal} rupees at {rate} percent for one year?",
                ]
            )
        elif category == "extraction":
            candidate = rng.choice(EXTRACT_TEMPLATES).format(text=_circular_text(rng))
        elif category == "summarisation":
            candidate = f"summarise this paragraph in two sentences: {_summary_text(rng)}"
        else:
            template, slot = OPEN_TEMPLATES[rng.randrange(len(OPEN_TEMPLATES))]
            candidate = template.format(topic=rng.choice(TOPICS)) if slot else template
        if candidate in seen:
            continue
        seen.add(candidate)
        bases.append(candidate)
    return bases


def _paraphrase(text: str) -> str:
    for original, replacement in SYNONYMS:
        if original in text:
            return text.replace(original, replacement, 1)
    return f"rephrase request: {text}"


def build() -> list[dict]:
    rows: list[dict] = []
    serial = 0
    for category in CATEGORIES:
        bases = _bases_for_category(category)
        pool = list(range(len(bases)))
        picks_repeat = [pool[i % len(pool)] for i in range(REPEAT_COUNT)]
        picks_para = [pool[(i + REPEAT_COUNT) % len(pool)] for i in range(PARAPHRASE_COUNT)]

        category_rows: list[dict] = []
        for base in bases:
            category_rows.append(
                {"kind": "base", "category": category, "text": base, "dup_of": None}
            )
        for src in picks_repeat:
            category_rows.append(
                {"kind": "repeat", "category": category, "text": bases[src], "dup_of": src}
            )
        for src in picks_para:
            category_rows.append(
                {
                    "kind": "paraphrase",
                    "category": category,
                    "text": _paraphrase(bases[src]),
                    "dup_of": src,
                }
            )

        order = list(range(len(category_rows)))
        random.Random(f"{SEED}-order-{category}").shuffle(order)
        for idx in order:
            row = dict(category_rows[idx])
            row["id"] = f"{category}-{serial:05d}"
            serial += 1
            rows.append(row)

    assert len(rows) == PER_CATEGORY * len(CATEGORIES)
    return rows


def content_hash(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def render() -> str:
    rows = build()
    digest = content_hash(rows)
    meta = {
        "meta": {
            "name": "tollgate-replay-v1",
            "seed": SEED,
            "n_requests": len(rows),
            "categories": list(CATEGORIES),
            "per_category": PER_CATEGORY,
            "composition_per_category": {
                "unique_base": UNIQUE_PER_CATEGORY,
                "repeat_byte_identical": REPEAT_COUNT,
                "paraphrase": PARAPHRASE_COUNT,
                "note": (
                    "these counts cap any hit rate measured against this file; "
                    "the caps travel with every number quoted from its replays"
                ),
            },
            "content_hash_sha256": digest,
        }
    }
    lines = [json.dumps(meta, sort_keys=True)]
    lines += [json.dumps(r, sort_keys=True) for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the artifact")
    args = parser.parse_args()

    rendered = render()
    if not args.write:
        if ARTIFACT.exists():
            print("artifact matches script:", ARTIFACT.read_text(encoding="utf-8") == rendered)
        print(rendered.splitlines()[0])
        return

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(rendered, encoding="utf-8")
    print(f"wrote {ARTIFACT}")
    print(rendered.splitlines()[0])


if __name__ == "__main__":
    main()
