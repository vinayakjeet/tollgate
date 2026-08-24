"""Embeddings for the semantic cache, computed locally.

SPEC rejects an embedding API call outright: it would put a network round trip
and a second quota inside the one path that has to be faster than the upstream
call it is trying to avoid. The real backend loads a small sentence-transformers
model fetched by `scripts/fetch_embedding_model.py`; the stub exists for tests
and for demos that must not download weights.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path


class EmbeddingBackend:
    def embed(self, text: str) -> list[float]: ...


class HashingStubEmbedder(EmbeddingBackend):
    """Deterministic character-ngram feature hashing. Similarity is real enough
    to exercise thresholds in tests (shared substrings score high), and dishonest
    enough to never ship as a production similarity signal: it knows nothing about
    meaning, only about shared trigrams. Named stub so it cannot sneak past review."""

    def __init__(self, dims: int = 256) -> None:
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        for i in range(len(normalized) - 2):
            ngram = normalized[i : i + 3]
            digest = hashlib.md5(ngram.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self._dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class LocalEmbedder(EmbeddingBackend):
    """A pinned sentence-transformers snapshot from `models/`, loaded lazily so
    the base install (and every CI run) never pays for torch."""

    def __init__(self, model_dir: Path | str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "embedding_backend=local needs the 'semantic' dependency group: "
                "uv sync --group semantic"
            ) from exc
        self._model = SentenceTransformer(str(model_dir))

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
