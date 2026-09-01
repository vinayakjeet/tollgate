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


class NoSnapshot(RuntimeError):
    """No pinned snapshot to load, with enough detail to fix it in one command."""


def resolve_snapshot(model_dir: Path | str, models_root: Path | None = None) -> Path:
    """Find the pinned snapshot, whatever it happens to be called.

    `scripts/fetch_embedding_model.py` writes to `<repo>__<revision>`, because a
    directory that does not name its revision is a directory that silently
    measures one model this week and another next week. Its callers all pointed at
    an unsuffixed `models/all-MiniLM-L6-v2`, so the fetch script and everything
    that consumes it had never been run in the same place: the fetch reported
    success, the bench reported no model, and both were correct.

    An explicit directory still wins when it exists, so a caller that pins its own
    path keeps working. Otherwise the single snapshot under `models/` is used, and
    two snapshots are an error rather than a coin flip.
    """
    explicit = Path(model_dir)
    if (explicit / "config.json").exists():
        return explicit

    root = models_root or explicit.parent
    if root.is_dir():
        found = sorted(p for p in root.iterdir() if (p / "config.json").exists())
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = ", ".join(p.name for p in found)
            raise NoSnapshot(
                f"{len(found)} snapshots under {root}, so which one produced a number "
                f"would be ambiguous: {names}. Point embedding_model_dir at one."
            )
    raise NoSnapshot(
        f"no pinned snapshot under {root}. Fetch one with: uv run python "
        "scripts/fetch_embedding_model.py --repo sentence-transformers/"
        "all-MiniLM-L6-v2 --revision <sha>"
    )


def snapshot_label(path: Path) -> str:
    """How a snapshot names itself in a report.

    Carries the revision, because pinning a model and then publishing a number
    against an unversioned name throws away the thing the pin was for.
    """
    name = path.name
    if "__" in name:
        repo, _, revision = name.rpartition("__")
        return f"local:{repo.replace('__', '/')}@{revision[:12]}"
    return f"local:{name}"


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
        self.path = resolve_snapshot(model_dir)
        self._model = SentenceTransformer(str(self.path))

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
