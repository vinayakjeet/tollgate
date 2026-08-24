"""Per-request metering, appended to a store that survives restarts.

The metering table is the multi-day trace (SPEC proof artifact 1) and the raw
material for the estimate-versus-provider disagreement rate (BACKLOG M1.5), so it
is written even when every other subsystem is degraded: a metering failure must
cost a log line, never a user-facing error.

The default store appends JSONL locally because Neon credentials are an M0.1/M0.6
affair and the disagreement study needs data before it needs Postgres. The
Postgres-backed store joins when those credentials exist; both implement
`MeteringStore`, and `disagreement_summary` runs against either.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MeterRow:
    """One line of history about one request or one routing decision inside it.

    Kinds: `request` (a served response, success or mapped failure), `skip`
    (selection passed this provider over), `trip` (the provider refused with a
    429 that the estimate did not predict). The disagreement rate is trips
    divided by requests against skips whose chain went on to succeed.
    """

    ts: float
    kind: str
    provider: str
    detail: str = ""
    request_id: str = ""
    model: str | None = None
    cache_outcome: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    overhead_ms: float | None = None
    status: int | None = None
    tags: dict[str, str] = field(default_factory=dict)


def new_request_id() -> str:
    return uuid.uuid4().hex


class MeteringStore(Protocol):
    async def append(self, row: MeterRow) -> None: ...

    def read_all(self) -> list[MeterRow]: ...


class JsonlMeteringStore:
    """Append-only JSONL, one row per line, readable by anything.

    Writes go through a lock because uvicorn dispatches concurrent requests onto
    one event loop and interleaved `write`s would splice two rows together.
    """

    def __init__(self, path: Path | str = "metering.jsonl") -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, row: MeterRow) -> None:
        line = json.dumps(asdict(row), separators=(",", ":"))
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_all(self) -> list[MeterRow]:
        if not self._path.exists():
            return []
        rows: list[MeterRow] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                rows.append(
                    MeterRow(
                        **{k: v for k, v in payload.items() if k in MeterRow.__dataclass_fields__}
                    )
                )
            except (json.JSONDecodeError, TypeError):
                continue
        return rows


class NullMeteringStore:
    """For load runs where metering would be the measured thing."""

    async def append(self, row: MeterRow) -> None:
        return None

    def read_all(self) -> list[MeterRow]:
        return []


@dataclass(frozen=True)
class DisagreementSummary:
    requests: int
    trips: int
    skips: int
    skips_before_success: int

    @property
    def false_healthy_rate(self) -> float | None:
        """Trips over opportunities to trip: how often the estimate called a
        provider healthy that then refused. None when nothing was measured."""
        opportunities = self.requests + self.trips
        return self.trips / opportunities if opportunities else None

    @property
    def false_exhausted_rate(self) -> float | None:
        """Skips where the chain went on to succeed, over all skips. An upper
        bound on wrongly-exhausted calls: the skipped provider would probably
        have answered, though only a post-hoc probe could prove it."""
        return self.skips_before_success / self.skips if self.skips else None


def disagreement_summary(rows: list[MeterRow]) -> DisagreementSummary:
    """Answer "how often was the estimate wrong, in each direction" in one pass."""
    requests = sum(1 for r in rows if r.kind == "request")
    trips = sum(1 for r in rows if r.kind == "trip")
    skips = [r for r in rows if r.kind == "skip"]
    succeeded_requests = {
        r.request_id for r in rows if r.kind == "request" and (r.status or 0) < 400
    }
    skips_before_success = sum(1 for r in skips if r.request_id in succeeded_requests)
    return DisagreementSummary(
        requests=requests,
        trips=trips,
        skips=len(skips),
        skips_before_success=skips_before_success,
    )


def now_row(kind: str, provider: str, **fields: object) -> MeterRow:
    return MeterRow(ts=time.time(), kind=kind, provider=provider, **fields)  # type: ignore[arg-type]
