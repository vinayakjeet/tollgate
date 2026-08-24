from __future__ import annotations

import json

from app.metering import (
    JsonlMeteringStore,
    MeterRow,
    disagreement_summary,
    now_row,
)


def _row(kind: str, provider: str, request_id: str = "", status: int | None = None) -> MeterRow:
    return now_row(
        kind,
        provider,
        request_id=request_id,
        status=status,
        detail="",
        model="m",
        cache_outcome=None,
        tokens_in=1,
        tokens_out=2,
        overhead_ms=0.5,
    )


async def test_jsonl_store_round_trips_rows(tmp_path):
    store = JsonlMeteringStore(tmp_path / "metering.jsonl")
    row = _row("request", "groq", request_id="r1", status=200)
    await store.append(row)

    loaded = JsonlMeteringStore(tmp_path / "metering.jsonl").read_all()
    assert len(loaded) == 1
    assert loaded[0].kind == "request"
    assert loaded[0].provider == "groq"
    assert loaded[0].tokens_out == 2


async def test_jsonl_store_survives_a_corrupt_line(tmp_path):
    """A half-written line from a killed process must cost one row, not the file."""
    path = tmp_path / "metering.jsonl"
    path.write_text('{"ts": 1, "kind": "request", "provider": "x"}\nnot json at all\n')
    rows = JsonlMeteringStore(path).read_all()
    assert len(rows) == 1


async def test_concurrent_appends_do_not_splice_lines(tmp_path):
    import asyncio

    store = JsonlMeteringStore(tmp_path / "metering.jsonl")
    await asyncio.gather(
        *[store.append(_row("request", f"p{i}", status=200)) for i in range(20)]
    )
    lines = (tmp_path / "metering.jsonl").read_text().splitlines()
    assert len(lines) == 20
    assert all(json.loads(line) for line in lines)


class TestDisagreementSummary:
    """M1.5's acceptance: the store answers "how often was the estimate wrong,
    in each direction" without a bespoke query."""

    def test_trip_counts_a_provider_the_estimate_called_healthy(self):
        rows = [
            _row("request", "cerebras", request_id="r1", status=200),
            _row("trip", "groq", request_id="r1"),
        ]
        summary = disagreement_summary(rows)
        assert summary.trips == 1
        assert summary.false_healthy_rate is not None and summary.false_healthy_rate > 0

    def test_skip_before_success_counts_as_probably_wrong_exhaustion(self):
        rows = [
            _row("skip", "groq", request_id="r1"),
            _row("request", "cerebras", request_id="r1", status=200),
        ]
        summary = disagreement_summary(rows)
        assert summary.skips == 1
        assert summary.skips_before_success == 1
        assert summary.false_exhausted_rate == 1.0

    def test_skip_before_failure_is_not_counted_against_the_estimate(self):
        rows = [
            _row("skip", "groq", request_id="r2"),
            _row("trip", "cerebras", request_id="r2"),
        ]
        summary = disagreement_summary(rows)
        assert summary.false_exhausted_rate == 0.0

    def test_no_data_yields_none_not_zero(self):
        summary = disagreement_summary([])
        assert summary.false_healthy_rate is None
        assert summary.false_exhausted_rate is None


def test_kind_partitioning():
    rows = (
        [_row("request", "a", status=200)] * 3
        + [_row("trip", "b")]
        + [_row("skip", "c")] * 2
    )
    summary = disagreement_summary(rows)
    assert summary.requests == 3
    assert summary.trips == 1
    assert summary.skips == 2
