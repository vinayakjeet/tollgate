from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.budget import BudgetTracker
from app.gateway import Gateway
from llm.providers.registry import known_providers, quota_verification

router = APIRouter(tags=["budget"])


def _limit_payload(state, last_verified: str | None) -> dict:
    """One limit's public shape. `remaining: null` means the limit itself is
    unknown; `0` means it is known and spent. Collapsing one into the other would
    tell a caller to back off a provider that has plenty left."""
    return {
        "limit": state.limit,
        "used": state.used,
        "remaining": state.remaining,
        "window_start": state.window.start,
        "window_end": state.window.end,
        "reset_in_s": round(state.window.reset_in, 1),
        "last_verified": last_verified,
    }


@router.get("/budget")
async def budget(request: Request) -> dict:
    gateway: Gateway = request.app.state.gateway
    tracker: BudgetTracker = gateway.tracker

    providers = {}
    for name in sorted(known_providers()):
        estimate = await tracker.estimate(name)
        verified = quota_verification(name)
        providers[name] = {
            "rpm": _limit_payload(estimate.requests_per_minute, verified.get("rpm")),
            "tpm": _limit_payload(estimate.tokens_per_minute, verified.get("tpm")),
            "rpd": _limit_payload(estimate.requests_per_day, verified.get("rpd")),
            "degraded": estimate.degraded,
        }

    return {"as_of": time.time(), "providers": providers}
