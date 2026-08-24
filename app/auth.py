"""The one key at the edge.

Threat model, in short: Tollgate fronts free-tier quota that the whole portfolio
shares. An open endpoint means anyone who finds the URL spends the budget every
downstream project depends on, and reads cached answers to prompts it did not
send. One shared bearer key is the right size for a single-tenant deployment;
per-caller keys are a multi-tenant feature and SPEC names multi-tenancy a
non-goal.
"""

from __future__ import annotations

import hmac

import structlog
from fastapi import HTTPException, Request

from app.config import get_settings

logger = structlog.get_logger(__name__)

_warned = False


def maybe_warn_unauthenticated() -> None:
    """Say once, loudly, when the edge is open: an unauthenticated gateway must
    be a decision someone made, never a default nobody noticed."""
    global _warned
    if _warned or get_settings().edge_api_key:
        return
    _warned = True
    logger.warning(
        "auth.edge_disabled",
        detail="set EDGE_API_KEY before exposing this service beyond localhost",
    )


async def require_edge_key(request: Request) -> None:
    key = get_settings().edge_api_key
    if not key:
        return
    supplied = request.headers.get("authorization", "")
    # Compare in constant time so a wrong guess cannot be timed against the real
    # key character by character.
    if supplied and hmac.compare_digest(supplied.encode(), f"Bearer {key}".encode()):
        return
    raise HTTPException(
        status_code=401,
        detail={"error": {"message": "missing or invalid API key", "type": "auth_error"}},
    )
