from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import spans
from app.gateway import Gateway
from app.metering import new_request_id, now_row
from app.model_ref import parse_model_ref
from app.routing import Selection
from app.spans import stage_span
from llm.types import (
    ChatMessage,
    ChatResponse,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
)

router = APIRouter(prefix="/v1", tags=["openai"])

logger = structlog.get_logger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    # Null rather than zeros when the provider did not report token counts. Zeros
    # would be a number a caller could bill against, and a wrong one; null says the
    # provider did not tell us, which is the truth and is what the metering store
    # needs to be able to distinguish.
    usage: Usage | None = None


def _sampling_kwargs(request: ChatCompletionRequest) -> dict[str, object]:
    fields = ("temperature", "top_p", "max_tokens")
    return {name: value for name in fields if (value := getattr(request, name)) is not None}


async def _dispatch(
    gateway: Gateway, chain: list[str], messages: list[ChatMessage], kwargs: dict[str, object]
) -> tuple[ChatResponse | None, str]:
    """Walk the dispatching half of the chain.

    Rate limits and transient provider errors move down the chain; each one is a
    disagreement between the estimate that kept the provider in play and the
    provider itself, so both kinds land in the metering store. A 4xx from the
    provider is the caller's mistake and stops the walk, because re-sending the
    same bad request at another provider cannot fix it.
    """
    failure = ""
    for provider in chain:
        try:
            return await gateway.client.complete(provider, messages, **kwargs), ""
        except RateLimitError as exc:
            await gateway.metering.append(
                now_row("trip", provider, detail=f"429 while estimated healthy: {exc}")
            )
            failure = "rate_limited"
        except ProviderError as exc:
            logger.warning("dispatch.provider_error", provider=provider, error=str(exc))
            failure = "provider_error"
        except ProviderConfigError as exc:
            logger.error("dispatch.provider_config", provider=provider, error=str(exc))
            failure = "config_error"
            break
    return None, failure


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, http_request: Request, response: Response
):
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="streaming is not implemented yet; send stream=false",
        )

    gateway: Gateway = http_request.app.state.gateway
    request_id = new_request_id()

    with stage_span(
        spans.REQUEST,
        **{"tollgate.model_requested": request.model, "tollgate.streaming": request.stream},
    ) as request_span:
        request_started = time.perf_counter()

        # The cache probe is a miss until M2 builds the layers it probes. Its span
        # exists anyway: the overhead figure in M5 is a decomposition of exactly
        # this tree, and a stage reporting zero is the honest way to show what is
        # not built yet.
        with stage_span(spans.CACHE_PROBE) as cache_span:
            cache_outcome = "miss"
            cache_span.record(**{"tollgate.cache.outcome": cache_outcome})

        with stage_span(spans.ROUTE):
            pass

        ref = parse_model_ref(request.model, gateway.chain[0])
        allow_local = http_request.headers.get("x-tollgate-allow-local", "").lower() == "true"
        plan = gateway.selector.plan(ref.provider, gateway.chain, allow_local)

        with stage_span(spans.BUDGET) as budget_span:
            head = await gateway.tracker.estimate(plan[0])
            minute = head.requests_per_minute.window
            budget_span.record(
                **{
                    "tollgate.budget.requests_remaining": head.requests_per_minute.remaining,
                    "tollgate.budget.tokens_remaining": head.tokens_per_minute.remaining,
                    "tollgate.budget.window": f"{minute.start}-{minute.end}",
                    "tollgate.budget.degraded": head.degraded,
                }
            )

        with stage_span(spans.SELECT) as select_span:
            selection: Selection = await gateway.selector.choose(plan)
            select_span.record(
                **{
                    "tollgate.select.chosen": selection.chosen or "",
                    "tollgate.select.skipped": ",".join(
                        f"{s.provider}:{s.reason}" for s in selection.skipped
                    ),
                    "tollgate.select.chain_length": len(plan),
                }
            )
            for skip in selection.skipped:
                await gateway.metering.append(now_row("skip", skip.provider, detail=skip.reason))

        if selection.chosen is None:
            return _exhausted(selection.retry_after, request_id)

        skipped_names = {s.provider for s in selection.skipped}
        remaining_chain = [selection.chosen] + [
            p for p in plan if p != selection.chosen and p not in skipped_names
        ]

        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
        kwargs = _sampling_kwargs(request)
        if ref.model is not None:
            kwargs["model"] = ref.model

        dispatch_started = time.perf_counter()
        try:
            with stage_span(spans.DISPATCH) as dispatch_span:
                completion, failure = await _dispatch(gateway, remaining_chain, messages, kwargs)
                dispatch_span.record(**{"tollgate.dispatch.attempts": len(remaining_chain)})
        except ProviderClientError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dispatch_elapsed = time.perf_counter() - dispatch_started

        if completion is None:
            # The status has to say what actually happened. Exhaustion gets a 429
            # and a computed Retry-After; a broken or misconfigured upstream gets
            # a 502, because telling a caller to slow down will not help them.
            if failure == "provider_error":
                raise HTTPException(status_code=502, detail="upstream provider failed")
            if failure == "config_error":
                raise HTTPException(status_code=502, detail="no provider is configured correctly")
            retry_after = await gateway.selector.earliest_retry_after(remaining_chain)
            return _exhausted(retry_after, request_id)

        usage = None
        if completion.tokens_in is not None and completion.tokens_out is not None:
            usage = Usage(
                prompt_tokens=completion.tokens_in,
                completion_tokens=completion.tokens_out,
                total_tokens=completion.tokens_in + completion.tokens_out,
            )

        body = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=completion.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=completion.text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

        # Overhead is the request minus the dispatch, which is the definition written
        # down and hashed in bench/stages.md. Computing it as a subtraction rather
        # than as a sum of the parts means a stage added later is counted without
        # anyone having to remember to add it here.
        overhead_s = (time.perf_counter() - request_started) - dispatch_elapsed

        with stage_span(spans.METER) as meter_span:
            await gateway.metering.append(
                now_row(
                    "request",
                    completion.provider,
                    request_id=request_id,
                    model=completion.model,
                    cache_outcome=cache_outcome,
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    overhead_ms=round(overhead_s * 1000, 3),
                    status=200,
                )
            )
            meter_span.record(**{"tollgate.meter.persisted": True})

        await gateway.tracker.record(
            completion.provider, tokens=(completion.tokens_in or 0) + (completion.tokens_out or 0)
        )

        request_span.record(
            **{
                "tollgate.provider": completion.provider,
                "tollgate.cache": cache_outcome,
                "tollgate.overhead_ms": round(overhead_s * 1000, 3),
            }
        )

    response.headers["x-tollgate-overhead-ms"] = f"{overhead_s * 1000:.3f}"
    response.headers["x-tollgate-provider"] = completion.provider
    response.headers["x-tollgate-cache"] = cache_outcome
    response.headers["x-request-id"] = request_id
    return body


def _exhausted(retry_after: int, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "all providers are exhausted",
                "type": "rate_limit_error",
                "code": "all_providers_exhausted",
            }
        },
        headers={"retry-after": str(max(retry_after, 1)), "x-request-id": request_id},
    )
