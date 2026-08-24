from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from pydantic import BaseModel, Field

from app import spans
from app.auth import require_edge_key
from app.caches import CacheEntry, Lookup, MissWithVector
from app.gateway import Gateway
from app.metering import new_request_id, now_row
from app.model_ref import ModelRef, parse_model_ref
from app.routing import Selection
from app.spans import stage_span
from llm.types import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
)

router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(require_edge_key)])

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


@dataclass
class Prepared:
    """Everything both response paths need decided before bytes move."""

    request_id: str
    lookup: Lookup
    ref: ModelRef
    messages: list[ChatMessage]
    selection: Selection | None = None  # None when served from cache
    chain: list[str] = field(default_factory=list)  # dispatch order after skips
    kwargs: dict[str, object] = field(default_factory=dict)


async def _prepare(
    gateway: Gateway, request: ChatCompletionRequest, http_request: Request
) -> tuple[Prepared, float]:
    """Cache probe, route, budget, select: the stages that decide where a request
    goes, in the order bench/stages.md fixes them."""
    request_started = time.perf_counter()
    prepared = Prepared(
        request_id=new_request_id(),
        lookup=await gateway.cache.lookup(_key_fields(request)),
        ref=parse_model_ref(request.model, gateway.chain[0]),
        messages=[ChatMessage(role=m.role, content=m.content) for m in request.messages],
    )

    with stage_span(spans.ROUTE):
        pass

    if prepared.lookup.outcome != "miss":
        return prepared, request_started

    allow_local = http_request.headers.get("x-tollgate-allow-local", "").lower() == "true"
    plan = gateway.selector.plan(prepared.ref.provider, gateway.chain, allow_local)

    with stage_span(spans.BUDGET) as budget_span:
        head = await gateway.tracker.estimate(plan[0])
        minute = head.requests_per_minute.window
        budget_attrs: dict[str, object] = {
            "tollgate.budget.window": f"{minute.start}-{minute.end}",
            "tollgate.budget.degraded": head.degraded,
        }
        # Unknown limits stay off the span rather than becoming nulls: an
        # attribute that silently drops is the honest representation of
        # "unknown", which is what None means here.
        if head.requests_per_minute.remaining is not None:
            budget_attrs["tollgate.budget.requests_remaining"] = head.requests_per_minute.remaining
        if head.tokens_per_minute.remaining is not None:
            budget_attrs["tollgate.budget.tokens_remaining"] = head.tokens_per_minute.remaining
        budget_span.record(**budget_attrs)

    with stage_span(spans.SELECT) as select_span:
        selection = await gateway.selector.choose(plan)
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

    prepared.selection = selection
    if selection.chosen is not None:
        skipped_names = {s.provider for s in selection.skipped}
        prepared.chain = [selection.chosen] + [
            p for p in plan if p != selection.chosen and p not in skipped_names
        ]
        kwargs = _sampling_kwargs(request)
        if prepared.ref.model is not None:
            kwargs["model"] = prepared.ref.model
        prepared.kwargs = kwargs
    return prepared, request_started


async def _dispatch_chain(gateway: Gateway, prepared: Prepared) -> tuple[ChatResponse | None, str]:
    """Walk the dispatching half of the chain (non-streaming).

    Rate limits and transient provider errors move down the chain; each one is a
    disagreement between the estimate that kept the provider in play and the
    provider itself, so both kinds land in the metering store. A 4xx from the
    provider is the caller's mistake and stops the walk, because re-sending the
    same bad request at another provider cannot fix it.
    """
    failure = ""
    for provider in prepared.chain:
        try:
            return (
                await gateway.client.complete(provider, prepared.messages, **prepared.kwargs),
                "",
            )
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
    gateway: Gateway = http_request.app.state.gateway

    if request.stream:
        return await _streaming_response(gateway, request, http_request)

    with stage_span(
        spans.REQUEST,
        **{"tollgate.model_requested": request.model, "tollgate.streaming": False},
    ) as request_span:
        prepared, request_started = await _prepare(gateway, request, http_request)
        cache_outcome = prepared.lookup.outcome
        dispatch_elapsed = 0.0

        if prepared.selection is None:
            assert prepared.lookup.entry is not None
            completion = _completion_from_entry(prepared.lookup.entry)
        else:
            if prepared.selection.chosen is None:
                return _exhausted(prepared.selection.retry_after, prepared.request_id)

            dispatch_started = time.perf_counter()
            try:
                with stage_span(spans.DISPATCH) as dispatch_span:
                    completion, failure = await _dispatch_chain(gateway, prepared)
                    dispatch_span.record(**{"tollgate.dispatch.attempts": len(prepared.chain)})
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
                    raise HTTPException(
                        status_code=502, detail="no provider is configured correctly"
                    )
                retry_after = await gateway.selector.earliest_retry_after(prepared.chain)
                return _exhausted(retry_after, prepared.request_id)

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
        # anyone having to remember to add it here. A cache hit never dispatched,
        # so its overhead is the whole wall time: all of it is Tollgate's.
        overhead_s = (time.perf_counter() - request_started) - dispatch_elapsed

        await _after_completion(gateway, prepared, completion, cache_outcome, overhead_s)

        request_span.record(
            **{
                "tollgate.provider": completion.provider,
                "tollgate.cache": cache_outcome,
                "tollgate.overhead_ms": round(overhead_s * 1000, 3),
            }
        )
        answered_provider = completion.provider

    response.headers["x-tollgate-overhead-ms"] = f"{overhead_s * 1000:.3f}"
    response.headers["x-tollgate-provider"] = answered_provider
    response.headers["x-tollgate-cache"] = cache_outcome
    response.headers["x-request-id"] = prepared.request_id
    similarity = prepared.lookup.similarity
    if cache_outcome == "semantic" and similarity is not None:
        response.headers["x-tollgate-cache-similarity"] = f"{similarity:.4f}"
    return body


# ------------------------------------------------------------------ streaming


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk_frame(chunk_id: str, created: int, model: str, delta: dict, finish: str | None) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _streaming_response(
    gateway: Gateway, request: ChatCompletionRequest, http_request: Request
) -> StreamingResponse:
    """SSE out.

    Preparation runs eagerly here so the response headers can name the selected
    provider and cache outcome before the first byte. The REQUEST span is opened
    manually (open_stage, not a context manager) because it must stay open while
    the body generator drains in some other task: a CM's exit would detach a
    contextvars token created in this task and fail. The body generator may run
    in a different task, where no context of ours is current, so child stages
    anchor through an explicit context instead.

    The REQUEST span ends when the body ends, however it ends: normally, on a
    client disconnect (marked, ended early, no error status), or on failure.
    """
    request_span, finish_request = spans.open_stage(
        spans.REQUEST,
        {"tollgate.model_requested": request.model, "tollgate.streaming": True},
    )

    try:
        prepared, request_started = await _prepare(gateway, request, http_request)
    except BaseException as exc:
        finish_request(exc)
        raise

    outcome = prepared.lookup.outcome
    answered_now = prepared.selection is not None and prepared.selection.chosen is not None
    provider_header = (
        prepared.selection.chosen
        if answered_now
        else (prepared.lookup.entry.provider if prepared.lookup.entry else "")
    )
    headers = {
        "x-tollgate-cache": outcome,
        "x-tollgate-provider": provider_header,
        "x-request-id": prepared.request_id,
    }
    similarity = prepared.lookup.similarity
    if outcome == "semantic" and similarity is not None:
        headers["x-tollgate-cache-similarity"] = f"{similarity:.4f}"

    parent_ctx = trace.set_span_in_context(request_span.span)

    async def generate() -> AsyncIterator[str]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        completion: ChatResponse | None = None
        dispatch_elapsed = 0.0
        upstream: AsyncIterator[ChatChunk] | None = None

        try:
            yield f": tollgate cache={outcome}\n"

            if prepared.selection is None:
                entry = prepared.lookup.entry
                assert entry is not None
                # A cached answer is re-streamed so the caller's code path does
                # not change (M3.3). Its time-to-first-token profile differs from
                # live; the README discloses that rather than hiding it.
                words = entry.text.split(" ")
                for i, piece in enumerate(words):
                    piece += " " if i < len(words) - 1 else ""
                    delta: dict[str, str] = {"content": piece}
                    if i == 0:
                        delta = {"role": "assistant", "content": piece}
                    yield _sse_frame(
                        _chunk_frame(
                            chunk_id,
                            created,
                            entry.model,
                            delta,
                            "stop" if i == len(words) - 1 else None,
                        )
                    )
                completion = _completion_from_entry(entry)

            elif prepared.selection.chosen is None:
                yield _exhausted_event(prepared.selection.retry_after)
                return

            else:
                upstream = gateway.client.stream_complete(
                    prepared.selection.chosen, prepared.messages, **prepared.kwargs
                )
                parts: list[str] = []
                tokens_in: int | None = None
                tokens_out: int | None = None
                model_seen = prepared.ref.model or ""
                first_frame = True
                with stage_span(spans.DISPATCH, context=parent_ctx) as dispatch_span:
                    dispatch_started = time.perf_counter()
                    async for chunk in upstream:
                        if chunk.model:
                            model_seen = chunk.model
                        if chunk.tokens_in is not None:
                            tokens_in = chunk.tokens_in
                        if chunk.tokens_out is not None:
                            tokens_out = chunk.tokens_out
                        if not chunk.text_delta and chunk.finish_reason is None:
                            continue
                        delta = {"content": chunk.text_delta}
                        if first_frame:
                            delta = {"role": "assistant", **delta}
                            first_frame = False
                        yield _sse_frame(
                            _chunk_frame(chunk_id, created, model_seen, delta, chunk.finish_reason)
                        )
                    dispatch_elapsed = time.perf_counter() - dispatch_started
                    dispatch_span.record(**{"tollgate.dispatch.attempts": 1})
                completion = ChatResponse(
                    text="".join(parts),
                    provider=prepared.selection.chosen,
                    model=model_seen,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            overhead_s = (time.perf_counter() - request_started) - dispatch_elapsed
            provider_name = completion.provider if completion is not None else provider_header
            request_span.record(
                **{
                    "tollgate.provider": provider_name,
                    "tollgate.cache": outcome,
                    "tollgate.overhead_ms": round(overhead_s * 1000, 3),
                }
            )

            if completion is not None:
                await _after_completion(
                    gateway, prepared, completion, outcome, overhead_s, parent_ctx
                )
                tollgate_fields: dict[str, object] = {
                    "tollgate_overhead_ms": round(overhead_s * 1000, 3),
                    "tollgate_provider": completion.provider,
                    "tollgate_cache": outcome,
                }
            else:
                status = "no_provider_available"
                await gateway.metering.append(
                    now_row(
                        "request",
                        provider_header,
                        request_id=prepared.request_id,
                        cache_outcome=outcome,
                        detail=status,
                        status=None,
                    )
                )
                tollgate_fields = {
                    "tollgate_status": status,
                    "tollgate_provider": provider_header,
                    "tollgate_cache": outcome,
                }

            # The overhead cannot be a header (headers left long ago) and cannot
            # be a separately named SSE event either: the official SDK parses
            # every data line as a chat chunk. So the figure rides a final
            # chunk-shaped frame as extra fields, which the SDK tolerates and raw
            # readers can audit.
            final_frame = _chunk_frame(chunk_id, created, "", {}, None) | tollgate_fields
            yield _sse_frame(final_frame)
            yield "data: [DONE]\n\n"
            finish_request(None)

        except GeneratorExit:
            # The client hung up mid-stream. Close the upstream so nobody pays
            # for tokens nobody will read, and leave a mark a trace reader can
            # find. The span ending early IS the proof of cancellation.
            request_span.mark("client.disconnect")
            if upstream is not None:
                await upstream.aclose()
            finish_request(GeneratorExit())
            raise
        except (RateLimitError, ProviderError) as exc:
            # Bytes already went out; nothing can retry them un-sent. Name the
            # failure on the wire and end the stream honestly.
            yield _sse_frame({"error": {"message": str(exc), "type": "upstream_error"}})
            await gateway.metering.append(
                now_row(
                    "request",
                    provider_header,
                    request_id=prepared.request_id,
                    cache_outcome=outcome,
                    detail=f"mid_stream_{type(exc).__name__}",
                    status=None,
                )
            )
            error_frame = _chunk_frame(chunk_id, created, "", {}, None) | {
                "tollgate_status": f"upstream_failed:{type(exc).__name__}",
                "tollgate_provider": provider_header,
                "tollgate_cache": outcome,
            }
            yield _sse_frame(error_frame)
            yield "data: [DONE]\n\n"
            finish_request(None)
        except BaseException as exc:
            finish_request(exc)
            raise

    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


def _exhausted_event(retry_after: int) -> str:
    payload = {
        "error": {
            "message": "all providers are exhausted",
            "type": "rate_limit_error",
            "code": "all_providers_exhausted",
            "retry_after": retry_after,
        }
    }
    return _sse_frame(payload)


def _key_fields(request: ChatCompletionRequest) -> dict[str, object]:
    """The request as the cache sees it.

    `model` stays exactly as requested rather than resolved to a provider: the
    fallback chain already makes who answers best-effort, so pinning the key to a
    resolved provider would fragment the cache without making any answer more
    correct. `stream` is excluded upstream in app/cache_key.py on purpose.
    """
    return {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }


def _completion_from_entry(entry: CacheEntry) -> ChatResponse:
    return ChatResponse(
        text=entry.text,
        provider=entry.provider,
        model=entry.model,
        tokens_in=entry.tokens_in,
        tokens_out=entry.tokens_out,
        cost_usd=entry.cost_usd,
    )


async def _after_completion(
    gateway: Gateway,
    prepared: Prepared,
    completion: ChatResponse,
    cache_outcome: str,
    overhead_s: float,
    span_context=None,
) -> None:
    if cache_outcome == "miss" and prepared.lookup.key is not None:
        miss = (
            prepared.lookup
            if isinstance(prepared.lookup, MissWithVector)
            else MissWithVector(outcome="miss", key=prepared.lookup.key)
        )
        await gateway.cache.store(
            miss,
            CacheEntry(
                text=completion.text,
                provider=completion.provider,
                model=completion.model,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                cost_usd=completion.cost_usd,
            ),
        )

    with stage_span(spans.METER, context=span_context) as meter_span:
        await gateway.metering.append(
            now_row(
                "request",
                completion.provider,
                request_id=prepared.request_id,
                model=completion.model,
                cache_outcome=cache_outcome,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                overhead_ms=round(overhead_s * 1000, 3),
                status=200,
            )
        )
        meter_span.record(**{"tollgate.meter.persisted": True})

    if cache_outcome == "miss":
        await gateway.tracker.record(
            completion.provider,
            tokens=(completion.tokens_in or 0) + (completion.tokens_out or 0),
        )


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

