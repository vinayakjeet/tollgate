from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app import spans
from app.config import get_settings
from app.model_ref import parse_model_ref
from app.spans import stage_span
from llm import ChatClient, ChatMessage, ProviderClientError, ProviderConfigError, ProviderError

router = APIRouter(prefix="/v1", tags=["openai"])

# Module-level singleton so throttle cooldowns and retry state persist across
# requests within one process. A per-request client would forget every 429 the
# moment it answered, which is the whole mechanism the chassis exists to provide.
client = ChatClient(max_retry_attempts=get_settings().llm_max_retry_attempts)


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


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, response: Response):
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="streaming is not implemented yet; send stream=false",
        )

    settings = get_settings()

    with stage_span(
        spans.REQUEST,
        **{"tollgate.model_requested": request.model, "tollgate.streaming": request.stream},
    ) as request_span:
        request_started = time.perf_counter()

        # The cache, router and budget stages are no-ops until M1 and M2. Their spans
        # exist anyway, because the overhead figure in M5 is a decomposition of
        # exactly this tree, and a decomposition cannot be added afterwards. A stage
        # that reports zero is also the honest way to show what is not built yet.
        with stage_span(spans.CACHE_PROBE) as cache_span:
            cache_outcome = "miss"
            cache_span.record(**{"tollgate.cache.outcome": cache_outcome})

        with stage_span(spans.ROUTE):
            pass

        with stage_span(spans.BUDGET):
            pass

        with stage_span(spans.SELECT) as select_span:
            provider, model = parse_model_ref(request.model, settings.llm_provider)
            select_span.record(**{"tollgate.select.chosen": provider})

        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
        kwargs = _sampling_kwargs(request)
        if model is not None:
            kwargs["model"] = model

        dispatch_started = time.perf_counter()
        try:
            with stage_span(spans.DISPATCH) as dispatch_span:
                completion = await client.complete(provider, messages, **kwargs)
                dispatch_span.record(**{"tollgate.dispatch.attempts": 1})
        except ProviderConfigError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ProviderClientError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        dispatch_elapsed = time.perf_counter() - dispatch_started

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

        with stage_span(spans.METER) as meter_span:
            meter_span.record(**{"tollgate.meter.persisted": False})

        # Overhead is the request minus the dispatch, which is the definition written
        # down and hashed in bench/stages.md. Computing it as a subtraction rather
        # than as a sum of the parts means a stage added later is counted without
        # anyone having to remember to add it here.
        overhead_s = (time.perf_counter() - request_started) - dispatch_elapsed

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
    return body
