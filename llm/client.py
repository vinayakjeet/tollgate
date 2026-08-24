from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator

import structlog

from llm.providers.registry import get_provider
from llm.retry import retry_with_backoff
from llm.throttle import InMemoryThrottle, ThrottleBackend
from llm.types import ChatChunk, ChatMessage, ChatResponse, ProviderError, RateLimitError

logger = structlog.get_logger(__name__)


class ChatClient:
    """Provider-agnostic chat-completion client.

    Call flow: throttle gate (queue behind an active 429 cooldown) -> retry with
    exponential backoff + jitter (transient errors only) -> provider HTTP call ->
    structured cost/token/latency log.
    """

    def __init__(
        self,
        throttle: ThrottleBackend | None = None,
        max_retry_attempts: int = 5,
    ) -> None:
        self._throttle = throttle or InMemoryThrottle()
        self._max_retry_attempts = max_retry_attempts

    async def complete(
        self, provider: str, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse:
        provider_impl = get_provider(provider)

        @retry_with_backoff(max_attempts=self._max_retry_attempts)
        async def _attempt() -> ChatResponse:
            # The gate is checked inside the retry loop, not once before it.
            # A 429 trips the throttle with the delay the provider asked for, and
            # only a gate inside the loop makes the next attempt honour it.
            # Checking once outside meant retries fell back to exponential
            # backoff, which caps well below what a provider can ask for: a real
            # Gemini 429 requested 40 seconds while five attempts of backoff
            # totalled about 31, so every retry was spent while still rate
            # limited and the call failed with quota to spare.
            wait = await self._throttle.is_open(provider)
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                return await provider_impl.chat_completion(messages, **kwargs)
            except RateLimitError as exc:
                await self._throttle.trip(provider, exc.retry_after)
                raise

        start = time.monotonic()
        response = await _attempt()
        response.latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "llm.call",
            provider=response.provider,
            model=response.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        return response

    async def stream_complete(
        self, provider: str, messages: list[ChatMessage], **kwargs: object
    ) -> AsyncIterator[ChatChunk]:
        """Streaming dispatch with retry-until-first-chunk semantics.

        Before the first chunk arrives a failure is invisible to the caller, so
        the usual machinery applies: throttle gate inside the loop, backoff on
        transient errors, trip on 429. After it, bytes have been forwarded and no
        retry can unsend them; a mid-stream failure propagates and the API layer
        tells the caller honestly what happened.
        """
        provider_impl = get_provider(provider)
        attempts = 0
        first_chunk_sent = False

        while True:
            attempts += 1
            wait = await self._throttle.is_open(provider)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async for chunk in provider_impl.stream_completion(messages, **kwargs):
                    first_chunk_sent = True
                    yield chunk
                return
            except RateLimitError as exc:
                if first_chunk_sent or attempts >= self._max_retry_attempts:
                    raise
                await self._throttle.trip(provider, exc.retry_after)
                await self._backoff_sleep(attempts)
            except ProviderError as exc:
                logger.warning("llm.stream.retry", provider=provider, error=str(exc))
                if first_chunk_sent or attempts >= self._max_retry_attempts:
                    raise
                await self._backoff_sleep(attempts)

    @staticmethod
    async def _backoff_sleep(attempt: int) -> None:
        # Same shape as the decorator on complete(): exponential with jitter, so
        # simultaneous failures do not retry in lockstep.
        await asyncio.sleep(min(30.0, (2**attempt) + random.uniform(0, 1)))
