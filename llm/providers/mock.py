from __future__ import annotations

from collections.abc import AsyncIterator

from llm.types import ChatChunk, ChatMessage, ChatResponse


class MockProvider:
    """Deterministic, no-network provider. Always registered, needs no API key.

    This is the default LLM_PROVIDER so `docker compose up`, CI, and the demo
    route all work with zero API keys out of the box.
    """

    name = "mock"

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return ChatResponse(
            text=f"mock reply: {last_user}",
            provider=self.name,
            model=model or "mock-echo",
            tokens_in=len(last_user.split()),
            tokens_out=len(last_user.split()) + 2,
            cost_usd=0.0,
        )

    async def stream_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> AsyncIterator[ChatChunk]:
        """The same answer as chat_completion, one word per chunk. No sleeps:
        pacing belongs to whoever is measuring latency, not to the mock."""
        text = (await self.chat_completion(messages, model, **kwargs)).text
        resolved_model = model or "mock-echo"
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        words = text.split(" ")
        for i, word in enumerate(words):
            yield ChatChunk(
                text_delta=word + (" " if i < len(words) - 1 else ""),
                provider=self.name,
                model=resolved_model,
                finish_reason="stop" if i == len(words) - 1 else None,
            )
        # Token counts on a terminal chunk, mirroring what include_usage gives on
        # real OpenAI-compatible streams, with the same counting as above.
        yield ChatChunk(
            provider=self.name,
            model=resolved_model,
            tokens_in=len(last_user.split()),
            tokens_out=len(last_user.split()) + 2,
        )
