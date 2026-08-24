"""Streaming dispatch semantics in ChatClient (M3.1's chassis half).

The rule under test: failures before the first chunk are invisible to the caller
and get the usual machinery (throttle gate inside the loop, backoff, trip on 429).
Failures after it are un-retryable because bytes were already forwarded."""

from __future__ import annotations

import pytest

import llm.providers.registry as registry_module
from llm.client import ChatClient
from llm.throttle import InMemoryThrottle
from llm.types import ChatChunk, ChatMessage, ChatResponse, ProviderError, RateLimitError


class StreamStub:
    def __init__(self, script: list) -> None:
        self.name = "stub"
        self.script = list(script)
        self.calls = 0

    async def chat_completion(self, messages, **kwargs) -> ChatResponse:
        raise AssertionError("non-streaming path should not be used here")

    async def stream_completion(self, messages, **kwargs):
        self.calls += 1
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            yield item


def _chunk(text: str = "hi") -> ChatChunk:
    return ChatChunk(text_delta=text, provider="stub", model="m")


def _register(monkeypatch, stub: StreamStub) -> StreamStub:
    monkeypatch.setitem(registry_module._PROVIDERS, stub.name, stub)
    return stub


async def test_retries_a_transient_failure_before_the_first_chunk(monkeypatch):
    stub = _register(
        monkeypatch,
        StreamStub([ProviderError("boom"), _chunk("a"), _chunk("b")]),
    )
    chunks = [
        c
        async for c in ChatClient().stream_complete(
            "stub", [ChatMessage(role="user", content="x")]
        )
    ]
    assert "".join(c.text_delta for c in chunks) == "ab"
    assert stub.calls == 2


async def test_no_retry_after_bytes_were_forwarded(monkeypatch):
    """A mid-stream failure propagates as-is: retrying would splice two answers
    into one stream, which is worse than an honest error."""
    stub = _register(
        monkeypatch,
        StreamStub([_chunk("a"), _chunk("b"), RateLimitError("late", retry_after=1.0)]),
    )
    received: list[str] = []
    with pytest.raises(RateLimitError):
        async for chunk in ChatClient(max_retry_attempts=3).stream_complete(
            "stub", [ChatMessage(role="user", content="x")]
        ):
            received.append(chunk.text_delta)
            if chunk.text_delta.strip() == "b":
                # Keep consuming so the raising frame is actually reached.
                continue
    assert received[:2] == ["a", "b"]
    assert stub.calls == 1


async def test_rate_limit_before_first_chunk_trips_throttle_and_retries(monkeypatch):
    throttle = InMemoryThrottle()
    stub = _register(
        monkeypatch,
        StreamStub([RateLimitError("slow", retry_after=9.0), _chunk("ok")]),
    )
    chunks = [
        c
        async for c in ChatClient(throttle=throttle).stream_complete(
            "stub", [ChatMessage(role="user", content="x")]
        )
    ]
    assert chunks[-1].text_delta == "ok"
    assert await throttle.is_open("stub") > 0
    assert stub.calls == 2


async def test_exhausted_attempts_raise_before_anything_is_forwarded(monkeypatch):
    stub = _register(monkeypatch, StreamStub([RateLimitError("no")] * 5))
    with pytest.raises(RateLimitError):
        async for _ in ChatClient(max_retry_attempts=2).stream_complete(
            "stub", [ChatMessage(role="user", content="x")]
        ):
            pass
    assert stub.calls == 2
