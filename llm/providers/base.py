from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Callable
from typing import Protocol

import httpx

from llm.types import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
)


class Provider(Protocol):
    """Anything ChatClient can dispatch to. OpenAICompatibleProvider below covers
    groq/cerebras/openrouter/ollama/sarvam/gemini today; a future native provider
    with its own native message shape implements this Protocol directly instead
    of subclassing OpenAICompatibleProvider."""

    name: str

    async def chat_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse: ...

    async def stream_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> AsyncIterator[ChatChunk]: ...


def default_usage_parser(payload: dict) -> tuple[int | None, int | None]:
    """Standard OpenAI usage shape: {"usage": {"prompt_tokens", "completion_tokens"}}."""
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

# Reasoning models spend far longer than httpx's 5 second default before the first
# byte, so leaving it unset turns ordinary slow generations into network errors.
# Generous rather than unbounded: a genuinely hung connection must still fail.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def parse_retry_after(resp: httpx.Response) -> float | None:
    """How long to wait after a 429, from wherever the provider chose to put it.

    Three places, in order of reliability:

    1. The `Retry-After` header, which is the standard and which Gemini does not
       send.
    2. A `google.rpc.RetryInfo` entry in `error.details`, as `retryDelay: "48s"`.
    3. The human-readable error message: "Please retry in 48.63971551s."

    Falling back through all three matters more than it looks. Without a value the
    throttle uses a short default cooldown, so a limit that needs 49 seconds gets
    retried after 5, fails, and burns quota indefinitely without ever succeeding.
    Free tiers are exactly where this happens.
    """
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None

    # Gemini wraps its error object in a single-element list.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return None

    # `error` is an object in Google's shape but a bare string in plenty of
    # others, so it cannot be assumed to have fields.
    error = payload.get("error")
    if not isinstance(error, dict):
        return None

    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type", "").endswith("RetryInfo"):
            delay = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(delay)
            except ValueError:
                continue

    if match := _RETRY_IN.search(str(error.get("message", ""))):
        value = float(match.group(1))
        # Gemini reports sub-second waits in milliseconds. Reading 607ms as 607
        # seconds would stall a run for ten minutes over a delay of half a second.
        return value / 1000 if match.group(2).lower() == "ms" else value
    return None


def total_aware_usage_parser(payload: dict) -> tuple[int | None, int | None]:
    """Usage parser for providers that bill tokens absent from the itemised fields.

    Gemini's OpenAI-compatible layer reports reasoning tokens in `total_tokens`
    only. A real response looked like prompt=2, completion=9, total=197: the model
    spent 186 tokens thinking, and reading only the itemised fields undercounts
    the run by roughly eighteen times.

    So output is derived as total minus prompt whenever that exceeds the reported
    completion count. Overstating output slightly is the safe direction to be
    wrong in for a benchmark table and a cost ceiling.
    """
    usage = payload.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")

    if prompt is None or total is None:
        return prompt, completion

    derived = total - prompt
    if completion is None or derived > completion:
        return prompt, derived
    return prompt, completion


class OpenAICompatibleProvider:
    """Backs any provider exposing an OpenAI-compatible POST {base_url}/chat/completions.

    Some OpenAI-compatible layers (Gemini's, Sarvam's) diverge slightly in how usage
    is reported - pass a custom `usage_parser` for those instead of assuming
    `default_usage_parser` fits every provider verbatim.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        default_model: str,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        usage_parser: Callable[[dict], tuple[int | None, int | None]] = default_usage_parser,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._input_price_per_1m = input_price_per_1m
        self._output_price_per_1m = output_price_per_1m
        self._usage_parser = usage_parser
        self._client = client

    def _require_api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise ProviderConfigError(
                f"provider '{self.name}' requires env var {self._api_key_env} to be set"
            )
        return key

    def _cost_usd(self, tokens_in: int | None, tokens_out: int | None) -> float | None:
        if self._input_price_per_1m is None or self._output_price_per_1m is None:
            return None
        if tokens_in is None or tokens_out is None:
            return None
        return (tokens_in / 1_000_000) * self._input_price_per_1m + (
            tokens_out / 1_000_000
        ) * self._output_price_per_1m

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        api_key = self._require_api_key()
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump() for m in messages],
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=DEFAULT_TIMEOUT
        )
        try:
            try:
                resp = await client.post("/chat/completions", json=payload, headers=headers)
            except httpx.RequestError as exc:
                raise ProviderError(f"{self.name}: network error: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

        if resp.status_code == 429:
            raise RateLimitError(
                f"{self.name}: rate limited", retry_after=parse_retry_after(resp)
            )
        if resp.status_code >= 500:
            raise ProviderError(f"{self.name}: server error {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderClientError(f"{self.name}: client error {resp.status_code}: {resp.text}")

        data = _parse_json_body(self.name, resp)
        text = data["choices"][0]["message"]["content"]
        tokens_in, tokens_out = self._usage_parser(data)

        return ChatResponse(
            text=text,
            provider=self.name,
            model=resolved_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost_usd(tokens_in, tokens_out),
        )

    async def stream_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> AsyncIterator[ChatChunk]:
        """SSE passthrough. Chunks are yielded as they arrive; nothing buffers the
        body, because buffering would fold time-to-first-token into total latency
        and hand this project's headline number exactly the flattering bias SPEC
        says to measure against.

        Retries are not possible past the first chunk: bytes already forwarded
        cannot be unsent, so any mid-stream failure propagates to the caller as-is.
        """
        api_key = self._require_api_key()
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=DEFAULT_TIMEOUT
        )
        own_client = self._client is None
        try:
            async with client.stream(
                "POST", "/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code == 429:
                    body = await resp.aread()
                    raise RateLimitError(
                        f"{self.name}: rate limited",
                        retry_after=_retry_after_from_bytes(resp.headers, body),
                    )
                if resp.status_code >= 500:
                    raise ProviderError(f"{self.name}: server error {resp.status_code}")
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise ProviderClientError(
                        f"{self.name}: client error {resp.status_code}: {body}"
                    )

                async for chunk in _iter_sse_chunks(self.name, resp, resolved_model):
                    yield chunk
        except httpx.RequestError as exc:
            raise ProviderError(f"{self.name}: network error: {exc}") from exc
        finally:
            if own_client:
                await client.aclose()


def _retry_after_from_bytes(headers: httpx.Headers, body: bytes) -> float | None:
    """The stream path has a body only as bytes, so hand parse_retry_after a
    stand-in carrying just the two things it reads."""
    return parse_retry_after(_FakeResponse(headers, body))


class _FakeResponse:
    """Just enough of httpx.Response for parse_retry_after's JSON paths."""

    def __init__(self, headers: httpx.Headers, body: bytes) -> None:
        self.headers = headers
        self._body = body

    def json(self):
        return json.loads(self._body)


def _parse_json_body(name: str, resp: httpx.Response) -> dict:
    """A malformed 200 must be a retryable provider fault, not a 500 from our own
    JSON decoder. The fault-injection suite (M4.1) is what forced this to exist."""
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(f"{name}: malformed response body: {exc}") from exc


async def _iter_sse_chunks(
    name: str, resp: httpx.Response, model: str
) -> AsyncIterator[ChatChunk]:
    """Parse OpenAI-shaped SSE into ChatChunks.

    Tolerates the two line endings and the keep-alive comment lines real
    providers emit. A `data:` line that is neither JSON nor [DONE] is a protocol
    violation from upstream and surfaces as a ProviderError rather than silence.
    """
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise ProviderError(f"{name}: malformed stream frame: {exc}") from exc

        choices = event.get("choices") or []
        delta = (choices[0].get("delta") if choices else None) or {}
        usage = event.get("usage") or {}
        finish = choices[0].get("finish_reason") if choices else None
        if finish is None and not delta.get("content") and not usage:
            continue
        yield ChatChunk(
            text_delta=delta.get("content") or "",
            provider=name,
            model=event.get("model") or model,
            finish_reason=finish,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )
