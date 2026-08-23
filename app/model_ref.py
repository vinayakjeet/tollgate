from __future__ import annotations

from typing import NamedTuple

from llm.providers.registry import known_providers


class ModelRef(NamedTuple):
    provider: str
    model: str | None


def parse_model_ref(raw: str, default_provider: str) -> ModelRef:
    """Split an OpenAI `model` field into a provider and a model name.

    `groq/llama-3.3-70b-versatile` routes to groq. A bare `gpt-4o` routes to the
    configured default, so an existing OpenAI client works without editing its model
    string.

    Split on the first slash only. OpenRouter model names contain slashes of their
    own, so `openrouter/meta-llama/llama-3.1-8b-instruct:free` has to keep everything
    after the first separator intact. Splitting on the last slash, or on all of them,
    silently rewrites the model being asked for.

    A prefix that is not a known provider is treated as part of the model name rather
    than as a typo to reject, because that is what `meta-llama/llama-3.1` is when
    someone sends it without a provider.
    """
    head, sep, tail = raw.partition("/")
    if sep and head in known_providers():
        return ModelRef(head, tail or None)
    return ModelRef(default_provider, raw or None)
