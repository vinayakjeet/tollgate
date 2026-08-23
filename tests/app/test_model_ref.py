from __future__ import annotations

import pytest

from app.model_ref import parse_model_ref


@pytest.mark.parametrize(
    ("raw", "provider", "model"),
    [
        ("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b-versatile"),
        ("mock/demo", "mock", "demo"),
        ("gemini/gemini-flash-latest", "gemini", "gemini-flash-latest"),
    ],
)
def test_a_known_prefix_selects_the_provider(raw, provider, model):
    assert parse_model_ref(raw, "mock") == (provider, model)


def test_only_the_first_slash_separates_provider_from_model():
    """OpenRouter model names carry their own slashes.

    Splitting anywhere but the first separator rewrites the model being asked for,
    which is a silent substitution rather than an error.
    """
    ref = parse_model_ref("openrouter/meta-llama/llama-3.1-8b-instruct:free", "mock")
    assert ref == ("openrouter", "meta-llama/llama-3.1-8b-instruct:free")


def test_a_bare_model_name_uses_the_default_provider():
    assert parse_model_ref("gpt-4o", "groq") == ("groq", "gpt-4o")


def test_an_unknown_prefix_stays_part_of_the_model_name():
    assert parse_model_ref("meta-llama/llama-3.1-8b", "mock") == (
        "mock",
        "meta-llama/llama-3.1-8b",
    )


def test_a_provider_with_no_model_leaves_the_provider_default_in_charge():
    """`groq/` names a provider and no model, so the registry's default applies."""
    assert parse_model_ref("groq/", "mock") == ("groq", None)
