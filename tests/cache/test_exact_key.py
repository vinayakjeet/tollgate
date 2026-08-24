from __future__ import annotations

from app.cache_key import canonical_payload, exact_cache_key


def _fields(**overrides) -> dict:
    base = {
        "model": "groq/llama-3.3-70b",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
    }
    base.update(overrides)
    return base


def test_byte_identical_requests_share_a_key():
    assert canonical_payload(_fields()) == canonical_payload(_fields())


def test_every_sampling_parameter_gets_its_own_key():
    """The acceptance criterion for M2.1. A key that ignores any of these serves
    one answer to requests that asked for different generations."""
    baseline = exact_cache_key("s", canonical_payload(_fields()))
    variants = [
        _fields(temperature=0.7),
        _fields(top_p=0.9),
        _fields(max_tokens=16),
        _fields(messages=[{"role": "user", "content": "goodbye"}]),
        _fields(model="cerebras/llama-3.1-8b"),
    ]
    for fields in variants:
        assert exact_cache_key("s", canonical_payload(fields)) != baseline


def test_stream_is_deliberately_absent_from_the_key():
    """stream changes transport, not content: excluding it is what will let a
    cached completion be re-streamed (M3.3)."""
    assert canonical_payload(_fields()) == canonical_payload(_fields(stream=True))


def test_unknown_fields_are_dropped_visibly():
    """A field not in KEY_FIELDS cannot silently participate in the key. If a new
    sampling parameter appears, the whitelist makes the decision explicit."""
    assert canonical_payload(_fields(seed=42)) == canonical_payload(_fields())


def test_salt_changes_the_key():
    payload = canonical_payload(_fields())
    assert exact_cache_key("salt-a", payload) != exact_cache_key("salt-b", payload)


def test_key_is_hex_sha256():
    import hashlib

    payload = canonical_payload(_fields())
    expected = hashlib.sha256(b"s\x00" + payload).hexdigest()
    assert exact_cache_key("s", payload) == expected
