"""The exact-cache key: a pure function with its own tests.

A cache key that accidentally ignores `temperature` serves a deterministic answer
to a creative request forever, which is why the fields are a whitelist and not a
blacklist. A whitelist misses when a new sampling parameter appears (safe: one
avoidable upstream call); a blacklist hits wrongly (unsafe: a wrong answer,
served confidently, forever). Misses are cheap. Wrong hits are not.
"""

from __future__ import annotations

import hashlib
import json

# Every field that changes the answer the provider returns. `stream` is absent on
# purpose: it changes how an answer travels, never what the answer is, which is
# what lets a cached completion be re-streamed to a streaming caller.
KEY_FIELDS = ("model", "messages", "temperature", "top_p", "max_tokens")


def canonical_payload(request_fields: dict) -> bytes:
    """The request's answer-relevant content in one stable byte form.

    Sorted keys and tight separators make two semantically identical requests
    produce identical bytes regardless of dict construction order or client
    whitespace. Anything not in KEY_FIELDS is dropped here, visibly.
    """
    picked = {name: request_fields.get(name) for name in KEY_FIELDS}
    return json.dumps(picked, sort_keys=True, separators=(",", ":")).encode()


def exact_cache_key(salt: str, payload: bytes) -> str:
    """SHA-256 over salt then payload.

    The salt exists so a key cannot be brute-forced from a known prompt (M4.2):
    without it, anyone who can read Redis can confirm a prompt was served by
    replaying the hash. With it, the map from prompt to key needs the running
    process's secret.
    """
    return hashlib.sha256(salt.encode() + b"\x00" + payload).hexdigest()
