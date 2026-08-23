from __future__ import annotations

import os

import pytest

# Set before anything imports `app.main`, which builds the app at module scope and
# loads `.env` into the environment on the way. `load_dotenv_into_environ` uses
# `setdefault`, so a value pinned here wins over the file.
#
# The reason this exists: adding dotenv loading turned three passing tests red,
# because a developer `.env` naming a real provider silently became the default under
# test. A suite whose result depends on whether an untracked file happens to exist is
# a suite that passes here and fails in CI, and the version that fails in CI is the
# lucky one.
os.environ["LLM_PROVIDER"] = "mock"
for _key in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
    os.environ[_key] = ""

from app import spans  # noqa: E402  imported after the environment is pinned


@pytest.fixture(autouse=True)
def strict_span_contract(monkeypatch):
    """Strict in CI, lenient in production, and the asymmetry is deliberate.

    `app/spans.py` drops an undeclared attribute and logs it rather than raising,
    because instrumentation that throws turns a mistyped attribute into an outage,
    and every other project in this portfolio calls a model through this process.

    That leniency would also let a contract breach ship silently. So the suite turns
    strict on: a stage carrying an attribute the table does not declare fails here,
    where it is cheap, instead of producing an empty dashboard panel later, where an
    empty panel and a wrong metric name and an idle service all look identical.
    """
    monkeypatch.setattr(spans, "strict", True)
