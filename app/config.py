from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def load_dotenv_into_environ(path: Path = Path(".env")) -> None:
    """Put `.env` values into the process environment, not just into Settings.

    `Settings` reads `.env` for its own fields, but provider API keys are not Settings
    fields: `llm/providers/base.py` resolves them with `os.environ.get(api_key_env)`,
    because the name of the variable is configuration in `quotas.yaml` rather than a
    fixed attribute. So a `.env` holding `GROQ_API_KEY` loaded only into Settings is
    invisible to the code that needs it, and the failure is a 502 saying the variable
    is unset while the file plainly sets it.

    That gap is not theoretical. `.env.example` documents exactly those keys, so the
    documented local setup did not work until this existed. Real deployments set real
    environment variables and never hit it, which is why it survived four forks.

    Existing environment variables win, because a value exported deliberately should
    beat a file left lying in a working directory.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "local"
    log_level: str = "INFO"
    git_sha: str = "dev"

    database_url: str | None = None
    redis_url: str | None = None

    llm_provider: str = "mock"
    llm_max_retry_attempts: int = 5

    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None


def get_settings() -> Settings:
    """Not cached on purpose: reading env vars is cheap and tests need to be able
    to monkeypatch env per-test without fighting a cached singleton."""
    return Settings()
