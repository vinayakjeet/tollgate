from __future__ import annotations

import os
import re
from pathlib import Path

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

# Names a .env may set. Provider keys are matched by shape because which variable
# holds a key is configuration in quotas.yaml rather than a fixed list, but the shape
# is still narrow enough to exclude PATH, PYTHONPATH and LD_PRELOAD.
_LOADABLE = re.compile(
    r"[A-Z0-9_]+_API_KEY|DATABASE_URL|REDIS_URL|TOLLGATE_URL|EMBEDDING_MODEL"
    r"|DASTAVEZ_MODEL|ENVIRONMENT|LOG_LEVEL|GIT_SHA|LLM_PROVIDER"
    r"|LLM_MAX_RETRY_ATTEMPTS|OTEL_EXPORTER_OTLP_ENDPOINT|OTEL_EXPORTER_OTLP_HEADERS"
)


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

    Only names this project expects are loaded. A `.env` is an untrusted file: it sits
    in whatever directory the process happens to start in, it is not tracked, and
    nobody reviews it. Copying arbitrary names out of it into the process environment
    would let one set PATH, PYTHONPATH or LD_PRELOAD, which turns "the app reads its
    config" into "the app runs code the config chose". The allowlist keeps this to
    what `quotas.yaml` and Settings actually resolve.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _LOADABLE.fullmatch(key):
            logger.warning("dotenv.ignored", key=key)
            continue
        os.environ.setdefault(key, value.strip())


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
