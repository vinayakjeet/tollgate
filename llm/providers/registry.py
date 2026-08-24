from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from llm.providers.base import (
    OpenAICompatibleProvider,
    Provider,
    default_usage_parser,
    total_aware_usage_parser,
)
from llm.providers.mock import MockProvider
from llm.types import ProviderConfigError

DEFAULT_QUOTAS_PATH = Path(__file__).parent / "quotas.yaml"


USAGE_PARSERS = {
    "default": default_usage_parser,
    "total_aware": total_aware_usage_parser,
}


class ProviderQuotaConfig(BaseModel):
    base_url: str
    api_key_env: str
    default_model: str
    reset_window: str
    rpm_limit: int | None = None
    # Requests per day and tokens per minute. Nullable because no provider here
    # publishes every limit, and an honest schema represents "unknown" rather than
    # inventing a number: a None limit is never treated as zero (see app/budget.py).
    # Each carries its own verification date so one field can be re-measured
    # without claiming the others were checked too.
    tpm_limit: int | None = None
    rpd_limit: int | None = None
    tpm_last_verified: str | None = None
    rpd_last_verified: str | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    usage_parser: str = "default"
    last_verified: str


def _load_everything(
    path: Path = DEFAULT_QUOTAS_PATH,
) -> tuple[dict[str, Provider], dict[str, ProviderQuotaConfig]]:
    """Parse quotas.yaml once into both the provider map and the raw quota configs.

    Fails fast with a clear ProviderConfigError on a malformed entry - a broken
    config should crash at import time, not on the first real request.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("providers", {}) or {}

    providers: dict[str, Provider] = {"mock": MockProvider()}
    configs: dict[str, ProviderQuotaConfig] = {}
    for name, entry in entries.items():
        try:
            config = ProviderQuotaConfig.model_validate(entry)
        except ValidationError as exc:
            raise ProviderConfigError(
                f"{path}: invalid entry for provider '{name}': {exc}"
            ) from exc
        configs[name] = config
        providers[name] = OpenAICompatibleProvider(
            name=name,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            default_model=config.default_model,
            input_price_per_1m=config.input_price_per_1m,
            output_price_per_1m=config.output_price_per_1m,
            usage_parser=USAGE_PARSERS[config.usage_parser],
        )
    return providers, configs


def load_providers(path: Path = DEFAULT_QUOTAS_PATH) -> dict[str, Provider]:
    """Build the provider map from a quotas.yaml file. `mock` is always included."""
    return _load_everything(path)[0]


_PROVIDERS, _QUOTA_CONFIGS = _load_everything()


def known_providers() -> frozenset[str]:
    """Provider names the registry can actually dispatch to, including `mock`."""
    return frozenset(_PROVIDERS)


def get_provider(name: str) -> Provider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        valid = ", ".join(sorted(_PROVIDERS))
        raise ProviderConfigError(
            f"unknown llm provider '{name}'. Valid providers: {valid}"
        ) from None


def quota_limits(provider: str) -> tuple[int | None, int | None, int | None]:
    """The (rpm, tpm, rpd) limits recorded for `provider`, unknowns as None.

    Reads the same validated config the registry built at import, so a limit and
    the provider it constrains can never disagree about what quotas.yaml says.
    """
    config = _QUOTA_CONFIGS.get(provider)
    if config is None:
        return None, None, None
    return config.rpm_limit, config.tpm_limit, config.rpd_limit


def quota_verification(provider: str) -> dict[str, str | None]:
    """Verification dates for each limit, for /budget to publish beside figures."""
    config = _QUOTA_CONFIGS.get(provider)
    if config is None:
        return {}
    return {
        "rpm": config.last_verified,
        "tpm": config.tpm_last_verified or config.last_verified,
        "rpd": config.rpd_last_verified or config.last_verified,
    }
