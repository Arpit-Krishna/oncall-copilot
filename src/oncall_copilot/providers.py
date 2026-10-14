"""Which LLM providers exist, how to reach them, and in what order to try them."""

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Try local first (free), then the hosted one.
FALLBACK_ORDER = ("ollama", "groq")


class ConfigError(RuntimeError):
    """A provider is unknown or missing a required setting."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None  # the NAME of the env var holding the key; None = no key needed
    extra: dict = field(default_factory=dict)  # provider-specific request settings
    price: tuple[float, float] | None = None  # USD per million tokens (input, output); None = free/local

    def api_key(self) -> str:
        if self.api_key_env is None:
            return "not-needed"  # the SDK insists on a value; local Ollama ignores it
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(f"{self.api_key_env} is not set (needed for provider '{self.name}')")
        return key

    def is_configured(self) -> bool:
        return self.api_key_env is None or bool(os.environ.get(self.api_key_env))


def load_providers() -> dict[str, Provider]:
    """Built from the environment at call time, so tests and CI need no settings."""
    return {
        "ollama": Provider(
            name="ollama",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b"),
            extra={"reasoning_effort": "none"},
        ),
        "groq": Provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
            api_key_env="GROQ_API_KEY",
            # qwen3.8-27b accepts "none"; Groq's gpt-oss models only accept low/medium/high.
            extra={"reasoning_effort": "none"},
            price=(0.80, 4.00),
        ),
    }


def provider_chain(first: str | None = None) -> list[Provider]:
    """The chosen provider first, then the others in fallback order.

    A provider you asked for explicitly must be configured. Fallback providers
    that aren't configured (e.g. no Groq key) are skipped with a warning.
    """
    providers = load_providers()
    name = (first or os.environ.get("LLM_PROVIDER") or FALLBACK_ORDER[0]).strip().lower()
    if name not in providers:
        raise ConfigError(f"unknown provider '{name}' (choose from: {', '.join(providers)})")

    chosen = providers[name]
    chosen.api_key()  # fail now with a clear message if the chosen provider has no key

    chain = [chosen]
    for other in FALLBACK_ORDER:
        if other == name:
            continue
        if providers[other].is_configured():
            chain.append(providers[other])
        else:
            log.warning("fallback provider '%s' skipped: %s is not set", other, providers[other].api_key_env)
    return chain
