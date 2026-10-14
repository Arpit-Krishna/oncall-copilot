import pytest

from oncall_copilot.providers import ConfigError, provider_chain


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "GROQ_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "GROQ_MODEL"):
        monkeypatch.delenv(name, raising=False)


def names(chain):
    return [p.name for p in chain]


def test_default_is_ollama_then_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    assert names(provider_chain()) == ["ollama", "groq"]


def test_llm_provider_env_picks_the_first(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    assert names(provider_chain()) == ["groq", "ollama"]


def test_explicit_choice_beats_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    assert names(provider_chain("ollama")) == ["ollama", "groq"]


def test_fallback_without_key_is_skipped():
    assert names(provider_chain()) == ["ollama"]


def test_chosen_provider_without_key_fails_clearly():
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        provider_chain("groq")


def test_unknown_provider():
    with pytest.raises(ConfigError, match="unknown provider 'openai'"):
        provider_chain("openai")


def test_ollama_defaults_and_overrides(monkeypatch):
    assert provider_chain()[0].model == "qwen3.5:9b"
    assert provider_chain()[0].base_url.endswith("/v1")

    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    assert provider_chain()[0].model == "gemma4:e4b"


def test_each_provider_uses_its_own_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    ollama, groq = provider_chain()

    assert groq.api_key() == "groq-secret"
    assert ollama.api_key() != "groq-secret"
