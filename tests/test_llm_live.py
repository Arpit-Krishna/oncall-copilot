"""Calls the real local model. Skipped by default; run with `uv run --env-file .env pytest -m llm`."""

import asyncio
import os
import urllib.request

import pytest

from oncall_copilot.llm import explain_line, make_client
from oncall_copilot.providers import load_providers

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except OSError:
        return False


def check(result) -> None:
    # The wording changes run to run, so check properties, not exact text.
    assert result.finish_reason == "stop"  # not cut off
    assert len(result.text.split()) <= 150  # roughly respects the 120-word limit
    assert "mod_jk" in result.text or "worker" in result.text.lower()  # the right topic
    assert "Thinking Process" not in result.text and "<think>" not in result.text  # no leaked reasoning
    assert result.input_tokens > 0 and result.output_tokens > 0


@pytest.mark.llm
@pytest.mark.skipif(not ollama_up(), reason="Ollama is not running")
def test_ollama_explains_real_error_line():
    provider = load_providers()["ollama"]
    check(asyncio.run(explain_line(LINE, make_client(provider), provider)))


@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set (use --env-file .env)")
def test_groq_explains_real_error_line():
    provider = load_providers()["groq"]
    result = asyncio.run(explain_line(LINE, make_client(provider), provider))
    check(result)
    assert result.cost_usd is not None and result.cost_usd > 0
