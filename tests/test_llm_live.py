"""Calls the real local model. Skipped by default; run with `uv run --env-file .env pytest -m llm`."""

import os
import urllib.request

import pytest

from oncall_copilot.llm import LLMConfig, explain_line, make_client

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except OSError:
        return False


@pytest.mark.llm
@pytest.mark.skipif(not ollama_up(), reason="Ollama is not running")
@pytest.mark.skipif(not os.getenv("LLM_MODEL"), reason="LLM settings not loaded (use --env-file .env)")
def test_explains_real_error_line():
    config = LLMConfig.from_env()

    r = explain_line(LINE, make_client(config), config.model)

    # The wording changes run to run, so check properties, not exact text.
    assert r.finish_reason == "stop"  # not cut off
    assert len(r.text.split()) <= 150  # roughly respects the 120-word limit
    assert "mod_jk" in r.text or "worker" in r.text.lower()  # talks about the right thing
    assert "Thinking Process" not in r.text  # no reasoning leaked into the answer
    assert r.input_tokens > 0 and r.output_tokens > 0
