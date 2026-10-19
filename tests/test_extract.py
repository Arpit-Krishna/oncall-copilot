import asyncio
import json

import pytest

from oncall_copilot.llm import (
    StructuredOutputError,
    extract_event,
    extract_with_fallback,
)
from tests.fakes import (
    FakeCompletionClient,
    completion_response,
    connection_error,
    provider,
)

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"

VALID_JSON = json.dumps(
    {
        "severity": "high",
        "component": "mod_jk",
        "event_type": "worker_error",
        "summary": "Worker environment in error state 6",
        "actionable": True,
    }
)

BAD_JSON = '{"severity": "high", component: broken}'

BAD_ENUM = json.dumps(
    {
        "severity": "catastrophic",
        "component": "mod_jk",
        "event_type": "worker_error",
        "summary": "Bad",
        "actionable": True,
    }
)

FENCED_JSON = f"```json\n{VALID_JSON}\n```"


def run(coro):
    return asyncio.run(coro)


# --- extract_event: one provider ----------------------------------------------


def test_valid_first_try():
    fake = FakeCompletionClient([completion_response(VALID_JSON)])
    log = []

    r = run(extract_event(LINE, fake, provider(), log=log.append))

    assert r.event.severity == "high"
    assert r.event.component == "mod_jk"
    assert r.event.event_type == "worker_error"
    assert r.event.actionable is True
    assert r.attempts == 1
    assert r.input_tokens == 20 and r.output_tokens == 10
    assert len(fake.calls) == 1
    # One log line: attempt 1, valid
    assert len(log) == 1
    assert log[0]["attempt"] == 1 and log[0]["valid"] is True


def test_invalid_json_then_valid():
    fake = FakeCompletionClient([
        completion_response(BAD_JSON),
        completion_response(VALID_JSON),
    ])
    log = []

    r = run(extract_event(LINE, fake, provider(), log=log.append))

    assert r.attempts == 2
    assert r.event.severity == "high"
    # The retry request contains the validation error text
    retry_messages = fake.calls[1]["messages"]
    error_msg = retry_messages[-1]["content"]
    assert "Invalid JSON or schema:" in error_msg
    # The bad response was fed back as assistant message
    assert retry_messages[-2]["content"] == BAD_JSON
    # Two log lines: attempt 1 invalid, attempt 2 valid
    assert len(log) == 2
    assert log[0]["attempt"] == 1 and log[0]["valid"] is False
    assert log[1]["attempt"] == 2 and log[1]["valid"] is True


def test_bad_enum_then_valid():
    fake = FakeCompletionClient([
        completion_response(BAD_ENUM),
        completion_response(VALID_JSON),
    ])
    log = []

    r = run(extract_event(LINE, fake, provider(), log=log.append))

    assert r.attempts == 2
    assert r.event.severity == "high"
    # Retry message mentions the invalid enum value
    error_msg = fake.calls[1]["messages"][-1]["content"]
    assert "catastrophic" in error_msg or "Invalid" in error_msg
    # Two log lines
    assert len(log) == 2
    assert log[0]["valid"] is False and log[1]["valid"] is True


def test_fenced_json_parses():
    fake = FakeCompletionClient([completion_response(FENCED_JSON)])
    log = []

    r = run(extract_event(LINE, fake, provider(), log=log.append))

    assert r.attempts == 1
    assert r.event.component == "mod_jk"
    assert len(log) == 1 and log[0]["valid"] is True


def test_three_failures_raises_structured_output_error():
    fake = FakeCompletionClient([
        completion_response(BAD_JSON),
        completion_response(BAD_JSON),
        completion_response(BAD_JSON),
    ])
    log = []

    with pytest.raises(StructuredOutputError, match="3 attempts"):
        run(extract_event(LINE, fake, provider(), log=log.append))

    assert len(fake.calls) == 3
    # Three log lines, all invalid
    assert len(log) == 3
    assert all(entry["valid"] is False for entry in log)


def test_tokens_accumulate_across_retries():
    fake = FakeCompletionClient([
        completion_response(BAD_JSON, prompt_tokens=15, completion_tokens=8),
        completion_response(VALID_JSON, prompt_tokens=30, completion_tokens=12),
    ])

    r = run(extract_event(LINE, fake, provider()))

    assert r.input_tokens == 45  # 15 + 30
    assert r.output_tokens == 20  # 8 + 12


def test_each_attempt_logged():
    """Per-attempt log lines carry attempt number, valid flag, and tokens."""
    fake = FakeCompletionClient([
        completion_response(BAD_ENUM, prompt_tokens=18, completion_tokens=9),
        completion_response(VALID_JSON, prompt_tokens=25, completion_tokens=11),
    ])
    log = []

    run(extract_event(LINE, fake, provider("groq"), log=log.append))

    assert len(log) == 2
    # First attempt: invalid
    assert log[0] == {
        "event": "extract_attempt",
        "provider": "groq",
        "model": "groq-model",
        "attempt": 1,
        "valid": False,
        "error": log[0]["error"],  # non-empty error string
        "input_tokens": 18,
        "output_tokens": 9,
    }
    assert "catastrophic" in log[0]["error"]
    # Second attempt: valid
    assert log[1] == {
        "event": "extract_attempt",
        "provider": "groq",
        "model": "groq-model",
        "attempt": 2,
        "valid": True,
        "input_tokens": 25,
        "output_tokens": 11,
    }


# --- extract_with_fallback ----------------------------------------------------


def _fallback(clients: dict, providers: list, log: list):
    return run(
        extract_with_fallback(
            providers,
            LINE,
            client_factory=lambda p: clients[p.name],
            log=log.append,
        )
    )


def test_fallback_logs_both_attempt_and_provider_level():
    """extract_with_fallback passes log to extract_event, so both levels appear."""
    fake = FakeCompletionClient([completion_response(VALID_JSON)])
    log = []

    _fallback({"p1": fake}, [provider("p1")], log)

    # Per-attempt log from extract_event + provider-level log from extract_with_fallback
    attempt_logs = [e for e in log if e["event"] == "extract_attempt"]
    provider_logs = [e for e in log if e["event"] == "extract_call"]
    assert len(attempt_logs) == 1
    assert attempt_logs[0]["valid"] is True
    assert len(provider_logs) == 1
    assert provider_logs[0]["status"] == "ok"


def test_fallback_on_connection_error():
    class FailClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": type("CC", (), {"create": self._create})()})()

        async def _create(self, **kw):
            raise connection_error()

    log = []
    r = _fallback(
        {"p1": FailClient(), "p2": FakeCompletionClient([completion_response(VALID_JSON)])},
        [provider("p1"), provider("p2")],
        log,
    )

    assert r.event.severity == "high"
    provider_logs = [e for e in log if e["event"] == "extract_call"]
    assert [e["status"] for e in provider_logs] == ["error", "ok"]
    assert provider_logs[1]["fallback_from"] == "p1"
