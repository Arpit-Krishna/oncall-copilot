import asyncio

import pytest

from oncall_copilot.llm import (
    AllProvidersFailedError,
    EmptyResponseError,
    PartialResponseError,
    cost_usd,
    explain_line,
    explain_with_fallback,
    format_footer,
)
from tests.fakes import (
    FakeClient,
    connection_error,
    ok_client,
    provider,
    status_error,
    text_chunk,
    usage_chunk,
)

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


def run(coro):
    return asyncio.run(coro)


# --- explain_line: one provider ----------------------------------------------


def test_streams_text_and_captures_usage():
    fake = FakeClient([text_chunk("mod_jk "), text_chunk("is down", finish="stop"), usage_chunk(42, 7)])
    seen = []

    r = run(explain_line(LINE, fake, provider(), on_text=seen.append))

    assert seen == ["mod_jk ", "is down"]  # really streamed, piece by piece
    assert r.text == "mod_jk is down"
    assert (r.input_tokens, r.output_tokens) == (42, 7)
    assert r.finish_reason == "stop"
    assert r.ttft_s is not None and r.total_s >= r.ttft_s


def test_sends_model_and_provider_specific_settings():
    fake = ok_client()

    run(explain_line(LINE, fake, provider("groq")))

    request = fake.calls[0]
    assert request["model"] == "groq-model"
    assert request["reasoning_effort"] == "none"  # from provider.extra, not hardcoded
    assert request["stream"] is True and request["temperature"] == 0
    assert request["stream_options"] == {"include_usage": True}
    assert request["messages"][-1] == {"role": "user", "content": LINE}


def test_cut_off_answer_is_reported():
    fake = FakeClient([text_chunk("half an ans", finish="length"), usage_chunk(5, 300)])

    assert run(explain_line(LINE, fake, provider())).finish_reason == "length"


def test_empty_response_raises():
    fake = FakeClient([text_chunk("", finish="stop"), usage_chunk(5, 0)])

    with pytest.raises(EmptyResponseError):
        run(explain_line(LINE, fake, provider()))


def test_failure_after_text_is_a_partial_response():
    fake = FakeClient([text_chunk("half "), text_chunk("answer")], error=connection_error(), fail_after=1)

    with pytest.raises(PartialResponseError):
        run(explain_line(LINE, fake, provider()))


# --- cost and footer -----------------------------------------------------------


def test_cost_is_per_million_tokens():
    assert cost_usd((0.10, 0.50), 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd((0.80, 4.00), 99, 121) == pytest.approx(0.0005632)


def test_paid_provider_reports_real_cost():
    r = run(explain_line(LINE, ok_client(), provider("groq", price=(0.80, 4.00))))

    assert r.cost_usd == pytest.approx(cost_usd((0.80, 4.00), 10, 5))
    assert "$0.000028" in format_footer(r)


def test_local_provider_footer_shows_reference_cost():
    r = run(explain_line(LINE, ok_client(), provider("ollama")))

    assert r.cost_usd is None
    assert "$0 local (≈ $" in format_footer(r)


# --- explain_with_fallback: the two rules --------------------------------------


def fallback(clients: dict, providers: list, log: list, on_text=None):
    return run(
        explain_with_fallback(
            providers,
            LINE,
            on_text=on_text,
            client_factory=lambda p: clients[p.name],
            log=log.append,
        )
    )


@pytest.mark.parametrize(
    "error",
    [connection_error(), status_error(429), status_error(500)],
    ids=["connection", "rate-limited", "server-error"],
)
def test_falls_back_on_errors_the_next_provider_might_not_have(error):
    clients = {"ollama": FakeClient(error=error), "groq": ok_client("recovered")}
    log = []

    r = fallback(clients, [provider("ollama"), provider("groq")], log)

    assert r.provider == "groq" and r.text == "recovered"
    assert [entry["status"] for entry in log] == ["error", "ok"]
    assert log[1]["fallback_from"] == "ollama"


def test_does_not_fall_back_on_bad_request():
    clients = {"ollama": FakeClient(error=status_error(400)), "groq": ok_client()}
    log = []

    with pytest.raises(Exception, match="HTTP 400"):
        fallback(clients, [provider("ollama"), provider("groq")], log)

    assert clients["groq"].calls == []  # a bug in our request would fail everywhere
    assert log[0]["error"] == "BadRequestError"


def test_does_not_fall_back_after_first_token():
    broken = FakeClient([text_chunk("half "), text_chunk("answer")], error=connection_error(), fail_after=1)
    clients = {"ollama": broken, "groq": ok_client()}
    seen = []

    with pytest.raises(PartialResponseError):
        fallback(clients, [provider("ollama"), provider("groq")], [], on_text=seen.append)

    assert seen == ["half "]  # nothing from a second provider stitched on
    assert clients["groq"].calls == []


def test_all_providers_failing_names_them():
    clients = {"ollama": FakeClient(error=connection_error()), "groq": FakeClient(error=status_error(500))}

    with pytest.raises(AllProvidersFailedError, match="ollama, groq"):
        fallback(clients, [provider("ollama"), provider("groq")], [])


def test_successful_call_logs_tokens_latency_and_cost():
    log = []

    fallback({"groq": ok_client()}, [provider("groq", price=(0.80, 4.00))], log)

    entry = log[0]
    assert entry["event"] == "llm_call" and entry["status"] == "ok"
    assert entry["input_tokens"] == 10 and entry["output_tokens"] == 5
    assert entry["cost_usd"] == pytest.approx(cost_usd((0.80, 4.00), 10, 5))
    assert entry["fallback_from"] is None
    assert {"ttft_s", "total_s", "model", "provider"} <= entry.keys()
