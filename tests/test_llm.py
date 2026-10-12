import httpx2
import openai
import pytest

from oncall_copilot import cli
from oncall_copilot.llm import (
    ConfigError,
    EmptyResponseError,
    ExplainResult,
    LLMConfig,
    cost_usd,
    explain_line,
    format_footer,
)
from tests.fakes import FakeClient, text_chunk, usage_chunk

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


# --- explain_line against a fake client -------------------------------------


def test_streams_text_and_captures_usage():
    fake = FakeClient([text_chunk("mod_jk "), text_chunk("is down", finish="stop"), usage_chunk(42, 7)])
    seen = []

    r = explain_line(LINE, fake, "test-model", on_text=seen.append)

    assert seen == ["mod_jk ", "is down"]  # really streamed, piece by piece
    assert r.text == "mod_jk is down"
    assert (r.input_tokens, r.output_tokens) == (42, 7)
    assert r.finish_reason == "stop"
    assert r.ttft_s is not None and r.total_s >= r.ttft_s


def test_sends_expected_request():
    fake = FakeClient([text_chunk("ok", finish="stop"), usage_chunk(1, 1)])

    explain_line(LINE, fake, "test-model")

    request = fake.calls[0]
    assert request["model"] == "test-model"
    assert request["stream"] is True
    assert request["reasoning_effort"] == "none"
    assert request["temperature"] == 0
    assert request["stream_options"] == {"include_usage": True}
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][-1] == {"role": "user", "content": LINE}


def test_cut_off_answer_is_reported():
    fake = FakeClient([text_chunk("half an ans", finish="length"), usage_chunk(5, 300)])

    assert explain_line(LINE, fake, "m").finish_reason == "length"


def test_empty_response_raises():
    fake = FakeClient([text_chunk("", finish="stop"), usage_chunk(5, 0)])

    with pytest.raises(EmptyResponseError):
        explain_line(LINE, fake, "m")


# --- pure helpers -----------------------------------------------------------


def test_cost_is_per_million_tokens():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("claude-haiku-5-5", 142, 187) == pytest.approx(0.0001077)


def test_cost_of_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("no-such-model", 1, 1)


def test_footer_for_local_model_shows_reference_cost():
    r = ExplainResult("qwen3.5:9b", "x", 142, 187, "stop", ttft_s=0.61, total_s=2.9)

    footer = format_footer(r)

    assert "142 in / 187 out" in footer
    assert "first token 0.61s" in footer
    assert "$0 local" in footer and "0.000108" in footer


def test_config_reports_every_missing_variable(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError, match="LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"):
        LLMConfig.from_env()


# --- the explain command, with the client swapped for a fake -----------------


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def use_fake(monkeypatch, fake: FakeClient) -> None:
    monkeypatch.setattr(cli, "make_client", lambda config: fake)


def test_explain_streams_and_prints_footer(apache_log, llm_env, monkeypatch, capsys):
    fake = FakeClient([text_chunk("Worker "), text_chunk("crashed.", finish="stop"), usage_chunk(50, 9)])
    use_fake(monkeypatch, fake)

    exit_code = cli.main(["explain", str(apache_log), "--line", "2"])
    out = capsys.readouterr()

    assert exit_code == 0
    assert out.out == "Worker crashed.\n"  # only the explanation on stdout
    assert "50 in / 9 out" in out.err  # footer on stderr
    assert fake.calls[0]["messages"][-1]["content"] == LINE  # line 2, in its original format


def test_explain_warns_when_answer_is_cut_off(apache_log, llm_env, monkeypatch, capsys):
    use_fake(monkeypatch, FakeClient([text_chunk("partial", finish="length"), usage_chunk(5, 300)]))

    assert cli.main(["explain", str(apache_log), "--line", "2"]) == 0
    assert "cut off" in capsys.readouterr().err


def test_explain_when_model_unreachable(apache_log, llm_env, monkeypatch, capsys):
    down = openai.APIConnectionError(request=httpx2.Request("POST", "http://fake/v1/chat/completions"))
    use_fake(monkeypatch, FakeClient(error=down))

    exit_code = cli.main(["explain", str(apache_log), "--line", "2"])

    assert exit_code == 1
    assert "Is Ollama running?" in capsys.readouterr().err


def test_explain_line_out_of_range(apache_log, llm_env, monkeypatch, capsys):
    use_fake(monkeypatch, FakeClient())

    assert cli.main(["explain", str(apache_log), "--line", "99999"]) == 1
    assert "no parsed log record at line 99999" in capsys.readouterr().err


def test_explain_without_settings(apache_log, monkeypatch, capsys):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    assert cli.main(["explain", str(apache_log), "--line", "2"]) == 1
    assert "missing environment variables" in capsys.readouterr().err


def test_explain_picks_record_by_line_number_not_position(tmp_path, llm_env, monkeypatch):
    log = tmp_path / "x.log"
    log.write_text(
        "garbage first line\n"
        "[Sun Dec 04 04:47:44 2005] [notice] second\n"
        "[Sun Dec 04 04:47:45 2005] [error] third\n"
    )
    fake = FakeClient([text_chunk("ok", finish="stop"), usage_chunk(1, 1)])
    use_fake(monkeypatch, fake)

    assert cli.main(["explain", str(log), "--line", "3"]) == 0
    assert fake.calls[0]["messages"][-1]["content"].endswith("third")
