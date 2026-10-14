import pytest

from oncall_copilot.cli import main
from tests.fakes import FakeClient, connection_error, ok_client, text_chunk, usage_chunk


def test_stats_on_real_file(apache_log, capsys):
    exit_code = main(["stats", str(apache_log)])
    out = capsys.readouterr()

    assert exit_code == 0
    assert "2000" in out.out
    assert "notice" in out.out and "1405" in out.out
    assert "error" in out.out and "595" in out.out
    assert "workerEnv.init() ok" in out.out
    assert out.err == ""


def test_stats_top_limits_messages(apache_log, capsys):
    main(["stats", str(apache_log), "--top", "1"])
    out = capsys.readouterr().out

    assert "workerEnv.init() ok" in out
    assert "error state 6" not in out


def test_stats_missing_file_fails_on_stderr(tmp_path, capsys):
    missing = tmp_path / "nope.log"

    exit_code = main(["stats", str(missing)])
    out = capsys.readouterr()

    assert exit_code == 1
    assert "nope.log" in out.err
    assert out.out == ""


# --- explain, with the model client swapped for a fake ------------------------



APACHE_LINE_2 = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


@pytest.fixture
def no_llm_env(monkeypatch):
    for name in ("LLM_PROVIDER", "GROQ_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def use_fakes(monkeypatch, clients: dict) -> None:
    monkeypatch.setattr("oncall_copilot.llm.make_client", lambda provider: clients[provider.name])


def test_explain_streams_and_prints_footer(apache_log, no_llm_env, monkeypatch, capsys):
    fake = FakeClient([text_chunk("Worker "), text_chunk("crashed.", finish="stop"), usage_chunk(50, 9)])
    use_fakes(monkeypatch, {"ollama": fake})

    exit_code = main(["explain", str(apache_log), "--line", "2"])
    out = capsys.readouterr()

    assert exit_code == 0
    assert out.out == "Worker crashed.\n"  # only the explanation on stdout
    assert "ollama qwen3.5:9b · 50 in / 9 out" in out.err  # footer on stderr
    assert fake.calls[0]["messages"][-1]["content"] == APACHE_LINE_2  # original format


def test_explain_falls_back_to_groq(apache_log, no_llm_env, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    use_fakes(monkeypatch, {"ollama": FakeClient(error=connection_error()), "groq": ok_client("from groq")})

    assert main(["explain", str(apache_log), "--line", "2"]) == 0
    out = capsys.readouterr()
    assert out.out == "from groq\n"
    assert "groq qwen/qwen3.8-27b" in out.err


def test_explain_provider_flag_puts_groq_first(apache_log, no_llm_env, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    clients = {"ollama": ok_client("from ollama"), "groq": ok_client("from groq")}
    use_fakes(monkeypatch, clients)

    assert main(["explain", str(apache_log), "--line", "2", "--provider", "groq"]) == 0
    assert capsys.readouterr().out == "from groq\n"
    assert clients["ollama"].calls == []


def test_explain_groq_without_key(apache_log, no_llm_env, capsys):
    assert main(["explain", str(apache_log), "--line", "2", "--provider", "groq"]) == 1
    assert "GROQ_API_KEY is not set" in capsys.readouterr().err


def test_explain_when_everything_is_down(apache_log, no_llm_env, monkeypatch, capsys):
    use_fakes(monkeypatch, {"ollama": FakeClient(error=connection_error())})

    assert main(["explain", str(apache_log), "--line", "2"]) == 1
    assert "all providers failed: ollama" in capsys.readouterr().err


def test_explain_cut_off_warns(apache_log, no_llm_env, monkeypatch, capsys):
    cut = FakeClient([text_chunk("partial", finish="length"), usage_chunk(5, 300)])
    use_fakes(monkeypatch, {"ollama": cut})

    assert main(["explain", str(apache_log), "--line", "2"]) == 0
    assert "cut off" in capsys.readouterr().err


def test_explain_line_out_of_range(apache_log, no_llm_env, capsys):
    assert main(["explain", str(apache_log), "--line", "99999"]) == 1
    assert "no parsed log record at line 99999" in capsys.readouterr().err


def test_explain_picks_record_by_line_number_not_position(tmp_path, no_llm_env, monkeypatch):
    log = tmp_path / "x.log"
    log.write_text(
        "garbage first line\n"
        "[Sun Dec 04 04:47:44 2005] [notice] second\n"
        "[Sun Dec 04 04:47:45 2005] [error] third\n"
    )
    fake = ok_client()
    use_fakes(monkeypatch, {"ollama": fake})

    assert main(["explain", str(log), "--line", "3"]) == 0
    assert fake.calls[0]["messages"][-1]["content"].endswith("third")
