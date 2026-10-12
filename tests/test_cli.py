from oncall_copilot.cli import main


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
