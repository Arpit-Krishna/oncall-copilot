import logging
from collections import Counter
from datetime import datetime

import pytest

from oncall_copilot.loader import load_file, parse_line, read_lines
from oncall_copilot.models import Level

VALID_LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


def test_parse_valid_line():
    record = parse_line(VALID_LINE, 2)

    assert record is not None
    assert record.line_no == 2
    assert record.timestamp == datetime(2005, 12, 4, 4, 47, 44)
    assert record.level is Level.ERROR
    assert record.message == "mod_jk child workerEnv in error state 6"
    assert record.source == "apache"


def test_parse_uppercase_level_is_normalised():
    record = parse_line("[Sun Dec 04 04:47:44 2005] [NOTICE] hello", 1)

    assert record is not None
    assert record.level is Level.NOTICE


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("this is not a log line", id="garbage"),
        pytest.param("", id="empty"),
        pytest.param("[Sun Dec 04 04:47:44 2005] [emerg] boom", id="unknown-level"),
        pytest.param("[Sun Feb 30 04:47:44 2005] [error] boom", id="impossible-date"),
        pytest.param("[Sun Dec 04 04:47:44 2005] [error] ", id="empty-message"),
        pytest.param("Sun Dec 04 04:47:44 2005 error no brackets", id="no-brackets"),
    ],
)
def test_bad_lines_return_none(line):
    assert parse_line(line, 1) is None


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("this is not a log line", "does not match log format"),
        ("[Sun Dec 04 04:47:44 2005] [emerg] boom", "level"),
        ("[Sun Feb 30 04:47:44 2005] [error] boom", "timestamp"),
    ],
)
def test_skipped_line_is_logged_with_line_number_and_reason(caplog, line, reason):
    with caplog.at_level(logging.WARNING):
        parse_line(line, 42)

    assert "line 42 skipped" in caplog.text
    assert reason in caplog.text


def test_read_lines_numbers_from_one_and_strips_newlines(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("first\nsecond\n")

    assert list(read_lines(f)) == [(1, "first"), (2, "second")]


def test_load_file_counts_skips(tmp_path, caplog):
    f = tmp_path / "mixed.log"
    f.write_text(
        "[Sun Dec 04 04:47:44 2005] [notice] good one\n"
        "garbage in the middle\n"
        "[Sun Dec 04 04:51:08 2005] [error] good two\n"
    )

    with caplog.at_level(logging.WARNING):
        result = load_file(f)

    assert [r.line_no for r in result.records] == [1, 3]
    assert result.skipped == 1
    assert result.count == 2
    assert "skipped 1 lines" in caplog.text


def test_load_real_apache_sample(apache_log, caplog):
    with caplog.at_level(logging.WARNING):
        result = load_file(apache_log)

    assert len(result.records) == 2000
    assert result.skipped == 0
    assert Counter(r.level for r in result.records) == {Level.NOTICE: 1405, Level.ERROR: 595}
    assert caplog.text == ""  # a clean file logs nothing
