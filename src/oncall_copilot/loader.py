import logging
import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from oncall_copilot.models import LoadResult, LogRecord

log = logging.getLogger(__name__)

# Compiled once at import time, not on every call.
LOG_PATTERN = re.compile(
    r"^\[(?P<timestamp>\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4})\] "
    r"\[(?P<level>\w+)\] "
    r"(?P<message>.+)$"
)


def parse_line(line: str, line_no: int) -> LogRecord | None:
    """Parse one raw line. Returns None for a bad line, but always logs why."""
    match = LOG_PATTERN.match(line)
    if match is None:
        log.warning("line %d skipped: does not match log format: %r", line_no, line[:120])
        return None

    try:
        return LogRecord(
            line_no=line_no,
            timestamp=match["timestamp"],
            level=match["level"].lower(),
            message=match["message"],
        )
    except ValidationError as e:
        # Bad timestamp, unknown level, etc. Log every field that failed.
        reasons = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        log.warning("line %d skipped: %s", line_no, reasons)
        return None


def read_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (line_no, line) one at a time, without loading the whole file."""
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            yield line_no, line.rstrip("\n")


def load_file(path: Path) -> LoadResult:
    result = LoadResult()
    for line_no, line in read_lines(Path(path)):
        record = parse_line(line, line_no)
        if record is not None:
            result.records.append(record)
        else:
            result.skipped += 1
    result.count = len(result.records)

    if result.skipped:
        log.warning("%s: loaded %d records, skipped %d lines", path, result.count, result.skipped)
    return result
