import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import openai

from oncall_copilot.llm import (
    ConfigError,
    EmptyResponseError,
    LLMConfig,
    explain_line,
    format_footer,
    make_client,
)
from oncall_copilot.loader import load_file
from oncall_copilot.models import LogRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OnCall Copilot CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Generate statistics")
    stats_parser.add_argument("path", type=Path, help="Path to the data file")
    stats_parser.add_argument("--top", type=int, default=5, help="Top N records to display")

    explain_parser = subparsers.add_parser("explain", help="Explain one log line with an LLM")
    explain_parser.add_argument("path", type=Path, help="Path to the data file")
    explain_parser.add_argument("--line", type=int, required=True, help="Line number in the file (1-based)")

    return parser


def error(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def format_record(record: LogRecord) -> str:
    """Rebuild the line in its original Apache format to send to the model."""
    return f"[{record.timestamp:%a %b %d %H:%M:%S %Y}] [{record.level}] {record.message}"


def cmd_stats(args: argparse.Namespace) -> int:
    if not args.path.exists():
        return error(f"{args.path} does not exist")

    result = load_file(args.path)
    print(f"Total: {result.count}")
    print(f"Skipped: {result.skipped}")

    level_counts = Counter(record.level for record in result.records)
    for level, count in level_counts.most_common():
        print(f"{level:<8}{count:>6}")

    message_counts = Counter(record.message for record in result.records)
    print(f"Top {args.top} messages: {message_counts.most_common(args.top)}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if not args.path.exists():
        return error(f"{args.path} does not exist")

    result = load_file(args.path)
    # Match on the record's own line number: indexing breaks when lines were skipped.
    record = next((r for r in result.records if r.line_no == args.line), None)
    if record is None:
        return error(f"no parsed log record at line {args.line} of {args.path}")

    try:
        config = LLMConfig.from_env()
    except ConfigError as e:
        return error(str(e))

    client = make_client(config)
    try:
        explanation = explain_line(
            format_record(record),
            client,
            config.model,
            on_text=lambda piece: print(piece, end="", flush=True),
        )
    except openai.APIConnectionError:
        print()
        return error(f"could not reach the model at {config.base_url}. Is Ollama running?")
    except openai.APIStatusError as e:
        print()
        return error(f"model API returned HTTP {e.status_code}: {e.message}")
    except EmptyResponseError as e:
        return error(str(e))

    print()
    if explanation.finish_reason == "length":
        print("Warning: the answer was cut off at max_tokens", file=sys.stderr)
    # Metadata goes to stderr so stdout holds only the explanation.
    print(format_footer(explanation), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = build_parser().parse_args(argv)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "explain":
        return cmd_explain(args)
    return error(f"unknown command {args.command}")
