import argparse
import logging
from collections import Counter
from pathlib import Path

from .loader import load_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OnCall Copilot CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the "stats" command
    stats_parser = subparsers.add_parser("stats", help="Generate statistics")
    stats_parser.add_argument(
        "path", type=Path, help="Path to the data file"
    )
    stats_parser.add_argument(
        "--top", type=int, default=5, help="Top N records to display"
    )

    return parser

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "stats":
        if not args.path.exists():
            print(f"Error: {args.path} does not exist", file=__import__('sys').stderr)
            return 1
        
        result = load_file(args.path)
        print(f"Total: {result.count}")
        print(f"Skipped: {result.skipped}")

        level_counts = Counter(record.level for record in result.records)
        for level, count in level_counts.most_common():
            print(f"{level:<8}{count:>6}")
        
        message_counts = Counter(record.message for record in result.records)
        print(f"Top {args.top} messages: {message_counts.most_common(args.top)}")

    return 0