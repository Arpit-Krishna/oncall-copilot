import logging
import sys


def configure_logging() -> None:
    """Warnings for humans, plus the per-call JSON lines from oncall_copilot.calls.

    Call this from entry points (CLI, API), never from library code.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    calls = logging.getLogger("oncall_copilot.calls")
    if not calls.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))  # the message is already JSON
        calls.addHandler(handler)
    calls.setLevel(logging.INFO)
    calls.propagate = False
