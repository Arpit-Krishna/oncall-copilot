import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an SRE explaining Apache log lines to an on-call engineer: for the given line, "
    "state what the line means, the likely cause, and one next step to check. "
    "Keep the reply under 120 words."
)

# USD per million tokens: (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
}
# Local models are free. We price their tokens at this paid model's rates to show
# what the same work would cost on a hosted API.
REFERENCE_MODEL = "claude-haiku-5-5"


class ConfigError(RuntimeError):
    """A required setting is missing."""


class EmptyResponseError(RuntimeError):
    """The model finished without returning any text."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise ConfigError(
                f"missing environment variables: {', '.join(missing)} "
                "(set them in .env and run with `uv run --env-file .env ...`)"
            )
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
        )


def make_client(config: LLMConfig) -> OpenAI:
    """Created on demand, never at import time, so tests and CI need no settings."""
    return OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=60.0, max_retries=2)


@dataclass
class ExplainResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None  # "stop" = finished, "length" = cut off by max_tokens
    ttft_s: float | None  # time to first token
    total_s: float


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a call in USD. Raises KeyError for a model with no price."""
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def build_messages(line: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": line},
    ]


def explain_line(
    line: str,
    client: OpenAI,
    model: str,
    on_text: Callable[[str], None] | None = None,
    max_tokens: int = 300,
) -> ExplainResult:
    """Stream an explanation of one log line, calling on_text as each piece arrives."""
    start = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    usage = None
    finish_reason: str | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(line),
        temperature=0,
        reasoning_effort="none",  # no thinking: this is a simple task
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices:  # the final usage-only chunk has no choices
            choice = chunk.choices[0]
            piece = choice.delta.content
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - start
                parts.append(piece)
                if on_text is not None:
                    on_text(piece)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if chunk.usage:
            usage = chunk.usage

    text = "".join(parts).strip()
    if not text:
        raise EmptyResponseError(f"{model} returned no text (finish_reason={finish_reason})")

    return ExplainResult(
        model=model,
        text=text,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        finish_reason=finish_reason,
        ttft_s=ttft,
        total_s=time.perf_counter() - start,
    )


def format_footer(result: ExplainResult) -> str:
    tokens = f"{result.input_tokens} in / {result.output_tokens} out tokens"
    ttft = f"{result.ttft_s:.2f}s" if result.ttft_s is not None else "n/a"
    timing = f"first token {ttft} · total {result.total_s:.2f}s"
    if result.model in PRICES:
        cost = f"${cost_usd(result.model, result.input_tokens, result.output_tokens):.6f}"
    else:
        ref = cost_usd(REFERENCE_MODEL, result.input_tokens, result.output_tokens)
        cost = f"$0 local (≈ ${ref:.6f} on {REFERENCE_MODEL})"
    return f"model {result.model} · {tokens} · {timing} · {cost}"
