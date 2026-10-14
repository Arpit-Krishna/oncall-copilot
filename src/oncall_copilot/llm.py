import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from oncall_copilot.providers import Provider

SYSTEM_PROMPT = (
    "You are an SRE explaining Apache log lines to an on-call engineer: for the given line, "
    "state what the line means, the likely cause, and one next step to check. "
    "Keep the reply under 120 words."
)

# Local runs are free; we also show what the same tokens would cost on this paid model.
REFERENCE_MODEL = "claude-haiku-5-5"
REFERENCE_PRICE = (0.10, 0.50)  # USD per million tokens (input, output)

call_log = logging.getLogger("oncall_copilot.calls")


# --- errors -----------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for our own LLM failures."""


class EmptyResponseError(LLMError):
    """The model finished without returning any text."""


class PartialResponseError(LLMError):
    """The stream broke after text was already sent. Never retried elsewhere."""


class AllProvidersFailedError(LLMError):
    """Every provider in the chain failed with a fallback-worthy error."""


# Errors another provider might not have: worth trying the next one.
# APITimeoutError is a subclass of APIConnectionError, so timeouts are covered too.
FALLBACK_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


def should_fall_back(exc: BaseException) -> bool:
    if isinstance(exc, FALLBACK_ERRORS):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


# --- client, cost, result ---------------------------------------------------


def make_client(provider: Provider) -> AsyncOpenAI:
    """The SDK retries connection errors, 429s and 5xx itself (max_retries) with backoff."""
    return AsyncOpenAI(
        base_url=provider.base_url, api_key=provider.api_key(), timeout=60.0, max_retries=2
    )


def cost_usd(price: tuple[float, float], input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = price
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class ExplainResult:
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None  # "stop" = finished, "length" = cut off by max_tokens
    ttft_s: float | None  # time to first token
    total_s: float
    cost_usd: float | None  # None for a free/local provider
    reference_cost_usd: float  # the same tokens at REFERENCE_PRICE


def build_messages(line: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": line},
    ]


# --- one provider -----------------------------------------------------------


async def explain_line(
    line: str,
    client: AsyncOpenAI,
    provider: Provider,
    on_text: Callable[[str], None] | None = None,
    max_tokens: int = 300,
) -> ExplainResult:
    """Stream an explanation from one provider, calling on_text as each piece arrives."""
    start = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    usage = None
    finish_reason: str | None = None

    stream = await client.chat.completions.create(
        model=provider.model,
        messages=build_messages(line),
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        **provider.extra,
    )

    try:
        async for chunk in stream:
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
    except openai.OpenAIError as exc:
        if parts:  # text already went out: switching providers would stitch two answers
            raise PartialResponseError(
                f"{provider.name} failed after {len(parts)} pieces of text: {exc}"
            ) from exc
        raise

    text = "".join(parts).strip()
    if not text:
        raise EmptyResponseError(
            f"{provider.name}/{provider.model} returned no text (finish_reason={finish_reason})"
        )

    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return ExplainResult(
        provider=provider.name,
        model=provider.model,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        ttft_s=ttft,
        total_s=time.perf_counter() - start,
        cost_usd=cost_usd(provider.price, input_tokens, output_tokens) if provider.price else None,
        reference_cost_usd=cost_usd(REFERENCE_PRICE, input_tokens, output_tokens),
    )


# --- several providers ------------------------------------------------------


def log_call(record: dict) -> None:
    """One JSON line per attempt: the record of truth for cost and latency."""
    call_log.info(json.dumps(record))


async def explain_with_fallback(
    providers: list[Provider],
    line: str,
    on_text: Callable[[str], None] | None = None,
    client_factory: Callable[[Provider], AsyncOpenAI] | None = None,
    log: Callable[[dict], None] = log_call,
) -> ExplainResult:
    """Try providers in order. Fall back only on errors another provider might not have,
    and only before any text was sent."""
    if not providers:
        raise ValueError("providers must not be empty")
    factory = client_factory or make_client  # looked up at call time, so tests can swap it

    failed: list[str] = []
    for provider in providers:
        start = time.perf_counter()
        try:
            result = await explain_line(line, factory(provider), provider, on_text=on_text)
        except (openai.OpenAIError, LLMError) as exc:  # bugs in our own code are not caught
            log(
                {
                    "event": "llm_call",
                    "provider": provider.name,
                    "model": provider.model,
                    "status": "error",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                    "total_s": round(time.perf_counter() - start, 3),
                    "fallback_from": failed[-1] if failed else None,
                }
            )
            if should_fall_back(exc):
                failed.append(provider.name)
                continue
            raise

        log(
            {
                "event": "llm_call",
                "provider": result.provider,
                "model": result.model,
                "status": "ok",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "ttft_s": round(result.ttft_s, 3) if result.ttft_s is not None else None,
                "total_s": round(result.total_s, 3),
                "cost_usd": result.cost_usd,
                "finish_reason": result.finish_reason,
                "fallback_from": failed[-1] if failed else None,
            }
        )
        return result

    raise AllProvidersFailedError(f"all providers failed: {', '.join(failed)}")


def format_footer(result: ExplainResult) -> str:
    tokens = f"{result.input_tokens} in / {result.output_tokens} out tokens"
    ttft = f"{result.ttft_s:.2f}s" if result.ttft_s is not None else "n/a"
    timing = f"first token {ttft} · total {result.total_s:.2f}s"
    if result.cost_usd is not None:
        cost = f"${result.cost_usd:.6f}"
    else:
        cost = f"$0 local (≈ ${result.reference_cost_usd:.6f} on {REFERENCE_MODEL})"
    return f"{result.provider} {result.model} · {tokens} · {timing} · {cost}"
