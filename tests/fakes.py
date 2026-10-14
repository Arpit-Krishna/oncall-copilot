"""A stand-in for openai.AsyncOpenAI that replays chunks we choose and records requests."""

from types import SimpleNamespace as NS

import httpx2
import openai

from oncall_copilot.providers import Provider


def text_chunk(text: str, finish: str | None = None) -> NS:
    return NS(choices=[NS(delta=NS(content=text), finish_reason=finish)], usage=None)


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> NS:
    # The last streamed chunk: no choices, only token counts.
    return NS(choices=[], usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def provider(name: str = "p1", price: tuple[float, float] | None = None) -> Provider:
    return Provider(
        name=name,
        base_url=f"http://{name}/v1",
        model=f"{name}-model",
        extra={"reasoning_effort": "none"},
        price=price,
    )


# --- errors as the real SDK raises them --------------------------------------

_REQUEST = httpx2.Request("POST", "http://fake/v1/chat/completions")


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def status_error(code: int) -> openai.APIStatusError:
    response = httpx2.Response(code, request=_REQUEST)
    error_class = {
        400: openai.BadRequestError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }.get(code, openai.APIStatusError)
    return error_class(f"HTTP {code}", response=response, body=None)


class FakeClient:
    """`error` is raised before streaming starts; `fail_after` raises it after that many chunks."""

    def __init__(self, chunks=(), error: Exception | None = None, fail_after: int | None = None):
        self.chunks = list(chunks)
        self.error = error
        self.fail_after = fail_after
        self.calls: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and self.fail_after is None:
            raise self.error
        return self._stream()

    async def _stream(self):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise self.error
            yield chunk


def ok_client(text: str = "ok") -> FakeClient:
    return FakeClient([text_chunk(text, finish="stop"), usage_chunk(10, 5)])
