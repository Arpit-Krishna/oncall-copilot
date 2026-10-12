"""A stand-in for openai.OpenAI that replays chunks we choose and records requests."""

from types import SimpleNamespace as NS


def text_chunk(text: str, finish: str | None = None) -> NS:
    return NS(choices=[NS(delta=NS(content=text), finish_reason=finish)], usage=None)


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> NS:
    # The last streamed chunk: no choices, only token counts.
    return NS(choices=[], usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


class FakeClient:
    def __init__(self, chunks=(), error: Exception | None = None):
        self.chunks = list(chunks)
        self.error = error
        self.calls: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return iter(self.chunks)
