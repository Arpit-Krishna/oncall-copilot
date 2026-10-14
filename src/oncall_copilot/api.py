"""HTTP API: `uv run --env-file .env uvicorn oncall_copilot.api:app --reload`."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from oncall_copilot.llm import LLMError, explain_with_fallback, make_client
from oncall_copilot.logs import configure_logging
from oncall_copilot.providers import ConfigError, Provider, provider_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


app = FastAPI(title="On-call Copilot API", lifespan=lifespan)


class ExplainRequest(BaseModel):
    line: str = Field(min_length=1, description="One raw log line")
    provider: str | None = Field(default=None, description="Provider to try first")


# Dependencies: tests replace these through app.dependency_overrides.
def get_client_factory() -> Callable[[Provider], AsyncOpenAI]:
    return make_client


def sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/explain")
async def explain(
    req: ExplainRequest,
    client_factory: Annotated[Callable[[Provider], AsyncOpenAI], Depends(get_client_factory)],
) -> StreamingResponse:
    try:
        chain = provider_chain(req.provider)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def events() -> AsyncIterator[str]:
        # explain_with_fallback pushes text through a callback; a queue turns that into
        # something this generator can `async for` over and send as it arrives.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def run():
            try:
                return await explain_with_fallback(
                    chain, req.line, on_text=queue.put_nowait, client_factory=client_factory
                )
            finally:
                queue.put_nowait(None)  # tell the reader we're finished, success or not

        task = asyncio.create_task(run())
        try:
            while (piece := await queue.get()) is not None:
                yield sse({"text": piece})
            try:
                result = await task
            except (LLMError, openai.OpenAIError) as e:
                yield sse({"error": type(e).__name__, "detail": str(e)})
            else:
                yield sse(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "ttft_s": result.ttft_s,
                        "total_s": result.total_s,
                        "cost_usd": result.cost_usd,
                        "finish_reason": result.finish_reason,
                    }
                )
            yield sse("[DONE]")
        finally:
            if not task.done():  # client went away mid-stream: stop paying for tokens
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
