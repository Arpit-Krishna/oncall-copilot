# Learning log

- **Day 1 (12 Oct):** Built project skeleton, LogRecord model, Apache log loader with skip tracking, stats CLI, and first pytest suite. Learned Pydantic `field_validator` for custom timestamp parsing.

- **Day 2 (13 Oct):** Added streaming `explain` command with token counts, timing, and cost estimation. Learned OpenAI SDK streaming with `stream_options={"include_usage": True}` for token accounting.

- **Day 3 (14 Oct):** Added Ollama + Groq providers with automatic fallback chain, JSON call logging, and a streaming FastAPI `/explain` endpoint. Learned that `reasoning_effort` and provider-specific params go through `**extra` so the core code stays provider-agnostic.

- **Day 4 (19 Oct):** Added structured `LogEvent` extraction (`parse` command). Key lessons:
  - **Few-shot examples leak answers.** Putting a real dataset line in the prompt means the model copies it verbatim instead of reasoning. Use synthetic examples or exclude example lines from eval.
  - **Catch `ValidationError`, not `Exception`.** Broad except in a retry loop hides real bugs as "retry" — only catch what you intend to retry on.
  - **Provider-specific params need `extra_body`.** `reasoning_format` is Groq-only; the OpenAI SDK rejects unknown top-level kwargs. Tested without it and Groq returns valid JSON anyway.
  - **Per-attempt logging matters.** Logging only the final result hides how many retries happened and what failed. Each attempt gets its own log line with `attempt`, `valid`, and token counts.
  - **JSON output over labelled text.** `model_dump_json()` pipes into `jq` and downstream tools; labelled `print()` lines don't.
  - **Mistral classifies weakly** on this task (line 2: `severity=info, event_type=other` for an error-state line). qwen3.5:9b and Groq qwen3.8-27b both get it right. Model choice matters for structured extraction even when all models produce valid JSON.
