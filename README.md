# On-call Copilot

An AI assistant that investigates production incidents from logs, metrics and runbooks. Work in progress, built in public as a 4-week sprint.

uv sync
uv run copilot stats data/apache_2k.log
uv run pytest
## Explain a log line with an LLM

Copy `.env.example` to `.env`. Ollama runs locally for free; Groq is the fallback (free-tier key).

uv run --env-file .env copilot explain data/apache_2k.log --line 2
uv run --env-file .env copilot explain data/apache_2k.log --line 2 --provider groq

## HTTP API (server-sent events)

uv run --env-file .env uvicorn oncall_copilot.api:app --reload
curl -N localhost:8000/explain -H 'content-type: application/json' -d '{"line": "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"}'

## Tests

uv run pytest                          # fast, no network (what CI runs)
uv run --env-file .env pytest -m llm   # live calls to Ollama and Groq
