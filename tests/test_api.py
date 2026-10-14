import json

import pytest
from fastapi.testclient import TestClient

from oncall_copilot.api import app, get_client_factory
from tests.fakes import FakeClient, connection_error, ok_client, text_chunk, usage_chunk

LINE = "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6"


@pytest.fixture
def client(monkeypatch):
    for name in ("LLM_PROVIDER", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def use_fake(fake: FakeClient) -> None:
    app.dependency_overrides[get_client_factory] = lambda: (lambda provider: fake)


def events(response) -> list:
    lines = [ln.removeprefix("data: ") for ln in response.text.split("\n\n") if ln.startswith("data: ")]
    return [ln if ln == "[DONE]" else json.loads(ln) for ln in lines]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_explain_streams_text_then_summary_then_done(client):
    use_fake(FakeClient([text_chunk("Worker "), text_chunk("crashed.", finish="stop"), usage_chunk(50, 9)]))

    response = client.post("/explain", json={"line": LINE})
    got = events(response)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert got[0] == {"text": "Worker "} and got[1] == {"text": "crashed."}
    assert got[2]["provider"] == "ollama" and got[2]["output_tokens"] == 9
    assert got[-1] == "[DONE]"


def test_missing_line_is_422(client):
    assert client.post("/explain", json={}).status_code == 422


def test_empty_line_is_422(client):
    assert client.post("/explain", json={"line": ""}).status_code == 422


def test_unknown_provider_is_400(client):
    response = client.post("/explain", json={"line": LINE, "provider": "openai"})

    assert response.status_code == 400
    assert "unknown provider" in response.json()["detail"]


def test_model_failure_becomes_an_error_event(client):
    use_fake(FakeClient(error=connection_error()))  # only ollama in the chain: no GROQ_API_KEY

    got = events(client.post("/explain", json={"line": LINE}))

    assert got[0]["error"] == "AllProvidersFailedError"
    assert got[-1] == "[DONE]"


def test_interrupted_stream_reports_partial_answer(client):
    use_fake(FakeClient([text_chunk("half "), text_chunk("more")], error=connection_error(), fail_after=1))

    got = events(client.post("/explain", json={"line": LINE}))

    assert got[0] == {"text": "half "}
    assert got[1]["error"] == "PartialResponseError"


def test_sends_the_line_to_the_model(client):
    fake = ok_client()
    use_fake(fake)

    client.post("/explain", json={"line": LINE})

    assert fake.calls[0]["messages"][-1]["content"] == LINE
