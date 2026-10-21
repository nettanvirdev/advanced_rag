import pytest
from fastapi.testclient import TestClient

from advanced_rag.api import app


@pytest.fixture
def client(index, fake_llm, fake_redis):
    # TestClient talks to the app in memory, no real server. Starting it runs the lifespan;
    # then we swap in the test index, so tests don't depend on your storage/ folder
    with TestClient(app) as test_client:
        app.state.collection, app.state.chunks = index
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 210}


def test_ask_answers_and_then_comes_from_the_cache(client, fake_llm):
    first = client.post("/ask", json={"question": "How long do cats sleep?"})
    assert first.status_code == 200
    assert first.json() == {"answer": "fake reply", "cached": False}

    second = client.post("/ask", json={"question": "how long do cats sleep?"})
    assert second.json() == {"answer": "fake reply", "cached": True}
    assert len(fake_llm) == 2  # the LLM was called once (system + user message)


def test_ask_rejects_bad_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422  # too short
    assert client.post("/ask", json={"question": "hi", "k": 99}).status_code == 422  # k above 10
    assert client.post("/ask", json={}).status_code == 422  # question missing


def test_search_returns_ranked_sources(client):
    sources = client.get("/search", params={"q": "How long do cats sleep?"}).json()
    assert sources[0]["id"] == "cat.txt#25"
    assert set(sources[0]) == {"id", "score", "text"}


def test_search_for_nonsense_is_empty(client):
    assert client.get("/search", params={"q": "What is the capital of France?"}).json() == []
