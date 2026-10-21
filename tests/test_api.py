import openai
import pytest
from fastapi.testclient import TestClient

from advanced_rag import cache, config, llm, vectorstore
from advanced_rag.api import app


@pytest.fixture
def client(index_dir, fake_llm, monkeypatch, tmp_path):
    # TestClient talks to the app in memory, no real server. Starting it runs the lifespan,
    # which syncs the (already built) test index: nothing changed, so nothing is re-embedded
    monkeypatch.setattr(config, "CHROMA_DIR", index_dir)
    # the shared index must not share cached answers between tests
    monkeypatch.setattr(cache, "_semantic", vectorstore.get_collection(tmp_path, name=cache.SEMANTIC_COLLECTION))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 210, "redis": "ok"}


def test_health_is_503_without_an_index(monkeypatch, tmp_path, fake_llm):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # no .txt files -> nothing to index
    with TestClient(app) as empty:
        response = empty.get("/health")
    assert response.status_code == 503
    assert response.json()["chunks"] == 0


def test_ask_answers_and_then_comes_from_the_cache(client, fake_llm):
    first = client.post("/ask", json={"question": "How long do cats sleep?"})
    assert first.status_code == 200
    assert first.json() == {"answer": "fake reply", "cached": None}
    assert "X-Response-Time-Ms" in first.headers

    second = client.post("/ask", json={"question": "how long do cats sleep?"})
    assert second.json() == {"answer": "fake reply", "cached": "exact"}
    assert len(fake_llm) == 2  # the LLM was called once (system + user message)


def test_ask_rejects_bad_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422  # too short
    assert client.post("/ask", json={"question": "   "}).status_code == 422  # only spaces
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422  # too long
    assert client.post("/ask", json={"question": "hi", "k": 99}).status_code == 422  # k above 10
    assert client.post("/ask", json={}).status_code == 422  # question missing


def test_ask_can_be_limited_to_some_files(client, fake_llm):
    response = client.post("/ask", json={"question": "How long do cats sleep?", "sources": ["coffee.txt"]})
    assert response.json()["answer"] == "I don't know"  # the answer is in cat.txt, which was excluded
    assert fake_llm == []


def test_ask_stream_sends_the_answer_in_pieces(client):
    response = client.post("/ask/stream", json={"question": "How long do cats sleep?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "fake reply"
    # the finished stream was cached, so the normal endpoint now answers from the cache
    assert client.post("/ask", json={"question": "How long do cats sleep?"}).json()["cached"] == "exact"


def test_llm_outage_is_a_clean_503(client, monkeypatch):
    def broken(**kwargs):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(llm.client.chat.completions, "create", broken)
    for path in ["/ask", "/ask/stream"]:
        response = client.post(path, json={"question": "How long do cats sleep?"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]


def test_api_key_is_required_when_set(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "s3cret")
    assert client.get("/search", params={"q": "cats"}).status_code == 401
    assert client.get("/search", params={"q": "cats"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/search", params={"q": "cats"}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/ask", json={"question": "cats"}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_search_returns_ranked_sources(client):
    sources = client.get("/search", params={"q": "How long do cats sleep?"}).json()
    assert sources[0]["id"] == "cat.txt#25"
    assert set(sources[0]) == {"id", "score", "text"}


def test_search_source_filter(client):
    sources = client.get("/search", params={"q": "Who created Python?", "source": ["python_language.txt"], "k": 5}).json()
    assert sources and all(s["id"].startswith("python_language.txt#") for s in sources)


def test_search_rejects_bad_k(client):
    assert client.get("/search", params={"q": "cats", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "cats", "k": 500}).status_code == 422


def test_search_for_nonsense_is_empty(client):
    assert client.get("/search", params={"q": "What is the capital of France?"}).json() == []
