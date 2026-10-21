from types import SimpleNamespace

import fakeredis
import pytest

from advanced_rag import cache, config, llm, vectorstore
from advanced_rag.retrieval.bm25 import BM25Index


@pytest.fixture
def fake_llm(monkeypatch):
    """Swap the real API client for a fake one. Gives the test the list of messages the fake receives."""
    sent = []

    def fake_create(model, messages, stream=False, **kwargs):
        sent.extend(messages)
        if stream:
            # shaped like real stream events: event.choices[0].delta.content, a few characters at a time
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in ["fake ", "reply"]])
        # shaped like a real response: response.choices[0].message.content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fake reply"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm, "client", fake_client)  # undone automatically after the test
    return sent


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Every test gets an in-memory fake Redis, so no test can ever touch your real Redis."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache, "client", fake)
    return fake


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    """Every test gets its own empty storage folder (and semantic cache), never your real storage/."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(cache, "_semantic", None)  # reopened lazily inside tmp_path on first use
    monkeypatch.setattr(cache, "_down_until", 0.0)  # no test inherits another test's Redis outage


# building the real index takes a few seconds, so build it once for the whole test run
@pytest.fixture(scope="session")
def index_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("index")
    vectorstore.build_index(path)
    return path


@pytest.fixture(scope="session")
def index(index_dir):
    collection = vectorstore.get_collection(index_dir)
    return collection, BM25Index(vectorstore.all_chunks(collection))
