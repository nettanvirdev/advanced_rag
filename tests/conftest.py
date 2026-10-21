from types import SimpleNamespace

import fakeredis
import pytest

from advanced_rag import cache, llm, vectorstore


@pytest.fixture
def fake_llm(monkeypatch):
    """Swap the real API client for a fake one. Gives the test the list of messages the fake receives."""
    sent = []

    def fake_create(model, messages):
        sent.extend(messages)
        # shaped like a real response: response.choices[0].message.content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fake reply"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm, "client", fake_client)  # undone automatically after the test
    return sent


@pytest.fixture
def fake_redis(monkeypatch):
    """Swap the real Redis for an in-memory fake, so tests never touch your real Redis."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache, "client", fake)
    return fake


# building the real index takes a few seconds, so build it once for the whole test run
@pytest.fixture(scope="session")
def index(tmp_path_factory):
    collection = vectorstore.build_index(tmp_path_factory.mktemp("chroma"))
    return collection, vectorstore.all_chunks(collection)
