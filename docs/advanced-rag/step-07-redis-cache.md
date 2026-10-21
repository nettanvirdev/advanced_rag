---
title: "Step 7: Redis cache"
series: "Advanced RAG"
group: "steps"
step: 7
slug: "step-07-redis-cache"
date: "2026-10-21"
tag: "RAG"
excerpt: "Wire hybrid search to the LLM in rag.py, then cache every answer in Redis: a repeated question drops from 4,886 ms to 1 ms."
minutes: 15
---

# Step 7: Redis cache

**You'll build:** `src/advanced_rag/rag.py` (`build_context()`, `answer()`), `src/advanced_rag/cache.py` (`make_key()`, `get()`, `put()`, `clear()`), and `llm.py` copied from `simple_rag`.
**You'll learn:** what to cache and how to name it (key normalization, hashing, a prefix, expiry), Redis in Docker, and testing with a fake Redis.

---

## 1. Theory

An LLM call is **slow** (seconds) and **costs money**, and many people ask the same questions. So the first time a question is answered, store the answer; next time, return it instantly. **Redis** is an in-memory key → value store that answers in about a millisecond. The hard part is the **key**: different spellings of the same question should share one key, but anything that changes the answer (like the model) must not. ([Theory: Caching](concepts.md#caching))

---

## Before vs after (measured)

| | Before (no cache) | After (Redis cache) |
|---|---|---|
| "How long do cats sleep?" the first time | 4,886 ms (search + LLM) | 4,886 ms |
| The same question again, as "how long do  cats sleep?" | another 4,886 ms and another LLM call | **1 ms** from Redis, no LLM call |
| "asdfgh" (nothing relevant found) | n/a | "I don't know" in **80 ms**, no LLM call |
| Answers cite their chunk | n/a | yes, e.g. `[cat.txt#25]` |

About **5,000× faster** on a repeat. Through the API in Step 11 the same effect is 3.1 s → 3.7 ms.

---

## 2. Code example

### 2.1 Run Redis
Redis is a separate program (a server). The easiest way to run it is Docker:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```
`-d` runs it in the background, and `-p 6379:6379` makes it reachable at `localhost:6379`. That matches `REDIS_URL=redis://localhost:6379/0` in your `.env`. Then install the Python client, plus a fake Redis for tests:
```bash
uv add redis openai
```
```bash
uv add --dev fakeredis
```

### 2.2 Copy the LLM pieces from `simple_rag`
`src/ragi/llm.py` → `src/advanced_rag/llm.py`, and `tests/conftest.py` + `tests/test_llm.py` (the fake LLM from `simple_rag`'s Step 11), renaming `from ragi` to `from advanced_rag`. No other changes.

### 2.3 The cache key, line by line
```python
normalized = " ".join(question.lower().split())
```
Lowercase, then `split()` (which splits on any run of spaces) and re-join with single spaces. `"How long do  CATS sleep?"` and `"how long do cats sleep?"` become the same text.

```python
digest = hashlib.sha256(f"{config.LLM_MODEL}|{normalized}".encode()).hexdigest()
```
- The **model name** goes in front: switching models must not serve the old model's answers.
- **sha256** turns any text into a fixed-length fingerprint like `9f2c…`: same input, same code, every time. Keys stay short and safe however long the question is.
- `.encode()` turns the text into bytes, which is what the hash function reads.

```python
return "advanced_rag:answer:" + digest
```
A **prefix**, so this project only ever touches its own keys in a Redis that may hold other data.

### 2.4 Save with an expiry
```python
client.set(make_key(question), answer, ex=24 * 60 * 60)   # ex = expire after this many seconds
```
Answers disappear after a day, so they can't stay stale forever.

### 2.5 The answer flow in `rag.py`
```
question → 1. in the cache? → yes: return it (no search, no LLM)
                            → no:  2. hybrid search
                                     nothing found? → "I don't know" (no LLM call)
                                   3. LLM answers from the chunks, citing [cat.txt#25]
                                   4. save the answer in the cache
```
The context labels every chunk with its ID, so the LLM can cite exactly where an answer came from:
```python
"\n\n".join(f"[{vectorstore.chunk_id(chunk)}]\n{chunk.text}" for _, chunk in hits)
```
The rules now ask for the chunk ID: `Cite the chunk id in square brackets, like [cat.txt#25].`

---

## 3. Goal

1. Copy `llm.py`, `conftest.py` and `test_llm.py` from `simple_rag` (2.2).
2. Create **`src/advanced_rag/cache.py`**:
   - `client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)` (`decode_responses=True` gives back `str` instead of bytes)
   - `PREFIX = "advanced_rag:answer:"`, `TTL_SECONDS = 24 * 60 * 60`
   - **`make_key(question)`**: normalize, hash with the model, add the prefix (2.3)
   - **`get(question)`**: `client.get(make_key(question))`, which is `None` when unseen
   - **`put(question, answer)`**: `client.set(...)` with `ex=TTL_SECONDS` (2.4)
   - **`clear() -> int`**: `keys = list(client.scan_iter(match=PREFIX + "*"))`; if any, `client.delete(*keys)`; return how many. `scan_iter` walks the keys in small batches instead of all at once.
3. Create **`src/advanced_rag/rag.py`** with `RULES`, `NO_ANSWER = "I don't know"`, and:
   - **`build_context(hits) -> str`**: the join from 2.5
   - **`answer(question, collection, chunks, k=3) -> str`**: the flow from 2.5. Only call `cache.put` when there's a result.
   - a `__main__` box that loads the collection and chunks once, then answers each question and prints the time, plus `from cache` when it was cached
4. In `vectorstore.py`'s `__main__`, after `--build`, call `cache.clear()`. New chunks can change answers, so cached answers from the old index are no longer trustworthy.

### ✅ Checkpoint
```bash
uv run python -m advanced_rag.rag
```
Ask `How long do cats sleep?` twice, the second time as `how long do  cats sleep?`:
```
ask> How long do cats sleep?
Cats typically sleep 12–16 hours per day, averaging 13–14 hours; some may sleep as much as 20 hours [cat.txt#25].
  (4886 ms)
ask> how long do  cats sleep?
Cats typically sleep 12–16 hours per day, averaging 13–14 hours; some may sleep as much as 20 hours [cat.txt#25].
  (1 ms, from cache)
```
The LLM's wording may differ on your machine; the timing pattern and the `[cat.txt#25]` citation are what to look for. Then try `asdfgh`: "I don't know" in about 80 ms.

---

## 4. Test

Two new fixtures go in `tests/conftest.py`, next to the copied `fake_llm`:

**`fake_redis`** swaps the real Redis for an in-memory fake during a test, the same way `fake_llm` swaps the API client. Tests never touch your real Redis, and work without Docker running.
```python
import fakeredis

@pytest.fixture
def fake_redis(monkeypatch):
    """Swap the real Redis for an in-memory fake, so tests never touch your real Redis."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache, "client", fake)
    return fake
```

**`index`** moves here from `test_hybrid.py`, now with **`scope="session"`**: built once for the whole test run and shared by the hybrid **and** RAG tests.
```python
@pytest.fixture(scope="session")
def index(tmp_path_factory):
    collection = vectorstore.build_index(tmp_path_factory.mktemp("chroma"))
    return collection, vectorstore.all_chunks(collection)
```

### `tests/test_cache.py`

**This test checks:** spelling differences share one key. **How:** extra spaces and capitals.
```python
def test_key_ignores_case_and_extra_spaces():
    assert cache.make_key("How long do cats sleep?") == cache.make_key("  how LONG do   cats sleep?")
```

**This test checks:** a different model gets a different key. **How:** `monkeypatch` changes `config.LLM_MODEL` just for this test.
```python
def test_key_changes_with_the_model(monkeypatch):
    before = cache.make_key("Who created Python?")
    monkeypatch.setattr(config, "LLM_MODEL", "another-model")
    assert cache.make_key("Who created Python?") != before
```

**This test checks:** save then read back. **How:** unseen gives `None`; after `put`, a differently-spelled question finds it.
```python
def test_put_then_get(fake_redis):
    assert cache.get("Who created Python?") is None
    cache.put("Who created Python?", "Guido van Rossum")
    assert cache.get("who created python?") == "Guido van Rossum"
```

**This test checks:** answers expire. **How:** `ttl` = seconds left, at most one day.
```python
def test_answers_expire_after_a_day(fake_redis):
    cache.put("q", "a")
    assert 0 < fake_redis.ttl(cache.make_key("q")) <= 24 * 60 * 60
```

**This test checks:** `clear()` never deletes other people's data. **How:** a key without our prefix must survive.
```python
def test_clear_removes_only_our_keys(fake_redis):
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    fake_redis.set("someone-else:data", "keep me")
    assert cache.clear() == 2
    assert fake_redis.get("someone-else:data") == "keep me"
```

### `tests/test_rag.py`

**This test checks:** the context format. **How:** each chunk is labelled with its ID.
```python
def test_build_context_labels_chunks_with_their_id():
    hits = [(0.9, Chunk(text="Cats sleep a lot.", source="cat.txt", index=25))]
    assert build_context(hits) == "[cat.txt#25]\nCats sleep a lot."
```

**This test checks:** the rules and the right chunk reach the LLM. **How:** the fake LLM records what it was sent.
```python
def test_answer_sends_rules_and_the_right_chunk(index, fake_llm, fake_redis):
    collection, chunks = index
    assert answer("How long do cats sleep?", collection, chunks) == "fake reply"
    assert fake_llm[0] == {"role": "system", "content": RULES}
    assert "[cat.txt#25]" in fake_llm[1]["content"]
```

**This test checks:** the second ask never reaches the LLM. **How:** one call sends exactly 2 messages (system + user); two calls would send 4.
```python
def test_second_ask_comes_from_the_cache(index, fake_llm, fake_redis):
    collection, chunks = index
    answer("How long do cats sleep?", collection, chunks)
    answer("how long do  cats sleep?", collection, chunks)  # same question, different spacing
    assert len(fake_llm) == 2  # system + user from the FIRST call only: the LLM was asked once
```

**This test checks:** nothing relevant means no LLM call at all. **How:** the fake LLM's list stays empty.
```python
def test_nothing_relevant_skips_the_llm(index, fake_llm, fake_redis):
    collection, chunks = index
    assert answer("asdfgh", collection, chunks) == NO_ANSWER
    assert fake_llm == []  # no LLM call at all
    assert cache.get("asdfgh") == NO_ANSWER
```

```bash
uv run pytest -v
```
Expected: **76 passed**. Afterwards, your real Redis holds exactly the keys it held before: every test used the fake.

---

## 5. Make it a module

`rag.answer()` is the heart of the app: everything before this step feeds it, and everything after improves or serves it. Step 8 measures how good its answers are. Step 9 changes its retrieval line from hybrid search to hybrid search + reranking. Step 11 puts it behind a web API. See the full query path in the [diagrams](diagrams.md).

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/cache.py</code> and <code>src/advanced_rag/rag.py</code></strong></summary>

`src/advanced_rag/cache.py`:
```python
import hashlib

import redis

from advanced_rag import config

# one connection for the whole program; decode_responses=True gives back str instead of bytes
client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

PREFIX = "advanced_rag:answer:"  # every key we write starts with this, so we never touch other data
TTL_SECONDS = 24 * 60 * 60  # answers expire after a day, so they can't stay stale forever


def make_key(question: str) -> str:
    # "How long do cats sleep?" and "how long do  cats sleep?" are the same question
    normalized = " ".join(question.lower().split())
    # a different model would give a different answer, so the model is part of the key.
    # sha256 turns any text into a short, fixed-length, safe key
    digest = hashlib.sha256(f"{config.LLM_MODEL}|{normalized}".encode()).hexdigest()
    return PREFIX + digest


def get(question: str) -> str | None:
    return client.get(make_key(question))  # None when we've never seen this question


def put(question: str, answer: str) -> None:
    client.set(make_key(question), answer, ex=TTL_SECONDS)  # ex = expire after this many seconds


def clear() -> int:
    # delete only our own keys; scan_iter walks them in small batches instead of all at once
    keys = list(client.scan_iter(match=PREFIX + "*"))
    if keys:
        client.delete(*keys)
    return len(keys)
```

`src/advanced_rag/rag.py` (as of this step; Steps 8–10 extend it):
```python
from advanced_rag import cache, vectorstore
from advanced_rag.chunking import Chunk
from advanced_rag.llm import ask
from advanced_rag.retrieval.hybrid import hybrid_search

RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know\". "
    "Cite the chunk id in square brackets, like [cat.txt#25]."
)
NO_ANSWER = "I don't know"


def build_context(hits: list[tuple[float, Chunk]]) -> str:
    # label every chunk with its id, so the LLM can cite exactly where an answer came from
    return "\n\n".join(f"[{vectorstore.chunk_id(chunk)}]\n{chunk.text}" for _, chunk in hits)


def answer(question: str, collection, chunks: list[Chunk], k: int = 3) -> str:
    # 1. asked before? answer from Redis: no search, no LLM call
    cached = cache.get(question)
    if cached is not None:
        return cached

    # 2. retrieve; nothing relevant means there's no point paying for an LLM call
    hits = hybrid_search(collection, chunks, question, k)
    if not hits:
        result = NO_ANSWER
    else:
        prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
        result = ask(prompt, system=RULES)

    # 3. remember it for next time
    if result:
        cache.put(question, result)
    return result


if __name__ == "__main__":
    import time

    collection = vectorstore.get_collection()
    chunks = vectorstore.all_chunks(collection)
    print(f"{len(chunks)} chunks loaded")
    while True:
        question = input("ask> ")
        was_cached = cache.get(question) is not None
        start = time.perf_counter()
        reply = answer(question, collection, chunks)
        ms = (time.perf_counter() - start) * 1000
        print(f"{reply}\n  ({ms:.0f} ms{', from cache' if was_cached else ''})")
```

Added to the `--build` branch of `vectorstore.py`'s `__main__`:
```python
        # new chunks can change answers, so cached answers from the old index are no longer trustworthy
        from advanced_rag import cache

        print(f"cleared {cache.clear()} cached answers")
```

</details>

---

**Previous:** [← Step 6: Hybrid retrieval](step-06-hybrid-retrieval.md)

**Next:** [Step 8: Evaluation →](step-08-evaluation.md)
