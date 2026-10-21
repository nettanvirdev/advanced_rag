---
title: "Step 11: FastAPI"
series: "Advanced RAG"
group: "steps"
step: 11
slug: "step-11-fastapi"
date: "2026-10-21"
tag: "RAG"
excerpt: "Serve the whole pipeline as a web API with three endpoints, input validation and auto-generated docs: 3.1 s for a new question, 3.7 ms from the cache."
minutes: 14
---

# Step 11: FastAPI

**You'll build:** `src/advanced_rag/api.py` with `GET /health`, `POST /ask` and `GET /search`.
**You'll learn:** FastAPI, request/response models with pydantic, input validation, the `lifespan` startup hook, and testing an API with `TestClient`.

---

## 1. Theory

Until now, only you could use the RAG, by typing into a terminal. A **web API** lets any program use it, whether that's a website, a phone app or another service, by sending HTTP requests and getting JSON back. **FastAPI** turns plain Python functions into API endpoints, checks every incoming request against the shapes you declare, and writes interactive documentation for you at `/docs`. ([Theory: Serving with FastAPI](concepts.md#serving-with-fastapi))

---

## Before vs after (measured)

| | Before (Step 10) | After (FastAPI) |
|---|---|---|
| Who can use it | you, in a terminal | any program, over HTTP |
| New question (`POST /ask`) | — | **3.1 s** |
| Same question again | — | **3.7 ms** (`"cached": true`) |
| Unanswerable question | — | **324 ms**, "I don't know", no LLM call |
| Retrieval only (`GET /search?q=9 bar`) | — | **236 ms**, top hit `coffee.txt#45` |
| Bad input (empty question, `k=99`) | no checks | rejected with **422** before any code runs |
| Loading the index | once per terminal session | once per **server start** (`lifespan`) |

---

## 2. Code example

### 2.1 Install and run
```bash
uv add "fastapi[standard]"
```
`[standard]` also installs `uvicorn` (the web server that runs the app) and `httpx` (which `TestClient` needs).

### 2.2 The smallest possible endpoint
```python
from fastapi import FastAPI

app = FastAPI(title="advanced_rag")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```
| Piece | Meaning |
|---|---|
| `app = FastAPI(...)` | the application: one object that knows every endpoint |
| `@app.get("/health")` | a **decorator**: "when someone sends GET to `/health`, run the function below" |
| `return {...}` | a dictionary; FastAPI turns it into JSON automatically |

### 2.3 Declaring what a request must look like
```python
from pydantic import BaseModel, Field

class Question(BaseModel):
    question: str = Field(min_length=1, examples=["How long do cats sleep?"])
    k: int = Field(default=3, ge=1, le=10)
```
| Piece | Meaning |
|---|---|
| `class Question(BaseModel)` | a **model**: the shape of the JSON a client must send, like a dataclass with checks |
| `question: str` | required, and must be text |
| `min_length=1` | an empty string is rejected |
| `k: int = Field(default=3, ge=1, le=10)` | optional, defaults to 3; `ge` = "greater or equal", `le` = "less or equal", so 1 to 10 |
| `examples=[...]` | shown on the `/docs` page |

If a request breaks any rule, FastAPI answers **422 Unprocessable Entity** with an explanation, and your function never runs.

### 2.4 Loading things once: `lifespan`
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = vectorstore.get_collection()                 # before the server starts serving
    app.state.chunks = vectorstore.all_chunks(app.state.collection)
    yield                                                               # ← the server runs while we wait here

app = FastAPI(title="advanced_rag", lifespan=lifespan)
```
Everything **before** `yield` runs once at startup. `app.state` is a place to keep things every request can reuse, so the 210 chunks are read once, not on every question. You don't need to understand `async` here: copy the shape as it is.

### 2.5 Reading what was stored
An endpoint asks for the `request` to reach `app.state`:
```python
from fastapi import Request

@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.chunks)}
```

---

## 3. Goal

Create `src/advanced_rag/api.py` with the `lifespan` and `Question` model from above, plus two response models:
```python
class Answer(BaseModel):
    answer: str
    cached: bool

class Source(BaseModel):
    id: str
    score: float
    text: str
```
Then three endpoints:

| Endpoint | Steps |
|---|---|
| `GET /health` | return `{"status": "ok", "chunks": <how many chunks>}` |
| `POST /ask` (body: `Question`) | 1. `was_cached = cache.get(body.question) is not None` (check **before** answering, or it's always true afterwards) 2. `reply = rag.answer(body.question, collection, chunks, k=body.k)` 3. return `Answer(answer=reply, cached=was_cached)` |
| `GET /search?q=...&k=3` | `hits = rag.retrieve(q, collection, chunks, k)`, then return a `Source` per hit: `id=vectorstore.chunk_id(chunk)`, `score`, `text` |

`-> Answer` and `-> list[Source]` after a function tell FastAPI the response shape. It checks it and shows it on `/docs`.

### ✅ Checkpoint
Start the server (it loads the models and index first; wait for "Application startup complete"):
```bash
uv run uvicorn advanced_rag.api:app --port 8000
```
or, with auto-reload while you edit:
```bash
uv run fastapi dev src/advanced_rag/api.py
```
In a second terminal:
```bash
curl http://localhost:8000/health
```
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How long do cats sleep?\"}"
```
Run the `/ask` command **twice**. Expected:
```
{"status":"ok","chunks":210}
{"answer":"Cats usually sleep **12–16 hours per day**, averaging **13–14 hours**; some may sleep as much as **20 hours** [cat.txt#25].","cached":false}   ← ~3.1 s
{"answer":"Cats usually sleep **12–16 hours per day**, ...","cached":true}                                                                             ← ~3.7 ms
```
The exact wording depends on your LLM.
```bash
curl "http://localhost:8000/search?q=9%20bar&k=1"
```
Expected: one source with `"id":"coffee.txt#45"`. Then open **http://localhost:8000/docs** in a browser: every endpoint, with a **Try it out** button.

---

## 4. Test

`tests/test_api.py` never starts a real server. **`TestClient`** talks to the app in memory.
```python
import pytest
from fastapi.testclient import TestClient

from advanced_rag.api import app


@pytest.fixture
def client(index, fake_llm, fake_redis):
    with TestClient(app) as test_client:
        app.state.collection, app.state.chunks = index
        yield test_client
```
`with TestClient(app)` runs the `lifespan`, like a real start. Then the fixture swaps in the shared test index from `conftest.py`, so the tests don't depend on your `storage/` folder, and `fake_llm` + `fake_redis` keep them free and offline. `yield` hands the client to the test and cleans up afterwards.

**This test checks:** the server is up and sees every chunk. **How:** the test index has 210 chunks.
```python
def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 210}
```

**This test checks:** `/ask` answers, then serves the repeat from the cache. **How:** the second request (different capitals) must say `cached: true`, and the fake LLM must have received only one call (one system + one user message).
```python
def test_ask_answers_and_then_comes_from_the_cache(client, fake_llm):
    first = client.post("/ask", json={"question": "How long do cats sleep?"})
    assert first.status_code == 200
    assert first.json() == {"answer": "fake reply", "cached": False}

    second = client.post("/ask", json={"question": "how long do cats sleep?"})
    assert second.json() == {"answer": "fake reply", "cached": True}
    assert len(fake_llm) == 2
```

**This test checks:** bad input is refused. **How:** an empty question, `k` above 10, and a missing question must each get 422.
```python
def test_ask_rejects_bad_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "k": 99}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
```

**This test checks:** `/search` returns ranked sources in the promised shape. **How:** the cat question's first source is `cat.txt#25`, with exactly the fields `id`, `score`, `text`.
```python
def test_search_returns_ranked_sources(client):
    sources = client.get("/search", params={"q": "How long do cats sleep?"}).json()
    assert sources[0]["id"] == "cat.txt#25"
    assert set(sources[0]) == {"id", "score", "text"}
```

**This test checks:** nothing relevant means an empty list, not random chunks. **How:** the reranker scores everything about France below 0.
```python
def test_search_for_nonsense_is_empty(client):
    assert client.get("/search", params={"q": "What is the capital of France?"}).json() == []
```

```bash
uv run pytest -v
```
The whole project: **98 passed**.

---

## 5. Make it a module

`api.py` adds **no new logic**. It's a thin door in front of `rag.answer()` and `rag.retrieve()`, the same functions the terminal tools and `evaluate.py` use. That's the point: everything you measured in Steps 8–10 is exactly what the API serves. To go further (async endpoints, streaming answers word by word, Docker), see the [Optimization guide](optimization/README.md).

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/api.py</code></strong></summary>

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from advanced_rag import cache, rag, vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts: open the index and read the chunks for BM25.
    # Every request reuses them instead of loading them again.
    app.state.collection = vectorstore.get_collection()
    app.state.chunks = vectorstore.all_chunks(app.state.collection)
    yield


app = FastAPI(title="advanced_rag", description="Hybrid search + reranking + Redis cache over your .txt files.", lifespan=lifespan)


# request and response shapes: FastAPI checks incoming JSON against them and documents them at /docs
class Question(BaseModel):
    question: str = Field(min_length=1, examples=["How long do cats sleep?"])
    k: int = Field(default=3, ge=1, le=10)


class Answer(BaseModel):
    answer: str
    cached: bool


class Source(BaseModel):
    id: str
    score: float
    text: str


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.chunks)}


@app.post("/ask")
def ask(body: Question, request: Request) -> Answer:
    was_cached = cache.get(body.question) is not None
    reply = rag.answer(body.question, request.app.state.collection, request.app.state.chunks, k=body.k)
    return Answer(answer=reply, cached=was_cached)


@app.get("/search")
def search(q: str, request: Request, k: int = 3) -> list[Source]:
    # retrieval only, no LLM: handy to see which chunks an answer would be built from
    hits = rag.retrieve(q, request.app.state.collection, request.app.state.chunks, k)
    return [Source(id=vectorstore.chunk_id(chunk), score=score, text=chunk.text) for score, chunk in hits]
```

</details>

---

## 🎉 You built an advanced RAG

From an empty folder to a measured, cached, reranked RAG behind a web API:

| Step | What you built | Headline: before → after (measured) |
|---|---|---|
| 1. Project setup | uv project, config + loaders copied from simple_rag | 0 → **12 tests** passing |
| 2. Embeddings | `embed`, `dimension`, `count_tokens` | 210 chunks one by one 2.82 s → **one batch 1.48 s** |
| 3. Chunking: split and pack | `split_sentences`, `chunk_text` | model sees 2.1–10.5% of each article → **100%** (185 chunks) |
| 4. Chunking: overlap | `Chunk`, `chunk_documents` | naive overlap 203 tokens (over the limit) → **max 200**, 210 chunks |
| 5. ChromaDB | `vectorstore.py` | re-embed everything 1,482 ms → **~9 ms** per question |
| 6. Hybrid retrieval | BM25 + semantic, fused with RRF | MRR 0.76 → **0.79**; "Guido van Rossum" wrong → **right** |
| 7. Redis cache | `cache.py`, `rag.py` | repeat question 4,886 ms → **1 ms** |
| 8. Evaluation | `eval/questions.json`, `evaluate.py` | judged by eye → **measured**; found a real bug |
| 9. Reranking | cross-encoder `rerank()` | hit rate 89% → **95%**; answers 17/19 → **19/19** |
| 10. Query rewriting | `rewrite()`, `retrieve()` | 379 ms → 4,868 ms, no gain → **ships off** |
| 11. FastAPI | `api.py` | terminal only → **HTTP**: 3.1 s new, 3.7 ms cached |

**Keep going:**
- [Guide overview](README.md): all steps on one page
- [Advanced RAG Concepts](concepts.md): the theory and formulas behind every step
- [Diagrams](diagrams.md): where every function and model fits
- [Optimization guide](optimization/README.md): proven ways to make it faster and cheaper
- [Evaluation results](../../eval/results.md): the full numbers, every answer, so you can compare your run

---

**Previous:** [← Step 10: Query rewriting](step-10-query-rewriting.md)

**Next:** [Optimization guide →](optimization/README.md)
