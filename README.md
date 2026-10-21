# Advanced RAG: production branch

![Advanced RAG: chunks are stored once, searched with hybrid search, reranked, and answered with a cited source](docs/advanced-rag/thumbnail.svg)

> **You're on the `optimization` branch:** the guide's optimizations implemented, measured and tested, packaged to run as a service. Clone it, drop your `.txt` files into `data/`, and run `docker compose up`.
> Learning RAG step by step? The [`main`](https://github.com/nettanvirdev/advanced_rag/tree/main) branch keeps the code simple, with an 11-step guide.

Hybrid search (BM25 + semantic) → cross-encoder reranking → an LLM answer that cites its source, behind a FastAPI service. Exact and semantic answer caches, streaming, an incremental index, and a CI quality gate.

```
POST /ask {"question": "How long do cats sleep?"}
→ {"answer": "Cats usually sleep 12–16 hours per day … [cat.txt#25]", "cached": null}         fresh: retrieval + LLM
→ {"answer": "Cats usually sleep 12–16 hours per day … [cat.txt#25]", "cached": "exact"}      same question again
POST /ask {"question": "How many hours a day does a cat sleep?"}
→ {"answer": "Cats usually sleep 12–16 hours per day … [cat.txt#25]", "cached": "semantic"}   same meaning, no LLM call
POST /ask {"question": "What is the capital of France?"}
→ {"answer": "I don't know", "cached": null}                                                 no LLM call
```

---

## Quick start (Docker)

```bash
git clone -b optimization https://github.com/nettanvirdev/advanced_rag.git
cd advanced_rag
cp .env.example .env        # set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
docker compose up --build
```

Open **http://localhost:8000/docs**. The first build downloads PyTorch (CPU-only) and bakes both models into the image, so containers start without any downloads. On startup the API indexes `data/`. Only new or changed text is embedded, so a restart with unchanged files takes about a second.

**Use your own documents:** put `.txt` files in `data/` (replacing the 4 practice articles), then `docker compose restart api`. Removed files leave the index; edited files are re-embedded; cached answers are cleared when anything changed.

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) and, optionally, Redis.

```bash
uv sync
cp .env.example .env
uv run advanced-rag build
uv run uvicorn advanced_rag.api:app --port 8000
```

No Redis? Leave `REDIS_URL` empty. Everything works without the exact cache.

---

## API

| Endpoint | What it does |
|---|---|
| `POST /ask` | `{"question", "k"?, "sources"?}` → `{"answer", "cached": "exact" \| "semantic" \| null}` |
| `POST /ask/stream` | Same input. The answer streams as plain text while the LLM writes it (`curl -N`) |
| `GET /search?q=&k=&source=` | Retrieval only, no LLM: the chunks an answer would be built from |
| `GET /health` | `{"status", "chunks", "redis"}`: **503** when the index is empty, so a broken deploy fails its health check |

- **Auth:** set `API_KEY` and every endpoint except `/health` requires an `X-API-Key` header.
- **Filters:** `"sources": ["cat.txt"]` (or `?source=cat.txt`) searches only those files. Filtered answers are cached separately.
- **Errors:** bad input is a `422` (empty, whitespace-only or over 1,000 characters; `k` outside 1–10). An LLM that times out or is down is a `503` after 2 retries, never a hanging request or a stack trace. If Redis goes down, answers keep working: Redis is skipped for 30 s after a failure, and `/health` reports `"redis": "down"`.
- **Logs:** one line per request with its total time, plus per-stage timings (`retrieve_ms`, `llm_ms`, `llm_first_token_ms`), the cache result, the top chunk and score, and the LLM's token usage. Every response carries an `X-Response-Time-Ms` header.

## Configuration

Everything has a tested default; only the three `LLM_*` values are required. See [`.env.example`](.env.example).

| Variable | Default | |
|---|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | | any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Ollama, …) |
| `LLM_TIMEOUT` / `LLM_MAX_TOKENS` | `60` / `0` (no cap) | seconds per call / answer length cap |
| `API_KEY` | empty (open) | require `X-API-Key` |
| `REDIS_URL` | empty (no exact cache) | Compose sets it for you |
| `CACHE_TTL_SECONDS` | `86400` | both caches |
| `SEMANTIC_CACHE_THRESHOLD` | `0.85` | `0` turns the semantic cache off |
| `RERANK_CANDIDATES` | `20` | the measured sweet spot |
| `EMBEDDING_MODEL`, `RERANK_MODEL` | MiniLM-L6 models | change the embedding model → `advanced-rag build` |
| `DATA_DIR`, `CHROMA_DIR` | `data`, `storage/chroma` | |

---

## What's optimized, and what it measured

Same machine, same 210 chunks, the 19 answerable questions in [`eval/questions.json`](eval/questions.json). Quality must not drop: retrieval stayed at **95% hit rate / MRR 0.84** and answers at **19/19 correct, 19/19 cited, 3/3 refusals** with everything below turned on.

| # | Optimization ([guide](docs/advanced-rag/optimization/README.md)) | Status | Before → after |
|---|---|---|---|
| 1 | Precomputed BM25 (an inverted index built once) | ✅ | **320 ms → 0.18 ms** per search, identical results on every eval question (tested) |
| 2 | Batching + embedding caches | ✅ | Repeat query embedding **~7 ms → ~0**; restart with unchanged data re-embeds **0 of 210** chunks; moved text reuses its vector |
| 3 | Semantic cache, with a safety guard | ✅ | A paraphrase skips the **3–5 s** LLM call. It's reused only if the new question retrieves the **same best chunk**, so "How long do *pythons* sleep?" never gets the cat answer (tested) |
| 4 | Rerank pool | ✅ | Kept at **20** (measured: 40 adds 158 ms for no gain); configurable |
| 5 | Faster models (ONNX) | ❌ skipped | `sentence-transformers[onnx]` can't install with sentence-transformers 6 (`optimum-onnx` pins `transformers<4.58`). The image uses **CPU-only PyTorch** instead |
| 6 | Streaming | ✅ | `POST /ask/stream`; the first piece is awaited before the 200 is sent, so outages are still a clean 503 |
| 7 | LLM cost and latency | ✅ | Rules first (prefix-cacheable), optional `max_tokens`, token usage logged, timeout + retries |
| 8 | Better retrieval | ◐ | Source filters ✅. Query rewriting measured **+4.5 s for 0 gain**, so it stays off; HyDE/multi-query add the same LLM cost and aren't included |
| 9 | Production | ✅ | Docker + Compose, models baked in, non-root, healthcheck, readiness `/health`, API key, CI with the quality gate |

**Where a fresh question's time goes now:** cache check ~1 ms → hybrid search **~11 ms** (was ~228) → rerank ~160 ms → LLM 3–5 s. Retrieval: **~170 ms** (was ~380).

Reproduce: `uv run python -m advanced_rag.evaluate --answers`.

## Production notes

- **One worker per container.** Chroma's on-disk index belongs to one process. FastAPI still serves requests concurrently in its thread pool. Scale out with more containers, each with its own `index` volume.
- **Bigger corpora:** BM25 lives in RAM (fine to roughly a million chunks), and startup re-chunks every file to detect changes. Beyond that, move keyword search to OpenSearch and track file modification times.
- **Rate limiting and TLS** belong in front of the app (a reverse proxy or API gateway).
- **GPU:** remove the `[tool.uv.sources]` block in `pyproject.toml` to get CUDA PyTorch.

## Tests

```bash
uv run pytest
```
126 tests, all offline: a fake LLM and `fakeredis`, and every test gets its own temporary storage, so your Redis and `storage/` are never touched. `tests/test_evaluate.py` is a quality gate: CI fails if retrieval drops below 90% hit rate. CI also builds the Docker image.

## Project structure

```
src/advanced_rag/
├── config.py          every setting, with defaults
├── loaders.py         files → Document(text, source)
├── chunking.py        sentence packing, ≤200 tokens, 1 overlap
├── embeddings.py      embed(), embed_query() (cached)
├── vectorstore.py     ChromaDB: incremental build_index(), search() with filters
├── retrieval/
│   ├── bm25.py        BM25Index: precomputed postings
│   ├── hybrid.py      semantic + BM25, fused with RRF
│   ├── rerank.py      cross-encoder
│   └── rewrite.py     LLM query rewriting (off)
├── cache.py           exact (Redis) + semantic (Chroma) caches, failure-tolerant
├── llm.py             ask(), stream(): timeout, retries, usage logging
├── rag.py             retrieve(), answer(), answer_stream()
├── evaluate.py        hit rate, MRR, answer checks
├── api.py             FastAPI service
└── __init__.py        CLI: advanced-rag build | clear-cache
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Data and license
The practice texts in `data/` are English Wikipedia articles, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (see [`data/SOURCES.md`](data/SOURCES.md)). The code and guides are licensed under the [Apache License 2.0](LICENSE).

The step-by-step guides are published at **[nettanvir.dev/docs](https://nettanvir.dev/docs/advanced-rag)**.
