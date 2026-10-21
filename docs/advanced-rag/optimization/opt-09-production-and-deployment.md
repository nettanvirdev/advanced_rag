---
title: "Optimization 9: Production and deployment"
series: "Advanced RAG"
group: "optimization"
step: 9
slug: "opt-09-production-and-deployment"
date: "2026-10-21"
tag: "RAG"
excerpt: "Package the API and Redis with Docker Compose, tune the vector index, log latency per stage, and run the evaluation in CI as a quality gate."
minutes: 8
---

# Optimization 9: Production and deployment

**You'll learn:** how to run this RAG as a real service: one-command startup, vector index tuning, monitoring, and keeping quality from regressing.

---

## 1. Theory

A fast pipeline on your laptop isn't a service. In production you need it to **start the same way everywhere** (containers), **stay fast as data grows** (index tuning), **tell you when it gets slow or wrong** (monitoring), and **never ship a change that silently lowers quality** (the evaluation as a CI gate).

---

## Before vs after

Not measured in this project. Baseline and what to measure:

| | Baseline (measured locally) | After |
|---|---|---|
| Startup | 3 manual steps: Redis in Docker, `--build`, `uvicorn` | measure it: `docker compose up` brings everything up with one command |
| Vector search | ~9 ms on 210 chunks | measure it: ms and hit rate at your real size while tuning `ef_search` |
| Visibility | timings printed in the terminal | measure it: p50/p95 latency per stage and cache hit rate on a dashboard |
| Quality protection | `tests/test_evaluate.py` run by hand | measure it: CI fails a pull request that drops the hit rate below 90% |

---

## 2. Code example

Docs-only `docker-compose.yml` for the API and Redis:
```yaml
services:
  api:
    build: .
    command: uvicorn advanced_rag.api:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    env_file: .env
    environment:
      REDIS_URL: redis://redis:6379/0        # inside compose, the host name is the service name
    volumes:
      - ./storage:/app/storage               # keep the Chroma index between restarts
    depends_on: [redis]
  redis:
    image: redis:7-alpine
```

Vector index tuning in ChromaDB (HNSW = the graph that makes nearest-neighbour search fast):
```python
client.get_or_create_collection(
    name="chunks",
    configuration={"hnsw": {"space": "cosine", "ef_search": 100, "max_neighbors": 16}},
    embedding_function=None,
)
```

Per-stage timing for logs:
```python
start = time.perf_counter()
hits = rag.retrieve(question, collection, chunks)
logger.info("retrieve_ms=%.0f cache_hit=%s top_score=%.2f", (time.perf_counter() - start) * 1000, was_cached, hits[0][0] if hits else -1)
```
In plain English:
- **Compose** describes both services. `REDIS_URL` points at the `redis` service by name. The `storage` volume keeps the index on the host disk, so restarting the container doesn't wipe it.
- **`ef_search`**: how many graph neighbours HNSW explores per query. Higher means more accurate but slower. **`max_neighbors`** (often called `M`): links per node. Higher means better recall, more memory and slower builds. The defaults (100 / 16, as printed by `col.configuration` in step 5) are fine for small data. Tune them only at scale, measuring both speed and hit rate.
- **Logs with numbers** (`retrieve_ms=...`) can be graphed. Track p95 latency (the slow tail users actually feel), cache hit rate, and how often `top_score` is below 0 (unanswered questions = missing documents).

---

## 3. Goal

1. Add a `Dockerfile` (Python 3.12 image, `uv sync`, copy `src/`) and the compose file above.
2. Run the retrieval evaluation in CI on every pull request. The quality gate test already exists:
   ```bash
   uv run pytest tests/test_evaluate.py
   ```
3. Log per-stage timings and the cache hit rate, then watch p95 latency before and after every change.
4. Health check: point your platform at `GET /health` (it already reports the chunk count).

**Check it:**
```bash
docker compose up --build
curl http://localhost:8000/health
```

---

## 4. Test

```python
def test_health_reports_a_loaded_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0     # a container that started without its index is NOT healthy
```
This turns the health check into a real readiness signal: a deployment with an empty or missing index fails instead of serving "I don't know" to everyone.

---

## 5. Trade-offs

- **Containers add a build step and image size** (PyTorch is large). Use a slim base image and CPU-only wheels when there's no GPU.
- **The models download on first start.** Bake them into the image or mount a cache volume, otherwise every new container waits on the download.
- **HNSW tuning matters at scale, not at 210 chunks.** At this size every setting is fast.
- **A CI quality gate needs a stable eval set and index.** Pin the data and models used in CI, or the gate will flicker.

---

**Previous:** [← Optimization 8: Better retrieval ideas](opt-08-better-retrieval-ideas.md)

**Next:** [Back to the optimization overview →](README.md)
