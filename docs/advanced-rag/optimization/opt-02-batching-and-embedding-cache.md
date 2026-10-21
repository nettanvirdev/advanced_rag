---
title: "Optimization 2: Batching and an embedding cache"
series: "Advanced RAG"
group: "optimization"
step: 2
slug: "opt-02-batching-and-embedding-cache"
date: "2026-10-21"
tag: "RAG"
excerpt: "Embed many texts in one call to halve indexing time, and remember query embeddings so a repeated question never runs the model twice."
minutes: 6
---

# Optimization 2: Batching and an embedding cache

**You'll learn:** why one big model call beats many small ones, and when caching embeddings is worth it.

---

## 1. Theory

Every call to the embedding model has a fixed overhead: Python setup, tokenizer, moving data into the model. **Batching** pays that overhead once for many texts, which is why `embed()` takes a list. An **embedding cache** goes further: the same text always gives the same vector, so a vector computed once can be stored and looked up instead of recomputed.

---

## Before vs after

| | Before | After |
|---|---|---|
| Embed 210 chunks (indexing) | one call per chunk: **2.82 s** | one batch call: **1.48 s** (1.9x faster) |
| Embed a repeated query | run the model: **6.6 ms** | dictionary lookup: **~0.001 ms** |

Measured in a scratch experiment on this project's 210 chunks. The project already batches: `add_chunks` calls `embed([c.text for c in chunks])` once.

---

## 2. Code example

The batching rule, already in `vectorstore.add_chunks`:
```python
embeddings = embed([c.text for c in chunks])      # ✅ one call for all chunks
# embeddings = [embed([c.text])[0] for c in chunks]  ❌ 210 calls: about 2x slower
```

A docs-only query-embedding cache:
```python
from functools import lru_cache

from advanced_rag.embeddings import embed


@lru_cache(maxsize=10_000)
def embed_query(query: str) -> tuple[float, ...]:
    return tuple(embed([query])[0])   # lru_cache needs a hashable result, so a tuple
```
In plain English:
- `@lru_cache` remembers the result for each different `query` string. The second time, Python returns the stored vector without touching the model.
- `maxsize=10_000` keeps the 10,000 most recently used queries, so memory can't grow forever. "LRU" means least recently used: the oldest entries are thrown out first.
- For a cache shared across server processes, or one that survives restarts, store the vector in Redis instead (the same idea as `cache.py`, with the vector saved as JSON).

---

## 3. Goal

1. Keep every embedding call batched: never call `embed()` inside a loop over chunks.
2. Wrap query embedding in `vectorstore.search` with `embed_query`.
3. If you index often, also cache **chunk** embeddings keyed by a hash of the chunk text, so only new or changed chunks are re-embedded on rebuild.

**Check it:** time `build_index()` before and after (the first rebuild is the same, later rebuilds only embed changed chunks), and time two identical `vectorstore.search` calls in a row.

---

## 4. Test

```python
def test_cached_query_embedding_is_identical_and_reused():
    embed_query.cache_clear()
    first = embed_query("How long do cats sleep?")
    second = embed_query("How long do cats sleep?")
    assert first == second
    assert embed_query.cache_info().hits == 1
```
This checks that a repeated query returns the same vector and is served from the cache (one hit), not recomputed.

---

## 5. Trade-offs

- **Small win for queries:** 6.6 ms is already fast next to a 3–5 s LLM call. Cache query embeddings when you embed through a **paid API** (each call costs money and ~100+ ms of network) or under high traffic.
- **Batch size limits:** very large batches can run out of memory, especially on a GPU. Embed in batches of a few hundred to a few thousand.
- **The model is part of the key:** after changing `EMBEDDING_MODEL`, every cached vector is wrong. Clear the cache or put the model name in the key.

---

**Previous:** [← Optimization 1: Precomputed BM25](opt-01-precomputed-bm25.md)

**Next:** [Optimization 3: Semantic cache →](opt-03-semantic-cache.md)
