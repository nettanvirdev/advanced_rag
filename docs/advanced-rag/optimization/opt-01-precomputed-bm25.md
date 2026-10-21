---
title: "Optimization 1: Precomputed BM25"
series: "Advanced RAG"
group: "optimization"
step: 1
slug: "opt-01-precomputed-bm25"
date: "2026-10-21"
tag: "RAG"
excerpt: "BM25 re-tokenizes every chunk and recounts document frequencies on every question. Doing that work once at startup makes it about 1,000x faster with identical results."
minutes: 8
---

# Optimization 1: Precomputed BM25

**You'll learn:** why the BM25 copied from `simple_rag` is slow, and how to move the repeated work to startup.

---

## 1. Theory

`bm25.search` recomputes everything on every question: it tokenizes all 210 chunks, then for every query word it scans every chunk to count document frequency (`idf`). None of that depends on the question. The chunks only change when the index is rebuilt. So compute it **once** at startup and every question becomes a few dictionary lookups.

---

## Before vs after

| | Before (`bm25.search`) | After (precomputed) |
|---|---|---|
| Time per question | **198 ms** | **0.2 ms** (~1,000x faster) |
| Results | top 10 chunks | **identical** top 10 for all 19 eval questions |
| Hybrid search per question | 228 ms | ~30 ms (estimate: hybrid minus the BM25 saving) |

Measured in a scratch experiment on this project's 210 chunks with the 19 answerable eval questions.

---

## 2. Code example

A docs-only sketch that reuses the project's own `tokenize` and `term_score`:

```python
import math
from collections import Counter

from advanced_rag.retrieval.bm25 import term_score
from advanced_rag.retrieval.keyword import tokenize


class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.words = [tokenize(c.text) for c in chunks]             # tokenize every chunk ONCE
        self.counts = [Counter(w) for w in self.words]               # word -> how many times, per chunk
        self.avg_len = sum(len(w) for w in self.words) / len(self.words)
        self.df = Counter(word for w in self.words for word in set(w))  # in how many chunks each word appears
        self.n = len(chunks)

    def search(self, query: str, k: int = 10):
        query_words = tokenize(query)
        results = []
        for chunk, counts, words in zip(self.chunks, self.counts, self.words):
            score = sum(
                math.log(self.n / self.df[w]) * term_score(counts[w], len(words), self.avg_len)
                for w in query_words
                if w in counts  # skip words this chunk doesn't have: they add 0
            )
            if score > 0:
                results.append((score, chunk))
        return sorted(results, key=lambda pair: pair[0], reverse=True)[:k]
```

In plain English:
- `__init__` does all the expensive work once: tokenize, count each word per chunk (`Counter`), and count how many chunks contain each word (`df`, the "document frequency" that `idf` used to recount every time).
- `set(w)` counts a word once per chunk, even if the chunk says it five times. That's the definition of document frequency.
- `search` only does lookups: `counts[w]` is how often the word is in this chunk, and `self.df[w]` is how rare it is. Same formula as `bm25.score`, with no rescanning.
- `if w in counts` skips work for words the chunk doesn't contain, since they would add 0 anyway.

**Libraries do this for you:** [`bm25s`](https://github.com/xhluca/bm25s) (fast, sparse matrices) and `rank-bm25` precompute the same statistics.

---

## 3. Goal

1. Build the index once where the chunks are loaded: in `api.py`'s `lifespan` (next to `all_chunks`), store `BM25Index(chunks)` on `app.state`.
2. Pass it into `hybrid_search` instead of the chunk list, and call `index.search(query, candidates)`.
3. Rebuild it whenever the Chroma index is rebuilt (the chunks changed).

**Check it:**
```bash
uv run python -m advanced_rag.evaluate
```
`bm25` should drop from ~198 to under 1 ms/question, with the **same** hit rate (79%) and MRR (0.68).

---

## 4. Test

```python
def test_precomputed_bm25_matches_the_original(index):
    collection, chunks = index
    fast = BM25Index(chunks)
    for q in load_questions():
        slow_ids = [chunk_id(c) for _, c in bm25.search(q["question"], chunks, 10)]
        fast_ids = [chunk_id(c) for _, c in fast.search(q["question"], 10)]
        assert fast_ids == slow_ids, q["question"]
```
This checks the optimization is a pure speed-up: every eval question returns exactly the same chunks, in the same order.

---

## 5. Trade-offs

- **Memory:** the counts live in RAM. That's trivial for 210 chunks, but for millions of chunks use an on-disk engine (Elasticsearch/OpenSearch, or `bm25s` with memory mapping).
- **Staleness:** the precomputed numbers are wrong as soon as chunks change. Tie the rebuild to the index rebuild.
- **Don't bother** if BM25 isn't on your hot path, for example if you run semantic-only search.

---

**Previous:** [← Optimization overview](README.md)

**Next:** [Optimization 2: Batching and embedding cache →](opt-02-batching-and-embedding-cache.md)
