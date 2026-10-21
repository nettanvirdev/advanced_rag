---
title: "Optimization: making RAG fast and cheap"
series: "Advanced RAG"
group: "optimization"
step: 0
slug: "optimization"
date: "2026-10-21"
tag: "RAG"
excerpt: "Where the time goes in this RAG pipeline, measured stage by stage, and nine proven optimizations ranked by how much they help."
minutes: 6
---

# Optimization: making RAG fast and cheap

> **On the `main` branch these ideas stay out of the code on purpose, to keep it simple.** The [`optimization` branch](https://github.com/nettanvirdev/advanced_rag/tree/optimization) implements, measures and tests them. Its README has the before/after numbers.

**You'll learn:** where the time goes in this pipeline, how to read a latency budget, and which optimizations are worth doing first.

---

## 1. Theory

An optimization that doesn't target the slowest stage is wasted work. So **measure first, then fix the biggest number.** Every page below follows the same loop: measure a baseline → change one thing → measure again with the same questions (`eval/questions.json`) → keep the change only if quality stays the same or better.

---

## Where the time goes (measured on this project)

All numbers are per question, CPU only, on the 4 practice articles (210 chunks).

| Stage | Function | Time |
|---|---|---|
| Embed the question | `embeddings.embed([question])` | ~7 ms |
| Vector search | `vectorstore.search` (ChromaDB) | ~9 ms |
| Keyword search | `bm25.search` | **198 ms** |
| Hybrid (both + RRF) | `hybrid_search` | 228 ms |
| Hybrid + rerank (20 candidates) | `rag.retrieve` | ~380 ms |
| **LLM answer** | `llm.ask` | **~3–5 s** |
| Cache hit | `cache.get` (Redis) | ~1–4 ms |

Reading the budget:
- **The LLM is 90%+ of the time** on a new question. Caching it (step 7) is the single biggest win: 4.9 s → 1 ms.
- **Inside retrieval, BM25 is the slowest stage**, slower than the vector database and the embedding model put together. That's the easiest fix in this list (Optimization 1).
- The reranker adds ~150 ms for a real quality gain (89% → 95% hit rate). It's worth keeping, but its pool size is a knob (Optimization 4).

---

## The pages

| # | Optimization | Targets | Measured here? | Expected impact |
|---|---|---|---|---|
| 1 | [Precomputed BM25](opt-01-precomputed-bm25.md) | keyword search | ✅ 198 ms → 0.2 ms | retrieval ~40% faster |
| 2 | [Batching and embedding cache](opt-02-batching-and-embedding-cache.md) | indexing, repeated queries | ✅ 2.82 s → 1.48 s; 6.6 ms → ~0.001 ms | faster indexing |
| 3 | [Semantic cache](opt-03-semantic-cache.md) | LLM calls for paraphrased questions | ✅ similarities measured | skips the LLM for rephrasings |
| 4 | [Tuning the rerank pool](opt-04-tuning-the-rerank-pool.md) | rerank latency vs. quality | ✅ 5/10/20/40 sweep | pick the cheapest pool that keeps quality |
| 5 | [Faster models](opt-05-faster-models.md) | embedding + reranking | ❌ measure it | faster CPU inference |
| 6 | [Streaming and async](opt-06-streaming-and-async.md) | perceived latency, throughput | ❌ measure it | first words in under a second |
| 7 | [Cutting LLM cost and latency](opt-07-cutting-llm-cost-and-latency.md) | the 3–5 s LLM stage | ❌ measure it | cheaper, faster answers |
| 8 | [Better retrieval ideas](opt-08-better-retrieval-ideas.md) | the 1 remaining retrieval miss | ❌ measure it | higher hit rate |
| 9 | [Production and deployment](opt-09-production-and-deployment.md) | running it for real | ❌ measure it | reliable, observable service |

"Measured here" means it was tried in a scratch experiment on this project's data with the real numbers shown. "Measure it" pages give this project's baseline and tell you exactly what to measure. No invented results.

---

## How to measure any change

```bash
uv run python -m advanced_rag.evaluate
```
The command prints the hit rate, MRR and ms per question for every method. Run it before and after a change. A speed-up that lowers the hit rate isn't an optimization, it's a trade. The quality gate in `tests/test_evaluate.py` fails if retrieval drops below its thresholds.

---

**Previous:** [← Step 11: FastAPI](../step-11-fastapi.md)

**Next:** [Optimization 1: Precomputed BM25 →](opt-01-precomputed-bm25.md)
