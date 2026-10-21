---
title: "Advanced RAG"
series: "Advanced RAG"
group: "reference"
step: 0
slug: "index"
date: "2026-10-21"
tag: "RAG"
excerpt: "Make a RAG production-shaped, one measured step at a time: chunking, ChromaDB, hybrid search, reranking, a Redis cache, evaluation and FastAPI."
minutes: 4
---

# Advanced RAG

Take a simple RAG and turn it into something production-shaped, one small step at a time: **chunk** long documents, store them in a **vector database**, search with **hybrid search**, **rerank** the results, **cache** answers in Redis, **evaluate** everything with a golden question set, and serve it with **FastAPI**.

**You need:** the ideas from **[Simple RAG](https://github.com/nettanvirdev/simple_rag)** (keyword search, TF-IDF, BM25, semantic search, a first RAG) and basic Python. Every new library and idea is explained the first time it appears.

## How every step works
1. **Theory**: 2–3 lines on the idea, plus real numbers.
2. **Before vs after (measured)**: what this step changes, in real numbers from this repo.
3. **Code example**: the new or tricky pieces, explained line by line in plain English.
4. **Goal**: what to build, as plain-English steps, with a ✅ **checkpoint** command and its expected output.
5. **Test**: pytest tests that prove it works, each with one sentence on *what* it checks and *why*.
6. **Make it a module**: how the next step uses what you just built.
7. **Show solution**: the complete file, collapsed. Open it only if you're stuck.

## The steps
| # | Step | You build | Before → after (measured) |
|---|---|---|---|
| 1 | [Project setup](step-01-project-setup.md) | uv project, copied config + loaders | 0 → 12 tests passing |
| 2 | [Embeddings and dimensions](step-02-embeddings-and-dimensions.md) | `embeddings.py` | one-by-one 2.82 s → batch 1.48 s for 210 chunks |
| 3 | [Chunking: split and pack](step-03-chunking-split-and-pack.md) | `chunk_text` | 2.5% → 100% of `cat.txt` embedded |
| 4 | [Chunking: overlap and records](step-04-chunking-overlap-and-records.md) | `Chunk`, `chunk_documents` | biggest chunk 203 → 200 tokens (guard) |
| 5 | [ChromaDB vector store](step-05-chromadb-vector-store.md) | `vectorstore.py` | 1,482 ms → ~9 ms per search |
| 6 | [Hybrid retrieval](step-06-hybrid-retrieval.md) | `retrieval/hybrid.py` | MRR 0.76 → 0.79; exact names found |
| 7 | [Redis cache](step-07-redis-cache.md) | `cache.py`, `rag.py` | 4,886 ms → 1 ms on a repeat |
| 8 | [Evaluation](step-08-evaluation.md) | `evaluate.py`, golden set | "looks fine" → hit rate, MRR, 17/19 |
| 9 | [Reranking](step-09-reranking.md) | `retrieval/rerank.py` | hit rate 89% → 95%; answers 17/19 → 19/19 |
| 10 | [Query rewriting](step-10-query-rewriting.md) | `retrieval/rewrite.py` | no gain, 379 ms → 4,868 ms, so kept off |
| 11 | [FastAPI](step-11-fastapi.md) | `api.py` | terminal → HTTP API: 3.1 s new, 3.7 ms cached |

## Beyond the steps
- ⚡ **[Optimization guide](optimization/README.md)**: proven ways to cut latency and cost (precomputed BM25, embedding and semantic caches, rerank pool tuning, faster models, streaming, cheaper LLM calls, better retrieval, deployment). Each one comes with how to measure it. These aren't in the code on purpose, so the code stays simple.
- 🗺️ **[Diagrams](diagrams.md)**: the indexing pipeline, the query pipeline and the evaluation loop, with every function and model labelled.
- 📐 **[Concepts](concepts.md)**: the theory and formulas behind every step, with a cheat sheet.
- 📊 **[Evaluation results](../../eval/results.md)**: the latest measured numbers, including every answer.

## Practice data
Four long English Wikipedia articles in `data/` (CC BY-SA 4.0, see `data/SOURCES.md`): **Cat**, **Coffee**, **Python (programming language)** and **Pythonidae**, about 25,000 words in total. Two different "pythons" share one word on purpose, the same trick as in Simple RAG.

**Start here:** [Step 1: Project setup →](step-01-project-setup.md)
