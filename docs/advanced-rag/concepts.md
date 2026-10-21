---
title: "Advanced RAG Concepts"
series: "Advanced RAG"
group: "reference"
step: 0
slug: "concepts"
date: "2026-10-21"
tag: "RAG"
excerpt: "The theory behind a production RAG pipeline: embeddings and dimensions, chunking, vector databases, hybrid search and RRF, caching, evaluation metrics, reranking, query rewriting and serving, with formulas and real measured numbers."
minutes: 25
---

# Advanced RAG Concepts

The theory behind every step of the [Advanced RAG guide](README.md), with the formulas and the numbers measured on this project's data: 4 Wikipedia articles (cat, coffee, the Python language, python snakes), 210 chunks, and a 22-question evaluation set.

Keyword search, TF-IDF, BM25 and the basics of embeddings and RAG are explained in the [Simple RAG concepts](https://github.com/nettanvirdev/simple_rag/blob/main/docs/concepts.md). This page builds on them.

## Contents
- [Cheat sheet](#cheat-sheet)
- [The two pipelines](#the-two-pipelines)
- [Embeddings and dimensions](#embeddings-and-dimensions)
- [Chunking](#chunking)
- [Vector databases](#vector-databases)
- [Hybrid search and RRF](#hybrid-search-and-rrf)
- [Caching](#caching)
- [Evaluation metrics](#evaluation-metrics)
- [Reranking](#reranking)
- [Query rewriting](#query-rewriting)
- [Serving with FastAPI](#serving-with-fastapi)
- [Glossary](#glossary)

---

## Cheat sheet

You don't need to memorize these. Remember the idea, and look up the formula here.

| Concept | The idea in one sentence | Formula / rule | Measured here |
|---|---|---|---|
| **Dimension** | how many numbers an embedding has | storage = dims × 4 bytes | 384 dims = 1.5 KB per chunk |
| **Max tokens** | an embedding model reads only so much; the rest is silently ignored | 256 tokens for all-MiniLM-L6-v2 | one embedding of cat.txt sees 2.5% of it |
| **Chunking** | cut long text into pieces under the limit | pack whole sentences up to 200 tokens | 210 chunks, 100% of the text embedded |
| **Overlap** | repeat the last sentence in the next chunk | keep it only if it still fits | 185 → 210 chunks |
| **Vector database** | store chunk vectors once, search them fast | cosine distance = 1 − similarity | 1,482 ms → ~9 ms per question |
| **Hybrid search** | combine meaning (semantic) and exact words (BM25) | RRF: Σ 1 / (60 + rank) | MRR 0.76 → 0.79 |
| **Exact-match cache** | same question → reuse the stored answer | key = hash(model + normalized question) | 4,886 ms → 1 ms |
| **Hit rate @k** | how often the right chunk is in the top k | hits ÷ questions | 95% (hybrid + rerank) |
| **MRR** | how *high* the right chunk ranks | mean of 1 / rank | 0.84 (hybrid + rerank) |
| **Reranking** | a cross-encoder re-reads the top candidates with the question | keep score ≥ 0, best k | 89% → 95%, answers 17/19 → 19/19 |
| **Query rewriting** | the LLM rewrites the question into search words | search both, RRF, rerank vs. original | no gain here, 379 → 4,868 ms |
| **API** | serve the pipeline over HTTP | validate input, load once at startup | 3.1 s new, 3.7 ms cached |

---

## The two pipelines

A production RAG is really **two pipelines** that meet in the vector database:

```
INDEXING  (once, or when documents change)
  .txt files → load → split into sentences → pack into chunks → embed every chunk → store in ChromaDB

QUERYING  (every question)
  question → Redis cache? ──hit──► answer in ~1 ms
                │ miss
                ▼
        hybrid search (semantic + BM25, RRF) → 20 candidates → rerank → best 3 → LLM → answer → save in Redis
```

The expensive work (embedding 210 chunks: 1.5–2.7 s) happens **once** at indexing. Each question only embeds itself.

---

## Embeddings and dimensions

An **embedding** is a fixed-length list of numbers that represents a text's meaning. Its **dimension** is how many numbers it has.

| Model | Dims | Size | Max tokens | Notes |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (used here) | 384 | ~90 MB | 256 | fast, good enough for most uses |
| all-mpnet-base-v2 | 768 | ~420 MB | 384 | better quality, about 5× slower |
| OpenAI text-embedding-3-small | 1536 | API | 8191 | paid, no local compute |
| OpenAI text-embedding-3-large | 3072 | API | 8191 | best quality, most storage |

**Storage cost.** Every number is a 4-byte float:

| Dims | Per chunk | 1 million chunks |
|---|---|---|
| 384 | 1.5 KB | 1.5 GB |
| 768 | 3 KB | 3 GB |
| 1536 | 6 KB | 6 GB |
| 3072 | 12 KB | 12 GB |

More dimensions leave more room to separate subtle meanings, but cost storage, memory and search time. Choose with your evaluation, not by the size of the number.

**Tokens.** Models read **tokens** (words or word pieces), not characters: `"Kittens sleep a lot."` is 6 tokens (`kitten`, `##s`, `sleep`, `a`, `lot`, `.`) but 20 characters.

**The silent cut-off.** Text past the maximum is ignored without any warning. A 320-token text and the same text with 20 extra sentences appended get **identical** embeddings (similarity 1.0).

**Normalized vectors.** A normalized vector has length 1. For two normalized vectors, cosine similarity equals the plain **dot product**:

$$
\cos(A, B) = \frac{A \cdot B}{\|A\|\,\|B\|} \quad\xrightarrow{\ \|A\| = \|B\| = 1\ }\quad A \cdot B = \sum_i A_i B_i
$$

"cat" · "kitten" = 0.7882, the same number cosine gives.

**Batching.** Embedding many texts in one call is much faster than one at a time: 210 chunks took 2.82 s one by one and **1.48 s** in one batch.

---

## Chunking

**Why.** One embedding of a whole article sees only its first 256 tokens:

| File | Tokens | Seen by one whole-file embedding |
|---|---|---|
| cat.txt | 10,352 | 2.5% |
| coffee.txt | 12,236 | 2.1% |
| python_language.txt | 8,718 | 2.9% |
| python_snake.txt | 2,433 | 10.5% |

**How (sentence packing).** Split the text into sentences, then fill a "box" with sentences until the next one wouldn't fit under the limit (200 tokens, leaving a margin below 256); seal the box and start a new one. Never cut inside a sentence. Result: 185 chunks, 100% of the text embedded.

**Overlap.** Copy the last sentence of each chunk into the next, so a fact that sits on a boundary is whole in at least one chunk. Guard it: keep the copy only if it still fits (without the guard one chunk reached 203 tokens). Result: 210 chunks (+13% storage and embedding time).

**Trade-offs.**
- Too big: the cut-off loses text, and one vector blurs several topics.
- Too small: chunks lose context ("It is a social species": what is?).
- Overlap is insurance, not a guaranteed gain: on this eval set it didn't improve retrieval (hybrid + rerank MRR 0.90 without, 0.84 with), because no question here depends on a boundary-crossing fact. Measure on your own data.

**IDs.** Each chunk keeps its file and position: `cat.txt#12`. That's the database key and what answers cite.

---

## Vector databases

A **vector database** (here ChromaDB) stores each chunk's vector, text and labels on disk, and finds the nearest vectors to a question without comparing against every one, using an approximate-nearest-neighbour index called **HNSW** (a layered graph of shortcuts between nearby vectors).

| | Re-embed on every question (Simple RAG) | ChromaDB |
|---|---|---|
| Per question | 1,482 ms | ~9 ms (first query after opening ~50 ms) |
| Survives a restart | no | yes |
| Build | — | 210 chunks in 2.7 s, once |

**Cosine distance.** Chroma reports *distance*, not similarity:

$$
\text{similarity} = 1 - \text{distance}
$$

**Upsert** = update or insert: adding a chunk whose ID already exists overwrites it, so re-running indexing never duplicates.

**Limit.** A vector database always returns k results: "asdfgh" still got 3 cat chunks (best similarity 0.235). Something downstream must decide that weak results mean "no answer".

---

## Hybrid search and RRF

Semantic search understands meaning but misses **exact terms**; BM25 is the opposite.

| Question | Semantic alone | Hybrid |
|---|---|---|
| Guido van Rossum | coffee.txt#73 (0.20), wrong | python_language.txt#7 |
| 9 bar | coffee.txt#68 (0.13), wrong | coffee.txt#45 |
| asdfgh | 3 random cat chunks | nothing |

**Reciprocal Rank Fusion (RRF).** BM25 scores (like 12.0) and cosine scores (like 0.65) are on different scales, so combine **ranks**, not scores:

$$
\text{RRF}(c) = \sum_{\text{lists}} \frac{1}{60 + \text{rank of } c \text{ in that list}}
$$

A chunk 2nd in both lists gets 1/62 + 1/62 = 0.0323 and beats a chunk 1st in only one list (1/61 = 0.0164). 60 is the constant from the original RRF paper; it stops first place from dominating.

**Minimum similarity.** Semantic hits below 0.3 are dropped (real questions score 0.5+, nonsense ~0.2), and BM25 already ignores chunks with no shared words, so nonsense returns nothing.

| Method | Hit rate @3 | MRR | ms / question |
|---|---|---|---|
| semantic | 89% | 0.76 | 9 |
| bm25 | 79% | 0.68 | 198 |
| hybrid | 89% | 0.79 | 228 |

---

## Caching

An LLM call is slow and costs money; many users ask the same thing. An **exact-match cache** stores each answer under a key and reuses it.

**The key** must make trivially different questions match, and keep anything that changes the answer apart:
1. normalize: lowercase, collapse spaces ("How long do cats sleep?" = "how long do  cats sleep?")
2. include the **model name** (a different model gives a different answer)
3. hash with **sha256**: same input, same short fixed-length key
4. prefix it (`advanced_rag:answer:`) so you only ever touch your own keys

**Expiry (TTL).** Answers expire after 1 day, and rebuilding the index clears them, so stale answers can't live forever.

**Measured:** 4,886 ms → **1 ms** for a repeated question.

**Never cache a failure.** If the LLM returns an empty reply, answer "I don't know" but don't store it, or a temporary glitch is remembered for a day.

**Redis** is an in-memory key → value store; lookups take about a millisecond. Tests use **fakeredis**, an in-memory fake, so they never touch a real server.

More caching ideas (semantic cache, embedding cache) are in the [Optimization guide](optimization/README.md).

---

## Evaluation metrics

A **golden set** is a list of questions with known answers, written once and re-run after every change. Here: 22 questions (19 answerable, 3 not).

**Retrieval** (no LLM, fast and free): for each question, find the **rank** of the first retrieved chunk that contains a known answer phrase.

$$
\text{hit rate@}k = \frac{\#\{\text{questions whose right chunk is in the top } k\}}{\#\text{questions}}
$$

$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i} \qquad (\text{missed} = 0)
$$

Example: ranks 1, 2 and a miss → hit rate@3 = 2/3, MRR = (1 + ½ + 0) / 3 = 0.5. Hit rate says *whether* it was found; MRR also rewards finding it *higher*.

**Answers** (real LLM): correct (contains an expected phrase), cited (mentions a chunk ID), and refused when it should ("I don't know" for the 3 unanswerable questions).

**What it taught this project:**
- Both early wrong answers were **retrieval misses**: the LLM never saw the right chunk. Retrieval was the bottleneck, not the LLM.
- It exposed a real bug: an empty LLM reply crashed the pipeline.
- **An evaluation can be wrong too.** The reranker found "Python 3.0 was released on 3 December 2008", a better answer than the single phrase in the golden set. So each question now accepts a *list* of phrases. Only add phrases that truly answer.

**Quality gate.** A test fails if retrieval drops below a floor (hybrid + rerank ≥ 90% hit rate, ≥ 0.8 MRR), so a future change can't silently make things worse.

Full results: [eval/results.md](../../eval/results.md).

---

## Reranking

| | Bi-encoder (embedding model) | Cross-encoder (reranker) |
|---|---|---|
| Reads | question and chunk **separately** | question and chunk **together** |
| Output | two vectors → cosine | one relevance score |
| Precompute chunks? | yes (stored in ChromaDB) | no, it needs the question |
| Speed | ~9 ms for all 210 chunks | ~150 ms for 20 candidates |
| Role | find candidates | order candidates |

**Two-stage retrieval:** a fast, rough stage casts a wide net (hybrid, 20 candidates), and a slow, precise stage picks the best 3.

**Pool size** (measured): 5 → 89% / 0.83; 10 → 89% / 0.83; **20 → 95% / 0.84**; 40 → 95% / 0.84 at +160 ms. The reranker can only promote what it's given.

**Scores** from `cross-encoder/ms-marco-MiniLM-L6-v2` run from about −11 to +11. Above 0 = answers, below = doesn't. Every answerable eval question's best chunk scored **> 3.7**, every unanswerable **< −9.8**, so `min_score = 0` turns "no answer" into a decision that needs no LLM call.

**Result:** hit rate 89% → 95%, MRR 0.79 → 0.84, answers 17/19 → 19/19, for ~150 ms more per question.

**Limit.** Its scores are model-specific: re-check the threshold if you change the reranker.

---

## Query rewriting

The LLM rewrites a question into the words an article would likely use ("Which country grows the most coffee?" → "largest coffee-producing country Brazil"), and that query is searched too.

**Safe design:** search with the original **and** the rewritten query, merge with RRF, and rerank against the **original** question. A bad rewrite can only add candidates, never remove good ones.

**Measured here:** hit rate 95% → 95%, MRR 0.84 → 0.84, time 379 ms → 4,868 ms. No gain, about 13× slower, so it ships **off**.

**When it helps:** vague or chatty questions, follow-ups in a conversation ("how long do *they* live?"), typos and slang. The clean, specific questions in this eval set are where it has the least to add.

**Variants** (not built here): multi-query (several rewrites at once) and HyDE (the LLM writes a hypothetical answer, and you search with *that*). See the [Optimization guide](optimization/README.md).

**The lesson:** a technique being popular doesn't make it a win on your data. Measure it.

---

## Serving with FastAPI

A **web API** lets any program use the pipeline over HTTP, sending JSON in and getting JSON back.

| Endpoint | Does | Measured |
|---|---|---|
| `GET /health` | is it up, how many chunks | — |
| `POST /ask` `{"question", "k"}` | full RAG answer + whether it came from the cache | 3.1 s new, **3.7 ms** cached, 324 ms for an unanswerable question |
| `GET /search?q=&k=` | retrieval only, no LLM | 236 ms |
| `GET /docs` | interactive documentation, generated automatically | — |

**Validation.** Request shapes are declared as pydantic models (`question` non-empty, `k` between 1 and 10). Bad input gets **422** before your code runs.

**Lifespan.** Expensive things (opening the index, reading the chunks for BM25) load **once** at server start, not per request.

**Thin layer.** The API adds no logic of its own. It calls the same `rag.answer()` and `rag.retrieve()` that the evaluation measures, so the measured numbers are exactly what's served.

---

## Glossary
| Term | Meaning |
|---|---|
| **Indexing pipeline** | the once-off work: load, chunk, embed, store |
| **Query pipeline** | the per-question work: cache, retrieve, rerank, generate |
| **Dimension** | the number of values in an embedding vector |
| **Token** | a word or word piece, the unit models read |
| **Max sequence length** | the most tokens a model reads; the rest is ignored |
| **Normalized vector** | a vector of length 1; its dot product equals cosine similarity |
| **Dot product** | multiply matching numbers and add them up |
| **Chunk** | a piece of a document small enough to embed whole |
| **Overlap** | sentences repeated between neighbouring chunks |
| **Vector database** | a store that finds nearest vectors fast (ChromaDB) |
| **HNSW** | Hierarchical Navigable Small World: the graph index that makes nearest-neighbour search fast |
| **Cosine distance** | 1 − cosine similarity |
| **Upsert** | insert, or overwrite if the ID already exists |
| **Hybrid search** | combining semantic and keyword (BM25) search |
| **RRF** | Reciprocal Rank Fusion: merge ranked lists by Σ 1/(60 + rank) |
| **Cache hit / miss** | the answer was / wasn't already stored |
| **TTL** | time to live: when a cached entry expires |
| **Hash (sha256)** | a fixed-length fingerprint of any text |
| **Golden set** | questions with known answers, used to evaluate |
| **Hit rate @k** | share of questions whose right chunk is in the top k |
| **MRR** | mean reciprocal rank: average of 1 / rank of the right chunk |
| **Quality gate** | a test that fails if a metric drops below a floor |
| **Bi-encoder** | embeds question and text separately (the embedding model) |
| **Cross-encoder** | reads question and text together and scores the pair (the reranker) |
| **Candidate pool** | the rough results handed to the reranker (20 here) |
| **Query rewriting** | the LLM turning a question into better search words |
| **Endpoint** | one URL + method the API answers, like `POST /ask` |
| **pydantic model** | a declared request/response shape that FastAPI validates |
| **422** | HTTP status for "your request didn't match the required shape" |
| **Lifespan** | FastAPI's hook for startup/shutdown work |
