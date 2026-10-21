---
title: "Optimization 4: Tuning the rerank pool"
series: "Advanced RAG"
group: "optimization"
step: 4
slug: "opt-04-tuning-the-rerank-pool"
date: "2026-10-21"
tag: "RAG"
excerpt: "The reranker's cost grows with every candidate it reads. A measured sweep shows where more candidates stop paying off."
minutes: 6
---

# Optimization 4: Tuning the rerank pool

**You'll learn:** how to choose how many candidates the reranker reads, using a measured quality-vs-latency sweep.

---

## 1. Theory

The cross-encoder reads the question together with **each** candidate, so its cost grows linearly with the pool size. Too few candidates and the right chunk never reaches the reranker. Too many and you pay for reading chunks that were never going to win. The best pool is the **smallest one that keeps quality at its best**.

---

## Before vs after

Measured on this project with the 19 answerable eval questions (hybrid search → rerank → top 3):

| Candidates | Hit rate @3 | MRR | ms / question |
|---|---|---|---|
| 5 | 89% | 0.83 | 273 |
| 10 | 89% | 0.83 | 318 |
| **20 (chosen)** | **95%** | **0.84** | **383** |
| 40 | 95% | 0.84 | 541 |

- **5 → 20:** +6 points of hit rate for +110 ms. Worth it.
- **20 → 40:** **no** quality gain for +158 ms. Pure waste.

That's why `rag.retrieve` asks hybrid search for 20 candidates.

---

## 2. Code example

The knob is one number in `rag.retrieve`:
```python
candidates = hybrid_search(collection, chunks, question, k=20, candidates=20)   # ← the pool size
hits = rerank(question, candidates, k)
```

A docs-only sweep script, the same one that produced the table:
```python
import time

from advanced_rag.evaluate import first_relevant_rank, load_questions, retrieval_scores

questions = [q for q in load_questions() if q["relevant"]]
for n in [5, 10, 20, 40]:
    start = time.perf_counter()
    ranks = [
        first_relevant_rank(rerank(q["question"], hybrid_search(collection, chunks, q["question"], k=n, candidates=n), k=10), q["relevant"])
        for q in questions
    ]
    ms = (time.perf_counter() - start) * 1000 / len(questions)
    print(n, retrieval_scores(ranks), f"{ms:.0f} ms")
```
In plain English: for each pool size, run every answerable question through hybrid search with that many candidates, rerank, find where the right chunk landed, then print the hit rate, MRR and the average time.

---

## 3. Goal

1. Re-run the sweep whenever the data grows. More chunks usually need a bigger pool.
2. Pick the smallest pool on the "flat top" of the quality curve.
3. Optional: cap the text each pair sends to the cross-encoder. It reads up to 512 tokens per pair, while our chunks are at most 200, so there's nothing to cut here, but long chunks elsewhere would benefit.

**Check it:**
```bash
uv run python -m advanced_rag.evaluate
```
Look at the `hybrid+rerank` row's hit rate, MRR and ms/question.

---

## 4. Test

```python
def test_pool_size_keeps_quality(index):
    collection, chunks = index
    report = evaluate_retrieval(collection, chunks, load_questions())
    assert report["hybrid+rerank"]["hit_rate"] >= 0.9
    assert report["hybrid+rerank"]["mrr"] >= 0.8
```
This is the existing quality gate: shrinking the pool to save time can't silently lower retrieval below the measured level.

---

## 5. Trade-offs

- **Don't tune on a tiny eval set alone.** 19 questions means one question is about 5 points of hit rate. Confirm with more questions before cutting the pool.
- **The reranker runs pairs in batches already** (`model.predict` gets the whole list). On a GPU, larger pools cost much less, so the sweet spot moves.
- **Skip reranking entirely** for latency-critical paths like autocomplete. Hybrid alone was 89% hit rate at 228 ms.

---

**Previous:** [← Optimization 3: Semantic cache](opt-03-semantic-cache.md)

**Next:** [Optimization 5: Faster models →](opt-05-faster-models.md)
