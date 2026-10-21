---
title: "Optimization 3: Semantic cache"
series: "Advanced RAG"
group: "optimization"
step: 3
slug: "opt-03-semantic-cache"
date: "2026-10-21"
tag: "RAG"
excerpt: "Reuse answers for questions that mean the same thing, not just questions spelled the same, and guard against the dangerous near-misses."
minutes: 8
---

# Optimization 3: Semantic cache

**You'll learn:** how to cache by meaning, how to choose the similarity threshold from real numbers, and how to stop the cache from serving a wrong answer.

---

## 1. Theory

The Redis cache from step 7 only helps when a question is the **same text** after lowercasing and squashing spaces. "How long do cats sleep?" and "How many hours a day does a cat sleep?" are the same question with different words, so it misses. A **semantic cache** stores each question's embedding with its answer. A new question that is **similar enough** to a stored one gets the stored answer, with no retrieval and no LLM call.

---

## Before vs after

Cosine similarity between question embeddings, measured with this project's model (`all-MiniLM-L6-v2`):

| Question A | Question B | Same meaning? | Similarity |
|---|---|---|---|
| How long do cats sleep? | How many hours a day does a cat sleep? | ✅ yes | **0.894** |
| Who created Python? | Who invented the Python language? | ✅ yes | **0.878** |
| How long do cats sleep? | How long do pythons sleep? | ❌ no | 0.632 |
| Who created Python? | Who created coffee? | ❌ no | 0.470 |

| | Exact cache (step 7) | Semantic cache (threshold 0.85) |
|---|---|---|
| "How many hours a day does a cat sleep?" after "How long do cats sleep?" | miss → full pipeline, 3–5 s | **hit → ~1–4 ms**, plus ~7 ms to embed the question |
| "How long do pythons sleep?" after "How long do cats sleep?" | miss | miss (0.632 < 0.85) ✅ correct |

The hit/miss rows follow from the measured similarities. End-to-end timings for a semantic cache weren't measured here: measure your hit rate on real traffic.

---

## 2. Code example

A docs-only sketch that stores question vectors in a separate Chroma collection:

```python
from advanced_rag.embeddings import embed

THRESHOLD = 0.85


def semantic_get(cache_collection, question: str) -> str | None:
    if cache_collection.count() == 0:
        return None
    result = cache_collection.query(query_embeddings=embed([question]), n_results=1)
    similarity = 1 - result["distances"][0][0]          # cosine distance -> similarity
    if similarity >= THRESHOLD:
        return result["metadatas"][0][0]["answer"]      # the stored answer
    return None


def semantic_put(cache_collection, question: str, answer: str) -> None:
    cache_collection.upsert(
        ids=[question.lower()],
        embeddings=embed([question]),
        metadatas=[{"answer": answer}],
    )
```
In plain English:
- `semantic_get` embeds the new question and asks for the **single nearest** cached question.
- If the two questions are at least 0.85 similar, it returns that question's stored answer. Otherwise it returns `None` and the normal pipeline runs.
- `semantic_put` saves the question's vector with its answer in the metadata, so the next similar question finds it.

**Tools:** GPTCache and Redis vector search (RediSearch) provide this as a ready-made layer.

---

## 3. Goal

1. Check the **exact** cache first (`cache.get`, ~1 ms), then the semantic cache, then run the full pipeline.
2. Choose the threshold from your own pairs: write 20+ paraphrase pairs and 20+ "similar words, different meaning" pairs, embed them, and pick a value between the two groups. Here, between 0.632 and 0.878.
3. Add guards against the danger zone (below).

**Check it:** collect paraphrases of the 19 eval questions, then run `evaluate_answers` with the semantic cache on. Correct answers must stay at 19/19, and count how many were served from the cache.

---

## 4. Test

```python
def test_paraphrase_hits_but_different_question_misses(tmp_path):
    cache_collection = get_collection(tmp_path)  # a separate collection just for the cache
    semantic_put(cache_collection, "How long do cats sleep?", "12–16 hours [cat.txt#25]")
    assert semantic_get(cache_collection, "How many hours a day does a cat sleep?") == "12–16 hours [cat.txt#25]"
    assert semantic_get(cache_collection, "How long do pythons sleep?") is None
```
This checks both sides of the threshold: a real paraphrase is served from the cache, and a question with similar words but a different subject is not.

---

## 5. Trade-offs

**The danger zone:** "cats sleep" vs. "pythons sleep" scored 0.632. A threshold set too low (say 0.6) would answer a question about snakes with the cat answer, **confidently and wrongly**. A wrong cached answer is worse than a slow correct one. Guards:
- **Keep the threshold high** (0.85+) and lower it only with evidence from your evaluation.
- **Same-source check:** store the chunk ids the answer was built from. On a hit, run the cheap retrieval (~380 ms, no LLM) and only reuse the answer if the new question retrieves the same top chunk.
- **Short TTL** (hours, not days), and clear the cache when the index is rebuilt.
- **Don't use it** when questions contain user-specific details (names, account numbers, dates): the cache would leak one user's answer to another.

---

**Previous:** [← Optimization 2: Batching and embedding cache](opt-02-batching-and-embedding-cache.md)

**Next:** [Optimization 4: Tuning the rerank pool →](opt-04-tuning-the-rerank-pool.md)
