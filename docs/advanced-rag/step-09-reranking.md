---
title: "Step 9: Reranking"
series: "Advanced RAG"
group: "steps"
step: 9
slug: "step-09-reranking"
date: "2026-10-21"
tag: "RAG"
excerpt: "Let a cross-encoder re-read the top 20 candidates together with the question and keep the best 3: hit rate 89% → 95%, and every answer on the eval set correct."
minutes: 14
---

# Step 9: Reranking

**You'll build:** `rerank()` in `src/advanced_rag/retrieval/rerank.py`, used by `rag.py` and measured by `evaluate.py`.
**You'll learn:** bi-encoders vs. cross-encoders, candidate pools, score thresholds, and why an evaluation set can be wrong too.

---

## 1. Theory

Search (Step 5) turns the question and each chunk into vectors **separately**, then compares them: fast, but rough. A **cross-encoder** reads the question **and** one chunk **together**, word by word, and scores how well that chunk answers it: much more accurate, but far too slow to run on every chunk. So we combine them: **hybrid search fetches 20 candidates fast, the reranker carefully picks the best 3.** ([Theory: Reranking](concepts.md#reranking))

```
question → hybrid search (fast, rough) → 20 candidates → reranker (slow, precise) → best 3 → LLM
```

| | Bi-encoder (Step 2's model) | Cross-encoder (this step) |
|---|---|---|
| Reads | question and chunk **separately** | question and chunk **together** |
| Output | two vectors, compared with cosine | one relevance score |
| Can be precomputed? | yes: chunk vectors are stored in ChromaDB | no: needs the question, so it runs at search time |
| Speed | ~9 ms for all 210 chunks | about 150 ms for 20 candidates |
| Use it for | finding candidates | **ordering** candidates |

---

## Before vs after (measured)

| | Before (hybrid, Step 6) | After (hybrid + rerank) |
|---|---|---|
| Retrieval hit rate @3 | 89% | **95%** |
| MRR (how high the right chunk ranks) | 0.79 | **0.84** |
| Correct answers (real LLM) | 17/19 | **19/19** |
| Cited a chunk | 19/19 | 19/19 |
| Said "I don't know" when it should | 3/3 | 3/3, **now without calling the LLM** |
| Time per question (retrieval) | 228 ms | ~380 ms |

You pay about 150 ms per question for answers that are all correct on the eval set. The Redis cache from Step 7 hides even that for repeated questions.

---

## 2. Code example

### 2.1 What a cross-encoder score looks like
Try it in the playground (`uv run python`):
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")    # ~90 MB, downloads once

model.predict([
    ("How long do cats sleep?", "Cats sleep 12 to 16 hours a day."),      # answers the question
    ("How long do cats sleep?", "Espresso is brewed at 9 bar."),          # doesn't
])
# [9.52, -11.15]
```

| Piece | Meaning |
|---|---|
| `CrossEncoder(...)` | the same library as Step 2, a different kind of model |
| `ms-marco-MiniLM-L6-v2` | trained on millions of real Bing search questions (MS MARCO), so it knows what a good answer looks like |
| `model.predict([(q, text), ...])` | a list of **(question, chunk)** pairs in, one score per pair out |
| the score | unbounded, about −11 to +11. **Above 0 = answers it; below 0 = doesn't** |

### 2.2 The score also says "nothing here answers"
On the 22 eval questions:

| Questions | Best chunk's reranker score |
|---|---|
| every answerable question | **above +3.7** |
| every unanswerable question ("What is the capital of France?") | **below −9.8** |

The gap is huge, so the rule **"drop anything below 0"** lets RAG answer "I don't know" **without spending an LLM call**.

### 2.3 How many candidates? (measured)
The reranker can only promote chunks that hybrid search handed it. A bigger pool finds more, but costs more:

| Candidates | Hit rate @3 | MRR | ms / question |
|---|---|---|---|
| 5 | 89% | 0.83 | 273 |
| 10 | 89% | 0.83 | 318 |
| **20** | **95%** | **0.84** | **383** |
| 40 | 95% | 0.84 | 541 |

20 is the sweet spot: 40 costs 160 ms more and finds nothing new.

---

## 3. Goal

**a) Create `src/advanced_rag/retrieval/rerank.py`.** Load the model once at the top:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```
Then write `rerank(query: str, hits: list[tuple[float, Chunk]], k: int = 3, min_score: float = 0.0) -> list[tuple[float, Chunk]]`:
1. If `hits` is empty, return `[]`.
2. Score every pair: `model.predict([(query, chunk.text) for _, chunk in hits])`.
3. Swap each chunk's old search score for its new reranker score: a list of `(float(score), chunk)`. (`zip` the scores with the hits.)
4. Keep only pairs with a score `>= min_score`.
5. Sort best first and return the top `k`.

**b) Use it in `rag.py`.** Retrieval becomes two lines, a wide rough net and then a careful pick:
```python
candidates = hybrid_search(collection, chunks, question, k=20, candidates=20)
hits = rerank(question, candidates, k)
```
If `hits` is empty, answer `NO_ANSWER` without calling the LLM, exactly as before.

**c) Measure it.** Add one entry to the `methods` dictionary in `evaluate.py`:
```python
"hybrid+rerank": lambda q: rerank(q, hybrid_search(collection, chunks, q, k=20, candidates=20), k=10),
```

### Evaluation can be wrong too
The first run said reranking **missed** "When was Python 3.0 released?". Reading the chunk it chose showed *"Python 3.0 was released on 3 December 2008"*: a **better** answer than the single phrase in the golden set. The reranker was right; the test data was too strict. So each question's `relevant` field is now a **list** of phrases, and `first_relevant_rank` accepts any of them:
```json
{"question": "Which country grows the most coffee?",
 "relevant": ["Brazil was the leading grower", "led by Brazil with 39% of the total"],
 "answer_contains": ["brazil"]}
```
When a "miss" looks suspicious, **read the chunks** before blaming the system. Only add phrases that truly answer the question: making the test easier to get a better number fools nobody but you.

### ✅ Checkpoint
```bash
uv run python -m advanced_rag.evaluate
```
Expected (times vary by machine):
```
Retrieval (19 answerable questions, top 3)
  semantic       hit rate 89%   MRR 0.76   9 ms/question
  bm25           hit rate 79%   MRR 0.68   198 ms/question
  hybrid         hit rate 89%   MRR 0.79   228 ms/question
  hybrid+rerank  hit rate 95%   MRR 0.84   379 ms/question
```
Add `--answers` to run all 22 questions through the real LLM: **19/19 correct, 19/19 cited, 3/3 "I don't know"**.

The retrieval pipeline changed, so clear old cached answers before trying it by hand:
```bash
uv run python -c "from advanced_rag import cache; print(cache.clear())"
```

---

## 4. Test

`tests/test_rerank.py` uses three tiny hand-made chunks:
```python
from advanced_rag.chunking import Chunk
from advanced_rag.retrieval.rerank import rerank

SLEEP = Chunk(text="Cats usually sleep between 12 and 16 hours a day.", source="cat.txt", index=25)
ESPRESSO = Chunk(text="Espresso is brewed under high pressure, typically 9 bar.", source="coffee.txt", index=45)
GUIDO = Chunk(text="Python was created by Guido van Rossum.", source="python_language.txt", index=7)
```

**This test checks:** the reranker fixes a bad order. **How:** search "hands over" the sleep chunk **last**; the reranker must put it first.
```python
def test_reranker_puts_the_answering_chunk_first():
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    assert rerank("How long do cats sleep?", hits)[0][1] == SLEEP
```

**This test checks:** at most `k` results, and the very same `Chunk` objects come back. **How:** `k=1` for an espresso question must return exactly `[ESPRESSO]`.
```python
def test_reranker_keeps_at_most_k_and_returns_the_same_chunks():
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    result = rerank("What pressure is espresso brewed at?", hits, k=1)
    assert [c for _, c in result] == [ESPRESSO]
```

**This test checks:** chunks that don't answer are dropped. **How:** nothing mentions France, so every score is below 0 and the result is empty.
```python
def test_chunks_that_do_not_answer_are_dropped():
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    assert rerank("What is the capital of France?", hits) == []


def test_no_candidates_means_no_results():
    assert rerank("anything", []) == []
```

**This test checks:** an unanswerable question never reaches the LLM. **How:** the fake LLM from `conftest.py` records every message; it must stay empty. (In `tests/test_rag.py`.)
```python
def test_unanswerable_question_skips_the_llm(index, fake_llm, fake_redis):
    collection, chunks = index
    assert answer("What is the capital of France?", collection, chunks) == NO_ANSWER
    assert fake_llm == []
```

**This test checks:** reranking never silently gets worse. **How:** the quality gate in `tests/test_evaluate.py` gains two lines, set just under what reranking measured (95% / 0.84).
```python
    assert report["hybrid+rerank"]["hit_rate"] >= 0.9
    assert report["hybrid+rerank"]["mrr"] >= 0.8
```

```bash
uv run pytest -v
```

---

## 5. Make it a module

`rerank` takes and returns the same shape as every search in this project, `[(score, Chunk), ...]`, so it slots in after any of them. Step 10 wraps "hybrid → rerank" into one function, `rag.retrieve()`, and tries adding LLM-rewritten queries to the candidate pool. Step 11 serves `retrieve()` over HTTP as `/search`.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/retrieval/rerank.py</code></strong></summary>

```python
from sentence_transformers import CrossEncoder

from advanced_rag.chunking import Chunk

# a small cross-encoder trained on real search questions (MS MARCO); ~90 MB, runs on CPU
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(query: str, hits: list[tuple[float, Chunk]], k: int = 3, min_score: float = 0.0) -> list[tuple[float, Chunk]]:
    # Search was fast but rough: it compared the question and each chunk SEPARATELY (two vectors).
    # A cross-encoder reads the question and one chunk TOGETHER and judges how well the chunk
    # answers it. Much more accurate, but too slow for all 210 chunks, so only the candidates get it.
    if not hits:
        return []
    scores = model.predict([(query, chunk.text) for _, chunk in hits])
    rescored = [(float(score), chunk) for score, (_, chunk) in zip(scores, hits)]
    # scores are unbounded (about -11 to +11): above 0 means "answers it", below means "doesn't".
    # On the eval set every answerable question's best chunk scored > 3.7, every unanswerable < -9.8.
    good = [pair for pair in rescored if pair[0] >= min_score]
    return sorted(good, key=lambda pair: pair[0], reverse=True)[:k]
```

The retrieval part of `answer()` in `src/advanced_rag/rag.py` (add `from advanced_rag.retrieval.rerank import rerank`):
```python
    # 2. retrieve: hybrid finds 20 rough candidates, the reranker picks the best k and drops
    #    the ones that don't answer the question. Nothing left = no point paying for an LLM call
    candidates = hybrid_search(collection, chunks, question, k=20, candidates=20)
    hits = rerank(question, candidates, k)
    if not hits:
        result = NO_ANSWER
```

</details>

---

**Previous:** [← Step 8: Evaluation](step-08-evaluation.md)

**Next:** [Step 10: Query rewriting →](step-10-query-rewriting.md)
