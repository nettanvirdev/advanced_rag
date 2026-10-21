---
title: "Step 6: Hybrid retrieval"
series: "Advanced RAG"
group: "steps"
step: 6
slug: "step-06-hybrid-retrieval"
date: "2026-10-21"
tag: "RAG"
excerpt: "Combine semantic search with BM25 using Reciprocal Rank Fusion, so exact names and numbers are found and nonsense returns nothing."
minutes: 15
---

# Step 6: Hybrid retrieval

**You'll build:** `src/advanced_rag/retrieval/hybrid.py` (`rrf()`, `hybrid_search()`), `all_chunks()` in `vectorstore.py`, and BM25 copied from `simple_rag`.
**You'll learn:** why meaning and exact words need different searches, Reciprocal Rank Fusion (RRF), a minimum similarity, and sharing a slow test setup with a fixture.

---

## 1. Theory

Semantic search understands **meaning** but is weak on **exact words**: names, numbers, codes. BM25 is the opposite. So run both and merge the results. Their scores can't be added (BM25 gives numbers like 12.0, cosine numbers like 0.65), so merge their **ranks** instead, with **Reciprocal Rank Fusion**. A **minimum similarity** of 0.3 drops weak semantic matches, so a nonsense question gets nothing instead of random chunks. ([Theory: Hybrid search and RRF](concepts.md#hybrid-search-and-rrf))

---

## Before vs after (measured)

Retrieval on the 19 answerable questions in `eval/questions.json` (Step 8 builds this test set):

| | Before (semantic only, Step 5) | After (hybrid) |
|---|---|---|
| Hit rate @3 (right chunk in the top 3) | 89% | 89% |
| MRR (how high the right chunk ranks) | 0.76 | **0.79** |
| "Guido van Rossum" | `coffee.txt#73` (0.20) ❌ | `python_language.txt#7` ✅ |
| "9 bar" | `coffee.txt#68` (0.13) ❌ | `coffee.txt#45` ✅ (espresso, 9 bar) |
| "asdfgh" | 3 random cat chunks | **nothing** ✅ |
| Time per question | 9 ms | 228 ms (BM25 alone: 198 ms) |
| BM25 alone, for comparison | | 79% / 0.68 |

A person's name carries little "meaning" for an embedding model. BM25 just sees the rare words "guido" and "rossum" and finds them. Hybrid gets the best of both.

The extra time is almost all BM25, which re-tokenizes every chunk on every question. That's fine for learning; the Optimization section shows how precomputing it makes BM25 about 1,000× faster with identical results.

---

## 2. Code example

### 2.1 Copy BM25 from `simple_rag`
Copy these three files into `src/advanced_rag/retrieval/` (plus an empty `__init__.py`), and their tests into `tests/`:

| File | Why it comes along |
|---|---|
| `keyword.py` | `tokenize()`, which BM25 uses |
| `tfidf.py` | `idf()`, which BM25 uses |
| `bm25.py` | the BM25 `search()` itself |

The only change is the imports: `from ragi.` becomes `from advanced_rag.`. Drop the two copied tests that checked numbers on `simple_rag`'s four tiny files (`test_search_beats_keyword_tie_on_real_data` and `test_score_matches_checkpoint_on_real_data`), because this project's data is different.

BM25 works on chunks without any change: it only reads `.text`, and a `Chunk` has `.text` just like a `Document`. It searches all 210 chunks per question.

**`semantic.py` is not copied.** ChromaDB's `vectorstore.search()` from Step 5 replaces it.

### 2.2 BM25 needs the texts: `all_chunks()`
```python
result = collection.get(include=["documents", "metadatas"])   # every stored row, without the vectors
```
Reading the chunks back out of the index guarantees that semantic search and BM25 search **exactly the same chunks**.

### 2.3 Reciprocal Rank Fusion, worked by hand
Each method votes for its top results. 1st place earns 1/61 points, 2nd place 1/62, and so on: `1 / (60 + rank)`. Add up each chunk's points from both lists.
```
semantic: 1st A,  2nd B
bm25:     1st C,  2nd B

B = 1/62 + 1/62 = 0.0323   ← found by both, so it wins
A = 1/61        = 0.0164
C = 1/61        = 0.0164
```
**Why 60?** It's the standard constant from the paper that introduced RRF. It stops 1st place from completely dominating 2nd place.

In Python, a dictionary keeps a running total per chunk:
```python
scores = {}                                           # chunk id → points so far
for hits in result_lists:                             # each method's list
    for rank, (_, chunk) in enumerate(hits, start=1): # rank starts at 1, not 0
        cid = vectorstore.chunk_id(chunk)
        scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)   # .get(cid, 0): 0 if not seen yet
```
The key is the chunk **ID**, not the chunk, so the same chunk from two lists lands in **one** place.

### 2.4 The minimum similarity
```python
semantic = [(s, c) for s, c in vectorstore.search(collection, query, candidates) if s >= min_similarity]
```
Real questions score 0.5 and up; nonsense scores about 0.2. BM25 already drops chunks that share no words with the question, so "asdfgh" ends with two empty lists, and RRF of two empty lists is empty.

---

## 3. Goal

1. Copy `keyword.py`, `tfidf.py`, `bm25.py` (and an empty `__init__.py`) into `src/advanced_rag/retrieval/`, and `test_keyword.py`, `test_tfidf.py`, `test_bm25.py` into `tests/`. Rename the imports, and drop the two old-data tests.
2. Add **`all_chunks(collection) -> list[Chunk]`** to `vectorstore.py`: `collection.get(include=["documents", "metadatas"])`, then build a `Chunk` from each `(text, meta)` pair.
3. Create **`src/advanced_rag/retrieval/hybrid.py`** with `RRF_K = 60` and:
   - **`rrf(result_lists, k=3)`**: the loop from 2.3; also keep `chunks[cid] = chunk`; then sort the IDs by points, best first, keep `k`, and return `(points, chunk)` pairs.
   - **`hybrid_search(collection, chunks, query, k=3, candidates=10, min_similarity=0.3)`**: semantic top `candidates` minus weak scores, BM25 top `candidates`, then `rrf([semantic, keyword], k)`.
   - a `__main__` comparison box that prints semantic, BM25 and hybrid results for each question, and "(nothing good enough)" for an empty list.

### ✅ Checkpoint
```bash
uv run python -c "from advanced_rag import vectorstore; from advanced_rag.retrieval.hybrid import hybrid_search; c = vectorstore.get_collection(); ch = vectorstore.all_chunks(c); [print(q, [vectorstore.chunk_id(x) for _, x in hybrid_search(c, ch, q)]) for q in ['Guido van Rossum', '9 bar', 'asdfgh']]"
```
Expected:
```
Guido van Rossum ['python_language.txt#7', 'python_language.txt#1', 'python_language.txt#0']
9 bar ['coffee.txt#45', 'coffee.txt#44', 'python_language.txt#10']
asdfgh []
```
Then compare all three methods side by side:
```bash
uv run python -m advanced_rag.retrieval.hybrid
```
Type `Guido van Rossum` and watch semantic get it wrong while hybrid gets it right.

---

## 4. Test

**This test checks:** the RRF formula. **How:** one list, one chunk in 1st place: exactly 1/61.
```python
def test_rrf_first_place_scores_one_over_61():
    assert rrf([[(0.9, A)]])[0] == (1 / 61, A)
```

**This test checks:** a chunk found by both methods beats chunks found by one. **How:** B is 2nd in both lists; A and C are 1st in one each.
```python
def test_rrf_rewards_chunks_found_by_both_methods():
    semantic = [(0.9, A), (0.8, B)]
    keyword = [(12.0, C), (7.0, B)]
    # B is 2nd in both lists: 1/62 + 1/62 beats A's and C's single 1/61
    assert [c for _, c in rrf([semantic, keyword])] == [B, A, C]
```

**This test checks:** only rank counts, never the raw score. **How:** BM25's 12.0 and cosine's 0.9 give the winner the same points either way round.
```python
def test_rrf_ignores_score_scales():
    # BM25's 12.0 doesn't outweigh a cosine 0.9: only the rank counts
    assert rrf([[(0.9, A)], [(12.0, C)]])[0][0] == rrf([[(0.9, C)], [(12.0, A)]])[0][0]
```
(`A`, `B`, `C` are tiny chunks: `A = Chunk(text="a", source="a.txt", index=0)`, and so on.)

The next tests need the real index. Building it takes a few seconds, so a **fixture with `scope="module"`** builds it **once** and shares it with every test in the file, instead of rebuilding it per test:
```python
import pytest

@pytest.fixture(scope="module")
def index(tmp_path_factory):
    collection = vectorstore.build_index(tmp_path_factory.mktemp("chroma"))
    return collection, vectorstore.all_chunks(collection)
```
`tmp_path_factory` is the module-wide version of `tmp_path`. (In Step 7 this fixture moves into `tests/conftest.py` with `scope="session"`, so it's built once for the **whole** test run and shared with the RAG tests.)

**This test checks:** BM25 sees exactly what's stored. **How:** both counts must be 210.
```python
def test_all_chunks_reads_every_stored_chunk(index):
    collection, chunks = index
    assert len(chunks) == collection.count() == 210
```

**This test checks:** hybrid finds a name that semantic search alone misses. **How:** both halves are asserted, so the test also documents the weakness.
```python
def test_hybrid_finds_exact_names_that_semantic_misses(index):
    collection, chunks = index
    # semantic search alone ranks a coffee chunk first for this (score 0.20)
    assert vectorstore.search(collection, "Guido van Rossum")[0][1].source == "coffee.txt"
    assert hybrid_search(collection, chunks, "Guido van Rossum")[0][1].source == "python_language.txt"
```

**This test checks:** exact numbers, nonsense, and that good semantic answers survive. **How:** three real questions with known chunk IDs.
```python
def test_hybrid_finds_exact_numbers(index):
    collection, chunks = index
    assert vectorstore.chunk_id(hybrid_search(collection, chunks, "9 bar")[0][1]) == "coffee.txt#45"


def test_hybrid_returns_nothing_for_nonsense(index):
    collection, chunks = index
    assert hybrid_search(collection, chunks, "asdfgh") == []


def test_hybrid_keeps_good_semantic_answers(index):
    collection, chunks = index
    assert vectorstore.chunk_id(hybrid_search(collection, chunks, "How long do cats sleep?")[0][1]) == "cat.txt#25"
```

```bash
uv run pytest -v
```
Expected: **64 passed** (including the copied keyword, TF-IDF and BM25 tests).

---

## 5. Make it a module

`hybrid_search()` has the same output shape as every search so far: `[(score, Chunk), ...]`, best first. Step 7's `rag.py` calls it to fetch the context for the LLM, and Step 9 widens it to 20 candidates and hands them to a reranker. See where it sits in the [diagrams](diagrams.md).

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/retrieval/hybrid.py</code> and <code>all_chunks()</code></strong></summary>

`src/advanced_rag/retrieval/hybrid.py`:
```python
from advanced_rag.chunking import Chunk
from advanced_rag.retrieval import bm25
from advanced_rag import vectorstore

RRF_K = 60  # the standard constant from the original Reciprocal Rank Fusion paper


def rrf(result_lists: list[list[tuple[float, Chunk]]], k: int = 3) -> list[tuple[float, Chunk]]:
    # Reciprocal Rank Fusion: ignore each method's own scores (they're on different scales)
    # and use only the RANK. 1st place earns 1/61, 2nd 1/62, ... A chunk found by both
    # methods earns points twice, so it rises to the top.
    scores = {}
    chunks = {}
    for hits in result_lists:
        for rank, (_, chunk) in enumerate(hits, start=1):
            cid = vectorstore.chunk_id(chunk)
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
            chunks[cid] = chunk

    best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    return [(scores[cid], chunks[cid]) for cid in best]


def hybrid_search(
    collection,
    chunks: list[Chunk],
    query: str,
    k: int = 3,
    candidates: int = 10,
    min_similarity: float = 0.3,
) -> list[tuple[float, Chunk]]:
    # meaning: semantic hits, minus the weak ones (nonsense scores ~0.2, real questions 0.5+)
    semantic = [(s, c) for s, c in vectorstore.search(collection, query, candidates) if s >= min_similarity]
    # exact words: BM25 already drops chunks that share no words with the question
    keyword = bm25.search(query, chunks, candidates)
    return rrf([semantic, keyword], k)


if __name__ == "__main__":
    collection = vectorstore.get_collection()
    chunks = vectorstore.all_chunks(collection)
    print(f"{len(chunks)} chunks loaded")
    while True:
        query = input("compare> ")
        for name, hits in [
            ("semantic", vectorstore.search(collection, query)),
            ("bm25", bm25.search(query, chunks, 3)),
            ("hybrid", hybrid_search(collection, chunks, query)),
        ]:
            print(f"--- {name}")
            for score, chunk in hits:
                print(f"  {score:7.3f}  {vectorstore.chunk_id(chunk):24} {chunk.text[:70]}...")
            if not hits:
                print("  (nothing good enough)")
```

Added to `src/advanced_rag/vectorstore.py`, just above `build_index`:
```python
def all_chunks(collection) -> list[Chunk]:
    # every stored chunk, without vectors: BM25 needs the texts, and reading them from
    # the index guarantees both searches look at exactly the same chunks
    result = collection.get(include=["documents", "metadatas"])
    return [
        Chunk(text=text, source=meta["source"], index=meta["index"])
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
```

The copied files (`keyword.py`, `tfidf.py`, `bm25.py`) are unchanged from `simple_rag` apart from their imports:
```python
from advanced_rag.loaders import Document
from advanced_rag.retrieval.keyword import tokenize
from advanced_rag.retrieval.tfidf import idf
```

</details>

---

**Previous:** [← Step 5: ChromaDB vector store](step-05-chromadb-vector-store.md)

**Next:** [Step 7: Redis cache →](step-07-redis-cache.md)
