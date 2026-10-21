---
title: "Optimization 8: Better retrieval ideas"
series: "Advanced RAG"
group: "optimization"
step: 8
slug: "opt-08-better-retrieval-ideas"
date: "2026-10-21"
tag: "RAG"
excerpt: "HyDE, multi-query, small-to-big retrieval, contextual retrieval and metadata filters: what each one fixes, and how to prove it with the eval set."
minutes: 9
---

# Optimization 8: Better retrieval ideas

**You'll learn:** five proven retrieval techniques, what kind of miss each one fixes, and how to test each against `eval/questions.json`.

---

## 1. Theory

Retrieval quality caps answer quality: the LLM can't cite a chunk it never saw. This project is at a 95% hit rate. The remaining miss is "Where did people first drink coffee?", whose answer is worded as "Sufis of Yemen". Each technique below attacks a different cause of misses: **wording gaps**, **chunks too small to stand alone**, **chunks missing their context**, and **searching where the answer can't be**.

---

## Before vs after

Not measured in this project. Baseline and what to measure:

| Technique | Fixes | Baseline (measured) | After |
|---|---|---|---|
| HyDE | question worded unlike the answer | 95% hit rate, MRR 0.84, ~380 ms | measure it: `evaluate` hit rate/MRR + ms (adds an LLM call, ~3–5 s) |
| Multi-query | one phrasing misses | same | measure it: same (adds an LLM call + several searches) |
| Small-to-big | chunk found, but too little context for the answer | 19/19 correct | measure it: `evaluate --answers` |
| Contextual retrieval | chunks that only make sense inside their article ("It is a social species") | 95% / 0.84 | measure it: rebuild the index, then `evaluate`. Anthropic reported large drops in retrieval failures with this on their benchmarks |
| Metadata filters | searching files that can't contain the answer | ~9 ms vector search | measure it: hit rate and ms with a `where` filter |

Related result from this project: **query rewriting** (step 10) was measured. It made no gain (95% → 95%) and cost +4.5 s per question. Measure before you adopt any of these.

---

## 2. Code example

Docs-only sketches, one per idea.

**HyDE (Hypothetical Document Embeddings):** search with a fake answer instead of the question.
```python
fake_answer = ask(question, system="Write a short encyclopedia paragraph that answers this.")
hits = vectorstore.search(collection, fake_answer, 20)   # an answer looks more like the chunks than a question does
```

**Multi-query:** several phrasings, merged with the RRF you already have.
```python
variants = ask(question, system="Give 3 different search queries for this question, one per line.").splitlines()
candidates = rrf([hybrid_search(collection, chunks, v, k=20, candidates=20) for v in [question, *variants]], k=30)
hits = rerank(question, candidates)                    # always judge against the ORIGINAL question
```

**Small-to-big:** search small chunks, but send the neighbours to the LLM.
```python
def expand(chunk, collection):
    ids = [f"{chunk.source}#{i}" for i in (chunk.index - 1, chunk.index, chunk.index + 1)]
    found = collection.get(ids=ids)                    # missing ids (before #0 or after the last) are skipped
    return " ".join(found["documents"])
```

**Contextual retrieval:** at indexing time, prepend a sentence that places the chunk in its article.
```python
context = ask(f"Article: {doc.text[:4000]}\n\nChunk: {chunk.text}",
              system="In one sentence, say what this chunk is about within the article.")
chunk_text_to_embed = context + " " + chunk.text      # embed (and BM25-index) this, but show the LLM the original
```

**Metadata filters:** only search where the answer can be.
```python
collection.query(query_embeddings=embed([question]), n_results=10, where={"source": "coffee.txt"})
```

In plain English: HyDE and multi-query fix **wording gaps** at question time. Small-to-big fixes **too little context** at answer time. Contextual retrieval fixes **context-free chunks** at indexing time. Filters shrink the search space using the labels stored by `add_chunks`.

---

## 3. Goal

1. Pick **one** technique, aimed at the miss you actually have. Here that's a wording gap, so try HyDE or multi-query first.
2. Add it behind a flag (like `rewrite_query` in `rag.retrieve`), off by default.
3. Measure it with:
   ```bash
   uv run python -m advanced_rag.evaluate
   ```
   Add it to `evaluate_retrieval` as a new method row, the way `+rewrite` was added.
4. Keep it only if it beats `hybrid+rerank` by more than one question (one question is about 5 points here), at an acceptable latency.

---

## 4. Test

```python
def test_small_to_big_includes_the_neighbours(index):
    collection, chunks = index
    middle = next(c for c in chunks if c.source == "cat.txt" and c.index == 25)
    text = expand(middle, collection)
    assert middle.text in text
    assert len(text) > len(middle.text)       # the chunks before and after were added
```
This checks the expansion really adds surrounding context. The same pattern (a small, real-index check) applies to each technique, while the evaluation measures whether it helps.

---

## 5. Trade-offs

- **Every LLM-based technique (HyDE, multi-query, contextual retrieval) adds LLM calls.** At query time that's seconds per question. At indexing time it's one call per chunk (210 here, millions elsewhere), which is cost to budget for.
- **Small-to-big makes prompts longer** (see Optimization 7).
- **Filters can hide the answer** when a question spans files. Only filter when the user or the question clearly limits the scope.
- **A 19-question eval set can't prove small gains.** Grow it with real user questions before adopting complex techniques.

---

**Previous:** [← Optimization 7: Cutting LLM cost and latency](opt-07-cutting-llm-cost-and-latency.md)

**Next:** [Optimization 9: Production and deployment →](opt-09-production-and-deployment.md)
