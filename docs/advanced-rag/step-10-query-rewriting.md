---
title: "Step 10: Query rewriting"
series: "Advanced RAG"
group: "steps"
step: 10
slug: "step-10-query-rewriting"
date: "2026-10-21"
tag: "RAG"
excerpt: "Let the LLM rewrite a question into better search words, merge both searches safely, and measure it: no gain on this eval set, +4.5 s per question, so it ships switched off."
minutes: 12
---

# Step 10: Query rewriting

**You'll build:** `rewrite()` in `src/advanced_rag/retrieval/rewrite.py`, and `retrieve()` in `rag.py` with a `rewrite_query` switch.
**You'll learn:** query rewriting, merging two searches safely with RRF, and deciding with numbers instead of hype.

---

## 1. Theory

People don't ask questions in the words an article uses: "Which country **grows the most** coffee?" vs. "Brazil was the **leading grower**". **Query rewriting** asks the LLM to turn the question into the words an article would most likely use, and searches with those too. It's a popular technique, but it costs one extra LLM call per question, so it has to **earn** its place on your evaluation. ([Theory: Query rewriting](concepts.md#query-rewriting))

---

## Before vs after (measured)

| | Before (hybrid + rerank, Step 9) | After (+ rewrite) |
|---|---|---|
| Retrieval hit rate @3 | 95% | 95% |
| MRR | 0.84 | 0.84 |
| Time per question | 379 ms | **4,868 ms** |
| Extra LLM calls | 0 | 1 per question (~3.7 s on average) |

**No gain, 13× slower.** So rewriting ships **switched off** (`rewrite_query=False`), and the evaluation is the reason. That's the lesson of this step: a technique being popular doesn't make it a win on *your* data. Measure.

---

## 2. Code example

### 2.1 What the rewrites look like (real LLM output)
| Question | Rewritten query | Effect |
|---|---|---|
| Which country grows the most coffee? | largest coffee-producing country Brazil | already found without it |
| Who wrote the Zen of Python? | Zen of Python author Tim Peters | already found without it |
| Where did people first drink coffee? | history of coffee origin first consumption Ethiopia | **worse**: the right chunk dropped from rank 6 to 7 |

The rewrites are sensible. They just don't find anything the reranked hybrid search hadn't already found on these clean, specific questions. The third one even guessed a different story (Ethiopia) than the article's answer (the Sufis of Yemen).

### 2.2 The rewrite function
```python
from advanced_rag.llm import ask

RULES = (
    "Rewrite the user's question as a short search query for an encyclopedia. "
    "Use the most likely words an article would use, add key synonyms, and drop filler words. "
    "Reply with the query only."
)

def rewrite(question: str) -> str:
    query = " ".join((ask(question, system=RULES) or "").split())
    return query or question
```
| Piece | Meaning |
|---|---|
| `ask(question, system=RULES)` | your `llm.ask` from Step 7, with rewriting rules instead of answering rules |
| `... or ""` | the LLM can reply with nothing (Step 8 found that); treat it as an empty string |
| `" ".join(x.split())` | `split()` cuts on any spaces **and line breaks**, `join` glues back with single spaces: a multi-line reply becomes one line |
| `return query or question` | an empty rewrite falls back to the original question, so search never runs on nothing |

### 2.3 The safe design: add, never replace
```
question ──► hybrid search ──► 20 candidates ─┐
   │                                           ├─► RRF merge ─► rerank against the ORIGINAL question ─► best 3
   └─► rewrite() ─► hybrid search ─► 20 ───────┘
```
- The rewritten query only **adds** candidates to the pool. The original search still runs.
- The two lists are merged with your `rrf()` from Step 6.
- The reranker judges everything against the **original** question, because that's what the user actually asked.

So a bad rewrite can't push out a good chunk; at worst it wastes time. A test proves it (section 4).

---

## 3. Goal

**a) Create `src/advanced_rag/retrieval/rewrite.py`** with `RULES` and `rewrite()` from section 2.2.

**b) Give retrieval its own function in `rag.py`.** Move the two retrieval lines out of `answer()` into:
```python
def retrieve(question: str, collection, chunks: list[Chunk], k: int = 3, rewrite_query: bool = False) -> list[tuple[float, Chunk]]:
```
Steps:
1. `candidates = hybrid_search(collection, chunks, question, k=20, candidates=20)`
2. If `rewrite_query` is on:
   - `rewritten = hybrid_search(collection, chunks, rewrite(question), k=20, candidates=20)`
   - `candidates = rrf([candidates, rewritten], k=30)` (up to 30, since the two lists overlap)
3. Return `rerank(question, candidates, k)`, with the **original** question.

Then `answer()` calls `hits = retrieve(question, collection, chunks, k, rewrite_query)`, and gains a `rewrite_query: bool = False` parameter that it passes through. Imports: `rrf` from `hybrid`, `rewrite` from `rewrite`.

**c) Measure it.** In `evaluate.py`, `evaluate_retrieval` gets an `include_rewrite: bool = False` parameter. When it's on, add a fifth method:
```python
methods["+rewrite"] = lambda q: retrieve(q, collection, chunks, k=10, rewrite_query=True)
```
It's off unless asked for, because it makes one real LLM call per question. The `__main__` block switches it on with `--rewrite`.

### ✅ Checkpoint
```bash
uv run python -m advanced_rag.evaluate --rewrite
```
Expected (times vary by machine):
```
Retrieval (19 answerable questions, top 3)
  semantic       hit rate 89%   MRR 0.76   9 ms/question
  bm25           hit rate 79%   MRR 0.68   198 ms/question
  hybrid         hit rate 89%   MRR 0.79   228 ms/question
  hybrid+rerank  hit rate 95%   MRR 0.84   379 ms/question
  +rewrite       hit rate 95%   MRR 0.84   4868 ms/question
```

### When rewriting *does* help
Keep the switch: on other data it can pay off. It helps most with:
- **vague or chatty questions**: "so umm what's the deal with those big snakes and how they eat stuff?"
- **follow-ups in a chat**: "and how long do *they* live?" needs the earlier question to know what "they" is
- **typos and slang** that neither BM25 nor the embedding model recognizes

The eval set here has clean, specific questions, which is exactly where rewriting has the least to add. Write a few chatty questions into `eval/questions.json` and measure again to see the difference on your own data.

---

## 4. Test

`tests/test_rewrite.py` uses the fake LLM, so it makes no real calls.

**This test checks:** `rewrite` asks the LLM with the rewriting rules. **How:** the fake LLM records the system message and replies "fake reply".
```python
def test_rewrite_asks_the_llm_with_the_rewrite_rules(fake_llm):
    assert rewrite("Which country grows the most coffee?") == "fake reply"
    assert fake_llm[0] == {"role": "system", "content": RULES}
```

**This test checks:** an empty LLM reply falls back to the question. **How:** `monkeypatch` swaps `ask` *inside the rewrite module* for a function that returns `None`.
```python
def test_empty_reply_falls_back_to_the_question(monkeypatch):
    monkeypatch.setattr(rewrite_module, "ask", lambda question, system="": None)
    assert rewrite("Are pythons venomous?") == "Are pythons venomous?"
```

**This test checks:** a multi-line reply becomes one clean line. **How:** a fake reply with line breaks must come out as single-spaced words.
```python
def test_multi_line_reply_becomes_one_line(monkeypatch):
    monkeypatch.setattr(rewrite_module, "ask", lambda question, system="": "python snakes\nvenom\n")
    assert rewrite("Are pythons venomous?") == "python snakes venom"
```

**This test checks:** a useless rewrite can't hurt. **How:** the fake LLM "rewrites" every question into `"fake reply"`, which matches nothing. The cat question must still get its sleep chunk first, because the original search still runs and the reranker uses the original question.
```python
def test_a_useless_rewrite_cannot_hurt(index, fake_llm):
    collection, chunks = index
    hits = retrieve("How long do cats sleep?", collection, chunks, rewrite_query=True)
    assert vectorstore.chunk_id(hits[0][1]) == "cat.txt#25"
```

Imports: `from advanced_rag import vectorstore`, `from advanced_rag.rag import retrieve`, `from advanced_rag.retrieval import rewrite as rewrite_module`, `from advanced_rag.retrieval.rewrite import RULES, rewrite`.

**Why `rewrite_module`?** `rewrite.py` did `from advanced_rag.llm import ask`, so it holds its own name `ask`. To fake it, patch the name where it's **used** (the rewrite module), not where it was defined.

```bash
uv run pytest -v
```

---

## 5. Make it a module

`rag.retrieve()` is now the one place that decides how chunks are found: hybrid → (optional rewrite) → rerank. `answer()` uses it, `evaluate.py` measures it, and Step 11 serves it over HTTP as `GET /search`, so all three always use the same retrieval.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/retrieval/rewrite.py</code> and <code>src/advanced_rag/rag.py</code></strong></summary>

`src/advanced_rag/retrieval/rewrite.py`:
```python
from advanced_rag.llm import ask

RULES = (
    "Rewrite the user's question as a short search query for an encyclopedia. "
    "Use the most likely words an article would use, add key synonyms, and drop filler words. "
    "Reply with the query only."
)


def rewrite(question: str) -> str:
    # ask the LLM for better search words: "Which country grows the most coffee?" ->
    # "largest coffee-producing country Brazil". Falls back to the question if the reply is empty.
    query = " ".join((ask(question, system=RULES) or "").split())  # also flattens multi-line replies
    return query or question
```

`src/advanced_rag/rag.py`:
```python
from advanced_rag import cache, vectorstore
from advanced_rag.chunking import Chunk
from advanced_rag.llm import ask
from advanced_rag.retrieval.hybrid import hybrid_search, rrf
from advanced_rag.retrieval.rerank import rerank
from advanced_rag.retrieval.rewrite import rewrite

RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know\". "
    "Cite the chunk id in square brackets, like [cat.txt#25]."
)
NO_ANSWER = "I don't know"


def build_context(hits: list[tuple[float, Chunk]]) -> str:
    # label every chunk with its id, so the LLM can cite exactly where an answer came from
    return "\n\n".join(f"[{vectorstore.chunk_id(chunk)}]\n{chunk.text}" for _, chunk in hits)


def retrieve(question: str, collection, chunks: list[Chunk], k: int = 3, rewrite_query: bool = False) -> list[tuple[float, Chunk]]:
    # hybrid finds 20 rough candidates; the reranker picks the best k and drops the ones that don't answer
    candidates = hybrid_search(collection, chunks, question, k=20, candidates=20)
    if rewrite_query:
        # also search with LLM-rewritten words, and merge both candidate lists.
        # Off by default: on the eval set it didn't find anything new, and costs ~3.7 s per question
        rewritten = hybrid_search(collection, chunks, rewrite(question), k=20, candidates=20)
        candidates = rrf([candidates, rewritten], k=30)
    # always rerank against the ORIGINAL question: that's what the user actually asked
    return rerank(question, candidates, k)


def answer(
    question: str, collection, chunks: list[Chunk], k: int = 3, use_cache: bool = True, rewrite_query: bool = False
) -> str:
    # 1. asked before? answer from Redis: no search, no LLM call
    # (evaluation turns the cache off, to measure fresh answers every time)
    cached = cache.get(question) if use_cache else None
    if cached is not None:
        return cached

    # 2. retrieve. Nothing relevant = no point paying for an LLM call
    hits = retrieve(question, collection, chunks, k, rewrite_query)
    if not hits:
        result = NO_ANSWER
    else:
        prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
        result = ask(prompt, system=RULES)
        if not result:
            # the LLM sent back nothing (found by evaluation): answer safely, and don't cache a failure
            return NO_ANSWER

    # 3. remember it for next time
    if result and use_cache:
        cache.put(question, result)
    return result


if __name__ == "__main__":
    import time

    collection = vectorstore.get_collection()
    chunks = vectorstore.all_chunks(collection)
    print(f"{len(chunks)} chunks loaded")
    while True:
        question = input("ask> ")
        was_cached = cache.get(question) is not None
        start = time.perf_counter()
        reply = answer(question, collection, chunks)
        ms = (time.perf_counter() - start) * 1000
        print(f"{reply}\n  ({ms:.0f} ms{', from cache' if was_cached else ''})")
```

</details>

---

**Previous:** [← Step 9: Reranking](step-09-reranking.md)

**Next:** [Step 11: FastAPI →](step-11-fastapi.md)
