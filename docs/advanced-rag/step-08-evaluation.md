---
title: "Step 8: Evaluation"
series: "Advanced RAG"
group: "steps"
step: 8
slug: "step-08-evaluation"
date: "2026-10-21"
tag: "RAG"
excerpt: "Stop judging by eye: a 22-question golden set scores retrieval (hit rate, MRR) and answers, finds a real bug, and shows retrieval is the bottleneck."
minutes: 20
---

# Step 8: Evaluation

**You'll build:** `eval/questions.json` (the golden set) and `src/advanced_rag/evaluate.py` (`first_relevant_rank()`, `retrieval_scores()`, `evaluate_retrieval()`, `is_correct()`, `evaluate_answers()`).
**You'll learn:** golden sets, hit rate @k, MRR, checking answers automatically, and a quality-gate test that fails if retrieval gets worse.

---

## 1. Theory

Write down, once, a set of questions whose answers you already know: a **golden set**. Then measure the two halves of RAG **separately**. **Retrieval**: did search put a chunk that answers the question near the top? **Generation**: did the LLM give the right fact, cite a chunk, and say "I don't know" when it should? From now on every change (chunk size, search method, model) gets a **number**, not a feeling. ([Theory: Evaluation metrics](concepts.md#evaluation-metrics))

---

## Before vs after (measured)

| | Before | After |
|---|---|---|
| How quality is judged | by eye, one question at a time | 22 fixed questions, scored automatically |
| Retrieval, hit rate @3 / MRR | "looks right" | semantic **89% / 0.76**, BM25 **79% / 0.68**, hybrid **89% / 0.79** |
| Answers (hybrid + LLM) | "looks right" | **17/19** correct, **19/19** cited a chunk, **3/3** said "I don't know" |
| An empty reply from the LLM | undetected crash waiting to happen | **found on the first run**, fixed |
| A change makes retrieval worse | you find out much later | a test fails immediately |
| Cost of a retrieval check | n/a | no LLM calls, about a second |

Both wrong answers had the same cause: **retrieval never found the right chunk**, so the LLM never saw the answer.

| Question | The article says | Why search missed |
|---|---|---|
| Where did people **first drink** coffee? | "the **Sufis of Yemen**" | different wording |
| Which country **grows the most** coffee? | "Brazil was the **leading grower**" | different wording |

**Retrieval is the bottleneck, not the LLM.** A bigger LLM wouldn't have helped at all. That's exactly what Steps 9 and 10 go after.

---

## 2. Code example

### 2.1 The golden set: `eval/questions.json`
```json
[
  {"question": "How many hours a day do cats sleep?", "relevant": ["usually between 12 and 16 hours"], "answer_contains": ["16"]},
  {"question": "Which country grows the most coffee?", "relevant": ["Brazil was the leading grower", "led by Brazil with 39% of the total"], "answer_contains": ["brazil"]},
  {"question": "What is the capital of France?", "relevant": null, "answer_contains": ["i don't know"]}
]
```
| Field | Meaning |
|---|---|
| `question` | what a user would type |
| `relevant` | **a list** of phrases; a chunk containing **any** of them answers the question. `null` = the articles can't answer it |
| `answer_contains` | words a correct answer must include (any of them), lowercase |

Why phrases instead of chunk IDs like `cat.txt#25`? IDs change whenever chunking changes; a sentence from the article doesn't. The set has **22 questions**: 19 the articles answer, and 3 they can't (capital of France, the 2018 World Cup, Mars's moons).

**Why a list?** Some facts appear in more than one place. "Which country grows the most coffee?" is answered by "Brazil was the leading grower" **and** by "led by Brazil with 39% of the total". With a single phrase, a search that finds the second sentence would be scored as a miss even though it's right. **The golden set can be wrong too**: when a "miss" looks suspicious, read the chunk that was found before blaming the search. In Step 9 the reranker found a better answer than the one first written down. Only add phrases that truly answer the question; making the test easier just to get a better number is fooling yourself.

### 2.2 Hit rate @k and MRR, worked by hand
For each question, find the **rank** (1, 2, 3…) of the first chunk containing a relevant phrase. Say three questions give ranks **1**, **2** and **not found**:

| Metric | Question | Calculation | Result |
|---|---|---|---|
| **Hit rate @3**: is it in the top 3? | yes, yes, no | 2 / 3 | **67%** |
| **MRR**: 1/rank, missed = 0, averaged | 1, ½, 0 | (1 + 0.5 + 0) / 3 | **0.50** |

Hit rate says **whether** the right chunk is found; MRR says **how high**. A chunk at rank 5 doesn't count as a hit @3, but still earns 1/5 for MRR.

### 2.3 Checking answers
```python
text = answer.lower().replace("’", "'")    # some models write a curly apostrophe in "don’t"
return any(expected in text for expected in answer_contains)
```
Lowercase the answer, then check whether **any** accepted phrase appears in it. Simple, free and repeatable. (Asking a second LLM to grade answers, "LLM-as-a-judge", catches more nuance but costs calls and adds its own errors.)

### 2.4 The bug the first run found
On the very first `--answers` run, the LLM sent back an **empty reply** (`None`) for one question, and `answer()` passed it straight on, crashing the evaluation on `None.lower()`. It was invisible until something asked 22 questions in a row. Two changes to `rag.answer()` fix it:
```python
def answer(question, collection, chunks, k=3, use_cache=True):   # NEW: use_cache
    cached = cache.get(question) if use_cache else None             # evaluation turns the cache off
    ...
        result = ask(prompt, system=RULES)
        if not result:
            return NO_ANSWER                # NEW: answer safely, and don't cache a temporary failure
    ...
    if result and use_cache:
        cache.put(question, result)
```
`use_cache=False` makes the evaluation measure **fresh** answers every time, instead of reading yesterday's from Redis.

---

## 3. Goal

1. Create **`eval/questions.json`**: 19 answerable questions spread over the 4 articles (with a `relevant` phrase list copied **exactly** from the article text) and 3 unanswerable ones. The full set is in the repo.
2. Create **`src/advanced_rag/evaluate.py`** with `QUESTIONS_FILE = config.ROOT / "eval" / "questions.json"` and:
   - **`load_questions()`**: `json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))`
   - **`first_relevant_rank(hits, relevant)`**: walk `enumerate(hits, start=1)`; return the rank of the first chunk whose text contains any phrase; `None` if none does
   - **`retrieval_scores(ranks, k=3)`**: hit rate = share of ranks that are not `None` and ≤ k; MRR = average of `1 / rank` (0 for `None`)
   - **`evaluate_retrieval(collection, chunks, questions, k=3)`**: for semantic (top 10), BM25 (top 10) and hybrid (`k=10`), compute the ranks over the answerable questions and score them
   - **`is_correct(answer, answer_contains)`**: 2.3
   - **`evaluate_answers(collection, chunks, questions)`**: call `answer(..., use_cache=False)` for every question; record `correct`, `cited` (contains `[` and `#`), and the answer
   - a `__main__` that prints the retrieval table, and with `--answers` also the answer scores and every wrong answer
3. Apply the two `rag.py` changes from 2.4.

### ✅ Checkpoint
Retrieval only (free, about a second):
```bash
uv run python -m advanced_rag.evaluate
```
Expected:
```
Retrieval (19 answerable questions, top 3)
  semantic  hit rate 89%   MRR 0.76
  bm25      hit rate 79%   MRR 0.68
  hybrid    hit rate 89%   MRR 0.79
```
Answers too (22 real LLM calls, 1–2 minutes):
```bash
uv run python -m advanced_rag.evaluate --answers
```
Expected at this step: **17/19** correct, **19/19** cited, **3/3** "I don't know", with the two coffee questions listed as wrong. (If you run the finished repo, you'll also see the `hybrid+rerank` and `+rewrite` rows that Steps 9 and 10 add; the final committed numbers are in [`eval/results.md`](../../eval/results.md).)

---

## 4. Test

**This test checks:** ranks count from 1, and any listed phrase counts. **How:** two tiny chunks.
```python
def test_first_relevant_rank_counts_from_one():
    hits = [(0.9, Chunk("cats nap", "cat.txt", 0)), (0.8, Chunk("espresso uses 9 bar", "coffee.txt", 45))]
    assert first_relevant_rank(hits, ["9 bar"]) == 2
    assert first_relevant_rank(hits, ["Guido", "cats nap"]) == 1  # any of the phrases counts
    assert first_relevant_rank(hits, ["Guido"]) is None
```

**This test checks:** the hit-rate and MRR math. **How:** the worked example from 2.2. `approx` allows tiny rounding differences.
```python
def test_hit_rate_and_mrr():
    scores = retrieval_scores([1, 2, None], k=3)
    assert scores["hit_rate"] == approx(2 / 3)
    assert scores["mrr"] == approx((1 + 1 / 2 + 0) / 3)
```

**This test checks:** a hit below k isn't a hit, but still earns MRR. **How:** rank 5 with k = 3.
```python
def test_a_hit_below_k_does_not_count_but_still_scores_mrr():
    scores = retrieval_scores([5], k=3)
    assert scores["hit_rate"] == 0
    assert scores["mrr"] == approx(1 / 5)
```

**This test checks:** answer checking ignores case, accepts any option, and handles the curly apostrophe. **How:** one right, one curly, one wrong.
```python
def test_is_correct_is_case_insensitive_and_accepts_any_option():
    assert is_correct("Pythons are NOT VENOMOUS.", ["non-venomous", "not venomous"])
    assert is_correct("I don’t know", ["i don't know"])  # curly apostrophe
    assert not is_correct("They are venomous.", ["non-venomous", "not venomous"])
```

**This test checks:** the golden set itself has no typos. **How:** every `relevant` phrase must really exist in the data; a phrase that isn't there could never be found.
```python
def test_every_relevant_phrase_really_exists_in_the_data():
    # guards the golden set against typos: a phrase that isn't in the data could never be found
    corpus = " ".join(p.read_text(encoding="utf-8") for p in config.DATA_DIR.glob("*.txt"))
    for q in load_questions():
        assert q["answer_contains"], q["question"]
        for phrase in q["relevant"] or []:
            assert phrase in corpus, phrase
```

**This test checks:** retrieval never silently gets worse. **How:** a **quality gate**: if a future change to chunking or search drops hybrid below these numbers, the test suite goes red.
```python
def test_retrieval_quality_does_not_drop(index):
    # a quality gate: if a change to chunking or search makes retrieval worse, this fails
    collection, chunks = index
    report = evaluate_retrieval(collection, chunks, load_questions())
    assert report["hybrid"]["hit_rate"] >= 0.85
    assert report["hybrid"]["mrr"] >= 0.75
```

Two more in `tests/test_rag.py`, for the fixes from 2.4:

**This test checks:** an empty LLM reply becomes "I don't know" and isn't cached. **How:** `monkeypatch` replaces `rag.ask` with a function that returns `None`.
```python
def test_empty_llm_reply_becomes_i_dont_know_and_is_not_cached(index, fake_redis, monkeypatch):
    # found by evaluation: the LLM sometimes sends back nothing at all
    collection, chunks = index
    monkeypatch.setattr(rag, "ask", lambda prompt, system="": None)
    assert answer("How long do cats sleep?", collection, chunks) == NO_ANSWER
    assert cache.get("How long do cats sleep?") is None
```

**This test checks:** `use_cache=False` neither reads nor writes the cache. **How:** a stale answer sits in the cache; the call must ignore it and leave it untouched.
```python
def test_use_cache_false_never_reads_or_writes_the_cache(index, fake_llm, fake_redis):
    collection, chunks = index
    cache.put("How long do cats sleep?", "an old cached answer")
    assert answer("How long do cats sleep?", collection, chunks, use_cache=False) == "fake reply"
    assert cache.get("How long do cats sleep?") == "an old cached answer"
```

```bash
uv run pytest -v
```
Expected: **84 passed**.

---

## 5. Make it a module

`evaluate_retrieval()` takes a dictionary of named search methods, so comparing a new idea is one more line. Step 9 adds `"hybrid+rerank"` and raises the quality gate; Step 10 adds `"+rewrite"` (behind `--rewrite`, since it costs an LLM call per question); and a `--save` flag writes everything to [`eval/results.md`](../../eval/results.md), committed with the code so anyone can compare their numbers. See where evaluation sits in the [diagrams](diagrams.md).

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/evaluate.py</code> (as of this step)</strong></summary>

```python
import json

from advanced_rag import config, vectorstore
from advanced_rag.chunking import Chunk
from advanced_rag.retrieval import bm25
from advanced_rag.retrieval.hybrid import hybrid_search

QUESTIONS_FILE = config.ROOT / "eval" / "questions.json"


def load_questions() -> list[dict]:
    # each item: question, relevant (phrases that a chunk answering it contains; null = not answerable),
    # answer_contains (any of these words in the answer counts as correct)
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


# ---------- retrieval: did search find the right chunk? (no LLM, fast, free) ----------

def first_relevant_rank(hits: list[tuple[float, Chunk]], relevant: list[str]) -> int | None:
    # the position (1, 2, 3, ...) of the first retrieved chunk that contains any of the phrases
    for rank, (_, chunk) in enumerate(hits, start=1):
        if any(phrase in chunk.text for phrase in relevant):
            return rank
    return None


def retrieval_scores(ranks: list[int | None], k: int = 3) -> dict:
    # hit rate @k: how often the right chunk is somewhere in the top k
    # MRR (mean reciprocal rank): 1st place = 1, 2nd = 1/2, 3rd = 1/3, missed = 0, then averaged
    hits = [r is not None and r <= k for r in ranks]
    reciprocal = [1 / r if r is not None else 0 for r in ranks]
    return {"hit_rate": sum(hits) / len(ranks), "mrr": sum(reciprocal) / len(ranks)}


def evaluate_retrieval(collection, chunks: list[Chunk], questions: list[dict], k: int = 3) -> dict:
    methods = {
        "semantic": lambda q: vectorstore.search(collection, q, 10),
        "bm25": lambda q: bm25.search(q, chunks, 10),
        "hybrid": lambda q: hybrid_search(collection, chunks, q, k=10),
    }
    answerable = [q for q in questions if q["relevant"]]
    report = {}
    for name, search in methods.items():
        ranks = [first_relevant_rank(search(q["question"]), q["relevant"]) for q in answerable]
        report[name] = retrieval_scores(ranks, k)
    return report


# ---------- answers: did the LLM say the right thing? (real LLM calls, slower) ----------

def is_correct(answer: str, answer_contains: list[str]) -> bool:
    text = answer.lower().replace("’", "'")  # some models write a curly apostrophe in "don’t"
    return any(expected in text for expected in answer_contains)


def evaluate_answers(collection, chunks: list[Chunk], questions: list[dict]) -> list[dict]:
    from advanced_rag.rag import answer

    results = []
    for q in questions:
        reply = answer(q["question"], collection, chunks, use_cache=False)
        results.append({
            "question": q["question"],
            "answerable": q["relevant"] is not None,
            "correct": is_correct(reply, q["answer_contains"]),
            "cited": "#" in reply and "[" in reply,  # mentions a chunk id like [cat.txt#25]
            "answer": reply,
        })
    return results


if __name__ == "__main__":
    import sys

    collection = vectorstore.get_collection()
    chunks = vectorstore.all_chunks(collection)
    questions = load_questions()

    print(f"Retrieval ({sum(1 for q in questions if q['relevant'])} answerable questions, top 3)")
    for name, scores in evaluate_retrieval(collection, chunks, questions).items():
        print(f"  {name:9} hit rate {scores['hit_rate']:.0%}   MRR {scores['mrr']:.2f}")

    if "--answers" in sys.argv:
        results = evaluate_answers(collection, chunks, questions)
        answerable = [r for r in results if r["answerable"]]
        refusals = [r for r in results if not r["answerable"]]
        print(f"\nAnswers ({len(results)} LLM calls)")
        print(f"  correct      {sum(r['correct'] for r in answerable)}/{len(answerable)}")
        print(f"  cited        {sum(r['cited'] for r in answerable)}/{len(answerable)}")
        print(f"  said I don't know when it should  {sum(r['correct'] for r in refusals)}/{len(refusals)}")
        for r in results:
            if not r["correct"]:
                print(f"  ✗ {r['question']}\n      {r['answer'].strip()[:200]}")
```

`src/advanced_rag/rag.py`'s `answer()` after the 2.4 fixes:
```python
def answer(question: str, collection, chunks: list[Chunk], k: int = 3, use_cache: bool = True) -> str:
    # 1. asked before? answer from Redis: no search, no LLM call
    # (evaluation turns the cache off, to measure fresh answers every time)
    cached = cache.get(question) if use_cache else None
    if cached is not None:
        return cached

    # 2. retrieve; nothing relevant means there's no point paying for an LLM call
    hits = hybrid_search(collection, chunks, question, k)
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
```

</details>

---

**Previous:** [← Step 7: Redis cache](step-07-redis-cache.md)

**Next:** [Step 9: Reranking →](step-09-reranking.md)
