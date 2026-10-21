---
title: "Optimization 7: Cutting LLM cost and latency"
series: "Advanced RAG"
group: "optimization"
step: 7
slug: "opt-07-cutting-llm-cost-and-latency"
date: "2026-10-21"
tag: "RAG"
excerpt: "The LLM is 90% of the time on a new question. Shorter prompts, provider-side prompt caching, capped answers and routing easy questions to a smaller model all cut it down."
minutes: 7
---

# Optimization 7: Cutting LLM cost and latency

**You'll learn:** where an LLM call's time and money go, and five proven levers to reduce both.

---

## 1. Theory

An LLM call costs time and money in proportion to **tokens**: the tokens you send (system prompt + context + question) and the tokens it writes back. Writing is the slow part, one token at a time. So the levers are: **send fewer tokens**, **let the provider reuse work on repeated prompt prefixes**, **write fewer tokens**, and **use a smaller model when a smaller model is good enough**.

---

## Before vs after

Not measured in this project. Baseline and what to measure:

| | Baseline (measured) | After |
|---|---|---|
| LLM time per new question | ~3–5 s (4.9 s and 3.1 s measured) | measure it: average `llm.ask` time over the 22 eval questions |
| Context sent | 3 chunks × up to 200 tokens + the rules | measure it: prompt tokens from `response.usage.prompt_tokens` |
| Answer quality | 19/19 correct, 19/19 cited, 3/3 refusals | measure it: `evaluate --answers` must stay at these scores |

---

## 2. Code example

Docs-only examples of each lever against `llm.ask`:
```python
response = client.chat.completions.create(
    model=config.LLM_MODEL,
    messages=[
        {"role": "system", "content": RULES},     # 1. stable text FIRST: providers cache repeated prefixes
        {"role": "user", "content": prompt},      #    the changing part (context + question) goes last
    ],
    max_tokens=200,                               # 2. cap the answer length: fewer tokens to write
)
print(response.usage.prompt_tokens, response.usage.completion_tokens)   # 3. measure what you send and get
```

```python
hits = rag.retrieve(question, collection, chunks, k=2)     # 4. fewer chunks = a shorter prompt (k=3 today)
```

```python
def choose_model(hits) -> str:
    # 5. routing: a strong single match means an easy question, so a small fast model is enough
    best_score = hits[0][0]
    return SMALL_MODEL if best_score > 7 else config.LLM_MODEL
```
In plain English:
1. **Prompt caching:** many providers automatically (or via a flag) cache the processing of a prompt **prefix** they've seen before. That makes repeated prefixes cheaper and faster. It only works if the identical part (the rules) comes **before** the part that changes.
2. **`max_tokens`** stops the model from writing a long essay when two sentences will do.
3. **`usage`** tells you exactly how many tokens each call used. Log it and you can see your costs.
4. **Fewer chunks:** each chunk is up to 200 tokens of input. The reranker already puts the best chunk first, so check whether `k=2` keeps quality.
5. **Routing:** reranker scores above ~7 usually meant a clean, direct match in this project's eval set. A cheaper model may answer those just as well. Prove it with the evaluation before trusting it.

---

## 3. Goal

1. Log `prompt_tokens`, `completion_tokens` and time for every `ask` call.
2. Try one lever at a time: `max_tokens`, then `k=2`, then routing.
3. After each change, run:
   ```bash
   uv run python -m advanced_rag.evaluate --answers
   ```
   Keep the change only if it stays at 19/19 correct, 19/19 cited and 3/3 refusals.

---

## 4. Test

```python
def test_answer_sends_at_most_two_chunks(index, fake_llm, fake_redis):
    collection, chunks = index
    answer("How long do cats sleep?", collection, chunks, k=2)
    assert fake_llm[1]["content"].count("[") <= 2   # only two chunk labels like [cat.txt#25] in the prompt
```
This checks that the prompt really got shorter: the fake LLM records what was sent, so you can count the chunk labels without a real call.

---

## 5. Trade-offs

- **`max_tokens` too low cuts answers off mid-sentence.** Check the answers in `eval/results.md`, not just the scores.
- **Fewer chunks risks missing context** for questions whose answer spans two chunks.
- **Routing adds a second model to maintain and evaluate.** It's worth it at real traffic and cost, not for a demo.
- **Prompt caching rules differ per provider** (minimum prefix length, explicit cache markers, pricing). Read your provider's docs.

---

**Previous:** [← Optimization 6: Streaming and async](opt-06-streaming-and-async.md)

**Next:** [Optimization 8: Better retrieval ideas →](opt-08-better-retrieval-ideas.md)
