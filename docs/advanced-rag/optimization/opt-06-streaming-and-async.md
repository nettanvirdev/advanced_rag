---
title: "Optimization 6: Streaming and async"
series: "Advanced RAG"
group: "optimization"
step: 6
slug: "opt-06-streaming-and-async"
date: "2026-10-21"
tag: "RAG"
excerpt: "Send the answer word by word as the LLM writes it, so users see text almost immediately, and keep the server responsive while it waits on slow calls."
minutes: 8
---

# Optimization 6: Streaming and async

**You'll learn:** time-to-first-token, how to stream an LLM answer through FastAPI, and when async endpoints help.

---

## 1. Theory

`POST /ask` returns nothing until the whole answer is written (~3–5 s). Most of that time the LLM is producing text **token by token**, and it can hand each piece over as soon as it exists. **Streaming** forwards those pieces to the user immediately. Total time stays the same, but the user sees the answer start almost at once. The metric that matters becomes **time to first token (TTFT)**, not total time.

---

## Before vs after

Not measured in this project. Baseline and what to measure:

| | Before (measured: `POST /ask`, new question) | After (streaming) |
|---|---|---|
| Time until the first word is visible | ~3.1 s (the whole answer) | measure it: time to the first chunk of the stream (TTFT). Widely reported to be well under a second for typical models |
| Total time | ~3.1 s | measure it: should be about the same |
| Retrieval before the LLM | ~380 ms | unchanged: still happens before the first token |

---

## 2. Code example

Docs-only: stream from the OpenAI-compatible API and forward it with FastAPI.
```python
from fastapi.responses import StreamingResponse

from advanced_rag import config, llm, rag


def stream_answer(prompt: str):
    stream = llm.client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": rag.RULES}, {"role": "user", "content": prompt}],
        stream=True,                                  # ← ask for pieces as they are written
    )
    for event in stream:
        piece = event.choices[0].delta.content if event.choices else None
        if piece:
            yield piece                               # hand each piece to FastAPI right away


@app.post("/ask/stream")
def ask_stream(body: Question, request: Request):
    hits = rag.retrieve(body.question, request.app.state.collection, request.app.state.chunks, body.k)
    if not hits:
        return StreamingResponse(iter([rag.NO_ANSWER]), media_type="text/plain")
    prompt = f"Context:\n{rag.build_context(hits)}\n\nQuestion: {body.question}"
    return StreamingResponse(stream_answer(prompt), media_type="text/plain")
```
In plain English:
- `stream=True` makes the API send many small events instead of one finished reply. Each one carries a few new characters in `delta.content`.
- `yield` turns `stream_answer` into a **generator** (the "ticket machine" from simple_rag): FastAPI pulls one piece at a time and sends it straight to the browser.
- `StreamingResponse` keeps the HTTP connection open while pieces arrive. Browsers usually consume this as **server-sent events** (`media_type="text/event-stream"` with `data:` lines).
- Retrieval still runs first (~380 ms), so the first token can't arrive sooner than that.

**Async:** declaring endpoints with `async def` and using the async OpenAI client (`AsyncOpenAI`) lets one server process wait on many slow LLM calls at the same time, instead of tying up a worker thread per request. Plain `def` endpoints (like this project's) run in a thread pool, which is fine at low traffic.

---

## 3. Goal

1. Add a streaming endpoint next to `/ask` and keep `/ask` for clients that want one JSON reply.
2. Save the full streamed text to the Redis cache when the stream finishes. Cache hits can return the whole answer at once.
3. Add timeouts: `OpenAI(..., timeout=30)` so a stuck LLM call can't hang a request forever.

**Check it:**
```bash
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" -d "{\"question\": \"How long do cats sleep?\"}"
```
`-N` turns off curl's buffering, so the text appears piece by piece. Measure TTFT with a small script that records the time of the first received chunk.

---

## 4. Test

```python
def test_stream_returns_the_answer_in_pieces(client, monkeypatch):
    monkeypatch.setattr(api_module, "stream_answer", lambda prompt: iter(["Cats ", "sleep ", "a lot."]))
    response = client.post("/ask/stream", json={"question": "How long do cats sleep?"})
    assert response.status_code == 200
    assert response.text == "Cats sleep a lot."
```
This checks the endpoint forwards every streamed piece in order. The LLM stream is faked, as in the other API tests, so no real call is made.

---

## 5. Trade-offs

- **Harder to post-process:** you can't check or fix the whole answer before the user sees it (for example, verifying the citations). Do those checks after the stream, or don't stream.
- **Proxies and load balancers** sometimes buffer responses, which breaks streaming. Test it through your real deployment path.
- **Not worth it for cached answers** (~1–4 ms), or for machine clients that need the full JSON anyway.

---

**Previous:** [← Optimization 5: Faster models](opt-05-faster-models.md)

**Next:** [Optimization 7: Cutting LLM cost and latency →](opt-07-cutting-llm-cost-and-latency.md)
