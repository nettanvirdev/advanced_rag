import logging
import secrets
import time
from contextlib import asynccontextmanager
from itertools import chain
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException, Query, Request, Response, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from advanced_rag import cache, config, rag, vectorstore
from advanced_rag.retrieval.bm25 import BM25Index

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("advanced_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts: bring the index in line with data/ (only new or changed text
    # is embedded, so an unchanged restart takes about a second), then precompute BM25.
    # Every request reuses them instead of loading them again.
    collection, changes = vectorstore.build_index()
    if changes:
        cache.clear()  # answers built from the old chunks can't be trusted anymore
    app.state.collection = collection
    app.state.bm25 = BM25Index(vectorstore.all_chunks(collection))
    log.info("ready: %d chunks", len(app.state.bm25.chunks))
    yield


app = FastAPI(
    title="advanced_rag",
    description="Hybrid search + reranking + exact and semantic caches over your .txt files.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    # one log line per request with its total time; the pipeline logs each stage separately
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.0f}"
    log.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(openai.APIError)
async def llm_unavailable(request: Request, error: openai.APIError):
    # timeouts, rate limits after retries, a bad key, the provider being down: a clear 503, not a stack trace
    log.error("llm call failed: %s", error)
    return JSONResponse(status_code=503, content={"detail": "The language model is unavailable. Try again later."})


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)) -> None:
    # off unless API_KEY is set; compare_digest takes the same time for any wrong key (no timing leaks)
    if config.API_KEY and not (key and secrets.compare_digest(key, config.API_KEY)):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header")


# request and response shapes: FastAPI checks incoming JSON against them and documents them at /docs
class Question(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # "   " is not a question

    question: str = Field(min_length=1, max_length=1000, examples=["How long do cats sleep?"])
    k: int = Field(default=3, ge=1, le=10)
    sources: list[str] | None = Field(default=None, max_length=50, examples=[None], description="Only search these files, e.g. [\"cat.txt\"]")


class Answer(BaseModel):
    answer: str
    cached: Literal["exact", "semantic"] | None


class Source(BaseModel):
    id: str
    score: float
    text: str


def scope(sources: list[str] | None) -> set[str] | None:
    return set(sources) if sources else None


@app.get("/health")
def health(request: Request, response: Response) -> dict:
    # a readiness signal: with no chunks every answer would be "I don't know", so report unhealthy.
    # Redis being down is reported but not fatal: answers still work, just without the exact cache
    chunks = len(request.app.state.bm25.chunks)
    if chunks == 0:
        response.status_code = 503
    return {"status": "ok" if chunks else "empty index", "chunks": chunks, "redis": cache.ping()}


@app.post("/ask", dependencies=[Security(require_key)])
def ask(body: Question, request: Request) -> Answer:
    reply = rag.answer(body.question, request.app.state.collection, request.app.state.bm25, k=body.k, sources=scope(body.sources))
    return Answer(answer=reply.text, cached=reply.cached)


@app.post("/ask/stream", dependencies=[Security(require_key)])
def ask_stream(body: Question, request: Request) -> StreamingResponse:
    # plain text, sent piece by piece as the LLM writes it (curl -N shows it live)
    pieces = rag.answer_stream(body.question, request.app.state.collection, request.app.state.bm25, k=body.k, sources=scope(body.sources))
    # run retrieval and wait for the first piece HERE, so an LLM outage is still a clean 503
    # instead of a stream that dies after the 200 has been sent
    first = next(pieces, "")
    return StreamingResponse(chain([first], pieces), media_type="text/plain; charset=utf-8")


@app.get("/search", dependencies=[Security(require_key)])
def search(
    request: Request,
    q: str = Query(min_length=1, max_length=1000),
    k: int = Query(default=3, ge=1, le=10),
    source: list[str] | None = Query(default=None, description="Only search these files; repeat for several"),
) -> list[Source]:
    # retrieval only, no LLM: handy to see which chunks an answer would be built from
    hits = rag.retrieve(q, request.app.state.collection, request.app.state.bm25, k, sources=scope(source))
    return [Source(id=vectorstore.chunk_id(chunk), score=score, text=chunk.text) for score, chunk in hits]
