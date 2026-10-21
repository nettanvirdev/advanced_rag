from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from advanced_rag import cache, rag, vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts: open the index and read the chunks for BM25.
    # Every request reuses them instead of loading them again.
    app.state.collection = vectorstore.get_collection()
    app.state.chunks = vectorstore.all_chunks(app.state.collection)
    yield


app = FastAPI(title="advanced_rag", description="Hybrid search + reranking + Redis cache over your .txt files.", lifespan=lifespan)


# request and response shapes: FastAPI checks incoming JSON against them and documents them at /docs
class Question(BaseModel):
    question: str = Field(min_length=1, examples=["How long do cats sleep?"])
    k: int = Field(default=3, ge=1, le=10)


class Answer(BaseModel):
    answer: str
    cached: bool


class Source(BaseModel):
    id: str
    score: float
    text: str


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.chunks)}


@app.post("/ask")
def ask(body: Question, request: Request) -> Answer:
    was_cached = cache.get(body.question) is not None
    reply = rag.answer(body.question, request.app.state.collection, request.app.state.chunks, k=body.k)
    return Answer(answer=reply, cached=was_cached)


@app.get("/search")
def search(q: str, request: Request, k: int = 3) -> list[Source]:
    # retrieval only, no LLM: handy to see which chunks an answer would be built from
    hits = rag.retrieve(q, request.app.state.collection, request.app.state.chunks, k)
    return [Source(id=vectorstore.chunk_id(chunk), score=score, text=chunk.text) for score, chunk in hits]
