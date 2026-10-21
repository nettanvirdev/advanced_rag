import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from advanced_rag import cache, config, llm, vectorstore
from advanced_rag.chunking import Chunk
from advanced_rag.retrieval.bm25 import BM25Index
from advanced_rag.retrieval.hybrid import hybrid_search, rrf
from advanced_rag.retrieval.rerank import rerank
from advanced_rag.retrieval.rewrite import rewrite

RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know\". "
    "Cite the chunk id in square brackets, like [cat.txt#25]."
)
NO_ANSWER = "I don't know"

log = logging.getLogger("advanced_rag")


@dataclass
class Reply:
    text: str
    cached: str | None = None  # "exact", "semantic", or None for a fresh answer


def build_context(hits: list[tuple[float, Chunk]]) -> str:
    # label every chunk with its id, so the LLM can cite exactly where an answer came from
    return "\n\n".join(f"[{vectorstore.chunk_id(chunk)}]\n{chunk.text}" for _, chunk in hits)


def retrieve(
    question: str, collection, bm25: BM25Index, k: int = 3, rewrite_query: bool = False, sources: set[str] | None = None
) -> list[tuple[float, Chunk]]:
    # hybrid finds rough candidates; the reranker picks the best k and drops the ones that don't answer
    pool = config.RERANK_CANDIDATES
    candidates = hybrid_search(collection, bm25, question, k=pool, candidates=pool, sources=sources)
    if rewrite_query:
        # also search with LLM-rewritten words, and merge both candidate lists.
        # Off by default: on the eval set it didn't find anything new, and costs ~3.7 s per question
        rewritten = hybrid_search(collection, bm25, rewrite(question), k=pool, candidates=pool, sources=sources)
        candidates = rrf([candidates, rewritten], k=pool + pool // 2)
    # always rerank against the ORIGINAL question: that's what the user actually asked
    return rerank(question, candidates, k)


def _prepare(question, collection, bm25, k, use_cache, rewrite_query, sources) -> tuple[Reply | None, str, str]:
    """Everything before the LLM. Returns (a finished reply, "", "") or (None, the prompt, the top chunk id)."""
    start = time.perf_counter()
    # 1. asked before, worded the same? answer from Redis: no search, no LLM call
    # (evaluation turns the caches off, to measure fresh answers every time)
    cached = cache.get(question, sources) if use_cache else None
    if cached is not None:
        log.info("answer cache=exact ms=%.0f", (time.perf_counter() - start) * 1000)
        return Reply(cached, "exact"), "", ""

    # 2. retrieve. Nothing relevant = no point paying for an LLM call
    hits = retrieve(question, collection, bm25, k, rewrite_query, sources)
    retrieve_ms = (time.perf_counter() - start) * 1000
    if not hits:
        log.info("answer cache=none retrieve_ms=%.0f hits=0", retrieve_ms)
        if use_cache:
            cache.put(question, NO_ANSWER, sources)
        return Reply(NO_ANSWER), "", ""

    # 3. asked before in other words? Reuse that answer only if it was built from the SAME best chunk
    # we just found: a similar-sounding question about something else retrieves a different chunk
    top = vectorstore.chunk_id(hits[0][1])
    if use_cache and not sources:
        similar = cache.semantic_get(question)
        if similar is not None and similar[1] == top:
            log.info("answer cache=semantic retrieve_ms=%.0f", retrieve_ms)
            cache.put(question, similar[0], sources)
            return Reply(similar[0], "semantic"), "", ""

    log.info("retrieve_ms=%.0f hits=%d top=%s top_score=%.2f", retrieve_ms, len(hits), top, hits[0][0])
    return None, f"Context:\n{build_context(hits)}\n\nQuestion: {question}", top


def _remember(question: str, result: str, top: str, sources: set[str] | None) -> None:
    cache.put(question, result, sources)
    if not sources:  # a filtered answer only holds inside its filter, so it never goes in the shared semantic cache
        cache.semantic_put(question, result, top)


def answer(
    question: str, collection, bm25: BM25Index, k: int = 3,
    use_cache: bool = True, rewrite_query: bool = False, sources: set[str] | None = None,
) -> Reply:
    ready, prompt, top = _prepare(question, collection, bm25, k, use_cache, rewrite_query, sources)
    if ready is not None:
        return ready

    start = time.perf_counter()
    result = llm.ask(prompt, system=RULES)
    log.info("llm_ms=%.0f", (time.perf_counter() - start) * 1000)
    if not result:
        # the LLM sent back nothing (found by evaluation): answer safely, and don't cache a failure
        return Reply(NO_ANSWER)

    # 4. remember it for next time
    if use_cache:
        _remember(question, result, top, sources)
    return Reply(result)


def answer_stream(
    question: str, collection, bm25: BM25Index, k: int = 3, sources: set[str] | None = None
) -> Iterator[str]:
    # the same pipeline, but the LLM's answer is handed on piece by piece as it's written.
    # Cached answers and "I don't know" arrive whole, in a single piece
    ready, prompt, top = _prepare(question, collection, bm25, k, True, False, sources)
    if ready is not None:
        yield ready.text
        return

    start = time.perf_counter()
    pieces = []
    for piece in llm.stream(prompt, system=RULES):
        if not pieces:
            log.info("llm_first_token_ms=%.0f", (time.perf_counter() - start) * 1000)
        pieces.append(piece)
        yield piece
    result = "".join(pieces)
    log.info("llm_ms=%.0f", (time.perf_counter() - start) * 1000)
    if not result:
        yield NO_ANSWER
        return
    _remember(question, result, top, sources)  # only a complete stream is cached


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(message)s")
    collection = vectorstore.get_collection()
    bm25 = BM25Index(vectorstore.all_chunks(collection))
    print(f"{len(bm25.chunks)} chunks loaded")
    while True:
        question = input("ask> ")
        start = time.perf_counter()
        reply = answer(question, collection, bm25)
        ms = (time.perf_counter() - start) * 1000
        print(f"{reply.text}\n  ({ms:.0f} ms{f', from the {reply.cached} cache' if reply.cached else ''})")
