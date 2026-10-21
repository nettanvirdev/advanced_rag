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
