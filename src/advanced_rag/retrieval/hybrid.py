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
