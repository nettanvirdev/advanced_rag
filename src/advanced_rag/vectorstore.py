from pathlib import Path

import chromadb

from advanced_rag import config
from advanced_rag.chunking import Chunk, chunk_documents
from advanced_rag.embeddings import embed
from advanced_rag.loaders import load_folder

COLLECTION = "chunks"


def get_collection(path: Path = config.CHROMA_DIR):
    # a database that lives in a folder on disk, so it survives restarts
    client = chromadb.PersistentClient(path=str(path))
    # cosine = compare directions, like simple_rag. We bring our own embeddings,
    # so Chroma doesn't need an embedding model of its own.
    return client.get_or_create_collection(
        name=COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source}#{chunk.index}"  # e.g. "cat.txt#12": unique, and readable in answers


def add_chunks(collection, chunks: list[Chunk]) -> None:
    # upsert = insert, or overwrite if the id already exists, so running it twice never duplicates
    # ponytail: one call for everything; Chroma caps a single batch (~5k items), so batch it when the corpus grows
    collection.upsert(
        ids=[chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )


def search(collection, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
    # only the question is embedded now; the chunks were embedded once, when they were added
    result = collection.query(query_embeddings=embed([query]), n_results=k)

    # Chroma can answer many questions at once, so every field is a list per question; we asked one: [0]
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        chunk = Chunk(text=text, source=meta["source"], index=meta["index"])
        hits.append((1 - distance, chunk))  # cosine distance 0 = same direction = similarity 1
    return hits


def all_chunks(collection) -> list[Chunk]:
    # every stored chunk, without vectors: BM25 needs the texts, and reading them from
    # the index guarantees both searches look at exactly the same chunks
    result = collection.get(include=["documents", "metadatas"])
    return [
        Chunk(text=text, source=meta["source"], index=meta["index"])
        for text, meta in zip(result["documents"], result["metadatas"])
    ]


def build_index(path: Path = config.CHROMA_DIR):
    # start from an empty collection, so chunks of deleted or shortened files don't linger
    client = chromadb.PersistentClient(path=str(path))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)

    collection = get_collection(path)
    add_chunks(collection, chunk_documents(load_folder()))
    return collection


if __name__ == "__main__":
    import sys
    import time

    if "--build" in sys.argv:
        start = time.perf_counter()
        collection = build_index()
        print(f"indexed {collection.count()} chunks in {time.perf_counter() - start:.1f}s")
        # new chunks can change answers, so cached answers from the old index are no longer trustworthy
        from advanced_rag import cache

        print(f"cleared {cache.clear()} cached answers")
    else:
        collection = get_collection()
        print(f"{collection.count()} chunks in the index (run with --build to rebuild it)")

    while True:
        query = input("search> ")
        start = time.perf_counter()
        hits = search(collection, query)
        print(f"  ({(time.perf_counter() - start) * 1000:.0f} ms)")
        for score, chunk in hits:
            print(f"  {score:.3f}  {chunk_id(chunk):24} {chunk.text[:90]}...")
