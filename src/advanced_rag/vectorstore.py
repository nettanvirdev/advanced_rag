import hashlib
import logging
from pathlib import Path

import chromadb

from advanced_rag import config
from advanced_rag.chunking import Chunk, chunk_documents
from advanced_rag.embeddings import embed, embed_query
from advanced_rag.loaders import load_folder

COLLECTION = "chunks"
BATCH = 1000  # embed and write in batches: bounded memory, and under Chroma's max batch size (~5k)

log = logging.getLogger("advanced_rag")


def get_collection(path: Path | None = None, name: str = COLLECTION):
    # a database that lives in a folder on disk, so it survives restarts
    # telemetry off: a service shouldn't phone home
    client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    # cosine = compare directions, like simple_rag. We bring our own embeddings,
    # so Chroma doesn't need an embedding model of its own.
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source}#{chunk.index}"  # e.g. "cat.txt#12": unique, and readable in answers


def content_hash(text: str) -> str:
    # the embedding model is part of the hash: a new model means every stored vector is stale
    return hashlib.sha256(f"{config.EMBEDDING_MODEL}|{text}".encode()).hexdigest()[:32]


def add_chunks(collection, chunks: list[Chunk], reuse: dict[str, list[float]] | None = None) -> None:
    # upsert = insert, or overwrite if the id already exists, so running it twice never duplicates.
    # `reuse` maps content hash -> an embedding we already have, so unchanged text is never re-embedded
    reuse = reuse or {}
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        hashes = [content_hash(c.text) for c in batch]
        missing = [c.text for c, h in zip(batch, hashes) if h not in reuse]
        fresh = iter(embed(missing) if missing else [])  # one model call for everything new in the batch
        collection.upsert(
            ids=[chunk_id(c) for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"source": c.source, "index": c.index, "hash": h} for c, h in zip(batch, hashes)],
            embeddings=[reuse[h] if h in reuse else next(fresh) for h in hashes],
        )


def search(collection, query: str, k: int = 3, sources: set[str] | None = None) -> list[tuple[float, Chunk]]:
    # only the question is embedded now; the chunks were embedded once, when they were added
    where = {"source": {"$in": sorted(sources)}} if sources else None
    result = collection.query(query_embeddings=[list(embed_query(query))], n_results=k, where=where)

    # Chroma can answer many questions at once, so every field is a list per question; we asked one: [0]
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        chunk = Chunk(text=text, source=meta["source"], index=meta["index"])
        hits.append((1 - distance, chunk))  # cosine distance 0 = same direction = similarity 1
    return hits


def all_chunks(collection) -> list[Chunk]:
    # every stored chunk, without vectors: BM25 needs the texts, and reading them from
    # the index guarantees both searches look at exactly the same chunks.
    # Sorted, so the order (and BM25 tie-breaks) never depends on the order of past updates
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        Chunk(text=text, source=meta["source"], index=meta["index"])
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
    return sorted(chunks, key=lambda c: (c.source, c.index))


def build_index(path: Path | None = None, folder: Path | None = None):
    """Bring the index in line with the .txt files: add new chunks, update changed ones, delete removed ones.

    Only chunks whose text changed are embedded; vectors for text we've seen before are reused.
    Returns (collection, number of chunks that changed).
    """
    # ponytail: re-chunks every file on each run; for huge corpora track file mtimes and skip unchanged files
    collection = get_collection(path)
    folder = folder or config.DATA_DIR
    chunks = chunk_documents(load_folder(folder)) if folder.is_dir() else []
    if not chunks and collection.count():
        # a missing or empty mount must not wipe a working index; delete storage/ to really start over
        log.warning("no .txt files in %s: keeping the existing index", folder)
        return collection, 0
    wanted = {chunk_id(c): c for c in chunks}

    stored = collection.get(include=["metadatas"])
    have = {i: m.get("hash") for i, m in zip(stored["ids"], stored["metadatas"])}

    removed = [i for i in have if i not in wanted]
    changed = [c for i, c in wanted.items() if have.get(i) != content_hash(c.text)]

    # a chunk that only moved (same text, new id) keeps its vector: look the hashes up before deleting
    reuse = {}
    hashes = sorted({content_hash(c.text) for c in changed})
    for start in range(0, len(hashes), BATCH):
        found = collection.get(where={"hash": {"$in": hashes[start:start + BATCH]}}, include=["metadatas", "embeddings"])
        reuse.update({m["hash"]: list(e) for m, e in zip(found["metadatas"], found["embeddings"])})

    for start in range(0, len(removed), BATCH):
        collection.delete(ids=removed[start:start + BATCH])
    add_chunks(collection, changed, reuse)

    log.info("index synced: %d chunks, %d changed, %d removed, %d vectors reused",
             len(wanted), len(changed), len(removed), len(reuse))
    return collection, len(changed) + len(removed)


if __name__ == "__main__":
    import sys
    import time

    if "--build" in sys.argv:
        start = time.perf_counter()
        collection, changes = build_index()
        print(f"indexed {collection.count()} chunks ({changes} changed) in {time.perf_counter() - start:.1f}s")
        if changes:
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
