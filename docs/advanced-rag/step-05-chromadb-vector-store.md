---
title: "Step 5: ChromaDB vector store"
series: "Advanced RAG"
group: "steps"
step: 5
slug: "step-05-chromadb-vector-store"
date: "2026-10-21"
tag: "RAG"
excerpt: "Embed every chunk once and save it in ChromaDB on disk, so each question only embeds itself: 1,482 ms per question becomes about 9 ms."
minutes: 15
---

# Step 5: ChromaDB vector store

**You'll build:** `src/advanced_rag/vectorstore.py`: `get_collection()`, `chunk_id()`, `add_chunks()`, `search()`, `build_index()`.
**You'll learn:** what a vector database is, indexing vs. querying, cosine distance, `upsert`, and persistence on disk.

---

## 1. Theory

RAG is really **two pipelines**. **Indexing** runs once: chunk every file, embed every chunk, and store the text, its labels and its vector. **Querying** runs on every question: embed **only the question** and ask the database for the closest chunks. A **vector database** like ChromaDB stores everything on disk, so it survives restarts, and finds nearest vectors with a smart index (HNSW) instead of comparing the question to every chunk one by one. ([Theory: Vector databases](concepts.md#vector-databases))

```
INDEXING (once)      load .txt → chunk → embed all 210 chunks → store in ChromaDB
QUERYING (each time) embed the question → ChromaDB finds the nearest chunks → results
```

---

## Before vs after (measured)

Before this step, every question meant embedding all 210 chunks again, the way `simple_rag` worked.

| | Before (re-embed on every question) | After (ChromaDB) |
|---|---|---|
| What gets embedded per question | all 210 chunks + the question | **only the question** |
| Time per question | **1,482 ms** | **~9 ms** (about 50 ms for the first query after opening) |
| One-time indexing | none | 2.7 s for 210 chunks |
| After a restart | embed everything again | reopen the saved index in **13 ms** |
| Adding the same chunks twice | n/a | still 210 rows (`upsert`, no duplicates) |

About **160× faster** per question, and the work you did once is kept.

Real searches on the 4 articles:

| Question | Top hit | Similarity |
|---|---|---|
| How long do cats sleep? | `cat.txt#25`: "Cats conserve energy by sleeping more than most animals…" | 0.745 |
| What is espresso? | `coffee.txt#45`: "As a result of brewing under high pressure (typically 9 bar)…" | 0.633 |
| How do pythons kill their prey? | `python_snake.txt#6` | 0.725 |
| asdfgh | `cat.txt#16` | **0.235** |

The last row matters: nonsense still returns hits, just with a low score. Step 6 adds a minimum score to drop them.

---

## 2. Code example

```bash
uv add chromadb
```

### 2.1 Open (or create) the collection
A **collection** is like a table. It lives in a folder on disk:
```python
import chromadb

client = chromadb.PersistentClient(path=str(path))     # a database in a folder, e.g. storage/chroma
collection = client.get_or_create_collection(
    name="chunks",
    configuration={"hnsw": {"space": "cosine"}},       # compare directions: the cosine similarity you know
    embedding_function=None,                           # we bring our own vectors from embeddings.py
)
```
| Piece | Meaning |
|---|---|
| `PersistentClient(path=...)` | saves everything in that folder, so it's still there after a restart |
| `get_or_create_collection` | opens the table if it exists, creates it if it doesn't |
| `space: cosine` | Chroma returns **cosine distance** = 1 − cosine similarity |
| `embedding_function=None` | "Chroma, don't embed anything yourself." We stay in control of the model. |

### 2.2 Every row needs a unique ID
```python
def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source}#{chunk.index}"     # "cat.txt#25": unique, and readable in answers
```

### 2.3 Add chunks: four lists side by side
```python
collection.upsert(
    ids=[chunk_id(c) for c in chunks],                                  # "cat.txt#0", "cat.txt#1", ...
    documents=[c.text for c in chunks],                                 # the text itself
    metadatas=[{"source": c.source, "index": c.index} for c in chunks], # the labels
    embeddings=embed([c.text for c in chunks]),                         # one batch call for all vectors
)
```
**`upsert`** = **up**date or in**sert**: "if this ID already exists, overwrite it; otherwise add it." Running it twice never creates duplicates.

### 2.4 Search: distance back to similarity
```python
result = collection.query(query_embeddings=embed([query]), n_results=k)
```
Chroma can answer many questions in one call, so every field comes back as **a list per question**:
```python
result["documents"]   # [["chunk text 1", "chunk text 2", "chunk text 3"]]  ← one inner list: one question
result["distances"]   # [[0.255, 0.420, 0.477]]
```
We asked one question, so we read `[0]` of each. Then `1 - distance` turns distance back into similarity: distance 0 means the same direction, which means similarity 1.

---

## 3. Goal

Create `src/advanced_rag/vectorstore.py` with `COLLECTION = "chunks"` and:

1. **`get_collection(path: Path = config.CHROMA_DIR)`**: the code from 2.1, returning the collection.
2. **`chunk_id(chunk) -> str`**: from 2.2.
3. **`add_chunks(collection, chunks: list[Chunk]) -> None`**: the `upsert` from 2.3.
4. **`search(collection, query: str, k: int = 3) -> list[tuple[float, Chunk]]`**:
   - query with `embed([query])` and `n_results=k`
   - loop over `zip(result["documents"][0], result["metadatas"][0], result["distances"][0])`
   - rebuild each `Chunk(text=..., source=meta["source"], index=meta["index"])`
   - append `(1 - distance, chunk)` and return the list
5. **`build_index(path=config.CHROMA_DIR)`**: start fresh, then fill it:
   - open a `PersistentClient`; if `"chunks"` is in `[c.name for c in client.list_collections()]`, call `client.delete_collection("chunks")`, so chunks from deleted or shortened files never linger
   - `collection = get_collection(path)`, then `add_chunks(collection, chunk_documents(load_folder()))`
   - return the collection
6. A `__main__` search box: `--build` rebuilds and prints the time, otherwise it opens the saved index; then loop over `input("search> ")` and print each hit's score, id and the first 90 characters.

### ✅ Checkpoint
```bash
uv run python -m advanced_rag.vectorstore --build
```
Expected first line:
```
indexed 210 chunks in 2.7s
```
Then type `How long do cats sleep?`:
```
  (9 ms)
  0.745  cat.txt#25               Cats conserve energy by sleeping more than most animals, especially as they grow...
```
Run it again **without** `--build`: it says `210 chunks in the index` straight away, because nothing is embedded again.

`storage/` is in `.gitignore`. The index is rebuildable, so it never goes into git.

---

## 4. Test

Every test except the last uses three tiny chunks in pytest's `tmp_path`, so it never touches your real `storage/` folder.
```python
from advanced_rag.chunking import Chunk
from advanced_rag.vectorstore import get_collection, add_chunks, search, chunk_id, build_index

CHUNKS = [
    Chunk(text="Kittens sleep up to twenty hours a day.", source="cat.txt", index=0),
    Chunk(text="Espresso is strong coffee brewed under pressure.", source="coffee.txt", index=0),
    Chunk(text="Python is a programming language created by Guido van Rossum.", source="python.txt", index=0),
]
```

**This test checks:** the ID format. **How:** source, `#`, index.
```python
def test_chunk_id_is_source_and_index():
    assert chunk_id(CHUNKS[0]) == "cat.txt#0"
```

**This test checks:** the collection compares directions. **How:** read its configuration back.
```python
def test_collection_uses_cosine(tmp_path):
    assert get_collection(tmp_path).configuration["hnsw"]["space"] == "cosine"
```

**This test checks:** search returns the whole matching chunk, best first. **How:** "How long do cats sleep?" shares no exact sentence with the kitten chunk, but means the same thing.
```python
def test_search_finds_the_matching_chunk(tmp_path):
    collection = get_collection(tmp_path)
    add_chunks(collection, CHUNKS)
    hits = search(collection, "How long do cats sleep?", k=3)
    assert hits[0][1] == CHUNKS[0]  # the whole Chunk comes back: text, source and index
    assert [s for s, _ in hits] == sorted([s for s, _ in hits], reverse=True)  # best first
```

**This test checks:** `upsert` never duplicates. **How:** add the same 3 chunks twice and count.
```python
def test_adding_twice_does_not_duplicate(tmp_path):
    collection = get_collection(tmp_path)
    add_chunks(collection, CHUNKS)
    add_chunks(collection, CHUNKS)
    assert collection.count() == 3
```

**This test checks:** the data really is on disk. **How:** open the same folder again, like restarting the program.
```python
def test_index_survives_reopening(tmp_path):
    add_chunks(get_collection(tmp_path), CHUNKS)
    reopened = get_collection(tmp_path)  # like restarting the program
    assert reopened.count() == 3
```

**This test checks:** the full real index. **How:** all 4 articles give 210 chunks, and the snake question lands in the snake file.
```python
def test_build_index_on_real_articles(tmp_path):
    collection = build_index(tmp_path)
    assert collection.count() == 210
    assert search(collection, "How do pythons kill their prey?")[0][1].source == "python_snake.txt"
```

ChromaDB prints a harmless deprecation warning about `embedding_function=None`. Silence just that one in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
filterwarnings = ["ignore:legacy embedding function config:DeprecationWarning"]
```
```bash
uv run pytest -v
```
Expected: **30 passed**.

---

## 5. Make it a module

From now on, nothing re-embeds the articles on a question. Step 6 adds `all_chunks(collection)` to read the stored texts back out for BM25, and combines `vectorstore.search` with BM25 into hybrid search. Step 7's `rag.py` opens the collection once and searches it for every question. See the full picture in the [diagrams](diagrams.md).

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/vectorstore.py</code></strong></summary>

```python
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
```

</details>

---

**Previous:** [← Step 4: Chunking: overlap and records](step-04-chunking-overlap-and-records.md)

**Next:** [Step 6: Hybrid retrieval →](step-06-hybrid-retrieval.md)
