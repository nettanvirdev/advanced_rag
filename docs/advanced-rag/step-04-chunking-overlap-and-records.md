---
title: "Step 4: Chunking: overlap and records"
series: "Advanced RAG"
group: "steps"
step: 4
slug: "step-04-chunking-overlap-and-records"
date: "2026-10-21"
tag: "RAG"
excerpt: "Repeat one sentence across chunk borders, guard the size limit, and give every chunk a label: which file it came from and its position."
minutes: 15
---

# Step 4: Chunking: overlap and records

**You'll build:** an `overlap` option for `chunk_text()`, a `Chunk` record, and `chunk_documents()` in `src/advanced_rag/chunking.py`.
**You'll learn:** chunk overlap and its size trap, labelled records, `enumerate`, and why a popular technique still has to be measured.

---

## 1. Theory

**Overlap:** when a box is sealed, its **last sentence is copied** into the next box too, so an idea that sits right on the border between two chunks still appears whole in at least one of them. **Guard:** if the copied sentence plus the next sentence won't fit together, drop the copy. Without it, one coffee chunk grew to 203 tokens, over the limit. **Records:** every chunk remembers **which file** it came from and **its position** in that file, so answers can cite "cat.txt, chunk 12" and the database has a unique ID for each chunk. ([Theory: Chunking](concepts.md#chunking))

### What overlap looks like (real output from cat.txt)
```
chunk 0 ends:    "...such as those made by small mammals. It secretes and perceives pheromones."
chunk 1 starts:  "It secretes and perceives pheromones. Cat intelligence is evident in..."
                  ↑ the same sentence, in both chunks
```

---

## Before vs after (measured)

| | Step 3 (no overlap) | Plain overlap | **Overlap + guard** |
|---|---|---|---|
| Chunks for the 4 articles | 185 | 211 | **210** (+13% storage and embedding time) |
| Biggest chunk | 200 tokens | **203** ❌ over the limit | **200** ✅ |
| Sentences shared across a border | 0 | 1 | 1 |

**Retrieval quality, measured later with the evaluation from Step 8** (hit rate @3 / MRR on 19 answerable questions):

| Search method | Overlap 0 | Overlap 1 |
|---|---|---|
| semantic | 89% / 0.76 | 89% / 0.76 |
| hybrid + rerank (Step 9) | 95% / **0.90** | 95% / 0.84 |

An honest result: on this question set, overlap did **not** improve retrieval (MRR came out slightly lower with it). Overlap protects facts that cross a chunk border, and these 19 questions don't happen to test any. It's the standard default, and this guide keeps it, but the lesson is the important part: **measure on your own data before trusting a "best practice".**

---

## 2. Code example

### 2.1 The two new lines in `chunk_text`
Add an `overlap: int = 1` parameter. Inside the `if` that seals the box, replace `current = []` with:
```python
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []      # keep the last sentence(s) for the next box
            if current and count_tokens(" ".join(current + [sentence])) > max_tokens:
                current = []                                     # the copy doesn't fit with this sentence: drop it
```
| Piece | Meaning |
|---|---|
| `current[-overlap:]` | "the last `overlap` items of the list". With `overlap=1`, the last sentence. `[-1:]` counts from the end. |
| `if overlap else []` | with `overlap=0`, start with an empty box, exactly like Step 3 |
| the second `if` | the guard: "will the copied sentence **and** the new sentence fit in one box? If not, forget the copy." |

### 2.2 Watch overlap run
`chunk_text("One. Two. Three. Four.", max_tokens=4, overlap=1)`. Each sentence is 2 tokens:

| Round | sentence | Fits? | What happens | box after | shelf |
|---|---|---|---|---|---|
| 1 | One. | (empty box) | add | [One.] | |
| 2 | Two. | 4 ≤ 4 ✅ | add | [One., Two.] | |
| 3 | Three. | 6 > 4 ❌ | seal; **copy `Two.`** into the new box; Two.+Three. = 4 fits, keep it; add | [Two., Three.] | "One. Two." |
| 4 | Four. | 6 > 4 ❌ | seal; **copy `Three.`**; Three.+Four. = 4 fits; add | [Three., Four.] | + "Two. Three." |
| after loop | | | shelve the leftovers | | + "Three. Four." |

Result: `["One. Two.", "Two. Three.", "Three. Four."]`. Each chunk shares one sentence with its neighbour.

### 2.3 The `Chunk` record
```python
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str      # the chunk's words
    source: str    # which file, e.g. "cat.txt"
    index: int     # its position in that file: 0, 1, 2...
```
A labelled card: the same idea as `Document`, plus a number. Step 5 turns it into a database ID like `cat.txt#12`.

### 2.4 `enumerate`: the position and the item together
```python
for i, word in enumerate(["a", "b", "c"]):
    print(i, word)       # 0 a / 1 b / 2 c
```

---

## 3. Goal

In `chunking.py`:

**a)** Add `overlap: int = 1` to `chunk_text` and the two lines from 2.1.

**b)** Add the `Chunk` dataclass at the top, after the imports.

**c)** Add `chunk_documents(docs: list[Document], max_tokens: int = 200, overlap: int = 1) -> list[Chunk]`:
1. `chunks = []`
2. For each `doc` in `docs`: for each `i, text` in `enumerate(chunk_text(doc.text, max_tokens, overlap))`, append `Chunk(text=text, source=doc.source, index=i)`.
3. Return `chunks`.

Import `Document` from `advanced_rag.loaders` for the type hint. The numbering restarts at 0 for each file.

### ✅ Checkpoint
```bash
uv run python -c "from advanced_rag.loaders import load_folder; from advanced_rag.chunking import chunk_documents; from advanced_rag.embeddings import count_tokens; c = chunk_documents(load_folder()); print(len(c), max(count_tokens(x.text) for x in c)); print(c[0].source, c[0].index, c[-1].source, c[-1].index)"
```
Expected:
```
210 200
cat.txt 0 python_snake.txt 15
```

---

## 4. Test

Two Step 3 tests assumed no overlap, so they now pass `overlap=0` (`test_sentences_are_packed_until_the_limit`, `test_nothing_is_lost`). New tests:

**This test checks:** overlap repeats the last sentence. **How:** the traced example above.
```python
def test_overlap_repeats_the_last_sentence():
    assert chunk_text("One. Two. Three. Four.", max_tokens=4, overlap=1) == ["One. Two.", "Two. Three.", "Three. Four."]
```

**This test checks:** the guard keeps every chunk within the limit with overlap on. **How:** without the guard, one coffee chunk measures 203 tokens and this fails.
```python
def test_no_chunk_is_over_the_limit_on_real_articles():
    # with overlap on, the guard must still keep every chunk within the limit
    for doc in load_folder():
        assert max(count_tokens(c) for c in chunk_text(doc.text)) <= 200, doc.source
```

**This test checks:** each chunk keeps its file name, and the index counts from 0 **within each file**. **How:** two files; the second one must start again at 0.
```python
from advanced_rag.chunking import chunk_documents
from advanced_rag.loaders import Document


def test_chunk_documents_keeps_source_and_counts_index_per_file():
    docs = [
        Document(text="One. Two. Three.", source="a.txt"),
        Document(text="Four. Five.", source="b.txt"),
    ]
    chunks = chunk_documents(docs, max_tokens=3, overlap=0)
    assert [(c.source, c.index, c.text) for c in chunks] == [
        ("a.txt", 0, "One."),
        ("a.txt", 1, "Two."),
        ("a.txt", 2, "Three."),
        ("b.txt", 0, "Four."),
        ("b.txt", 1, "Five."),
    ]
```

```bash
uv run pytest -v
```
Expected: **24 passed**.

---

## 5. Make it a module

Step 5 does this:
```python
chunks = chunk_documents(load_folder())          # 210 labelled chunks
vectors = embed([c.text for c in chunks])        # one batch call
# stored in ChromaDB with the ID f"{c.source}#{c.index}", e.g. "cat.txt#12"
```
That ID is how an answer later cites exactly where it came from.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/chunking.py</code></strong></summary>

```python
import re
from dataclasses import dataclass

from advanced_rag.embeddings import count_tokens
from advanced_rag.loaders import Document


@dataclass
class Chunk:
    text: str
    source: str  # which file it came from, e.g. "cat.txt"
    index: int  # its position in that file: 0, 1, 2, ...


def split_sentences(text: str) -> list[str]:
    # split right after . ! ? (keeping them), or at line breaks (so headings stand alone)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = 200, overlap: int = 1) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current = []  # sentences in the chunk we are filling right now

    for sentence in sentences:
        # what the chunk would be if we added this sentence
        candidate = " ".join(current + [sentence])

        # too big? close the current chunk and start a new one
        # (`current` must not be empty, or one huge sentence would never be added)
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            # carry the last `overlap` sentences into the new chunk...
            current = current[-overlap:] if overlap else []
            # ...but only if they still fit together with this sentence
            if current and count_tokens(" ".join(current + [sentence])) > max_tokens:
                current = []

        current.append(sentence)

    # the leftovers after the loop are the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(docs: list[Document], max_tokens: int = 200, overlap: int = 1) -> list[Chunk]:
    chunks = []
    for doc in docs:
        # enumerate gives the position (0, 1, 2, ...) together with each chunk's text
        for i, text in enumerate(chunk_text(doc.text, max_tokens, overlap)):
            chunks.append(Chunk(text=text, source=doc.source, index=i))
    return chunks
```

</details>

---

**Previous:** [← Step 3: Chunking: split and pack](step-03-chunking-split-and-pack.md)

**Next:** [Step 5: ChromaDB vector store →](step-05-chromadb-vector-store.md)
