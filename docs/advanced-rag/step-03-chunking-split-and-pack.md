---
title: "Step 3: Chunking: split and pack"
series: "Advanced RAG"
group: "steps"
step: 3
slug: "step-03-chunking-split-and-pack"
date: "2026-10-21"
tag: "RAG"
excerpt: "Cut long articles into chunks that fit the model: split into sentences, then pack them into boxes of at most 200 tokens."
minutes: 15
---

# Step 3: Chunking: split and pack

**You'll build:** `split_sentences()` and `chunk_text()` in `src/advanced_rag/chunking.py`.
**You'll learn:** why chunking exists, splitting sentences with a regex lookbehind, and the "fill a box, shelve it, start a new one" loop.

---

## 1. Theory

Chunking cuts a long document into pieces that each fit the model, under its 256-token limit. Each piece gets its own embedding, so search can find the **exact part** that answers a question. Pieces that are too big get cut off and blur several topics together; pieces that are too small lose context ("It is a social species": what is?). ([Theory: Chunking](concepts.md#chunking))

### Why: what one embedding of a whole file sees
| File | Tokens | Sentences | One whole-file embedding sees |
|---|---|---|---|
| cat.txt | 10,352 | 455 | the first 256 tokens = **2.5%** |
| coffee.txt | 12,236 | 602 | 2.1% |
| python_language.txt | 8,718 | 431 | 2.9% |
| python_snake.txt | 2,433 | 112 | 10.5% |

### The strategy: pack whole sentences
Never cut in the middle of a sentence. Split the text into sentences, then **pack** them into a chunk until adding the next one would go over the limit:
```
sentences:  [S1 60 tok] [S2 70 tok] [S3 50 tok] [S4 40 tok] [S5 90 tok] ...
max 200:    chunk 1 = S1+S2+S3 (180)   ← adding S4 would make 220, so start a new chunk
            chunk 2 = S4+S5 ...
```

---

## Before vs after (measured)

| | Before (one embedding per file) | After (chunks) |
|---|---|---|
| Share of each article the model actually reads | 2.1% – 10.5% | **100%** |
| Vectors for the 4 articles | 4 | **185** (cat 56, coffee 67, python_language 48, python_snake 14) |
| Longest single input to the model | 12,236 tokens (cut to 256) | **200 tokens** (fits) |

---

## 2. Code example

### 2.1 Splitting into sentences
The regex part is given:
```python
import re

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)     # split after . ! ? + spaces, OR at line breaks
    return [p.strip() for p in parts if p.strip()]     # drop empty pieces and trim the edges
```
| Piece | Meaning |
|---|---|
| `re.split(pattern, text)` | cuts the text wherever the pattern matches |
| `(?<=[.!?])` | a **lookbehind**: "right after a `.`, `!` or `?`", which keeps the punctuation in the sentence |
| `\s+` | the spaces after it, which get removed |
| `\|\n+` | **or** one or more line breaks, so section headings become their own pieces |

```python
split_sentences("Hello there. How are you? I'm fine!\nHistory\nCats are great.")
# ['Hello there.', 'How are you?', "I'm fine!", 'History', 'Cats are great.']
```

### 2.2 Packing: the box and the shelf
Think of it as **filling boxes**. `current` is the box being filled; `chunks` is the shelf. When the next sentence won't fit, seal the box, put it on the shelf, and start a new box with that sentence.

```python
def chunk_text(text: str, max_tokens: int = 200) -> list[str]:
    sentences = split_sentences(text)
    chunks = []                       # the shelf
    current = []                      # the box being filled

    for sentence in sentences:
        candidate = " ".join(current + [sentence])            # PRETEND to add it: what would the box hold?

        if current and count_tokens(candidate) > max_tokens:  # box not empty AND it would overflow?
            chunks.append(" ".join(current))                  # seal the box, put it on the shelf
            current = []                                      # grab an empty box

        current.append(sentence)                              # the sentence ALWAYS goes in

    if current:                                               # leftovers after the loop
        chunks.append(" ".join(current))                     # shelve the last box too

    return chunks
```

**Line by line, in plain English:**
| Line | Meaning |
|---|---|
| `candidate = " ".join(current + [sentence])` | "Pretend to add this sentence and see what the box would hold." Nothing is added yet; it's a test fit. |
| `if current and count_tokens(candidate) > max_tokens:` | "Is the box holding something, **and** would this sentence overflow it?" Both must be true. An empty list counts as False. |
| `chunks.append(" ".join(current))` | "Seal the box: glue its sentences into one text and shelve it." It's `current`, not `candidate`: the new sentence is **not** in this chunk. |
| `current = []` | "Grab a new empty box." |
| `current.append(sentence)` | Outside the `if`, so it **always** runs: the sentence joins the box, or starts the fresh one. No sentence is ever skipped. |
| `if current: chunks.append(...)` | "The last box is probably half full and never overflowed, so nobody sealed it. Shelve it." Forgetting this loses the **end of every article**: the most common bug in this loop. |

**Why the `current and` part?** Picture one sentence of 300 tokens, bigger than the limit on its own. With the check, the empty box skips the overflow test, the sentence goes in, and it becomes its own (oversized) chunk. Without it, the code would seal an **empty box** and shelve a blank chunk. None of the four articles has such a sentence.

### 2.3 Watch it run
`chunk_text("One. Two. Three.", max_tokens=4)`. Each sentence is 2 tokens (`one` + `.`):

| Round | sentence | box before | candidate (tokens) | Overflow? | What happens | box after | shelf |
|---|---|---|---|---|---|---|---|
| 1 | `One.` | `[]` | `One.` (2) | box empty, no check | add | `[One.]` | `[]` |
| 2 | `Two.` | `[One.]` | `One. Two.` (4) | 4 > 4? **no** | add | `[One., Two.]` | `[]` |
| 3 | `Three.` | `[One., Two.]` | `One. Two. Three.` (6) | 6 > 4? **yes** | seal, then add | `[Three.]` | `["One. Two."]` |
| after loop | | `[Three.]` | | | leftovers shelved | | `["One. Two.", "Three."]` |

**The whole thing in one sentence:** keep adding sentences to a box; when the next one won't fit, shelve the box and start a new one with that sentence; at the end, shelve the last box.

**Why 200 and not 256?** A safety margin, and Step 4's overlap adds a little extra.

---

## 3. Goal

Create `src/advanced_rag/chunking.py` with `split_sentences` (given above) and `chunk_text(text, max_tokens=200)`. Import `count_tokens` from `advanced_rag.embeddings`.

Steps:
1. `sentences = split_sentences(text)`, `chunks = []`, `current = []`
2. For each `sentence`:
   - `candidate` = the box's sentences plus this one, joined with spaces
   - **if** `current` isn't empty **and** `count_tokens(candidate) > max_tokens`: append `" ".join(current)` to `chunks`, then reset `current = []`
   - append `sentence` to `current`
3. After the loop: if `current` still has sentences, append them as the last chunk.
4. Return `chunks`.

### ✅ Checkpoint
```bash
uv run python -c "from advanced_rag.loaders import load_folder; from advanced_rag.chunking import chunk_text; from advanced_rag.embeddings import count_tokens; docs = load_folder(); [print(d.source, len(c := chunk_text(d.text)), max(count_tokens(x) for x in c)) for d in docs]"
```
Expected:
```
cat.txt 56 200
coffee.txt 67 200
python_language.txt 48 200
python_snake.txt 14 199
```
185 chunks in total, none over 200 tokens. Print one to see what a chunk looks like: `print(chunk_text(docs[0].text)[0])`.

---

## 4. Test

**This test checks:** sentences split on `.` `!` `?` and on line breaks, keeping the punctuation. **How:** a heading on its own line must come out as its own piece.
```python
from advanced_rag.chunking import split_sentences, chunk_text
from advanced_rag.embeddings import count_tokens
from advanced_rag.loaders import load_folder


def test_split_sentences_keeps_punctuation_and_splits_lines():
    text = "Hello there. How are you? I'm fine!\nHistory\nCats are great."
    assert split_sentences(text) == ["Hello there.", "How are you?", "I'm fine!", "History", "Cats are great."]
```

**This test checks:** a tiny limit puts one sentence in each chunk. **How:** 3 tokens fits one 2-token sentence, never two.
```python
def test_tiny_limit_gives_one_sentence_per_chunk():
    assert chunk_text("One. Two. Three.", max_tokens=3) == ["One.", "Two.", "Three."]
```

**This test checks:** sentences are packed until the limit. **How:** the traced example above, round by round.
```python
def test_sentences_are_packed_until_the_limit():
    # "One. Two." is 4 tokens, so it fits in 4; adding "Three." would not
    assert chunk_text("One. Two. Three.", max_tokens=4) == ["One. Two.", "Three."]
```

**This test checks:** no chunk is over the limit on the real articles. **How:** every chunk of all 4 files is measured with `count_tokens`.
```python
def test_no_chunk_is_over_the_limit_on_real_articles():
    for doc in load_folder():
        assert max(count_tokens(c) for c in chunk_text(doc.text)) <= 200, doc.source
```

**This test checks:** nothing is lost, including the leftovers after the loop. **How:** joining all the chunks must give back every sentence, in order.
```python
def test_nothing_is_lost():
    for doc in load_folder():
        assert " ".join(chunk_text(doc.text)) == " ".join(split_sentences(doc.text)), doc.source
```

```bash
uv run pytest -v
```
Expected: **22 passed**.

---

## 5. Make it a module

Step 4 adds **overlap** (the last sentence of a chunk repeats at the start of the next) and a `Chunk(text, source, index)` record, so every chunk remembers which file it came from. Step 5 embeds those chunks and stores them in ChromaDB.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/chunking.py</code></strong></summary>

```python
import re

from advanced_rag.embeddings import count_tokens


def split_sentences(text: str) -> list[str]:
    # split right after . ! ? (keeping them), or at line breaks (so headings stand alone)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = 200) -> list[str]:
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
            current = []

        current.append(sentence)

    # the leftovers after the loop are the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks
```

</details>

---

**Previous:** [← Step 2: Embeddings and dimensions](step-02-embeddings-and-dimensions.md)

**Next:** [Step 4: Chunking: overlap and records →](step-04-chunking-overlap-and-records.md)
