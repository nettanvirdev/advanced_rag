---
title: "Step 2: Embeddings and dimensions"
series: "Advanced RAG"
group: "steps"
step: 2
slug: "step-02-embeddings-and-dimensions"
date: "2026-10-21"
tag: "RAG"
excerpt: "What the 384 numbers in an embedding are, what bigger dimensions cost, and the 256-token limit that silently throws text away."
minutes: 15
---

# Step 2: Embeddings and dimensions

**You'll build:** `src/advanced_rag/embeddings.py` with `embed()`, `dimension()` and `count_tokens()`.
**You'll learn:** what a dimension is, storage cost per dimension, tokens vs. characters, the maximum input length, normalized vectors, and why embedding in batches is faster.

---

## 1. Theory

An embedding's **dimension** is how many numbers are in its vector. Each number is a direction the model learned; more of them leave more room to tell subtle meanings apart, but every extra dimension costs storage, memory and search time. The limit people forget is the second one: a **maximum input length**, measured in **tokens**. Text past it is silently ignored. ([Theory: Embeddings and dimensions](concepts.md#embeddings-and-dimensions))

### The model used in this guide: `all-MiniLM-L6-v2`
| Property | Value |
|---|---|
| Dimensions | **384** |
| Max input | **256 tokens**: anything longer is cut off |
| Speed | 200 sentences in **0.11 s** on CPU |
| Vector length | already **1.0** (normalized) |

### Common models compared
| Model | Dims | Size | Max tokens | Notes |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | ~90 MB | 256 | fast, good enough for most uses |
| all-mpnet-base-v2 | 768 | ~420 MB | 384 | better quality, about 5× slower |
| OpenAI text-embedding-3-small | 1536 | API | 8191 | paid, no local compute |
| OpenAI text-embedding-3-large | 3072 | API | 8191 | best quality, most storage |

### Storage cost (each number is a 4-byte float)
| Dims | Per chunk | 1 million chunks |
|---|---|---|
| 384 | 1.5 KB | **1.5 GB** |
| 768 | 3 KB | 3 GB |
| 1536 | 6 KB | 6 GB |
| 3072 | 12 KB | 12 GB |

Bigger isn't automatically better. Choose with an **evaluation** (Step 8), not by the size of the number.

---

## Before vs after (measured)

| | Before (simple_rag style) | After this step |
|---|---|---|
| How much of `cat.txt` one whole-file embedding actually sees | assumed: all of it | **measured: 2.5%** (256 of 10,352 tokens) |
| Same for coffee / python_language / python_snake | assumed: all | 2.1% / 2.9% / 10.5% |
| Embedding 210 chunks | one call per chunk: **2.82 s** | one batch call: **1.48 s** (1.9× faster) |
| `count_tokens("Kittens sleep a lot.")` | counting characters: 20 (wrong) | counting tokens: **6** |

The first row is the reason Steps 3–4 exist: one embedding per file throws away more than 95% of each article.

---

## 2. Code example

Try it in `uv run python` first:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model.get_embedding_dimension()                     # 384
model.max_seq_length                                # 256
model.tokenizer.tokenize("Kittens sleep a lot.")    # the token pieces
model.encode(["cat", "kitten"], normalize_embeddings=True)   # a list in → one vector per text out
```
Use `get_embedding_dimension()`. The older name, `get_sentence_embedding_dimension()`, prints a "renamed" warning.

### 2.1 Tokens, not words
The model doesn't read words. It reads **tokens**: words or pieces of words.
```
"Kittens sleep a lot."  →  ['kitten', '##s', 'sleep', 'a', 'lot', '.']   = 6 tokens
```
`##s` means "this continues the previous token". The same sentence is 20 **characters**. Counting characters instead of tokens is a classic bug: limits would come out about 4× too small.

### 2.2 The silent cut-off, proven
A 320-token text, embedded once on its own and once with 20 extra sentences about kittens added at the end:
```
similarity(text, text + kitten sentences) = 1.0
```
Identical. The model never saw the kitten part.

### 2.3 Normalized vectors: cosine becomes a dot product
A **normalized** vector has length 1. For normalized vectors, cosine similarity is just "multiply matching numbers and add them up" (the **dot product**), which is much faster. That's why vector databases like them.
```python
a, b = model.encode(["cat", "kitten"], normalize_embeddings=True)
a @ b      # 0.7882 ← @ means dot product; the same number as cos_sim in simple_rag
```

### 2.4 Why `embed()` takes a list
```python
[model.encode([text]) for text in texts]   # 210 separate calls: 2.82 s
model.encode(texts)                        # 1 call with all 210:  1.48 s
```
The model processes a batch in one go, so one call with many texts is much faster than many calls with one text each.

---

## 3. Goal

```bash
uv add sentence-transformers
```
It downloads PyTorch. The model itself (~90 MB) downloads on first use and is cached after that.

Create `src/advanced_rag/embeddings.py`. Load the model **once** at the top with `config.EMBEDDING_MODEL`, then write:

| Function | Returns | Hint |
|---|---|---|
| `embed(texts: list[str]) -> list[list[float]]` | one vector per text | `model.encode(texts, normalize_embeddings=True)`, then `.tolist()` |
| `dimension() -> int` | 384 | `model.get_embedding_dimension()` |
| `count_tokens(text: str) -> int` | the number of **tokens** | `len(...)` of `model.tokenizer.tokenize(text)` |

**Why plain `list[list[float]]`?** Plain Python lists are what ChromaDB (Step 5) and API providers use. Keeping this exact shape means swapping to an API model later only changes the inside of `embed()`.

### ✅ Checkpoint
```bash
uv run python -c "from advanced_rag.embeddings import embed, dimension, count_tokens; v = embed(['cat', 'kitten']); print(dimension(), len(v), len(v[0]), count_tokens('Kittens sleep a lot.'))"
```
Expected: `384 2 384 6`

### 🧪 Experiment: see the cut-off
```bash
uv run python -c "from advanced_rag.embeddings import embed, count_tokens; head = 'Coffee is brewed from roasted beans. ' * 40; tail = ' Kittens sleep twenty hours a day.' * 20; a, b = embed([head, head + tail]); print(count_tokens(head), round(sum(x * y for x, y in zip(a, b)), 4))"
```
Expected: `320 1.0`. Change `* 40` to `* 5`: now the kitten part fits inside the 256-token window, and the similarity drops below 1. You can ignore the warnings about symlinks and HF_TOKEN.

---

## 4. Test

**This test checks:** one 384-number vector per text. **How:** three texts in, three vectors of length 384 out.
```python
from pytest import approx

from advanced_rag.embeddings import embed, dimension, count_tokens


def test_embed_returns_one_vector_per_text():
    vectors = embed(["cat", "kitten", "coffee"])
    assert len(vectors) == 3
    assert all(len(v) == dimension() == 384 for v in vectors)
```

**This test checks:** every vector is normalized. **How:** the squares of its numbers must add up to 1.
```python
def test_vectors_are_normalized():
    for v in embed(["cat", "a much longer sentence about coffee beans"]):
        assert sum(x * x for x in v) == approx(1.0, abs=1e-4)
```

**This test checks:** the dot product of normalized vectors is cosine similarity. **How:** cat · kitten must equal 0.788, the cosine from simple_rag.
```python
def test_dot_product_is_cosine_similarity():
    cat, kitten = embed(["cat", "kitten"])
    assert sum(a * b for a, b in zip(cat, kitten)) == approx(0.788, abs=0.001)
```

**This test checks:** tokens are counted, not characters. **How:** the sentence has 6 tokens but 20 characters, so `len(text)` fails this test.
```python
def test_count_tokens_counts_tokens_not_characters():
    # 'kitten', '##s', 'sleep', 'a', 'lot', '.'
    assert count_tokens("Kittens sleep a lot.") == 6
```

**This test checks:** text past 256 tokens is ignored. **How:** a 320-token text with or without extra text at the end gives the same embedding. This test documents the limit, so nobody forgets it.
```python
def test_text_past_256_tokens_is_ignored():
    head = "Coffee is brewed from roasted beans. " * 40
    tail = " Kittens sleep twenty hours a day." * 20
    a, b = embed([head, head + tail])
    assert sum(x * y for x, y in zip(a, b)) == approx(1.0, abs=1e-4)
```

```bash
uv run pytest -v
```
Expected: **17 passed**.

---

## 5. Make it a module

- **Steps 3–4 (chunking)** use `count_tokens` to keep every chunk under 200 tokens, so nothing gets cut off.
- **Step 5 (ChromaDB)** stores `embed()`'s output, all chunks in one batch call.
- Swapping to an API embedding model only changes the inside of `embed()`; its signature stays the same.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/embeddings.py</code></strong></summary>

```python
from sentence_transformers import SentenceTransformer
from advanced_rag import config


model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


def dimension() -> int | None:
    return model.get_embedding_dimension()


def count_tokens(text: str) -> int | None:
    return len(model.tokenizer.tokenize(text))
```

</details>

---

**Previous:** [← Step 1: Project setup](step-01-project-setup.md)

**Next:** [Step 3: Chunking: split and pack →](step-03-chunking-split-and-pack.md)
