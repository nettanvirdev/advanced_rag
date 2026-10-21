---
title: "Step 1: Project setup"
series: "Advanced RAG"
group: "steps"
step: 1
slug: "step-01-project-setup"
date: "2026-10-21"
tag: "RAG"
excerpt: "Start the advanced_rag project with uv, reuse the working config and loaders from simple_rag, and lay out the production folder structure the whole guide grows into."
minutes: 10
---

# Step 1: Project setup

**You'll build:** a new uv project `advanced_rag`, with `config.py` and `loaders.py` copied from simple_rag and their tests passing.
**You'll learn:** reusing code across projects, renaming imports, the production folder layout of an advanced RAG system, and what to keep out of git.

---

## 1. Theory

A new project, with the same production layout as simple_rag. Code that already works (settings, file loading) is **copied** instead of imported from the old project. That keeps `advanced_rag` self-contained and lets the copied code evolve: chunks will soon need fields that `Document` doesn't have. When code moves into a new package, its imports must be **renamed**, because the old package name doesn't exist here. Copying the tests along with the code is the fastest proof that nothing broke on the way over.

---

## Before vs after (measured)

| | Before | After |
|---|---|---|
| Project | empty folder | uv package `advanced_rag`, Python 3.12 |
| Working code | none | `config.py`, `loaders.py` (copied, imports renamed) |
| Documents loaded | 0 | 4 (`cat.txt`, `coffee.txt`, `python_language.txt`, `python_snake.txt`) |
| Tests | 0 | **12 passed** (8 config + 4 loaders) |

---

## 2. Code example

### 2.1 The folder structure you're building toward
This is the **final** project. Files appear when their step arrives, so don't create them all now.
```
advanced_rag/
├── pyproject.toml, uv.lock, .python-version
├── .env / .env.example / .gitignore / README.md / LICENSE
├── data/                     4 long Wikipedia articles (.txt) + SOURCES.md
├── storage/                  ChromaDB files, git-ignored                (Step 5)
├── eval/
│   ├── questions.json        the golden question set                    (Step 8)
│   └── results.md            the measured scores                        (Step 8)
├── src/advanced_rag/
│   ├── config.py             settings, read once from .env              ← today
│   ├── loaders.py            files → Document(text, source)            ← today
│   ├── embeddings.py         text → vectors, token counting             (Step 2)
│   ├── chunking.py           long text → Chunk(text, source, index)     (Steps 3–4)
│   ├── vectorstore.py        ChromaDB: store once, search fast          (Step 5)
│   ├── retrieval/
│   │   ├── keyword.py, tfidf.py, bm25.py   copied from simple_rag       (Step 6)
│   │   ├── hybrid.py         BM25 + semantic, fused with RRF            (Step 6)
│   │   ├── rerank.py         cross-encoder reranking                    (Step 9)
│   │   └── rewrite.py        LLM query rewriting (off by default)       (Step 10)
│   ├── llm.py                ask(): any OpenAI-compatible API           (Step 7)
│   ├── cache.py              Redis answer cache                         (Step 7)
│   ├── rag.py                retrieve → prompt → answer                 (Step 7)
│   ├── evaluate.py           hit rate, MRR, answer checks               (Step 8)
│   └── api.py                FastAPI server                             (Step 11)
└── tests/                    one test_<module>.py per module + conftest.py
```
The big picture of how these pieces connect is drawn in [Diagrams](diagrams.md).

### 2.2 Create the project
```bash
uv init --package --name advanced-rag --python 3.12
uv add python-dotenv
uv add --dev pytest
```

| Command | What it does |
|---|---|
| `--name advanced-rag` | the package folder becomes `src/advanced_rag/`: Python names can't contain `-`, so uv turns it into `_` |
| `--python 3.12` | machine-learning libraries (torch, chromadb) support 3.12 most reliably |
| `uv add --dev pytest` | the test framework, as a dev-only tool |

### 2.3 Copying code means renaming imports
In simple_rag the package was called `ragi`. Here it's `advanced_rag`, so every import of the old name must change:
```python
from ragi import config               # ❌ ModuleNotFoundError: No module named 'ragi'
from advanced_rag import config       # ✅
```
`config.py` itself imports nothing from `ragi`, so it copies over unchanged. `loaders.py` and the two test files each need one line renamed.

### 2.4 What stays out of git
```
.env            ← your API key and real settings
storage/        ← the vector database: generated, rebuildable (Step 5)
learning.md     ← private notes, if you keep any
```
Add these to `.gitignore`. Commit `.env.example` (same keys, no values) instead of `.env`.

---

## 3. Goal

Run these from your new, empty project folder, then copy the files below from simple_rag.

| From simple_rag | To | Then change |
|---|---|---|
| `.env`, `.env.example` | project root | nothing |
| `src/ragi/config.py` | `src/advanced_rag/config.py` | nothing |
| `src/ragi/loaders.py` | `src/advanced_rag/loaders.py` | `from ragi import config` → `from advanced_rag import config` |
| `tests/test_config.py`, `tests/test_loaders.py` | `tests/` | `from ragi…` → `from advanced_rag…` |

For data, use the four long Wikipedia articles from this repo's `data/` folder (`cat.txt`, `coffee.txt`, `python_language.txt`, `python_snake.txt`). `data/SOURCES.md` credits them (CC BY-SA 4.0). They are long on purpose: Step 3 needs text worth cutting into chunks.

`test_config.py` also checks that a `storage/chroma` folder exists, so create it: `mkdir -p storage/chroma`.

### ✅ Checkpoint
```bash
uv run python -c "from advanced_rag import config; from advanced_rag.loaders import load_folder; print(config.DATA_DIR.exists(), len(load_folder()))"
```
Expected: `True 4`

```bash
uv run pytest -v
```
Expected: **12 passed**. A `ModuleNotFoundError: No module named 'ragi'` means an import wasn't renamed.

---

## 4. Test

The tests came over with the code. If they pass in the new project, the copy worked.

**This test checks:** the project root really is the project root. **How:** the root must contain `pyproject.toml`.
```python
from advanced_rag import config


def test_root_has_pyproject():
    assert (config.ROOT / "pyproject.toml").exists()
```

**This test checks:** only `.txt` files are loaded. **How:** a folder with `a.txt` and `notes.md` must load `a.txt` only. Without `a.txt` the test could never fail, because an empty list never contains anything.
```python
from advanced_rag.loaders import load_txt, load_folder


def test_load_folder_ignores_non_txt(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "notes.md").write_text("test", encoding="utf-8")
    sources = [doc.source for doc in load_folder(tmp_path)]
    assert "notes.md" not in sources
```

**This test checks:** results come out in alphabetical order. **How:** `b.txt` is written before `a.txt`, so only real sorting puts `a.txt` first.
```python
def test_load_folder_is_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("test", encoding="utf-8")
    (tmp_path / "a.txt").write_text("test", encoding="utf-8")
    assert load_folder(tmp_path)[0].source == "a.txt"
```

The other tests check that `load_txt` reads text, empty files are skipped, and every `.env` value is set.

---

## 5. Make it a module

Every later step starts with:
```python
from advanced_rag import config
from advanced_rag.loaders import Document, load_folder
```
Step 2 reads `config.EMBEDDING_MODEL` to load the embedding model. In Step 6 you'll copy three more files from simple_rag (keyword, TF-IDF, BM25) the same way: copy, rename imports, run their tests.

---

<details>
<summary><strong>Show solution: <code>src/advanced_rag/loaders.py</code></strong></summary>

```python
from dataclasses import dataclass
from pathlib import Path
from advanced_rag import config


@dataclass
class Document:
    text: str
    source: str


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(text=text, source=path.name)


def load_folder(folder: Path = config.DATA_DIR) -> list[Document]:

    docs = []
    for path in sorted(folder.glob("*.txt")):
        doc = load_txt(path=path)
        if doc.text.strip() == "":
            continue
        docs.append(doc)
    return docs
```

`config.py` is unchanged from simple_rag:

```python
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")


CHROMA_DIR = ROOT / os.getenv("CHROMA_DIR", "storage/chroma")
REDIS_URL = os.getenv("REDIS_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
```

</details>

---

**Previous:** [← Advanced RAG overview](README.md)

**Next:** [Step 2: Embeddings and dimensions →](step-02-embeddings-and-dimensions.md)
