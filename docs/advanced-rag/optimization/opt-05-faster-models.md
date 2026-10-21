---
title: "Optimization 5: Faster models"
series: "Advanced RAG"
group: "optimization"
step: 5
slug: "opt-05-faster-models"
date: "2026-10-21"
tag: "RAG"
excerpt: "Run the same embedding and reranking models faster with ONNX or OpenVINO backends, int8 quantization, a GPU, or smaller distilled models, and check that quality holds."
minutes: 7
---

# Optimization 5: Faster models

**You'll learn:** four ways to make the embedder and the reranker faster, and why each needs an evaluation run before you trust it.

---

## 1. Theory

The embedder and the cross-encoder are neural networks running in PyTorch on the CPU. You can make them faster by running the **same model in a faster engine** (ONNX Runtime, OpenVINO), by using **lower-precision numbers** (int8 quantization), by using **better hardware** (a GPU), or by switching to a **smaller model**. The first three keep quality almost identical. A smaller model is a quality trade that must be measured.

---

## Before vs after

Not measured in this project. Here's this project's baseline, and what to measure:

| Stage | Baseline (measured, CPU, PyTorch) | After |
|---|---|---|
| Embed 210 chunks (batch) | 1.48 s | measure it: time `build_index()` with the new backend |
| Embed one query | ~7 ms | measure it: time `embed([question])` averaged over 100 calls |
| Rerank 20 candidates | ~150 ms (380 ms retrieve − 228 ms hybrid) | measure it: ms/question of `hybrid+rerank` in `evaluate` |
| Retrieval quality | 95% hit rate, MRR 0.84 | measure it: must stay the same (int8/smaller models can drop it) |

---

## 2. Code example

Docs-only: sentence-transformers can load the same models in a different engine with one argument.
```python
from sentence_transformers import CrossEncoder, SentenceTransformer

# same model, run by ONNX Runtime instead of PyTorch (needs: uv add "sentence-transformers[onnx]")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="onnx")

# Intel CPUs: OpenVINO (needs: uv add "sentence-transformers[openvino]")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="openvino")

# NVIDIA GPU: same code, the model just lives on the GPU
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
```
In plain English:
- `backend="onnx"` exports the network to ONNX, a portable format that ONNX Runtime executes with CPU-specific optimizations. The weights are the same, so the vectors are (nearly) the same.
- `backend="openvino"` does the same with Intel's engine.
- `device="cuda"` moves the work to a GPU. Big batches get dramatically faster. A single short query gains less, because copying data to the GPU has overhead.
- **int8 quantization** stores weights as 8-bit integers instead of 32-bit floats: smaller and faster, with a small accuracy cost. sentence-transformers ships helpers to export quantized ONNX models.
- **Smaller model:** for reranking, `cross-encoder/ms-marco-TinyBERT-L2-v2` has 2 layers vs. our model's 6. It's much faster, but expect lower quality, so measure it.

---

## 3. Goal

1. Change only the model loading line in `embeddings.py` (and/or `retrieval/rerank.py`).
2. Rebuild the index if the **embedding** model or backend changed, because stored vectors must come from the same model as query vectors:
   ```bash
   uv run python -m advanced_rag.vectorstore --build
   ```
3. Run the evaluation and compare speed **and** quality:
   ```bash
   uv run python -m advanced_rag.evaluate
   ```

---

## 4. Test

```python
def test_onnx_embeddings_match_pytorch():
    torch_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    onnx_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx")
    a = torch_model.encode("How long do cats sleep?", normalize_embeddings=True)
    b = onnx_model.encode("How long do cats sleep?", normalize_embeddings=True)
    assert float(a @ b) > 0.999
```
This checks the new engine produces the same vectors (dot product of normalized vectors ≈ 1). If it fails, the backend changed the model's behaviour, not just its speed.

---

## 5. Trade-offs

- **Extra dependencies and export time:** the first ONNX load converts the model. Cache the exported files.
- **Quantization and smaller models can lower quality.** The quality gate test exists for exactly this.
- **Don't optimize the 7 ms step first.** A new question spends ~3–5 s in the LLM. Faster models matter for indexing big corpora and high-traffic retrieval, not for this demo's latency.

---

**Previous:** [← Optimization 4: Tuning the rerank pool](opt-04-tuning-the-rerank-pool.md)

**Next:** [Optimization 6: Streaming and async →](opt-06-streaming-and-async.md)
