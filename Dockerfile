# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# dependencies first: this layer is reused until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# bake both models into the image, so a container starts without downloading anything
RUN python -c "import advanced_rag.embeddings, advanced_rag.retrieval.rerank"
ENV HF_HUB_OFFLINE=1

# the practice articles; mount your own folder over /app/data
COPY data ./data

RUN useradd --create-home --uid 1000 app && mkdir -p storage && chown app /app/storage
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

# one worker: Chroma's on-disk store belongs to one process. FastAPI still serves requests
# concurrently in its thread pool; scale out with more containers, each with its own index volume
CMD ["uvicorn", "advanced_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
