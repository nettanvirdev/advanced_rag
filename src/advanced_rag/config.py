from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
CHROMA_DIR = ROOT / os.getenv("CHROMA_DIR", "storage/chroma")

# ---- LLM (any OpenAI-compatible endpoint) ----
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # seconds; a stuck call can't hang a request forever
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "0")) or None  # 0 = no cap (reasoning models need room to think)

# ---- models ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# ---- retrieval ----
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # measured sweet spot: 20 (see optimization 4)
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.3"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))

# ---- caches ----
REDIS_URL = os.getenv("REDIS_URL", "")  # empty = exact cache off
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 60 * 60)))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85"))  # 0 = semantic cache off

# ---- API ----
API_KEY = os.getenv("API_KEY", "")  # set it to require an X-API-Key header on /ask and /search
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
