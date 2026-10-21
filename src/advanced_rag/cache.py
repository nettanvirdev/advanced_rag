import hashlib
import logging
import time

import redis

from advanced_rag import config

log = logging.getLogger("advanced_rag")

# one connection for the whole program; decode_responses=True gives back str instead of bytes.
# Short timeouts: a slow or dead Redis must cost milliseconds, never hang a request.
# No REDIS_URL = no exact cache (everything still works, just without it)
client = (
    redis.Redis.from_url(config.REDIS_URL, decode_responses=True, socket_timeout=0.5, socket_connect_timeout=0.5)
    if config.REDIS_URL else None
)

PREFIX = "advanced_rag:answer:"  # every key we write starts with this, so we never touch other data
SEMANTIC_COLLECTION = "answer_cache"
_semantic = None  # the Chroma collection for the semantic cache, opened on first use


def make_key(question: str, sources: set[str] | None = None) -> str:
    # "How long do cats sleep?" and "how long do  cats sleep?" are the same question
    normalized = " ".join(question.lower().split())
    # a different model would give a different answer, so the model is part of the key; so is a source filter.
    # sha256 turns any text into a short, fixed-length, safe key
    scope = ",".join(sorted(sources)) if sources else ""
    digest = hashlib.sha256(f"{config.LLM_MODEL}|{scope}|{normalized}".encode()).hexdigest()
    return PREFIX + digest


# ---------- exact cache (Redis): same question, any case or spacing ----------

RETRY_AFTER_SECONDS = 30
_down_until = 0.0  # after a failure, skip Redis for a while: a dead host can take seconds per attempt (DNS)


def _usable() -> bool:
    return client is not None and time.monotonic() >= _down_until


def _failed(action: str, error: Exception) -> None:
    # the cache is an optimization: when it's down, answer without it instead of failing
    global _down_until
    _down_until = time.monotonic() + RETRY_AFTER_SECONDS
    log.warning("redis %s failed, skipping the cache for %ds: %s", action, RETRY_AFTER_SECONDS, error)


def get(question: str, sources: set[str] | None = None) -> str | None:
    if not _usable():
        return None
    try:
        return client.get(make_key(question, sources))  # None when we've never seen this question
    except redis.RedisError as error:
        _failed("get", error)
        return None


def put(question: str, answer: str, sources: set[str] | None = None) -> None:
    if not _usable():
        return
    try:
        client.set(make_key(question, sources), answer, ex=config.CACHE_TTL_SECONDS)  # ex = expire after this many seconds
    except redis.RedisError as error:
        _failed("put", error)


def ping() -> str:
    if client is None:
        return "off"
    if not _usable():
        return "down"
    try:
        client.ping()
        return "ok"
    except redis.RedisError as error:
        _failed("ping", error)
        return "down"


# ---------- semantic cache (Chroma): same MEANING, different words ----------

def semantic_collection():
    global _semantic
    if _semantic is None:
        from advanced_rag.vectorstore import get_collection

        _semantic = get_collection(name=SEMANTIC_COLLECTION)
    return _semantic


def semantic_get(question: str) -> tuple[str, str] | None:
    """The closest earlier question's (answer, top chunk id) if it's similar enough and not expired.

    The caller must check the top chunk id against its own retrieval before trusting the answer:
    "How long do cats sleep?" vs "How long do pythons sleep?" is 0.63 similar but a different question.
    """
    if config.SEMANTIC_CACHE_THRESHOLD <= 0:
        return None
    from advanced_rag.embeddings import embed_query

    collection = semantic_collection()
    result = collection.query(
        query_embeddings=[list(embed_query(question))], n_results=1,
        where={"$and": [{"model": config.LLM_MODEL}, {"created": {"$gt": time.time() - config.CACHE_TTL_SECONDS}}]},
    )
    if not result["ids"][0]:
        return None
    similarity = 1 - result["distances"][0][0]  # cosine distance -> similarity
    meta = result["metadatas"][0][0]
    return (meta["answer"], meta["top_chunk"]) if similarity >= config.SEMANTIC_CACHE_THRESHOLD else None


def semantic_put(question: str, answer: str, top_chunk: str) -> None:
    if config.SEMANTIC_CACHE_THRESHOLD <= 0:
        return
    from advanced_rag.embeddings import embed_query

    collection = semantic_collection()
    now = time.time()
    collection.delete(where={"created": {"$lt": now - config.CACHE_TTL_SECONDS}})  # drop expired entries as we go
    collection.upsert(
        ids=[make_key(question)],
        embeddings=[list(embed_query(question))],
        metadatas=[{"answer": answer, "top_chunk": top_chunk, "model": config.LLM_MODEL, "created": now}],
    )


def clear() -> int:
    # delete only our own keys; scan_iter walks them in small batches instead of all at once
    removed = 0
    if client is not None:
        try:
            keys = list(client.scan_iter(match=PREFIX + "*"))
            if keys:
                client.delete(*keys)
            removed = len(keys)
        except redis.RedisError as error:
            log.warning("redis clear failed: %s", error)
    collection = semantic_collection()
    stored = collection.get(include=[])["ids"]
    if stored:
        collection.delete(ids=stored)
    return removed + len(stored)
