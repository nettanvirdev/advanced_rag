import hashlib

import redis

from advanced_rag import config

# one connection for the whole program; decode_responses=True gives back str instead of bytes
client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

PREFIX = "advanced_rag:answer:"  # every key we write starts with this, so we never touch other data
TTL_SECONDS = 24 * 60 * 60  # answers expire after a day, so they can't stay stale forever


def make_key(question: str) -> str:
    # "How long do cats sleep?" and "how long do  cats sleep?" are the same question
    normalized = " ".join(question.lower().split())
    # a different model would give a different answer, so the model is part of the key.
    # sha256 turns any text into a short, fixed-length, safe key
    digest = hashlib.sha256(f"{config.LLM_MODEL}|{normalized}".encode()).hexdigest()
    return PREFIX + digest


def get(question: str) -> str | None:
    return client.get(make_key(question))  # None when we've never seen this question


def put(question: str, answer: str) -> None:
    client.set(make_key(question), answer, ex=TTL_SECONDS)  # ex = expire after this many seconds


def clear() -> int:
    # delete only our own keys; scan_iter walks them in small batches instead of all at once
    keys = list(client.scan_iter(match=PREFIX + "*"))
    if keys:
        client.delete(*keys)
    return len(keys)
