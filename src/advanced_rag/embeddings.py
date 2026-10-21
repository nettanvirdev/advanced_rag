from functools import lru_cache

from sentence_transformers import SentenceTransformer
from advanced_rag import config


model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=10_000)
def embed_query(query: str) -> tuple[float, ...]:
    # the same question always gives the same vector: a repeat costs a dict lookup, not a model run.
    # A tuple because lru_cache results are shared and must not be changed by callers
    return tuple(embed([query])[0])


def dimension() -> int | None:
    return model.get_embedding_dimension()


def count_tokens(text: str) -> int | None:
    return len(model.tokenizer.tokenize(text))
