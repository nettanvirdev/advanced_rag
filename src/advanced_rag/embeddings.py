from sentence_transformers import SentenceTransformer
from advanced_rag import config


model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


def dimension() -> int | None:
    return model.get_embedding_dimension()


def count_tokens(text: str) -> int | None:
    return len(model.tokenizer.tokenize(text))
