from sentence_transformers import CrossEncoder

from advanced_rag import config
from advanced_rag.chunking import Chunk

# a small cross-encoder trained on real search questions (MS MARCO); ~90 MB, runs on CPU
model = CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, hits: list[tuple[float, Chunk]], k: int = 3, min_score: float | None = None) -> list[tuple[float, Chunk]]:
    # Search was fast but rough: it compared the question and each chunk SEPARATELY (two vectors).
    # A cross-encoder reads the question and one chunk TOGETHER and judges how well the chunk
    # answers it. Much more accurate, but too slow for all 210 chunks, so only the candidates get it.
    if not hits:
        return []
    scores = model.predict([(query, chunk.text) for _, chunk in hits])
    rescored = [(float(score), chunk) for score, (_, chunk) in zip(scores, hits)]
    # scores are unbounded (about -11 to +11): above 0 means "answers it", below means "doesn't".
    # On the eval set every answerable question's best chunk scored > 3.7, every unanswerable < -9.8.
    floor = config.MIN_RERANK_SCORE if min_score is None else min_score
    good = [pair for pair in rescored if pair[0] >= floor]
    return sorted(good, key=lambda pair: pair[0], reverse=True)[:k]
