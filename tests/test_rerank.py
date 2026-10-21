from advanced_rag.chunking import Chunk
from advanced_rag.retrieval.rerank import rerank

SLEEP = Chunk(text="Cats usually sleep between 12 and 16 hours a day.", source="cat.txt", index=25)
ESPRESSO = Chunk(text="Espresso is brewed under high pressure, typically 9 bar.", source="coffee.txt", index=45)
GUIDO = Chunk(text="Python was created by Guido van Rossum.", source="python_language.txt", index=7)


def test_reranker_puts_the_answering_chunk_first():
    # search handed them over in the wrong order; the reranker reads question + chunk together and fixes it
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    assert rerank("How long do cats sleep?", hits)[0][1] == SLEEP


def test_reranker_keeps_at_most_k_and_returns_the_same_chunks():
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    result = rerank("What pressure is espresso brewed at?", hits, k=1)
    assert [c for _, c in result] == [ESPRESSO]


def test_chunks_that_do_not_answer_are_dropped():
    # nothing here says anything about France, so every score is below 0
    hits = [(0.9, ESPRESSO), (0.8, GUIDO), (0.7, SLEEP)]
    assert rerank("What is the capital of France?", hits) == []


def test_no_candidates_means_no_results():
    assert rerank("anything", []) == []
