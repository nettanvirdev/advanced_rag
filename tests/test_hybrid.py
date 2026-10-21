from advanced_rag import vectorstore
from advanced_rag.chunking import Chunk
from advanced_rag.retrieval.hybrid import rrf, hybrid_search

A = Chunk(text="a", source="a.txt", index=0)
B = Chunk(text="b", source="b.txt", index=0)
C = Chunk(text="c", source="c.txt", index=0)


def test_rrf_first_place_scores_one_over_61():
    assert rrf([[(0.9, A)]])[0] == (1 / 61, A)


def test_rrf_rewards_chunks_found_by_both_methods():
    semantic = [(0.9, A), (0.8, B)]
    keyword = [(12.0, C), (7.0, B)]
    # B is 2nd in both lists: 1/62 + 1/62 beats A's and C's single 1/61
    assert [c for _, c in rrf([semantic, keyword])] == [B, A, C]


def test_rrf_ignores_score_scales():
    # BM25's 12.0 doesn't outweigh a cosine 0.9: only the rank counts
    assert rrf([[(0.9, A)], [(12.0, C)]])[0][0] == rrf([[(0.9, C)], [(12.0, A)]])[0][0]


# the `index` fixture (the real index, built once) lives in conftest.py

def test_all_chunks_reads_every_stored_chunk(index):
    collection, chunks = index
    assert len(chunks) == collection.count() == 210


def test_hybrid_finds_exact_names_that_semantic_misses(index):
    collection, chunks = index
    # semantic search alone ranks a coffee chunk first for this (score 0.20)
    assert vectorstore.search(collection, "Guido van Rossum")[0][1].source == "coffee.txt"
    assert hybrid_search(collection, chunks, "Guido van Rossum")[0][1].source == "python_language.txt"


def test_hybrid_finds_exact_numbers(index):
    collection, chunks = index
    assert vectorstore.chunk_id(hybrid_search(collection, chunks, "9 bar")[0][1]) == "coffee.txt#45"


def test_hybrid_returns_nothing_for_nonsense(index):
    collection, chunks = index
    assert hybrid_search(collection, chunks, "asdfgh") == []


def test_hybrid_keeps_good_semantic_answers(index):
    collection, chunks = index
    assert vectorstore.chunk_id(hybrid_search(collection, chunks, "How long do cats sleep?")[0][1]) == "cat.txt#25"
