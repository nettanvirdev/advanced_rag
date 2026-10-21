from pytest import approx

from advanced_rag import config
from advanced_rag.chunking import Chunk
from advanced_rag.evaluate import (
    evaluate_retrieval,
    first_relevant_rank,
    is_correct,
    load_questions,
    retrieval_scores,
)


def test_first_relevant_rank_counts_from_one():
    hits = [(0.9, Chunk("cats nap", "cat.txt", 0)), (0.8, Chunk("espresso uses 9 bar", "coffee.txt", 45))]
    assert first_relevant_rank(hits, ["9 bar"]) == 2
    assert first_relevant_rank(hits, ["Guido", "cats nap"]) == 1  # any of the phrases counts
    assert first_relevant_rank(hits, ["Guido"]) is None


def test_hit_rate_and_mrr():
    scores = retrieval_scores([1, 2, None], k=3)
    assert scores["hit_rate"] == approx(2 / 3)
    assert scores["mrr"] == approx((1 + 1 / 2 + 0) / 3)


def test_a_hit_below_k_does_not_count_but_still_scores_mrr():
    scores = retrieval_scores([5], k=3)
    assert scores["hit_rate"] == 0
    assert scores["mrr"] == approx(1 / 5)


def test_is_correct_is_case_insensitive_and_accepts_any_option():
    assert is_correct("Pythons are NOT VENOMOUS.", ["non-venomous", "not venomous"])
    assert is_correct("I don’t know", ["i don't know"])  # curly apostrophe
    assert not is_correct("They are venomous.", ["non-venomous", "not venomous"])


def test_every_relevant_phrase_really_exists_in_the_data():
    # guards the golden set against typos: a phrase that isn't in the data could never be found
    corpus = " ".join(p.read_text(encoding="utf-8") for p in config.DATA_DIR.glob("*.txt"))
    for q in load_questions():
        assert q["answer_contains"], q["question"]
        for phrase in q["relevant"] or []:
            assert phrase in corpus, phrase


def test_retrieval_quality_does_not_drop(index):
    # a quality gate: if a change to chunking or search makes retrieval worse, this fails
    collection, chunks = index
    report = evaluate_retrieval(collection, chunks, load_questions())
    assert report["hybrid"]["hit_rate"] >= 0.85
    assert report["hybrid"]["mrr"] >= 0.75
    # reranking must stay at least as good as it measured when it was added (95% / 0.84)
    assert report["hybrid+rerank"]["hit_rate"] >= 0.9
    assert report["hybrid+rerank"]["mrr"] >= 0.8
