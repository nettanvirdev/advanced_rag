import math

from pytest import approx

from advanced_rag.loaders import Document
from advanced_rag.retrieval.tfidf import tf, idf, score, search


def test_tf_is_count_divided_by_length():
    assert tf("a", ["a", "b", "a", "c"]) == 0.5


def test_tf_is_zero_for_missing_word():
    assert tf("zebra", ["a", "b"]) == 0.0


def test_tf_handles_empty_doc():
    assert tf("a", []) == 0.0


def test_idf_is_zero_for_word_in_every_doc():
    all_doc_words = [["a", "cat"], ["a", "dog"]]
    assert idf("a", all_doc_words) == 0.0


def test_idf_matches_formula():
    all_doc_words = [["python"], ["python"], ["coffee"], ["tea"]]
    assert idf("python", all_doc_words) == approx(math.log(4 / 2))


def test_idf_is_higher_for_rare_words():
    all_doc_words = [["python", "coffee"], ["python"], ["python"], ["tea"]]
    assert idf("coffee", all_doc_words) > idf("python", all_doc_words)


def test_idf_counts_documents_not_occurrences():
    all_doc_words = [["python", "python", "python"], ["tea"]]
    assert idf("python", all_doc_words) == approx(math.log(2 / 1))


def test_idf_is_zero_for_unknown_word():
    assert idf("zebra", [["a"], ["b"]]) == 0.0


def test_score_rewards_rare_words_over_common_ones():
    all_doc_words = [["a", "kitten"], ["a", "dog"], ["a", "cat"]]
    rare = score("a kitten", ["a", "kitten"], all_doc_words)
    common = score("a kitten", ["a", "dog"], all_doc_words)
    assert rare > common


def test_score_is_zero_without_shared_words():
    all_doc_words = [["coffee"], ["tea"]]
    assert score("python", ["coffee"], all_doc_words) == 0.0



def test_search_drops_zero_scores_and_limits_k():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="python snake", source="snake.txt"),
        Document(text="python tea", source="tea.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    assert len(search("snake", docs, k=3)) == 1
    assert len(search("python snake code tea", docs, k=2)) == 2
