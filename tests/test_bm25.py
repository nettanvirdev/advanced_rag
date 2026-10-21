from advanced_rag.loaders import Document
from advanced_rag.retrieval.bm25 import term_score, score, search


def test_term_score_saturates():
    once = term_score(1, doc_len=30, avg_len=30)
    ten_times = term_score(10, doc_len=30, avg_len=30)
    assert ten_times < 10 * once
    assert ten_times < 1.5 + 1  # can never pass k1 + 1


def test_term_score_penalizes_long_docs():
    short = term_score(1, doc_len=15, avg_len=30)
    long = term_score(1, doc_len=60, avg_len=30)
    assert short > long


def test_term_score_is_zero_when_word_missing():
    assert term_score(0, doc_len=30, avg_len=30) == 0.0


def test_score_is_zero_without_shared_words():
    all_doc_words = [["coffee"], ["tea"]]
    assert score("python", ["coffee"], all_doc_words) == 0.0


def test_score_handles_no_documents():
    assert score("python", [], []) == 0.0



def test_search_is_not_fooled_by_repetition():
    docs = [
        Document(text="python python python python python python python python", source="spam.txt"),
        Document(text="the python is a large snake", source="snake.txt"),
        Document(text="coffee beans", source="coffee.txt"),
        Document(text="felines purr", source="cat.txt"),
    ]
    results = search("python snake", docs)
    assert results[0][1].source == "snake.txt"


def test_search_drops_zero_scores_and_limits_k():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="python snake", source="snake.txt"),
        Document(text="python tea", source="tea.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    assert len(search("snake", docs, k=3)) == 1
    assert len(search("python snake code tea", docs, k=2)) == 2
