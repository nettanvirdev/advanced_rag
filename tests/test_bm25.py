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


# ---------- BM25Index: the precomputed version must be a pure speed-up ----------

from advanced_rag.evaluate import load_questions
from advanced_rag.retrieval.bm25 import BM25Index
from advanced_rag.vectorstore import chunk_id


def test_index_gives_exactly_the_same_results_as_search(index):
    _, bm25 = index
    for q in load_questions():
        slow = [(s, chunk_id(c)) for s, c in search(q["question"], bm25.chunks, 10)]
        fast = [(s, chunk_id(c)) for s, c in bm25.search(q["question"], 10)]
        assert fast == slow, q["question"]  # same chunks, same order, same scores


def test_index_matches_search_on_ties_and_repetition():
    docs = [
        Document(text="python python python python python python python python", source="spam.txt"),
        Document(text="the python is a large snake", source="snake.txt"),
        Document(text="a python snake", source="snake2.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    for query in ["python snake", "python", "coffee", "nothing here", "python python"]:
        assert BM25Index(docs).search(query, 4) == search(query, docs, 4), query


def test_index_source_filter_and_empty_index():
    docs = [Document(text="python snake", source="a.txt"), Document(text="python code", source="b.txt"), Document(text="coffee", source="c.txt")]
    assert [d.source for _, d in BM25Index(docs).search("python", 5, sources={"b.txt"})] == ["b.txt"]
    assert BM25Index([]).search("python") == []
