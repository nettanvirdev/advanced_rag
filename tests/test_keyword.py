from advanced_rag.loaders import Document
from advanced_rag.retrieval.keyword import tokenize, score, search


def test_tokenize_lowercases_and_drops_punctuation():
    data = tokenize("Hello, RAG!")
    assert data[0] == "hello"
    assert data[1] == "rag"


def test_score_counts_matching_words():
    doc = Document(text="the python is a snake", source="x.txt")
    scores = score("python snake", doc)
    assert scores == 2


def test_score_is_zero_for_synonyms():
    doc = Document(text="felines purr", source="x.txt")
    scores = score("cat", doc)
    assert scores == 0


def test_score_ignores_case():
    doc = Document(text="Python is great", source="x.txt")
    scores = score("python", doc)
    assert scores == 1


def test_score_matches_whole_words_only():
    doc = Document(text="we concatenate strings", source="x.txt")
    scores = score("cat", doc)
    assert scores == 0


def test_search_ranks_best_match_first():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="python snake", source="snake.txt"),
    ]
    results = search("python snake", docs)
    assert results[0][1].source == "snake.txt"


def test_search_drops_zero_scores():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    results = search("python", docs)
    assert len(results) == 1


def test_search_returns_at_most_k():
    docs = [Document(text="python", source=f"{i}.txt") for i in range(3)]
    results = search("python", docs, k=2)
    assert len(results) == 2
