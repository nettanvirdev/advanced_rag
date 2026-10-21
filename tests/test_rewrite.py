from advanced_rag import vectorstore
from advanced_rag.rag import retrieve
from advanced_rag.retrieval import rewrite as rewrite_module
from advanced_rag.retrieval.rewrite import RULES, rewrite


def test_rewrite_asks_the_llm_with_the_rewrite_rules(fake_llm):
    assert rewrite("Which country grows the most coffee?") == "fake reply"
    assert fake_llm[0] == {"role": "system", "content": RULES}


def test_empty_reply_falls_back_to_the_question(monkeypatch):
    monkeypatch.setattr(rewrite_module, "ask", lambda question, system="": None)
    assert rewrite("Are pythons venomous?") == "Are pythons venomous?"


def test_multi_line_reply_becomes_one_line(monkeypatch):
    monkeypatch.setattr(rewrite_module, "ask", lambda question, system="": "python snakes\nvenom\n")
    assert rewrite("Are pythons venomous?") == "python snakes venom"


def test_a_useless_rewrite_cannot_hurt(index, fake_llm):
    # the rewritten query only ADDS candidates; reranking still judges against the original question
    collection, chunks = index
    hits = retrieve("How long do cats sleep?", collection, chunks, rewrite_query=True)
    assert vectorstore.chunk_id(hits[0][1]) == "cat.txt#25"
