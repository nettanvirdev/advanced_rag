from advanced_rag import cache, rag
from advanced_rag.chunking import Chunk
from advanced_rag.rag import RULES, NO_ANSWER, build_context, answer


def test_empty_llm_reply_becomes_i_dont_know_and_is_not_cached(index, fake_redis, monkeypatch):
    # found by evaluation: the LLM sometimes sends back nothing at all
    collection, chunks = index
    monkeypatch.setattr(rag, "ask", lambda prompt, system="": None)
    assert answer("How long do cats sleep?", collection, chunks) == NO_ANSWER
    assert cache.get("How long do cats sleep?") is None


def test_use_cache_false_never_reads_or_writes_the_cache(index, fake_llm, fake_redis):
    collection, chunks = index
    cache.put("How long do cats sleep?", "an old cached answer")
    assert answer("How long do cats sleep?", collection, chunks, use_cache=False) == "fake reply"
    assert cache.get("How long do cats sleep?") == "an old cached answer"


def test_build_context_labels_chunks_with_their_id():
    hits = [(0.9, Chunk(text="Cats sleep a lot.", source="cat.txt", index=25))]
    assert build_context(hits) == "[cat.txt#25]\nCats sleep a lot."


def test_answer_sends_rules_and_the_right_chunk(index, fake_llm, fake_redis):
    collection, chunks = index
    assert answer("How long do cats sleep?", collection, chunks) == "fake reply"
    assert fake_llm[0] == {"role": "system", "content": RULES}
    assert "[cat.txt#25]" in fake_llm[1]["content"]


def test_second_ask_comes_from_the_cache(index, fake_llm, fake_redis):
    collection, chunks = index
    answer("How long do cats sleep?", collection, chunks)
    answer("how long do  cats sleep?", collection, chunks)  # same question, different spacing
    assert len(fake_llm) == 2  # system + user from the FIRST call only: the LLM was asked once


def test_nothing_relevant_skips_the_llm(index, fake_llm, fake_redis):
    collection, chunks = index
    assert answer("asdfgh", collection, chunks) == NO_ANSWER
    assert fake_llm == []  # no LLM call at all
    assert cache.get("asdfgh") == NO_ANSWER


def test_unanswerable_question_skips_the_llm(index, fake_llm, fake_redis):
    # the reranker scores every candidate below 0, so there is nothing worth sending to the LLM
    collection, chunks = index
    assert answer("What is the capital of France?", collection, chunks) == NO_ANSWER
    assert fake_llm == []
