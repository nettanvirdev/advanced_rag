from advanced_rag import cache, config, llm
from advanced_rag.chunking import Chunk
from advanced_rag.rag import RULES, NO_ANSWER, answer, answer_stream, build_context


def test_empty_llm_reply_becomes_i_dont_know_and_is_not_cached(index, monkeypatch):
    # found by evaluation: the LLM sometimes sends back nothing at all
    collection, bm25 = index
    monkeypatch.setattr(llm, "ask", lambda prompt, system="": None)
    assert answer("How long do cats sleep?", collection, bm25).text == NO_ANSWER
    assert cache.get("How long do cats sleep?") is None


def test_use_cache_false_never_reads_or_writes_the_cache(index, fake_llm):
    collection, bm25 = index
    cache.put("How long do cats sleep?", "an old cached answer")
    assert answer("How long do cats sleep?", collection, bm25, use_cache=False).text == "fake reply"
    assert cache.get("How long do cats sleep?") == "an old cached answer"
    assert cache.semantic_get("How long do cats sleep?") is None


def test_build_context_labels_chunks_with_their_id():
    hits = [(0.9, Chunk(text="Cats sleep a lot.", source="cat.txt", index=25))]
    assert build_context(hits) == "[cat.txt#25]\nCats sleep a lot."


def test_answer_sends_rules_and_the_right_chunk(index, fake_llm):
    collection, bm25 = index
    reply = answer("How long do cats sleep?", collection, bm25)
    assert (reply.text, reply.cached) == ("fake reply", None)
    assert fake_llm[0] == {"role": "system", "content": RULES}
    assert "[cat.txt#25]" in fake_llm[1]["content"]


def test_second_ask_comes_from_the_exact_cache(index, fake_llm):
    collection, bm25 = index
    answer("How long do cats sleep?", collection, bm25)
    reply = answer("how long do  cats sleep?", collection, bm25)  # same question, different spacing
    assert reply.cached == "exact"
    assert len(fake_llm) == 2  # system + user from the FIRST call only: the LLM was asked once


def test_a_paraphrase_comes_from_the_semantic_cache(index, fake_llm):
    collection, bm25 = index
    answer("How long do cats sleep?", collection, bm25)
    reply = answer("How many hours a day does a cat sleep?", collection, bm25)  # 0.89 similar, same best chunk
    assert (reply.text, reply.cached) == ("fake reply", "semantic")
    assert len(fake_llm) == 2  # still only one LLM call
    assert answer("How many hours a day does a cat sleep?", collection, bm25).cached == "exact"  # promoted


def test_a_similar_sounding_different_question_is_not_served_from_the_cache(index, fake_llm, monkeypatch):
    collection, bm25 = index
    answer("How long do cats sleep?", collection, bm25)
    # even with a careless threshold that lets the vectors match, the guard sees a different best chunk
    monkeypatch.setattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.6)
    # "cats sleep" vs "pythons sleep" is 0.63 similar: the vectors alone would serve the cat answer
    assert cache.semantic_get("How long do pythons sleep?") == ("fake reply", "cat.txt#25")
    assert answer("How long do pythons sleep?", collection, bm25).cached is None


def test_semantic_cache_can_be_turned_off(index, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_CACHE_THRESHOLD", 0)
    collection, bm25 = index
    answer("How long do cats sleep?", collection, bm25)
    assert answer("How many hours a day does a cat sleep?", collection, bm25).cached is None


def test_nothing_relevant_skips_the_llm(index, fake_llm):
    collection, bm25 = index
    assert answer("asdfgh", collection, bm25).text == NO_ANSWER
    assert fake_llm == []  # no LLM call at all
    assert cache.get("asdfgh") == NO_ANSWER


def test_unanswerable_question_skips_the_llm(index, fake_llm):
    # the reranker scores every candidate below 0, so there is nothing worth sending to the LLM
    collection, bm25 = index
    assert answer("What is the capital of France?", collection, bm25).text == NO_ANSWER
    assert fake_llm == []


def test_source_filter_only_uses_those_files(index, fake_llm):
    collection, bm25 = index
    answer("Who created Python?", collection, bm25, sources={"python_language.txt"})
    labels = [line for line in fake_llm[1]["content"].splitlines() if line.startswith("[")]
    assert labels and all(label.startswith("[python_language.txt#") for label in labels)
    # a filtered answer is cached under its filter only
    assert cache.get("Who created Python?") is None
    assert cache.get("Who created Python?", {"python_language.txt"}) == "fake reply"


def test_stream_yields_pieces_and_caches_the_whole_answer(index, fake_llm):
    collection, bm25 = index
    assert list(answer_stream("How long do cats sleep?", collection, bm25)) == ["fake ", "reply"]
    assert cache.get("How long do cats sleep?") == "fake reply"
    assert list(answer_stream("How long do cats sleep?", collection, bm25)) == ["fake reply"]  # cached: one piece


def test_stream_of_nothing_relevant_is_i_dont_know(index, fake_llm):
    collection, bm25 = index
    assert list(answer_stream("asdfgh", collection, bm25)) == [NO_ANSWER]
    assert fake_llm == []
