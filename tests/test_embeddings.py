from pytest import approx

from advanced_rag.embeddings import embed, dimension, count_tokens


def test_embed_returns_one_vector_per_text():
    vectors = embed(["cat", "kitten", "coffee"])
    assert len(vectors) == 3
    assert all(len(v) == dimension() == 384 for v in vectors)


def test_vectors_are_normalized():
    for v in embed(["cat", "a much longer sentence about coffee beans"]):
        assert sum(x * x for x in v) == approx(1.0, abs=1e-4)


def test_dot_product_is_cosine_similarity():
    cat, kitten = embed(["cat", "kitten"])
    assert sum(a * b for a, b in zip(cat, kitten)) == approx(0.788, abs=0.001)


def test_count_tokens_counts_tokens_not_characters():
    # 'kitten', '##s', 'sleep', 'a', 'lot', '.'
    assert count_tokens("Kittens sleep a lot.") == 6


def test_text_past_256_tokens_is_ignored():
    head = "Coffee is brewed from roasted beans. " * 40
    tail = " Kittens sleep twenty hours a day." * 20
    a, b = embed([head, head + tail])
    assert sum(x * y for x, y in zip(a, b)) == approx(1.0, abs=1e-4)


def test_query_embedding_is_cached_and_identical():
    from advanced_rag.embeddings import embed_query

    embed_query.cache_clear()
    first = embed_query("How long do cats sleep?")
    second = embed_query("How long do cats sleep?")
    assert first == second
    assert embed_query.cache_info().hits == 1
    assert list(first) == approx(embed(["How long do cats sleep?"])[0], abs=1e-6)
