from advanced_rag import cache, config


def test_key_ignores_case_and_extra_spaces():
    assert cache.make_key("How long do cats sleep?") == cache.make_key("  how LONG do   cats sleep?")


def test_key_changes_with_the_model(monkeypatch):
    before = cache.make_key("Who created Python?")
    monkeypatch.setattr(config, "LLM_MODEL", "another-model")
    assert cache.make_key("Who created Python?") != before


def test_put_then_get(fake_redis):
    assert cache.get("Who created Python?") is None
    cache.put("Who created Python?", "Guido van Rossum")
    assert cache.get("who created python?") == "Guido van Rossum"


def test_answers_expire_after_a_day(fake_redis):
    cache.put("q", "a")
    assert 0 < fake_redis.ttl(cache.make_key("q")) <= config.CACHE_TTL_SECONDS == 24 * 60 * 60


def test_clear_removes_only_our_keys(fake_redis):
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    fake_redis.set("someone-else:data", "keep me")
    assert cache.clear() == 2
    assert fake_redis.get("someone-else:data") == "keep me"


def test_key_changes_with_a_source_filter():
    assert cache.make_key("q", {"cat.txt"}) != cache.make_key("q")
    assert cache.make_key("q", {"a.txt", "b.txt"}) == cache.make_key("q", {"b.txt", "a.txt"})


def test_a_dead_redis_never_breaks_answering(monkeypatch):
    import redis

    dead = redis.Redis(host="127.0.0.1", port=1, socket_timeout=0.1, socket_connect_timeout=0.1)
    monkeypatch.setattr(cache, "client", dead)
    monkeypatch.setattr(cache, "_down_until", 0.0)
    assert cache.get("q") is None  # logged, not raised
    assert cache._down_until > 0  # ...and Redis is skipped for a while, so the next calls cost nothing
    calls = []
    monkeypatch.setattr(dead, "get", lambda key: calls.append(key))
    assert cache.get("q") is None and calls == []
    cache.put("q", "a")
    assert cache.ping() == "down"


def test_no_redis_url_means_no_exact_cache(monkeypatch):
    monkeypatch.setattr(cache, "client", None)
    cache.put("q", "a")
    assert cache.get("q") is None
    assert cache.ping() == "off"


# ---------- semantic cache ----------

def test_semantic_hit_and_miss():
    cache.semantic_put("How long do cats sleep?", "12-16 hours [cat.txt#25]", "cat.txt#25")
    assert cache.semantic_get("How many hours a day does a cat sleep?") == ("12-16 hours [cat.txt#25]", "cat.txt#25")
    assert cache.semantic_get("Who created coffee?") is None  # 0.47: far below the threshold


def test_semantic_entries_expire_and_belong_to_one_model(monkeypatch):
    cache.semantic_put("How long do cats sleep?", "a", "cat.txt#25")
    model = config.LLM_MODEL
    monkeypatch.setattr(config, "LLM_MODEL", "another-model")
    assert cache.semantic_get("How long do cats sleep?") is None  # another model's answer
    monkeypatch.setattr(config, "LLM_MODEL", model)
    assert cache.semantic_get("How long do cats sleep?") == ("a", "cat.txt#25")
    monkeypatch.setattr(config, "CACHE_TTL_SECONDS", -1)  # everything is already expired
    assert cache.semantic_get("How long do cats sleep?") is None


def test_clear_empties_both_caches(fake_redis):
    cache.put("q", "a")
    cache.semantic_put("How long do cats sleep?", "a", "cat.txt#25")
    assert cache.clear() == 2
    assert cache.semantic_get("How long do cats sleep?") is None
