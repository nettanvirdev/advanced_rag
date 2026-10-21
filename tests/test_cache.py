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
    assert 0 < fake_redis.ttl(cache.make_key("q")) <= 24 * 60 * 60


def test_clear_removes_only_our_keys(fake_redis):
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    fake_redis.set("someone-else:data", "keep me")
    assert cache.clear() == 2
    assert fake_redis.get("someone-else:data") == "keep me"
