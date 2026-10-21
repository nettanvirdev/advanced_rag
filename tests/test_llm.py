from advanced_rag import llm


def test_ask_returns_reply_text(fake_llm):
    assert llm.ask("hi") == "fake reply"


def test_ask_without_system_sends_only_user_message(fake_llm):
    llm.ask("hi")
    assert fake_llm == [{"role": "user", "content": "hi"}]


def test_ask_sends_the_given_system_rules(fake_llm):
    llm.ask("hi", system="Only use the context")
    assert fake_llm[0] == {"role": "system", "content": "Only use the context"}


def test_max_tokens_is_sent_only_when_set(monkeypatch):
    from advanced_rag import config

    assert "max_tokens" not in llm._request("hi", "")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 300)
    assert llm._request("hi", "")["max_tokens"] == 300


def test_stream_yields_the_pieces(fake_llm):
    assert list(llm.stream("hi")) == ["fake ", "reply"]


def test_client_has_a_timeout_and_retries():
    assert llm.client.timeout == llm.config.LLM_TIMEOUT
    assert llm.client.max_retries == 2
