from advanced_rag import llm


def test_ask_returns_reply_text(fake_llm):
    assert llm.ask("hi") == "fake reply"


def test_ask_without_system_sends_only_user_message(fake_llm):
    llm.ask("hi")
    assert fake_llm == [{"role": "user", "content": "hi"}]


def test_ask_sends_the_given_system_rules(fake_llm):
    llm.ask("hi", system="Only use the context")
    assert fake_llm[0] == {"role": "system", "content": "Only use the context"}
