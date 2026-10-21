import logging
from collections.abc import Iterator

from openai import OpenAI
from advanced_rag import config

log = logging.getLogger("advanced_rag")

# timeout: a stuck call fails instead of hanging; max_retries: the SDK retries 429s and 5xx with backoff.
# The placeholder key lets the app start without one: search works, only answering needs the LLM
client = OpenAI(
    api_key=config.LLM_API_KEY or "not-set",
    base_url=config.LLM_BASE_URL or None,
    timeout=config.LLM_TIMEOUT,
    max_retries=2,
)


def _request(prompt: str, system: str) -> dict:
    # the stable rules go FIRST: providers that cache prompt prefixes can reuse them across questions
    messages = []
    if system != "":
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    request = {"model": config.LLM_MODEL, "messages": messages}
    if config.LLM_MAX_TOKENS:
        request["max_tokens"] = config.LLM_MAX_TOKENS
    return request


def ask(prompt: str, system: str = "") -> str | None:
    response = client.chat.completions.create(**_request(prompt, system))
    usage = getattr(response, "usage", None)
    if usage:  # tokens are what you pay for: log them so cost is visible
        log.info("llm tokens prompt=%s completion=%s", usage.prompt_tokens, usage.completion_tokens)
    return response.choices[0].message.content


def stream(prompt: str, system: str = "") -> Iterator[str]:
    # the same call with stream=True: the reply arrives in small pieces as the model writes it
    for event in client.chat.completions.create(**_request(prompt, system), stream=True):
        piece = event.choices[0].delta.content if event.choices else None
        if piece:
            yield piece
