from openai import OpenAI
from advanced_rag import config

client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def ask(prompt: str, system: str = "") -> str | None:
    messages = []
    if system != "":
        messages.append(
            {"role": "system", "content": system})

    messages.append(
        {"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content
