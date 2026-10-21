from advanced_rag.llm import ask

RULES = (
    "Rewrite the user's question as a short search query for an encyclopedia. "
    "Use the most likely words an article would use, add key synonyms, and drop filler words. "
    "Reply with the query only."
)


def rewrite(question: str) -> str:
    # ask the LLM for better search words: "Which country grows the most coffee?" ->
    # "largest coffee-producing country Brazil". Falls back to the question if the reply is empty.
    query = " ".join((ask(question, system=RULES) or "").split())  # also flattens multi-line replies
    return query or question
