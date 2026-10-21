import re
from advanced_rag.loaders import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def score(query: str, doc: Document) -> int:
    query_words = tokenize(query)
    count = 0
    tokenized_doc_text = tokenize(doc.text)
    for word in query_words:
        if word.lower() in tokenized_doc_text:
            count = count + 1

    return count


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[int, Document]]:
    results = []

    for doc in docs:
        s = score(query=query,  doc=doc)
        if s > 0:
            results.append((s, doc))

    sorted_results = sorted(results, key=lambda s: s[0], reverse=True)
    return sorted_results[:k]


# only runs when you start this file directly, not when another file imports it
if __name__ == "__main__":
    from advanced_rag.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
