import math
from advanced_rag.loaders import Document
from advanced_rag.retrieval.keyword import tokenize


def tf(word: str, docs_words: list[str]) -> float:
    if len(docs_words) == 0:
        return 0.0
    doc_words_len = len(docs_words)
    count = 0
    for doc_word in docs_words:
        if doc_word == word:
            count = count + 1
    return count / doc_words_len


def idf(word: str, all_doc_words: list[list[str]]) -> float:
    if len(all_doc_words) == 0:
        return 0.0

    n = len(all_doc_words)
    df = 0

    for single_doc in all_doc_words:
        if word in single_doc:
            df = df + 1

    if df == 0:
        return 0.0

    return math.log(n / df)


def score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float:
    tokenized_query = tokenize(query)
    total = 0.0
    for word in tokenized_query:
        total = total + tf(word, doc_words) * idf(word, all_doc_words)

    return total


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]:
    all_doc_words = [tokenize(d.text) for d in docs]
    results = []
    for doc in docs:
        doc_words = tokenize(doc.text)
        s = score(query, doc_words, all_doc_words)
        if s > 0:
            results.append((s, doc))

    return sorted(results, key=lambda s: s[0], reverse=True)[:k]


# only runs when you start this file directly, not when another file imports it
if __name__ == "__main__":
    from advanced_rag.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
