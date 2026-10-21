
from advanced_rag.retrieval.keyword import tokenize
from advanced_rag.retrieval.tfidf import idf
from advanced_rag.loaders import Document


def term_score(f: int, doc_len: int, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    # 1 = average length, 2 = twice as long
    length_ratio = doc_len / avg_len
    return f * (k1 + 1) / (f + k1 * (1 - b + b * length_ratio))


def score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float:
    if len(all_doc_words) == 0:
        return 0.0

    query_words = tokenize(query)
    # total words of all docs / number of docs
    avg_len = sum([len(w) for w in all_doc_words]) / len(all_doc_words)
    total = 0.0
    for word in query_words:
        f = doc_words.count(word)
        total = total + idf(word, all_doc_words) * \
            term_score(f, len(doc_words), avg_len)

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
