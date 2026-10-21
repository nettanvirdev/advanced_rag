import heapq
import math
from collections import Counter

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


class BM25Index:
    """The same scores as `search`, with everything that doesn't depend on the question done once.

    `search` re-tokenizes every chunk and recounts document frequencies on every question (~198 ms
    on 210 chunks). Here that happens once, and a question only touches the chunks that contain
    one of its words (~0.2 ms). Works for anything with `.text` (Document or Chunk).
    """

    # ponytail: everything lives in RAM; fine to ~1M chunks, past that use an on-disk engine (OpenSearch, bm25s mmap)
    def __init__(self, docs: list):
        self.chunks = docs
        words = [tokenize(d.text) for d in docs]
        self.lengths = [len(w) for w in words]
        self.avg_len = sum(self.lengths) / len(docs) if docs else 0.0
        # postings: word -> [(chunk position, how many times the word is in that chunk), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, w in enumerate(words):
            for word, count in Counter(w).items():
                self.postings.setdefault(word, []).append((i, count))
        self.idf = {word: math.log(len(docs) / len(p)) for word, p in self.postings.items()}

    def search(self, query: str, k: int = 3, sources: set[str] | None = None) -> list[tuple[float, object]]:
        scores: dict[int, float] = {}
        for word in tokenize(query):
            for i, count in self.postings.get(word, ()):
                scores[i] = scores.get(i, 0.0) + self.idf[word] * term_score(count, self.lengths[i], self.avg_len)
        hits = [
            (s, i) for i, s in scores.items()
            if s > 0 and (sources is None or self.chunks[i].source in sources)
        ]
        # ties keep the original chunk order, exactly like `search`
        best = heapq.nlargest(k, sorted(hits, key=lambda pair: pair[1]), key=lambda pair: pair[0])
        return [(s, self.chunks[i]) for s, i in best]


# only runs when you start this file directly, not when another file imports it
if __name__ == "__main__":
    from advanced_rag.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
