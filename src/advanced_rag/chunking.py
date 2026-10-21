import re
from dataclasses import dataclass

from advanced_rag.embeddings import count_tokens
from advanced_rag.loaders import Document


@dataclass
class Chunk:
    text: str
    source: str  # which file it came from, e.g. "cat.txt"
    index: int  # its position in that file: 0, 1, 2, ...


def split_sentences(text: str) -> list[str]:
    # split right after . ! ? (keeping them), or at line breaks (so headings stand alone)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = 200, overlap: int = 1) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current = []  # sentences in the chunk we are filling right now

    for sentence in sentences:
        # what the chunk would be if we added this sentence
        candidate = " ".join(current + [sentence])

        # too big? close the current chunk and start a new one
        # (`current` must not be empty, or one huge sentence would never be added)
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            # carry the last `overlap` sentences into the new chunk...
            current = current[-overlap:] if overlap else []
            # ...but only if they still fit together with this sentence
            if current and count_tokens(" ".join(current + [sentence])) > max_tokens:
                current = []

        current.append(sentence)

    # the leftovers after the loop are the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(docs: list[Document], max_tokens: int = 200, overlap: int = 1) -> list[Chunk]:
    chunks = []
    for doc in docs:
        # enumerate gives the position (0, 1, 2, ...) together with each chunk's text
        for i, text in enumerate(chunk_text(doc.text, max_tokens, overlap)):
            chunks.append(Chunk(text=text, source=doc.source, index=i))
    return chunks
