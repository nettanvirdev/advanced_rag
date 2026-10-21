from advanced_rag.chunking import split_sentences, chunk_text, chunk_documents
from advanced_rag.embeddings import count_tokens
from advanced_rag.loaders import Document, load_folder


def test_split_sentences_keeps_punctuation_and_splits_lines():
    text = "Hello there. How are you? I'm fine!\nHistory\nCats are great."
    assert split_sentences(text) == ["Hello there.", "How are you?", "I'm fine!", "History", "Cats are great."]


def test_tiny_limit_gives_one_sentence_per_chunk():
    assert chunk_text("One. Two. Three.", max_tokens=3) == ["One.", "Two.", "Three."]


def test_sentences_are_packed_until_the_limit():
    # "One. Two." is 4 tokens, so it fits in 4; adding "Three." would not
    assert chunk_text("One. Two. Three.", max_tokens=4, overlap=0) == ["One. Two.", "Three."]


def test_overlap_repeats_the_last_sentence():
    assert chunk_text("One. Two. Three. Four.", max_tokens=4, overlap=1) == ["One. Two.", "Two. Three.", "Three. Four."]


def test_no_chunk_is_over_the_limit_on_real_articles():
    # with overlap on, the guard must still keep every chunk within the limit
    for doc in load_folder():
        assert max(count_tokens(c) for c in chunk_text(doc.text)) <= 200, doc.source


def test_nothing_is_lost():
    for doc in load_folder():
        assert " ".join(chunk_text(doc.text, overlap=0)) == " ".join(split_sentences(doc.text)), doc.source


def test_chunk_documents_keeps_source_and_counts_index_per_file():
    docs = [
        Document(text="One. Two. Three.", source="a.txt"),
        Document(text="Four. Five.", source="b.txt"),
    ]
    chunks = chunk_documents(docs, max_tokens=3, overlap=0)
    assert [(c.source, c.index, c.text) for c in chunks] == [
        ("a.txt", 0, "One."),
        ("a.txt", 1, "Two."),
        ("a.txt", 2, "Three."),
        ("b.txt", 0, "Four."),
        ("b.txt", 1, "Five."),
    ]
