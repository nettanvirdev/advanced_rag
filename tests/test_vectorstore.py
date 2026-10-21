from advanced_rag.chunking import Chunk
from advanced_rag.vectorstore import get_collection, add_chunks, search, chunk_id, build_index

CHUNKS = [
    Chunk(text="Kittens sleep up to twenty hours a day.", source="cat.txt", index=0),
    Chunk(text="Espresso is strong coffee brewed under pressure.", source="coffee.txt", index=0),
    Chunk(text="Python is a programming language created by Guido van Rossum.", source="python.txt", index=0),
]


def test_chunk_id_is_source_and_index():
    assert chunk_id(CHUNKS[0]) == "cat.txt#0"


def test_collection_uses_cosine(tmp_path):
    assert get_collection(tmp_path).configuration["hnsw"]["space"] == "cosine"


def test_search_finds_the_matching_chunk(tmp_path):
    collection = get_collection(tmp_path)
    add_chunks(collection, CHUNKS)
    hits = search(collection, "How long do cats sleep?", k=3)
    assert hits[0][1] == CHUNKS[0]  # the whole Chunk comes back: text, source and index
    assert [s for s, _ in hits] == sorted([s for s, _ in hits], reverse=True)  # best first


def test_adding_twice_does_not_duplicate(tmp_path):
    collection = get_collection(tmp_path)
    add_chunks(collection, CHUNKS)
    add_chunks(collection, CHUNKS)
    assert collection.count() == 3


def test_index_survives_reopening(tmp_path):
    add_chunks(get_collection(tmp_path), CHUNKS)
    reopened = get_collection(tmp_path)  # like restarting the program
    assert reopened.count() == 3


def test_build_index_on_real_articles(tmp_path):
    collection = build_index(tmp_path)
    assert collection.count() == 210
    assert search(collection, "How do pythons kill their prey?")[0][1].source == "python_snake.txt"
