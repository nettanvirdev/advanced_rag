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


def test_build_index_on_real_articles(index_dir):
    collection = get_collection(index_dir)
    assert collection.count() == 210
    assert search(collection, "How do pythons kill their prey?")[0][1].source == "python_snake.txt"


# ---------- incremental indexing ----------

def write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def test_build_index_reports_changes_and_skips_unchanged_text(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write(data, "cat.txt", "Cats sleep a lot. They purr.")
    write(data, "coffee.txt", "Espresso is brewed under pressure.")
    store = tmp_path / "chroma"

    assert build_index(store, data)[1] == 2  # two new chunks
    assert build_index(store, data)[1] == 0  # nothing changed: nothing embedded, nothing deleted

    write(data, "cat.txt", "Cats sleep twelve to sixteen hours a day.")
    (data / "coffee.txt").unlink()
    collection, changes = build_index(store, data)
    assert changes == 2  # cat.txt#0 updated, coffee.txt#0 removed
    assert collection.get()["ids"] == ["cat.txt#0"]
    assert search(collection, "How long do cats sleep?")[0][1].text == "Cats sleep twelve to sixteen hours a day."


def test_moved_text_reuses_its_vector(tmp_path, monkeypatch):
    from advanced_rag import vectorstore

    data = tmp_path / "data"
    data.mkdir()
    write(data, "a.txt", "Kittens sleep up to twenty hours a day.")
    store = tmp_path / "chroma"
    build_index(store, data)

    write(data, "b.txt", "Kittens sleep up to twenty hours a day.")  # same text under a new id
    (data / "a.txt").unlink()
    calls = []
    real_embed = vectorstore.embed
    monkeypatch.setattr(vectorstore, "embed", lambda texts: calls.append(texts) or real_embed(texts))
    collection, _ = build_index(store, data)
    assert calls == []  # the stored vector was reused, the model never ran
    assert collection.get()["ids"] == ["b.txt#0"]


def test_an_empty_data_folder_never_wipes_the_index(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write(data, "cat.txt", "Cats sleep a lot.")
    store = tmp_path / "chroma"
    build_index(store, data)
    (data / "cat.txt").unlink()
    collection, changes = build_index(store, data)
    assert (changes, collection.count()) == (0, 1)
    assert build_index(store, tmp_path / "missing")[0].count() == 1


def test_search_source_filter(index_dir):
    collection = get_collection(index_dir)
    hits = search(collection, "Who created Python?", k=5, sources={"coffee.txt"})
    assert hits and all(c.source == "coffee.txt" for _, c in hits)
