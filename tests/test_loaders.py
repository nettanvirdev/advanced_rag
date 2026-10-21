from advanced_rag.loaders import load_txt, load_folder


def test_load_txt_reads_text(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hello rag", encoding="utf-8")
    doc = load_txt(file)
    assert doc.text == "hello rag"


def test_load_folder_ignores_non_txt(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hello", encoding="utf-8")
    file2 = tmp_path / "notes.md"
    file2.write_text("test", encoding="utf-8")
    docs = load_folder(tmp_path)
    sources = []
    for doc in docs:
        sources.append(doc.source)

    assert "notes.md" not in sources

    # another way
    sources2 = [doc.source for doc in docs]
    assert "notes.md" not in sources2


def test_load_folder_skips_empty_files(tmp_path):
    # create empty.txt with text ""
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")
    # docs = load_folder(tmp_path)
    docs = load_folder(tmp_path)
    # assert the list has 0 items   (hint: len(docs))
    assert len(docs) == 0


def test_load_folder_is_sorted(tmp_path):
    # create b.txt first, then a.txt (any text)
    file = tmp_path / "b.txt"
    file.write_text("test", encoding="utf-8")
    file2 = tmp_path / "a.txt"
    file2.write_text("test", encoding="utf-8")
    # docs = load_folder(tmp_path)
    docs = load_folder(tmp_path)
    # assert the first doc's source is "a.txt"   (hint: docs[0].source)
    assert docs[0].source == "a.txt"
