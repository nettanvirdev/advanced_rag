from dataclasses import dataclass
from pathlib import Path
from advanced_rag import config


@dataclass
class Document:
    text: str
    source: str


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(text=text, source=path.name)


def load_folder(folder: Path = config.DATA_DIR) -> list[Document]:

    docs = []
    for path in sorted(folder.glob("*.txt")):
        doc = load_txt(path=path)
        if doc.text.strip() == "":
            continue
        docs.append(doc)
    return docs
