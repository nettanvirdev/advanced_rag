USAGE = """usage: advanced-rag <command>

  build         sync the index with data/ (only new or changed text is embedded) and clear stale caches
  clear-cache   delete every cached answer (exact and semantic)
"""


def main() -> None:
    import logging
    import sys
    import time

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in ("build", "clear-cache"):
        print(USAGE, end="")
        raise SystemExit(0 if command in ("-h", "--help") else 2)

    from advanced_rag import cache, config, vectorstore

    logging.basicConfig(level=config.LOG_LEVEL, format="%(message)s")
    if command == "build":
        start = time.perf_counter()
        collection, changes = vectorstore.build_index()
        print(f"{collection.count()} chunks in the index, {changes} changed, in {time.perf_counter() - start:.1f}s")
        if changes:
            print(f"cleared {cache.clear()} cached answers")
    else:
        print(f"cleared {cache.clear()} cached answers")
