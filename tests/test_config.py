import importlib

from advanced_rag import config


def test_root_has_pyproject():
    assert (config.ROOT / "pyproject.toml").exists()


def test_data_dir_exists():
    assert config.DATA_DIR.exists(), f"The data directory does not exist: {config.DATA_DIR}"
    assert config.DATA_DIR.name == "data"


def test_defaults_work_without_a_dotenv():
    # a fresh clone (or CI) has no .env: everything except answering must still have a working value
    assert config.EMBEDDING_MODEL and config.RERANK_MODEL
    assert config.RERANK_CANDIDATES == 20
    assert 0 < config.SEMANTIC_CACHE_THRESHOLD < 1


def test_env_overrides_and_zero_token_cap_means_none(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATES", "5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.RERANK_CANDIDATES == 5
        assert reloaded.LLM_MAX_TOKENS is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)
