from advanced_rag import config


def test_root_has_pyproject():
    assert (config.ROOT / "pyproject.toml").exists()


def test_env_has_llm_api_key():
    assert config.LLM_API_KEY != ""


def test_env_has_llm_model():
    assert config.LLM_MODEL != ""


def test_env_has_llm_base_url():
    assert config.LLM_BASE_URL != ""
    assert config.LLM_BASE_URL.startswith(("https://", "http://"))
    assert config.LLM_BASE_URL.endswith("v1")


def test_env_has_llm_chroma_dir():
    assert config.CHROMA_DIR.exists(
    ), f"The chroma db directory is missing: {config.CHROMA_DIR}"


def test_env_has_llm_redis_url():
    assert config.REDIS_URL != ""


def test_env_has_llm_embedding_model():
    assert config.EMBEDDING_MODEL != ""


def test_env_has_data_dir():
    assert config.DATA_DIR.exists(
    ), f"The data directory does not exist: {config.DATA_DIR}"
    assert config.DATA_DIR.name == "data", f"The data directory exists, however, the name is wrong:{config.DATA_DIR.name}"
