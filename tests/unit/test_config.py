"""Tests for config defaults and embedding arg building."""

from __future__ import annotations

import pytest

from shrike.cli.config import (
    DEFAULT_CONFIG,
    index_args,
    load_config,
    resolve_cache_dir,
    resolve_embedding,
    resolve_index_save,
    save_config,
)
from shrike.cli.config import embedding_args as _embedding_args


class TestEmbeddingConfig:
    def test_defaults_present(self) -> None:
        assert "embedding" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["embedding"]["model"] is None
        assert DEFAULT_CONFIG["embedding"]["port"] == 8373

    def test_load_config_includes_embedding(self, tmp_path) -> None:
        config = load_config(tmp_path / "nonexistent.yml")
        assert "embedding" in config
        assert config["embedding"]["port"] == 8373

    def test_load_config_merges_embedding(self, tmp_path) -> None:
        config_file = tmp_path / "config.yml"
        config_file.write_text("embedding:\n  model: /my/model.gguf\n  threads: 8\n")
        config = load_config(config_file)
        assert config["embedding"]["model"] == "/my/model.gguf"
        assert config["embedding"]["threads"] == 8
        assert config["embedding"]["port"] == 8373


class TestEmbeddingArgs:
    """_embedding_args takes an already-resolved flat dict (from resolve_embedding)."""

    def test_no_model_returns_empty(self) -> None:
        assert _embedding_args({}) == []
        assert _embedding_args({"model": None}) == []

    def test_model_only(self) -> None:
        args = _embedding_args({"model": "/m.gguf"})
        assert args == ["--embedding-model", "/m.gguf"]

    def test_model_with_port(self) -> None:
        args = _embedding_args({"model": "/m.gguf", "port": 9000})
        assert "--embedding-model" in args
        assert "--embedding-port" in args
        assert "9000" in args

    def test_all_options(self) -> None:
        resolved = {
            "model": "/m.gguf",
            "port": 9000,
            "context_size": 2048,
            "threads": 4,
            "gpu_layers": 33,
            "llama_server": "/bin/llama-server",
        }
        args = _embedding_args(resolved)
        assert "--embedding-model" in args
        assert "--embedding-port" in args
        assert "--embedding-context-size" in args
        assert "--embedding-threads" in args
        assert "--embedding-gpu-layers" in args
        assert "--llama-server" in args
        assert "/m.gguf" in args
        assert "2048" in args

    def test_no_embedding_flag(self) -> None:
        assert "--no-embedding" in _embedding_args({"model": "/m.gguf"}, no_embedding=True)
        assert "--no-embedding" in _embedding_args({}, no_embedding=True)
        assert "--no-embedding" not in _embedding_args({"model": "/m.gguf"})


class TestResolveEmbedding:
    """config → env → flag cascade (flag wins)."""

    def test_from_config(self) -> None:
        config = {"embedding": {"model": "/cfg.gguf", "port": 8500, "threads": 2}}
        resolved = resolve_embedding(config)
        assert resolved["model"] == "/cfg.gguf"
        assert resolved["port"] == 8500
        assert resolved["threads"] == 2

    def test_env_overrides_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_MODEL", "/env.gguf")
        monkeypatch.setenv("SHRIKE_EMBEDDING_PORT", "9999")
        config = {"embedding": {"model": "/cfg.gguf", "port": 8500}}
        resolved = resolve_embedding(config)
        assert resolved["model"] == "/env.gguf"
        assert resolved["port"] == 9999

    def test_flag_overrides_env_and_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_MODEL", "/env.gguf")
        config = {"embedding": {"model": "/cfg.gguf"}}
        resolved = resolve_embedding(config, model="/flag.gguf")
        assert resolved["model"] == "/flag.gguf"

    def test_llama_server_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/usr/bin/llama-server")
        resolved = resolve_embedding({"embedding": {}})
        assert resolved["llama_server"] == "/usr/bin/llama-server"

    def test_no_model_resolves_none(self) -> None:
        resolved = resolve_embedding({"embedding": {}})
        assert resolved["model"] is None

    def test_paths_expanded(self, monkeypatch) -> None:
        monkeypatch.delenv("SHRIKE_EMBEDDING_MODEL", raising=False)
        resolved = resolve_embedding({"embedding": {"model": "~/m.gguf"}})
        assert not resolved["model"].startswith("~")


class TestSaveConfigEmbedding:
    def test_persists_embedding_when_model_set(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        config["embedding"]["model"] = "/m.gguf"
        config["embedding"]["threads"] = 8
        path = save_config(config, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["embedding"]["model"] == "/m.gguf"
        assert reloaded["embedding"]["threads"] == 8

    def test_omits_embedding_when_no_model(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        path = save_config(config, tmp_path / "config.yml")
        text = path.read_text()
        assert "embedding" not in text


class TestResolveCacheDir:
    """config → env → flag cascade for the index cache directory."""

    def test_default_is_none(self) -> None:
        assert resolve_cache_dir({}) is None

    def test_from_config(self) -> None:
        assert resolve_cache_dir({"cache_dir": "/data/cache"}) == "/data/cache"

    def test_env_overrides_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_CACHE_DIR", "/env/cache")
        assert resolve_cache_dir({"cache_dir": "/cfg/cache"}) == "/env/cache"

    def test_flag_overrides_env_and_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_CACHE_DIR", "/env/cache")
        assert resolve_cache_dir({"cache_dir": "/cfg"}, "/flag/cache") == "/flag/cache"

    def test_path_expanded(self) -> None:
        assert not resolve_cache_dir({"cache_dir": "~/cache"}).startswith("~")


class TestResolveIndexSave:
    """config → env → flag cascade for the debounced-flush tuning."""

    def test_default_is_none(self) -> None:
        resolved = resolve_index_save({})
        assert resolved == {"save_delay": None, "save_threshold": None}

    def test_from_config(self) -> None:
        resolved = resolve_index_save({"index": {"save_delay": 30, "save_threshold": 50}})
        assert resolved == {"save_delay": 30, "save_threshold": 50}

    def test_env_overrides_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_INDEX_SAVE_DELAY", "15")
        monkeypatch.setenv("SHRIKE_INDEX_SAVE_THRESHOLD", "200")
        resolved = resolve_index_save({"index": {"save_delay": 30, "save_threshold": 50}})
        assert resolved["save_delay"] == 15.0
        assert resolved["save_threshold"] == 200

    def test_flag_overrides_env_and_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_INDEX_SAVE_DELAY", "15")
        resolved = resolve_index_save(
            {"index": {"save_delay": 30}}, save_delay=5.0, save_threshold=10
        )
        assert resolved["save_delay"] == 5.0
        assert resolved["save_threshold"] == 10


class TestIndexArgs:
    def test_empty_when_unset(self) -> None:
        assert index_args({"save_delay": None, "save_threshold": None}) == []

    def test_emits_set_values(self) -> None:
        args = index_args({"save_delay": 30, "save_threshold": 50})
        assert args == ["--index-save-delay", "30", "--index-save-threshold", "50"]

    def test_partial(self) -> None:
        assert index_args({"save_delay": 30, "save_threshold": None}) == [
            "--index-save-delay",
            "30",
        ]


class TestSaveConfigCacheAndIndex:
    def test_persists_cache_dir_and_index(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        config["cache_dir"] = "/data/cache"
        config["index"]["save_delay"] = 30
        config["index"]["save_threshold"] = 50
        path = save_config(config, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["cache_dir"] == "/data/cache"
        assert reloaded["index"]["save_delay"] == 30
        assert reloaded["index"]["save_threshold"] == 50

    def test_omits_index_when_unset(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        path = save_config(config, tmp_path / "config.yml")
        text = path.read_text()
        assert "index" not in text
        assert "cache_dir" not in text


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch) -> None:
    """Keep resolve tests independent of the ambient environment."""
    for var in (
        "SHRIKE_EMBEDDING_MODEL",
        "SHRIKE_EMBEDDING_PORT",
        "LLAMA_SERVER_PATH",
        "SHRIKE_CACHE_DIR",
        "SHRIKE_INDEX_SAVE_DELAY",
        "SHRIKE_INDEX_SAVE_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)
