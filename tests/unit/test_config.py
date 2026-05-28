"""Tests for config defaults and embedding arg building."""

from __future__ import annotations

import pytest

from shrike.cli.config import DEFAULT_CONFIG, load_config, resolve_embedding, save_config
from shrike.cli.server_cmd import _embedding_args


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


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch) -> None:
    """Keep resolve_embedding tests independent of the ambient environment."""
    for var in ("SHRIKE_EMBEDDING_MODEL", "SHRIKE_EMBEDDING_PORT", "LLAMA_SERVER_PATH"):
        monkeypatch.delenv(var, raising=False)
