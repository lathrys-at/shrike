"""Tests for config defaults and embedding arg building."""

from __future__ import annotations

from shrike.cli.config import DEFAULT_CONFIG, load_config
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
    def test_no_model_returns_empty(self) -> None:
        assert _embedding_args({}) == []
        assert _embedding_args({"embedding": {}}) == []
        assert _embedding_args({"embedding": {"model": None}}) == []

    def test_model_only(self) -> None:
        args = _embedding_args({"embedding": {"model": "/m.gguf"}})
        assert args == ["--embedding-model", "/m.gguf"]

    def test_model_with_port(self) -> None:
        args = _embedding_args({"embedding": {"model": "/m.gguf", "port": 9000}})
        assert "--embedding-model" in args
        assert "--embedding-port" in args
        assert "9000" in args

    def test_all_options(self) -> None:
        config = {
            "embedding": {
                "model": "/m.gguf",
                "port": 9000,
                "context_size": 2048,
                "threads": 4,
                "gpu_layers": 33,
            }
        }
        args = _embedding_args(config)
        assert "--embedding-model" in args
        assert "--embedding-port" in args
        assert "--embedding-context-size" in args
        assert "--embedding-threads" in args
        assert "--embedding-gpu-layers" in args
        assert "/m.gguf" in args
        assert "2048" in args
        assert "4" in args
        assert "33" in args
