"""Tests for embedding config defaults, arg building, and profile resolution."""

from __future__ import annotations

import os

import pytest

from shrike.cli.config import (
    DEFAULT_CONFIG,
    load_config,
    resolve_embedding,
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
            "pooling": "last",
            "llama_server": "/bin/llama-server",
        }
        args = _embedding_args(resolved)
        assert "--embedding-model" in args
        assert "--embedding-port" in args
        assert "--embedding-context-size" in args
        assert "--embedding-threads" in args
        assert "--embedding-gpu-layers" in args
        assert "--embedding-pooling" in args
        assert "--llama-server" in args
        assert "/m.gguf" in args
        assert "2048" in args

    def test_backend_emitted_only_when_non_llama(self) -> None:
        # None or the default "llama" emit nothing (keeps llama command lines
        # byte-identical and lets the server default apply); "onnx" is emitted.
        assert "--embedding-backend" not in _embedding_args({"model": "/m.gguf"})
        assert "--embedding-backend" not in _embedding_args(
            {"model": "/m.gguf", "backend": "llama"}
        )
        args = _embedding_args({"model": "/m.gguf", "backend": "onnx"})
        assert args[:2] == ["--embedding-backend", "onnx"]

    def test_onnx_providers_one_flag_per_value(self) -> None:
        args = _embedding_args(
            {"model": "/m.onnx", "backend": "onnx", "onnx_providers": ["CUDAExecutionProvider"]}
        )
        assert args.count("--embedding-onnx-provider") == 1
        assert "CUDAExecutionProvider" in args

    def test_pooling(self) -> None:
        args = _embedding_args({"model": "/m.gguf", "pooling": "last"})
        assert args[-2:] == ["--embedding-pooling", "last"]

    def test_no_pooling_omits_flag(self) -> None:
        assert "--embedding-pooling" not in _embedding_args({"model": "/m.gguf"})
        assert "--embedding-pooling" not in _embedding_args({"model": "/m.gguf", "pooling": None})

    def test_extra_args_one_flag_per_token(self) -> None:
        args = _embedding_args(
            {"model": "/m.gguf", "extra_args": ["--flash-attn", "--ubatch-size 256"]}
        )
        # Each raw entry is emitted as its own --embedding-arg (split happens
        # later, in the embedding service).
        assert args.count("--embedding-arg") == 2
        assert "--flash-attn" in args
        assert "--ubatch-size 256" in args

    def test_no_extra_args_emits_nothing(self) -> None:
        assert "--embedding-arg" not in _embedding_args({"model": "/m.gguf"})
        assert "--embedding-arg" not in _embedding_args({"model": "/m.gguf", "extra_args": []})

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

    def test_batch_size_resolves(self) -> None:
        assert resolve_embedding({"embedding": {"batch_size": 8}})["batch_size"] == 8
        assert resolve_embedding({"embedding": {}}, batch_size=4)["batch_size"] == 4

    @pytest.mark.parametrize("bad", [0, -5])
    def test_batch_size_below_one_rejected(self, bad: int) -> None:
        # A hand-edited config value bypasses the CLI's IntRange; reject it here too
        # (0 would otherwise be swallowed as "no cap", a negative would crash the index).
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            resolve_embedding({"embedding": {"batch_size": bad}})

    def test_no_model_resolves_none(self) -> None:
        resolved = resolve_embedding({"embedding": {}})
        assert resolved["model"] is None

    def test_backend_unset_resolves_none(self) -> None:
        # Must NOT default to "llama" here: `embedding start` only transmits
        # non-None values, so a None backend lets a running server keep the one it
        # booted with (the "llama" default is applied at the consumption sites).
        assert resolve_embedding({"embedding": {}})["backend"] is None

    def test_backend_from_flag_env_config(self, monkeypatch) -> None:
        assert resolve_embedding({"embedding": {}}, backend="onnx")["backend"] == "onnx"
        assert resolve_embedding({"embedding": {"backend": "onnx"}})["backend"] == "onnx"
        monkeypatch.setenv("SHRIKE_EMBEDDING_BACKEND", "onnx")
        assert resolve_embedding({"embedding": {}})["backend"] == "onnx"

    def test_pooling_from_config(self) -> None:
        resolved = resolve_embedding({"embedding": {"pooling": "last"}})
        assert resolved["pooling"] == "last"

    def test_pooling_env_overrides_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_POOLING", "cls")
        resolved = resolve_embedding({"embedding": {"pooling": "last"}})
        assert resolved["pooling"] == "cls"

    def test_pooling_flag_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_POOLING", "cls")
        resolved = resolve_embedding({"embedding": {"pooling": "mean"}}, pooling="last")
        assert resolved["pooling"] == "last"

    def test_pooling_defaults_none(self) -> None:
        assert resolve_embedding({"embedding": {}})["pooling"] is None

    def test_extra_args_from_config(self) -> None:
        resolved = resolve_embedding({"embedding": {"extra_args": ["--flash-attn"]}})
        assert resolved["extra_args"] == ["--flash-attn"]

    def test_extra_args_env_shlex_split(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_ARGS", "--flash-attn --ubatch-size 256")
        resolved = resolve_embedding({"embedding": {"extra_args": ["--cfg"]}})
        assert resolved["extra_args"] == ["--flash-attn", "--ubatch-size", "256"]

    def test_extra_args_flag_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_EMBEDDING_ARGS", "--env")
        resolved = resolve_embedding(
            {"embedding": {"extra_args": ["--cfg"]}}, extra_args=["--flag"]
        )
        assert resolved["extra_args"] == ["--flag"]

    def test_extra_args_defaults_empty(self) -> None:
        assert resolve_embedding({"embedding": {}})["extra_args"] == []

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

    def test_persists_pooling(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        config["embedding"]["model"] = "/m.gguf"
        config["embedding"]["pooling"] = "last"
        path = save_config(config, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["embedding"]["pooling"] == "last"

    def test_persists_extra_args(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        config["embedding"]["model"] = "/m.gguf"
        config["embedding"]["extra_args"] = ["--flash-attn", "--ubatch-size 256"]
        path = save_config(config, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["embedding"]["extra_args"] == ["--flash-attn", "--ubatch-size 256"]

    def test_omits_embedding_when_no_model(self, tmp_path) -> None:
        config = load_config(tmp_path / "none.yml")
        config["collection"] = "/c.anki2"
        path = save_config(config, tmp_path / "config.yml")
        text = path.read_text()
        assert "embedding" not in text


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch) -> None:
    """Keep resolve tests independent of the ambient environment."""
    for var in (
        "SHRIKE_EMBEDDING_MODEL",
        "SHRIKE_EMBEDDING_PORT",
        "SHRIKE_EMBEDDING_POOLING",
        "SHRIKE_EMBEDDING_ARGS",
        "LLAMA_SERVER_PATH",
        "SHRIKE_CACHE_DIR",
        "SHRIKE_INDEX_SAVE_DELAY",
        "SHRIKE_INDEX_SAVE_THRESHOLD",
        "SHRIKE_ALLOWED_HOSTS",
        "SHRIKE_ALLOWED_ORIGINS",
        "SHRIKE_NO_DNS_REBINDING_PROTECTION",
    ):
        monkeypatch.delenv(var, raising=False)


class TestEmbeddingProfileResolution:
    """A v2 config drives the legacy param shape through
    resolve_embedding_profile; legacy configs run the old cascade."""

    V2 = {
        "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "~/m", "pooling": "cls"}]
    }

    def test_v2_config_bridges_to_legacy_params(self) -> None:
        from shrike.cli.config import resolve_embedding_profile

        resolved = resolve_embedding_profile(dict(self.V2), None)
        assert resolved["backend"] == "onnx"
        assert resolved["model"] == os.path.expanduser("~/m")
        assert resolved["pooling"] == "cls"

    def test_v2_config_rejects_legacy_flags(self) -> None:
        from shrike.cli.config import resolve_embedding_profile
        from shrike.harness.profiles import ProfileError

        with pytest.raises(ProfileError, match="only home"):
            resolve_embedding_profile(dict(self.V2), {"model": "/elsewhere.gguf"})

    def test_v2_config_warns_on_ignored_env(self, monkeypatch, capsys) -> None:
        from shrike.cli.config import resolve_embedding_profile

        monkeypatch.setenv("SHRIKE_EMBEDDING_MODEL", "/ambient.gguf")
        resolved = resolve_embedding_profile(dict(self.V2), None)
        assert resolved["model"] == os.path.expanduser("~/m")  # env did NOT win
        assert "SHRIKE_EMBEDDING_MODEL" in capsys.readouterr().err

    def test_legacy_config_runs_the_old_cascade(self, capsys) -> None:
        from shrike.cli.config import resolve_embedding, resolve_embedding_profile

        legacy = {"embedding": {"model": "~/m.gguf", "pooling": "last"}}
        assert resolve_embedding_profile(dict(legacy), None) == resolve_embedding(dict(legacy))
        assert "deprecated" in capsys.readouterr().err

    def test_save_config_passes_v2_sections_through(self, tmp_path) -> None:
        cfg = {
            "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "/m"}],
            "managed": {"llama_server": {"manage": "auto"}},
        }
        path = save_config(cfg, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["embedders"] == cfg["embedders"]
        assert reloaded["managed"] == cfg["managed"]
