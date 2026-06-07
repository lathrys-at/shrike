"""Tests for config defaults and embedding arg building."""

from __future__ import annotations

import pytest

from shrike.cli.config import (
    DEFAULT_CONFIG,
    index_args,
    load_config,
    locking_args,
    resolve_cache_dir,
    resolve_embedding,
    resolve_index_save,
    resolve_locking,
    resolve_transport,
    save_config,
    transport_args,
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


class TestResolveTransport:
    def test_defaults_empty(self) -> None:
        r = resolve_transport({})
        assert r == {
            "allowed_hosts": [],
            "allowed_origins": [],
            "no_dns_rebinding_protection": False,
        }

    def test_flags_win(self) -> None:
        r = resolve_transport(
            {"server": {"allowed_hosts": ["from.config"]}},
            allowed_hosts=["from.flag"],
            no_dns_rebinding_protection=True,
        )
        assert r["allowed_hosts"] == ["from.flag"]
        assert r["no_dns_rebinding_protection"] is True

    def test_env_over_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_ALLOWED_HOSTS", "a.ts.net, b.ts.net")
        monkeypatch.setenv("SHRIKE_NO_DNS_REBINDING_PROTECTION", "true")
        r = resolve_transport({"server": {"allowed_hosts": ["cfg"]}})
        assert r["allowed_hosts"] == ["a.ts.net", "b.ts.net"]
        assert r["no_dns_rebinding_protection"] is True

    def test_config_fallback(self) -> None:
        r = resolve_transport(
            {"server": {"allowed_hosts": ["host.ts.net"], "no_dns_rebinding_protection": True}}
        )
        assert r["allowed_hosts"] == ["host.ts.net"]
        assert r["no_dns_rebinding_protection"] is True


class TestTransportArgs:
    def test_empty(self) -> None:
        assert transport_args(resolve_transport({})) == []

    def test_builds_flags(self) -> None:
        args = transport_args(
            {
                "allowed_hosts": ["h1", "h2"],
                "allowed_origins": ["https://o1"],
                "no_dns_rebinding_protection": True,
            }
        )
        assert args == [
            "--allowed-host",
            "h1",
            "--allowed-host",
            "h2",
            "--allowed-origin",
            "https://o1",
            "--no-dns-rebinding-protection",
        ]


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


class TestResolveLocking:
    """Cooperative-locking resolution (config → env → flag) and arg building (#64)."""

    def test_defaults_off(self) -> None:
        r = resolve_locking({})
        assert r["cooperative"] is False
        assert r["hold_seconds"] is None
        assert locking_args(r) == []

    def test_from_config(self) -> None:
        cfg = {"server": {"cooperative_lock": True, "lock_hold_seconds": 12.5}}
        r = resolve_locking(cfg)
        assert r["cooperative"] is True
        assert r["hold_seconds"] == 12.5
        assert locking_args(r) == ["--cooperative-lock", "--lock-hold-seconds", "12.5"]

    def test_env_overrides_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_COOPERATIVE_LOCK", "1")
        monkeypatch.setenv("SHRIKE_LOCK_HOLD_SECONDS", "3")
        r = resolve_locking({"server": {"cooperative_lock": False}})
        assert r["cooperative"] is True
        assert r["hold_seconds"] == 3.0

    def test_flag_overrides_env_and_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SHRIKE_COOPERATIVE_LOCK", "0")
        r = resolve_locking(
            {"server": {"cooperative_lock": False}}, cooperative=True, hold_seconds=7.0
        )
        assert r["cooperative"] is True
        assert r["hold_seconds"] == 7.0

    def test_cooperative_only_omits_hold_arg(self) -> None:
        assert locking_args({"cooperative": True, "hold_seconds": None}) == ["--cooperative-lock"]

    def test_save_config_persists_locking(self, tmp_path) -> None:
        cfg = {"server": {"cooperative_lock": True, "lock_hold_seconds": 8.0}}
        path = save_config(cfg, tmp_path / "config.yml")
        reloaded = load_config(path)
        assert reloaded["server"]["cooperative_lock"] is True
        assert reloaded["server"]["lock_hold_seconds"] == 8.0


def test_server_spec_includes_locking_args() -> None:
    from shrike.cli.config import build_server_spec

    spec = build_server_spec(
        {"collection": "/tmp/c.anki2"},
        locking_overrides={"cooperative": True, "hold_seconds": 5.0},
    )
    assert spec is not None
    assert "--cooperative-lock" in spec.locking_args
    assert "--lock-hold-seconds" in spec.locking_args
