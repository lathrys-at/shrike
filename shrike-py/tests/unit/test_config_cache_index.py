"""Tests for cache-dir and index-save config resolution + arg building."""

from __future__ import annotations

import pytest

from shrike.cli.config import (
    index_args,
    load_config,
    resolve_cache_dir,
    resolve_index_save,
    save_config,
)


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
