"""Tests for transport, cooperative-locking, and v2 server-spec config resolution."""

from __future__ import annotations

import pytest

from shrike.cli.config import (
    load_config,
    locking_args,
    resolve_locking,
    resolve_transport,
    save_config,
    transport_args,
)


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
    """Cooperative-locking resolution (config → env → flag) and arg building."""

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


class TestV2ServerSpec:
    """A v2 config rides --config to the daemon."""

    V2 = {
        "collection": "/tmp/c.anki2",
        "embedders": [{"modalities": ["text"], "runtime": "onnx", "model": "/m"}],
    }

    def test_spec_carries_config_path_and_no_embedding_flags(self) -> None:
        from shrike.cli.config import build_server_spec

        spec = build_server_spec(dict(self.V2), config_path="/etc/shrike/config.yml")
        assert spec is not None
        assert spec.config_path == "/etc/shrike/config.yml"
        assert spec.embedding_args == []

    def test_no_embedding_survives_under_v2(self) -> None:
        from shrike.cli.config import build_server_spec

        spec = build_server_spec(
            dict(self.V2), config_path="/etc/shrike/config.yml", no_embedding=True
        )
        assert spec is not None
        assert spec.embedding_args == ["--no-embedding"]

    def test_legacy_spec_has_no_config_path(self) -> None:
        from shrike.cli.config import build_server_spec

        spec = build_server_spec(
            {"collection": "/tmp/c.anki2", "embedding": {"model": "/m.gguf"}},
            config_path="/etc/shrike/config.yml",
        )
        assert spec is not None
        assert spec.config_path is None
        assert "--embedding-model" in spec.embedding_args
