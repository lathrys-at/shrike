"""RemoteBackend (#498): the unmanaged-endpoint embedding facade.

Unit tests over a faked native ``RemoteEmbedder`` — the real wire is covered
by the llama-server integration suite (an attached llama-server IS an
OpenAI-compatible endpoint). What's pinned here: the api_key_env contract
(referenced, never inline; unset → a clean start failure), the
connectivity-proof embed at start, fingerprint recipes (llama-style metadata
vs the model-name identity a cloud endpoint gets), and the runtime's
``remote`` kind wiring.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from shrike.embedding import EmbeddingRuntime, RemoteBackend


class _FakeRemoteEmbedder:
    """Stands in for shrike_native.RemoteEmbedder (constructor-compatible)."""

    instances: list[_FakeRemoteEmbedder] = []
    meta: dict = {}
    ident: str | None = "served-model"
    fail_embed: Exception | None = None

    def __init__(self, base_url: str, *, api_key: str | None = None, model: str | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        type(self).instances.append(self)

    def model_info(self) -> tuple[str | None, str]:
        return type(self).ident, json.dumps(type(self).meta)

    def embed_chunk(self, texts: list[str]) -> list[list[float]]:
        if type(self).fail_embed is not None:
            raise type(self).fail_embed
        return [[0.1, 0.2, 0.3] for _ in texts]

    def health_ok(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _fake_native(monkeypatch):
    _FakeRemoteEmbedder.instances = []
    _FakeRemoteEmbedder.meta = {}
    _FakeRemoteEmbedder.ident = "served-model"
    _FakeRemoteEmbedder.fail_embed = None
    with patch("shrike.embedding.shrike_native") as native:
        native.RemoteEmbedder = _FakeRemoteEmbedder
        yield native


class TestRemoteBackend:
    def test_start_builds_model_pinned_client_and_proves_connectivity(self):
        be = RemoteBackend(endpoint="https://api.example.com/v1/")
        be.start()
        assert be.running
        # Endpoint normalized (no trailing slash); pinned to the served model
        # when the entry names none.
        pinned = _FakeRemoteEmbedder.instances[-1]
        assert pinned.base_url == "https://api.example.com/v1"
        assert pinned.model == "served-model"

    def test_entry_model_wins_over_served_identity(self):
        be = RemoteBackend(endpoint="http://e", model="text-embedding-3-small")
        be.start()
        assert _FakeRemoteEmbedder.instances[-1].model == "text-embedding-3-small"

    def test_api_key_read_from_env_at_start(self, monkeypatch):
        monkeypatch.setenv("EXAMPLE_API_KEY", "sk-test")
        be = RemoteBackend(endpoint="http://e", api_key_env="EXAMPLE_API_KEY")
        be.start()
        assert all(i.api_key == "sk-test" for i in _FakeRemoteEmbedder.instances)

    def test_missing_api_key_env_fails_start_with_the_var_name(self, monkeypatch):
        monkeypatch.delenv("ABSENT_KEY", raising=False)
        be = RemoteBackend(endpoint="http://e", api_key_env="ABSENT_KEY")
        with pytest.raises(RuntimeError, match="ABSENT_KEY"):
            be.start()
        assert not be.running

    def test_dead_endpoint_fails_start(self):
        _FakeRemoteEmbedder.fail_embed = RuntimeError("connection refused")
        be = RemoteBackend(endpoint="http://down")
        with pytest.raises(RuntimeError, match="connection refused"):
            be.start()
        assert not be.running

    def test_fingerprint_prefers_llama_metadata(self):
        _FakeRemoteEmbedder.meta = {"n_params": 7, "n_embd": 384, "n_vocab": 1, "size": 9}
        be = RemoteBackend(endpoint="http://e")
        be.start()
        fp = be.model_fingerprint()
        assert fp.startswith("meta:7:384:1::9")
        assert ":textprep=" in fp

    def test_fingerprint_falls_back_to_model_name_not_endpoint(self):
        be = RemoteBackend(endpoint="https://api.example.com/v1", model="text-embedding-3-small")
        be.start()
        fp = be.model_fingerprint()
        # The name IS the identity for a cloud endpoint; the URL is excluded
        # (two endpoints serving one model share a vector space).
        assert fp.startswith("remote:text-embedding-3-small")
        assert "example.com" not in fp

    def test_stop_forgets_the_client_only(self):
        be = RemoteBackend(endpoint="http://e")
        be.start()
        be.stop()
        assert not be.running
        # No process management: health after stop is a plain unavailable.
        assert be.health() == {"available": False}


class TestRuntimeRemoteKind:
    def test_endpoint_alone_configures_the_runtime(self):
        rt = EmbeddingRuntime(backend="remote", endpoint="http://e")
        # No model, but endpoint-configured: state is stopped, never
        # not_configured (the endpoint's default model is a valid choice).
        assert rt.state == "stopped"

    def test_remote_kind_without_endpoint_is_not_configured(self):
        rt = EmbeddingRuntime(backend="remote")
        assert rt.state == "not_configured"
        with pytest.raises(ValueError, match="No embedding model configured"):
            rt.start()

    def test_start_constructs_remote_backend(self):
        rt = EmbeddingRuntime(backend="remote", endpoint="http://e", model="m")
        be = rt.start()
        assert isinstance(be, RemoteBackend)
        assert rt.state == "running"

    def test_failed_start_marks_failed(self, monkeypatch):
        monkeypatch.delenv("NOPE", raising=False)
        rt = EmbeddingRuntime(backend="remote", endpoint="http://e", api_key_env="NOPE")
        with pytest.raises(RuntimeError):
            rt.start()
        assert rt.state == "failed"


def test_runtime_health_reports_modalities_when_running():
    rt = EmbeddingRuntime(backend="remote", endpoint="http://e", model="m")
    assert "modalities" not in rt.health()  # down: shape-stable, no modalities
    rt.start()
    health = rt.health()
    assert health["modalities"] == ["text"]
    assert health["state"] == "running"
