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

from shrike.harness.engines.embedding.runtime import EmbeddingRuntime, RemoteBackend


class _FakeRemoteEmbedder:
    """Stands in for shrike_native.RemoteEmbedder (constructor-compatible)."""

    instances: list[_FakeRemoteEmbedder] = []
    meta: dict = {}
    ident: str | None = "served-model"
    fail_embed: Exception | None = None
    vision: bool = False
    # Router fakes (#567): a pinned-model → embedding-dim map. When a chunk is
    # embedded the vector length is taken from the INSTANCE's pinned model, so a
    # shared router serving two different-width models returns the right dim per
    # space. Empty (the default) = the legacy fixed dim-3 vector, so every
    # pre-#567 test is unchanged.
    model_dims: dict[str, int] = {}

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
        dim = type(self).model_dims.get(self.model or "", 3)
        return [[0.1] * dim for _ in texts]

    def embed_image_chunk(self, images: list[bytes]) -> list[list[float]]:
        return [[0.4, 0.5, 0.6] for _ in images]

    def vision_capable(self) -> bool:
        return type(self).vision

    def health_ok(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _fake_native(monkeypatch):
    _FakeRemoteEmbedder.instances = []
    _FakeRemoteEmbedder.meta = {}
    _FakeRemoteEmbedder.ident = "served-model"
    _FakeRemoteEmbedder.fail_embed = None
    _FakeRemoteEmbedder.vision = False
    _FakeRemoteEmbedder.model_dims = {}
    with patch("shrike.harness.engines.embedding.runtime.shrike_native") as native:
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


class TestRemoteImagePath:
    """#501B: a [text, image] remote entry serves images via the native
    dialect, gated on the endpoint actually loading vision."""

    def _entry(self, **kw):
        from shrike.harness.engines.embedding.base import IMAGE, TEXT

        return RemoteBackend(endpoint="http://e", modalities=frozenset({TEXT, IMAGE}), **kw)

    def test_image_entry_advertises_image_modality(self):
        from shrike.harness.engines.embedding.base import IMAGE

        assert IMAGE in self._entry().modalities

    def test_start_fails_when_endpoint_lacks_vision(self):
        _FakeRemoteEmbedder.vision = False
        be = self._entry()
        with pytest.raises(RuntimeError, match="does not serve image embeddings"):
            be.start()
        assert not be.running

    def test_start_succeeds_when_vision_capable(self):
        _FakeRemoteEmbedder.vision = True
        be = self._entry()
        be.start()
        assert be.running

    def test_embed_images_routes_through_the_native_dialect(self):
        _FakeRemoteEmbedder.vision = True
        be = self._entry()
        be.start()
        out = be.embed_images([b"png1", b"png2"])
        assert out == [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]]

    def test_native_embedder_composes_the_image_half(self, _fake_native):
        _FakeRemoteEmbedder.vision = True
        be = self._entry()
        be.start()
        be.native_embedder()
        _, kwargs = _fake_native.NativeEmbedder.from_remote.call_args
        assert kwargs["images"] is True

    def test_text_only_remote_has_no_image_path(self):
        be = RemoteBackend(endpoint="http://e")  # default text-only
        be.start()
        with pytest.raises(RuntimeError, match="does not serve images"):
            be.embed_images([b"x"])

    def test_text_only_native_embedder_does_not_compose_images(self, _fake_native):
        be = RemoteBackend(endpoint="http://e")
        be.start()
        be.native_embedder()
        _, kwargs = _fake_native.NativeEmbedder.from_remote.call_args
        assert kwargs["images"] is False


class TestRouterManagedRemoteBackend:
    """Router-managed remote (#567): N spaces share ONE llama.cpp router, each
    pinning its own model. The endpoint's /v1/models[0] is NOT this space's
    model, so dim + fingerprint MUST come from the pinned model — never the
    shared metadata (a dimension + vector-space collapse otherwise)."""

    def test_two_router_models_with_different_dims_each_get_their_own(self):
        # The HARD requirement: the index is built at THIS model's width. A
        # router-managed backend probes its pinned model (the returned vector
        # length is authoritative), never reads meta.n_embd from the shared
        # endpoint's data[0]. Two models of different widths on one endpoint
        # each report their own dim.
        _FakeRemoteEmbedder.model_dims = {"text-a": 384, "text-b": 768}
        # A shared-endpoint /v1/models[0] meta that would be WRONG for at least
        # one space if it were (incorrectly) consulted.
        _FakeRemoteEmbedder.meta = {"n_embd": 384}

        a = RemoteBackend(endpoint="http://127.0.0.1:8500", model="text-a", router_managed=True)
        b = RemoteBackend(endpoint="http://127.0.0.1:8500", model="text-b", router_managed=True)
        a.start()
        b.start()
        assert a.embedding_dim() == 384
        assert b.embedding_dim() == 768  # NOT 384 (the shared meta) — probed

    def test_distinct_fingerprints_from_pinned_names_on_one_endpoint(self):
        # Two router spaces share an endpoint whose /v1/models[0] meta is
        # identical for both — the `meta:` recipe would collapse them. Router
        # spaces pin remote:{pinned_model}, so the fingerprints stay distinct.
        _FakeRemoteEmbedder.meta = {"n_params": 7, "n_embd": 384, "n_vocab": 1, "size": 9}
        a = RemoteBackend(endpoint="http://127.0.0.1:8500", model="text-a", router_managed=True)
        b = RemoteBackend(endpoint="http://127.0.0.1:8500", model="text-b", router_managed=True)
        a.start()
        b.start()
        fp_a = a.model_fingerprint()
        fp_b = b.model_fingerprint()
        assert fp_a != fp_b
        assert fp_a.startswith("remote:text-a")
        assert fp_b.startswith("remote:text-b")
        # The shared meta: recipe is NOT used for a router space (it would be
        # identical across both).
        assert not fp_a.startswith("meta:")

    def test_router_space_pins_configured_model_not_data0_id(self):
        # _model_name for a router space is the configured pin, never the shared
        # endpoint's data[0] id (which names some OTHER served model).
        _FakeRemoteEmbedder.ident = "some-other-served-model"
        be = RemoteBackend(endpoint="http://127.0.0.1:8500", model="my-model", router_managed=True)
        be.start()
        assert be.health()["model"] == "my-model"
        # The pinned native client carries the configured model, not data[0].
        assert _FakeRemoteEmbedder.instances[-1].model == "my-model"

    def test_non_router_remote_still_uses_meta_recipe(self):
        # Control: a NON-router remote (an attached llama-server) keeps the
        # meta: fingerprint recipe + meta.n_embd dim — unchanged by #567.
        _FakeRemoteEmbedder.meta = {"n_params": 7, "n_embd": 512, "n_vocab": 1, "size": 9}
        be = RemoteBackend(endpoint="http://127.0.0.1:9000", model="attached")
        be.start()
        assert be.model_fingerprint().startswith("meta:7:512:1::9")
        assert be.embedding_dim() == 512  # read from meta, not probed

    def test_runtime_threads_router_managed_through_to_backend(self):
        # EmbeddingRuntime carries router_managed onto the constructed backend.
        _FakeRemoteEmbedder.meta = {"n_params": 7, "n_embd": 384, "n_vocab": 1, "size": 9}
        rt = EmbeddingRuntime(
            backend="remote",
            endpoint="http://127.0.0.1:8500",
            model="r-model",
            router_managed=True,
        )
        be = rt.start()
        # The fingerprint proves the flag reached the backend (router → pinned
        # name, not the meta: recipe the shared endpoint would otherwise give).
        assert be.model_fingerprint().startswith("remote:r-model")

    def test_router_wide_pooling_folds_into_the_fingerprint(self):
        # The router-wide pooling is vector-affecting, so it must fold into the
        # fingerprint — a pooling change rebuilds the space. Two backends on the
        # same pinned model but different pooling get DIFFERENT fingerprints.
        a = RemoteBackend(
            endpoint="http://127.0.0.1:8500", model="m", router_managed=True, pooling="last"
        )
        b = RemoteBackend(
            endpoint="http://127.0.0.1:8500", model="m", router_managed=True, pooling="mean"
        )
        a.start()
        b.start()
        assert a.model_fingerprint() != b.model_fingerprint()
        assert ":pool=last" in a.model_fingerprint()
        assert ":pool=mean" in b.model_fingerprint()

    def test_runtime_threads_router_pooling_to_the_backend(self):
        # EmbeddingRuntime(pooling=...) reaches a router-managed RemoteBackend's
        # fingerprint (the server passes the router-wide pooling here).
        rt = EmbeddingRuntime(
            backend="remote",
            endpoint="http://127.0.0.1:8500",
            model="r-model",
            router_managed=True,
            pooling="last",
        )
        be = rt.start()
        assert ":pool=last" in be.model_fingerprint()
