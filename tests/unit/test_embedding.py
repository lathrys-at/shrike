"""Tests for the shrike.embedding module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import shrike_native

from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding import EmbeddingRuntime, EmbeddingService


@pytest.fixture()
def svc(tmp_path: Path) -> EmbeddingService:
    return EmbeddingService(
        model="/fake/model.gguf",
        port=19999,
        log_dir=tmp_path / "logs",
    )


class _StubManager:
    """The native LlamaServerManager seam (#342 P4b): lifecycle logic
    (find-binary precedence, command construction, reserved-flag stripping,
    orphan reaping, stop escalation) is pinned in the Rust crate; these stubs
    pin the FACADE's delegation policy."""

    def __init__(self, *, running: bool = False, pid: int | None = None) -> None:
        self._running = running
        self._pid = pid
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self._running = True
        self._pid = self._pid or 123

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    def running(self) -> bool:
        return self._running

    def pid(self) -> int | None:
        return self._pid if self._running else None

    def passthrough_tokens(self) -> list[str]:
        return []


def _set_running(svc: EmbeddingService, pid: int = 123) -> None:
    """Make a service look like it has a live, batch-safe llama-server subprocess."""
    svc._manager = _StubManager(running=True, pid=pid)
    svc._safe_batch = 16  # as the startup probe would set for fp llama-server


class _StubClient:
    """The native RemoteEmbedder seam (#342 P4): the HTTP behaviours (index
    ordering, model pinning, auth, error mapping) are pinned in the Rust
    crate; these stubs pin the FACADE's policy around the client."""

    def __init__(
        self,
        *,
        healthy: bool = True,
        info: tuple[str | None, str] = (None, "{}"),
        vectors: list[list[float]] | None = None,
        embed_error: Exception | None = None,
        vision: bool = False,
    ) -> None:
        self.healthy = healthy
        self.info = info
        self.vectors = vectors or []
        self.embed_error = embed_error
        self.vision = vision
        self.health_calls = 0
        self.embed_calls: list[list[str]] = []

    def health_ok(self) -> bool:
        self.health_calls += 1
        return self.healthy

    def vision_capable(self) -> bool:
        return self.vision

    def model_info(self) -> tuple[str | None, str]:
        return self.info

    def embed_chunk(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        return self.vectors[: len(texts)]


class TestInit:
    def test_defaults(self) -> None:
        svc = EmbeddingService(model="/path/model.gguf")
        assert svc.url == "http://127.0.0.1:8373"
        assert svc.running is False

    def test_custom_host_port(self) -> None:
        svc = EmbeddingService(model="/m.gguf", host="0.0.0.0", port=9000)
        assert svc.url == "http://0.0.0.0:9000"

    def test_running_false_before_start(self, svc: EmbeddingService) -> None:
        assert svc.running is False


class TestStart:
    """The facade's start() = manager.start() + identity + pinned client +
    probe (the spawn/health/reap mechanics are Rust-pinned)."""

    def test_start_composes_manager_identity_and_probe(self, svc: EmbeddingService) -> None:
        manager = _StubManager()
        svc._manager = manager
        with (
            patch.object(svc, "model_info", return_value={"id": "m.gguf", "meta": {}}),
            patch("shrike.embedding.probe_max_safe_batch", return_value=16),
        ):
            svc.start()
        assert manager.start_calls == 1
        assert svc._model_name == "m.gguf"
        assert svc._remote is not None  # the model-pinned client was built
        assert svc._safe_batch == 16

    def test_start_noop_when_already_running(self, svc: EmbeddingService) -> None:
        manager = _StubManager(running=True, pid=7)
        svc._manager = manager
        svc.start()
        assert manager.start_calls == 0

    def test_probe_failure_degrades_to_serial_not_boot_failure(self, svc: EmbeddingService) -> None:
        svc._manager = _StubManager()
        with (
            patch.object(svc, "model_info", return_value={}),
            patch("shrike.embedding.probe_max_safe_batch", side_effect=RuntimeError("hiccup")),
        ):
            svc.start()
        assert svc._safe_batch == 1

    def test_stop_delegates_and_clears_the_pinned_client(self, svc: EmbeddingService) -> None:
        manager = _StubManager(running=True)
        svc._manager = manager
        svc._remote = object()
        svc.stop()
        assert manager.stop_calls == 1
        assert svc._remote is None

    def test_image_gate_failure_stops_the_spawned_server(self) -> None:
        # #501B: a managed image entry whose server loaded no vision projector
        # must NOT leave the spawned llama-server orphaned — start() stops it
        # before re-raising (a degraded boot has no later start() to reap it).
        from shrike.embedding_base import IMAGE, TEXT

        svc = EmbeddingService(model="/omni.gguf", modalities=frozenset({TEXT, IMAGE}))
        manager = _StubManager()
        svc._manager = manager
        with (
            patch(
                "shrike.embedding.shrike_native.RemoteEmbedder",
                return_value=_StubClient(vision=False),
            ),
            patch.object(svc, "model_info", return_value={"id": "omni", "meta": {}}),
            pytest.raises(RuntimeError, match="vision projector"),
        ):
            svc.start()
        assert manager.stop_calls == 1, "the orphaned server must be stopped"
        assert svc._remote is None
        assert svc.running is False


class TestHealth:
    def test_not_running(self, svc: EmbeddingService) -> None:
        result = svc.health()
        assert result == {"available": False}

    def test_running_and_healthy(self, svc: EmbeddingService) -> None:
        _set_running(svc, pid=456)
        svc._client = _StubClient(healthy=True)
        result = svc.health()

        assert result["available"] is True
        assert result["pid"] == 456
        assert result["model"] == "/fake/model.gguf"

    def test_running_but_unhealthy(self, svc: EmbeddingService) -> None:
        _set_running(svc, pid=456)
        svc._client = _StubClient(healthy=False)
        result = svc.health()

        assert result["available"] is False
        assert result["pid"] == 456


class TestEmbed:
    def test_raises_when_not_running(self, svc: EmbeddingService) -> None:
        with pytest.raises(RuntimeError, match="not running"):
            svc.embed_texts(["hello"])

    def test_returns_vectors(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        stub = _StubClient(vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        svc._remote = stub

        result = svc.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        # Batch-safe → both texts in ONE request.
        assert stub.embed_calls == [["hello", "world"]]

    def test_prefers_the_pinned_client(self, svc: EmbeddingService) -> None:
        # Once start() built the model-pinned client, embeds go through it
        # (index-ordering and pinning themselves are pinned in the Rust crate).
        _set_running(svc)
        pinned = _StubClient(vectors=[[1.0]])
        unpinned = _StubClient(vectors=[[9.9]])
        svc._remote, svc._client = pinned, unpinned

        assert svc.embed_texts(["hi"]) == [[1.0]]
        assert pinned.embed_calls and not unpinned.embed_calls

    def test_propagates_client_errors(self, svc: EmbeddingService) -> None:
        # A failed request surfaces (NativeUnavailableError from the client);
        # the facade never swallows it into fake vectors.
        _set_running(svc)
        svc._remote = _StubClient(
            embed_error=shrike_native.NativeUnavailableError("embeddings request failed")
        )
        with pytest.raises(shrike_native.NativeUnavailableError):
            svc.embed_texts(["hello"])


class TestModelInfo:
    def test_not_running_returns_empty(self, svc: EmbeddingService) -> None:
        assert svc.model_info() == {}

    def test_parses_client_model_info(self, svc: EmbeddingService) -> None:
        _set_running(svc)
        svc._client = _StubClient(info=("m.gguf", '{"n_embd": 384, "size": 100}'))
        info = svc.model_info()
        assert info["id"] == "m.gguf"
        assert info["meta"]["n_embd"] == 384

    def test_absent_identity_returns_empty(self, svc: EmbeddingService) -> None:
        # The client's graceful default — a down endpoint or one serving no
        # /v1/models — maps to the facade's {} (fingerprint falls back to file).
        _set_running(svc)
        svc._client = _StubClient(info=(None, "{}"))
        assert svc.model_info() == {}


class TestEmbeddingDim:
    _META = {"n_params": 1, "n_embd": 384, "n_vocab": 3, "n_ctx_train": 4, "size": 5}

    def test_from_meta(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.embedding_dim() == 384

    def test_probe_fallback_when_meta_missing(self, svc: EmbeddingService) -> None:
        # No n_embd in meta → probe with a tiny embed and measure the width.
        with (
            patch.object(svc, "model_info", return_value={"id": "m", "meta": {}}),
            patch.object(svc, "embed_texts", return_value=[[0.0] * 16]) as embed,
        ):
            assert svc.embedding_dim() == 16
        embed.assert_called_once()

    def test_none_when_both_routes_fail(self, svc: EmbeddingService) -> None:
        with (
            patch.object(svc, "model_info", return_value={}),
            patch.object(svc, "embed_texts", side_effect=RuntimeError("down")),
        ):
            assert svc.embedding_dim() is None


class TestModelFingerprint:
    _META = {"n_params": 1, "n_embd": 2, "n_vocab": 3, "n_ctx_train": 4, "size": 5}
    # The note-text normalization version is appended to every fingerprint.
    _TP = f":textprep={EMBED_TEXT_VERSION}"

    def test_from_meta(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_name_excluded(self, svc: EmbeddingService) -> None:
        # Same numeric meta, different name → identical fingerprint.
        with patch.object(svc, "model_info", return_value={"id": "A", "meta": self._META}):
            fp_a = svc.model_fingerprint()
        with patch.object(svc, "model_info", return_value={"id": "B", "meta": self._META}):
            fp_b = svc.model_fingerprint()
        assert fp_a == fp_b

    def test_fallback_to_file_size(self, tmp_path: Path) -> None:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"x" * 100)
        svc = EmbeddingService(model=str(model))
        with patch.object(svc, "model_info", return_value={}):
            assert svc.model_fingerprint() == "file:model.gguf:100" + self._TP

    def test_fallback_missing_file(self, svc: EmbeddingService) -> None:
        with patch.object(svc, "model_info", return_value={}):
            assert svc.model_fingerprint() == "file:model.gguf:-1" + self._TP

    def test_pooling_folded_in(self) -> None:
        svc = EmbeddingService(model="/m.gguf", pooling="last")
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5:pool=last" + self._TP

    def test_pooling_changes_fingerprint(self) -> None:
        # Different pooling on the same model → different identity → rebuild.
        mean = EmbeddingService(model="/m.gguf", pooling="mean")
        last = EmbeddingService(model="/m.gguf", pooling="last")
        with (
            patch.object(mean, "model_info", return_value={"id": "m", "meta": self._META}),
            patch.object(last, "model_info", return_value={"id": "m", "meta": self._META}),
        ):
            assert mean.model_fingerprint() != last.model_fingerprint()

    def test_unset_pooling_adds_no_pool_token(self, svc: EmbeddingService) -> None:
        # No pooling set → no pool= token (only the always-present textprep tail).
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_mmprojs_folded_in_with_size(self, tmp_path: Path) -> None:
        # #501B: the projector set is vector-affecting; folded as name:size
        # (size disambiguates two projectors sharing a basename), sorted, and
        # omitted entirely when none (a text-only fingerprint is unchanged).
        from shrike.embedding_base import IMAGE, TEXT

        proj = tmp_path / "vision.mmproj.gguf"
        proj.write_bytes(b"y" * 42)
        svc = EmbeddingService(
            model="/m.gguf", modalities=frozenset({TEXT, IMAGE}), mmprojs=[str(proj)]
        )
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert (
                svc.model_fingerprint() == "meta:1:2:3:4:5:mmproj=vision.mmproj.gguf:42" + self._TP
            )
        # No projectors → no mmproj token.
        text_only = EmbeddingService(model="/m.gguf")
        with patch.object(text_only, "model_info", return_value={"id": "m", "meta": self._META}):
            assert text_only.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_extra_args_folded_in(self) -> None:
        svc = EmbeddingService(model="/m.gguf", extra_args=["--flash-attn"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5:args=--flash-attn" + self._TP

    def test_extra_args_change_fingerprint(self) -> None:
        a = EmbeddingService(model="/m.gguf", extra_args=["--flash-attn"])
        b = EmbeddingService(model="/m.gguf", extra_args=["--ubatch-size 256"])
        with (
            patch.object(a, "model_info", return_value={"id": "m", "meta": self._META}),
            patch.object(b, "model_info", return_value={"id": "m", "meta": self._META}),
        ):
            assert a.model_fingerprint() != b.model_fingerprint()

    def test_reserved_extra_args_excluded_from_fingerprint(self) -> None:
        # A stripped reserved flag never reaches llama-server, so it must not
        # appear in the fingerprint (and thus can't force a needless rebuild).
        svc = EmbeddingService(model="/m.gguf", extra_args=["--host 0.0.0.0"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert svc.model_fingerprint() == "meta:1:2:3:4:5" + self._TP

    def test_pooling_and_extra_args_both_folded(self) -> None:
        svc = EmbeddingService(model="/m.gguf", pooling="last", extra_args=["--flash-attn"])
        with patch.object(svc, "model_info", return_value={"id": "m", "meta": self._META}):
            assert (
                svc.model_fingerprint() == "meta:1:2:3:4:5:pool=last:args=--flash-attn" + self._TP
            )


class TestEmbedModelPinning:
    def test_start_builds_the_model_pinned_client(self, svc: EmbeddingService) -> None:
        # start()'s tail caches the reported model name and constructs the
        # pinned embed client with it (the pin's wire effect — `"model"` in
        # the request body, omitted when None — is pinned in the Rust crate).
        captured: dict[str, Any] = {}

        class _CapturingCtor:
            def __init__(self, base_url: str, **kwargs: Any) -> None:
                captured["base_url"] = base_url
                captured.update(kwargs)

        svc._manager = _StubManager()
        with (
            patch.object(svc, "model_info", return_value={"id": "m.gguf", "meta": {}}),
            patch("shrike.embedding.probe_max_safe_batch", return_value=16),
            patch("shrike.embedding.shrike_native.RemoteEmbedder", _CapturingCtor),
        ):
            svc.start()
        assert captured["model"] == "m.gguf"
        assert captured["base_url"] == svc.url


class TestEmbeddingRuntime:
    def test_start_constructs_and_starts(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.LlamaServerBackend", return_value=fake_svc) as ctor:
            runtime.start()
        ctor.assert_called_once()
        fake_svc.start.assert_called_once()
        assert runtime.service is fake_svc
        assert runtime.running is True

    def test_start_no_model_raises(self) -> None:
        runtime = EmbeddingRuntime(model=None)
        with pytest.raises(ValueError, match="No embedding model"):
            runtime.start()

    def test_start_applies_override(self) -> None:
        runtime = EmbeddingRuntime(model=None)
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.LlamaServerBackend", return_value=fake_svc):
            runtime.start(model="/override.gguf")
        assert runtime.model == "/override.gguf"

    def test_start_passes_extra_args_to_service(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf", extra_args=["--flash-attn"])
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch("shrike.embedding.LlamaServerBackend", return_value=fake_svc) as ctor:
            runtime.start()
        assert ctor.call_args.kwargs["extra_args"] == ["--flash-attn"]

    def test_start_noop_if_running(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        existing = MagicMock()
        existing.running = True
        runtime._backend = existing
        with patch("shrike.embedding.LlamaServerBackend") as ctor:
            svc = runtime.start()
        ctor.assert_not_called()
        assert svc is existing

    def test_stop_stops_the_backend(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.running = True
        runtime._backend = fake_svc
        assert runtime.stop() is True
        fake_svc.stop.assert_called_once()
        assert runtime.service is None

    def test_stop_noop_if_not_running(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        assert runtime.stop() is False

    def test_health_no_service(self) -> None:
        runtime = EmbeddingRuntime()
        health = runtime.health()
        assert health["available"] is False
        assert health["state"] == "not_configured"

    def test_state_transitions(self) -> None:
        # No model → not_configured.
        assert EmbeddingRuntime().state == "not_configured"
        # Model present but not started → stopped.
        assert EmbeddingRuntime(model="/m.gguf").state == "stopped"

    def test_state_failed_after_start_error(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.start.side_effect = RuntimeError("boom")
        with (
            patch("shrike.embedding.LlamaServerBackend", return_value=fake_svc),
            pytest.raises(RuntimeError),
        ):
            runtime.start()
        assert runtime.state == "failed"

    def test_state_failed_on_construction_error(self) -> None:
        # A failure in _make_backend itself (here: an unknown backend kind) must
        # also mark the runtime failed, not leave it reporting "stopped".
        runtime = EmbeddingRuntime(backend="bogus", model="/m.gguf")
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            runtime.start()
        assert runtime.state == "failed"

    # --- #602: backend-alias normalization on the start(backend=) override ---
    # The ctor normalizes documented aliases ("onnx-rs"/"clip-rs") via
    # BACKEND_ALIASES; the start() override must do the same, or a documented
    # alias 400s on /embedding/start AND poisons _backend_kind for the daemon's
    # life (mutate-before-validate).

    def test_start_override_normalizes_documented_backend_alias(self) -> None:
        # start(backend="onnx-rs") must behave like the ctor: normalize to
        # "onnx" so it never raises "Unknown embedding backend" for a documented
        # alias. (Mock the backend so this is a pure normalization assertion.)
        runtime = EmbeddingRuntime(backend="llama", model="/m.gguf")
        fake_be = MagicMock()
        fake_be.running = True
        # _make_backend does `from shrike.embedding_onnx import OnnxBackend`, so
        # the patch target is the source module.
        with patch("shrike.embedding_onnx.OnnxBackend", return_value=fake_be):
            runtime.start(backend="onnx-rs")
        assert runtime.backend_kind == "onnx"

    def test_start_override_normalizes_alias_even_when_start_fails(self) -> None:
        # A documented alias must normalize BEFORE the kind is mutated, so even a
        # failed start leaves _backend_kind a valid kind (never the raw alias) —
        # otherwise subsequent no-override start()s also raise, bricking the
        # runtime for the daemon's life.
        runtime = EmbeddingRuntime(backend="onnx", model="/nonexistent")

        def _boom(*_a: object, **_k: object) -> MagicMock:
            raise RuntimeError("model load failed")

        with patch("shrike.embedding_onnx.OnnxBackend", side_effect=_boom):
            with pytest.raises(Exception) as ei:  # noqa: B017,PT011
                runtime.start(backend="onnx-rs")
        assert "Unknown embedding backend" not in str(ei.value)
        assert runtime.backend_kind in ("onnx", "clip", "llama", "remote")
