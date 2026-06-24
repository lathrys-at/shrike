"""Tests for the EmbeddingRuntime lifecycle in shrike.embedding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shrike.harness.engines.embedding.runtime import EmbeddingRuntime


class TestEmbeddingRuntime:
    def test_start_constructs_and_starts(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch(
            "shrike.harness.engines.embedding.runtime.LlamaServerBackend", return_value=fake_svc
        ) as ctor:
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
        with patch(
            "shrike.harness.engines.embedding.runtime.LlamaServerBackend", return_value=fake_svc
        ):
            runtime.start(model="/override.gguf")
        assert runtime.model == "/override.gguf"

    def test_start_passes_extra_args_to_service(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf", extra_args=["--flash-attn"])
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch(
            "shrike.harness.engines.embedding.runtime.LlamaServerBackend", return_value=fake_svc
        ) as ctor:
            runtime.start()
        assert ctor.call_args.kwargs["extra_args"] == ["--flash-attn"]

    def test_start_noop_if_running(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        existing = MagicMock()
        existing.running = True
        runtime._backend = existing
        with patch("shrike.harness.engines.embedding.runtime.LlamaServerBackend") as ctor:
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
            patch(
                "shrike.harness.engines.embedding.runtime.LlamaServerBackend", return_value=fake_svc
            ),
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

    # --- backend-alias normalization on the start(backend=) override ---
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
        # _make_backend does `from shrike.harness.engines.embedding.onnx import OnnxBackend`, so
        # the patch target is the source module.
        with patch("shrike.harness.engines.embedding.onnx.OnnxBackend", return_value=fake_be):
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

        with (
            patch("shrike.harness.engines.embedding.onnx.OnnxBackend", side_effect=_boom),
            pytest.raises(RuntimeError) as ei,
        ):
            runtime.start(backend="onnx-rs")
        # The failure is the real reason (model load), never the alias error.
        assert "Unknown embedding backend" not in str(ei.value)
        assert runtime.backend_kind in ("onnx", "clip", "llama", "remote")

    # --- endpoint/api_key_env are config-only; rejected as start() overrides ---
    # SSRF defense-in-depth — even a future careless POST /embedding/start body
    # that forwards these must not be able to point embedding traffic at an
    # attacker-chosen endpoint. They reach the runtime via the ctor only.

    def test_start_rejects_endpoint_override(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        with pytest.raises(ValueError, match="endpoint/api_key_env are config-only"):
            runtime.start(endpoint="http://evil.example/v1")

    def test_start_rejects_api_key_env_override(self) -> None:
        runtime = EmbeddingRuntime(model="/m.gguf")
        with pytest.raises(ValueError, match="endpoint/api_key_env are config-only"):
            runtime.start(api_key_env="STOLEN_KEY")

    def test_start_rejection_does_not_mutate_runtime(self) -> None:
        # The guard runs BEFORE the lock/mutation block, so a rejected override
        # leaves the runtime untouched (a normal start still works after).
        runtime = EmbeddingRuntime(model="/m.gguf")
        with pytest.raises(ValueError):
            runtime.start(endpoint="http://evil.example/v1")
        fake_svc = MagicMock()
        fake_svc.running = True
        with patch(
            "shrike.harness.engines.embedding.runtime.LlamaServerBackend", return_value=fake_svc
        ):
            runtime.start()
        assert runtime.running is True

    def test_start_endpoint_via_ctor_is_accepted(self) -> None:
        # The legitimate path: endpoint/api_key_env arrive at construction (the
        # config entry), NOT a start() override — so a remote runtime is fine.
        runtime = EmbeddingRuntime(backend="remote", endpoint="http://endpoint.example/v1")
        assert runtime.backend_kind == "remote"
        # No start() override of the endpoint → no rejection on a bare start
        # attempt (it proceeds to the configured-check / backend construction).
        fake_be = MagicMock()
        fake_be.running = True
        with patch("shrike.harness.engines.embedding.runtime.RemoteBackend", return_value=fake_be):
            runtime.start()
        assert runtime.running is True
