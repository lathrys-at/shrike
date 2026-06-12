"""Parameterized conformance + parity suite over EmbedderBackend implementations (#268).

Every registered backend configuration (``backend_cases.py``) runs through the same
checks: protocol conformance, lifecycle, health shape, fingerprint stability,
dimension consistency, the #174 batch-safety contract, and vector parity. Adding a
backend implementation — including the native ones (#270/#271) — is one
``BackendCase`` entry; its acceptance gate is this suite.

Direct in-process backends, no server (the end-to-end server paths stay in
``test_backends.py``). Embedding-gated like the existing lanes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from shrike.embed_batching import BATCH_PROBE_TEXTS
from shrike.embedding_base import IMAGE, TEXT, EmbedderBackend
from tests.integration.backend_cases import (
    PARITY_TEXTS,
    BackendCase,
    cases,
    conformance_params,
)

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

# Tolerance tier for non-byte-exact runtimes (GPU/Metal kernels): far above float
# noise would be a real divergence; this matches the probe's drift reasoning.
_APPROX_ATOL = 1e-4


@pytest.fixture(scope="module", params=conformance_params())
def case_backend(request: pytest.FixtureRequest) -> Iterator[tuple[BackendCase, EmbedderBackend]]:
    """One started backend per registered case, shared across this module's tests."""
    case: BackendCase = request.param
    backend = case.make(request)
    backend.start()
    yield case, backend
    backend.stop()


def _embed(backend: EmbedderBackend, texts: list[str]) -> np.ndarray:
    return np.array(backend.embed_texts(texts), dtype=np.float32)


class TestConformance:
    """The protocol surface every implementation must satisfy."""

    def test_satisfies_protocol(self, case_backend: tuple[BackendCase, EmbedderBackend]) -> None:
        _, backend = case_backend
        assert isinstance(backend, EmbedderBackend)

    def test_running_and_health(self, case_backend: tuple[BackendCase, EmbedderBackend]) -> None:
        _, backend = case_backend
        assert backend.running is True
        health = backend.health()
        assert isinstance(health, dict)
        assert health["available"] is True

    def test_modalities_cover_text(self, case_backend: tuple[BackendCase, EmbedderBackend]) -> None:
        _, backend = case_backend
        assert isinstance(backend.modalities, frozenset)
        assert TEXT in backend.modalities

    def test_embedding_dim_consistent(
        self, case_backend: tuple[BackendCase, EmbedderBackend]
    ) -> None:
        case, backend = case_backend
        vecs = _embed(backend, ["dimension probe"])
        assert vecs.shape == (1, case.ndim)
        assert backend.embedding_dim() == case.ndim

    def test_fingerprint_namespaced_and_nonempty(
        self, case_backend: tuple[BackendCase, EmbedderBackend]
    ) -> None:
        case, backend = case_backend
        fp = backend.model_fingerprint()
        assert fp
        assert fp.startswith(case.fingerprint_prefixes)

    def test_embeds_batch_of_inputs(
        self, case_backend: tuple[BackendCase, EmbedderBackend]
    ) -> None:
        case, backend = case_backend
        vecs = backend.embed_texts(PARITY_TEXTS)
        assert len(vecs) == len(PARITY_TEXTS)
        assert all(len(v) == case.ndim for v in vecs)

    def test_image_modality_shares_text_space(
        self, case_backend: tuple[BackendCase, EmbedderBackend]
    ) -> None:
        case, backend = case_backend
        if IMAGE not in backend.modalities:
            pytest.skip(f"{case.id} is text-only")
        from PIL import Image

        ivecs = backend.embed_images(  # type: ignore[attr-defined]
            [Image.new("RGB", (64, 64), (200, 30, 30))]
        )
        assert len(ivecs) == 1
        assert len(ivecs[0]) == case.ndim


class TestLifecycle:
    """start/stop/running transitions on a fresh (non-shared) instance."""

    def test_lifecycle_round_trip(self, request: pytest.FixtureRequest) -> None:
        # One representative non-subprocess backend keeps this cheap; the shared
        # per-case fixture already proves every case starts and stops cleanly.
        if not _onnx_available():
            pytest.skip("onnxruntime not installed")
        case = next(c for c in cases() if c.id == "onnx-minilm-int8")
        backend = case.make(request)
        assert backend.running is False
        assert backend.health()["available"] is False
        backend.start()
        try:
            assert backend.running is True
        finally:
            backend.stop()
        assert backend.running is False


def _onnx_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("onnxruntime") is not None
        and importlib.util.find_spec("tokenizers") is not None
    )


class TestBatchSafety:
    """The #174 contract, generic over implementations: what the probe proved is
    how the backend behaves — serial models are batch-independent, batch-safe
    models produce the same vector batched as serial."""

    def test_probe_ran_and_batching_is_consistent(
        self, case_backend: tuple[BackendCase, EmbedderBackend]
    ) -> None:
        case, backend = case_backend
        safe_batch = getattr(backend, "_safe_batch", 1)
        assert safe_batch >= 1

        text = BATCH_PROBE_TEXTS[0]
        alone = _embed(backend, [text])[0]
        batched = _embed(backend, [text, "an unrelated and deliberately long filler sentence"])[0]
        if safe_batch == 1 or case.batch_exact:
            # Serial models must be exactly batch-independent; bit-exact runtimes
            # must produce byte-identical vectors batched vs serial.
            assert np.array_equal(alone, batched)
        else:
            np.testing.assert_allclose(alone, batched, atol=_APPROX_ATOL)


class TestParity:
    """Golden parity (epic convention 7): a case is compared against its reference
    implementation — itself (a fresh instance) when ``parity_ref`` is None, the
    Python implementation it replaces otherwise. Byte-equal vectors + identical
    fingerprint is what justifies keeping a fingerprint namespace on a runtime
    swap; anything less must namespace itself."""

    def test_vectors_and_fingerprint_reproduce(
        self,
        request: pytest.FixtureRequest,
        case_backend: tuple[BackendCase, EmbedderBackend],
    ) -> None:
        case, backend = case_backend
        vecs = _embed(backend, PARITY_TEXTS)
        fingerprint = backend.model_fingerprint()

        if case.parity_ref is None and not case.restart_parity:
            # Single-instance form (#441): no second instantiation — embed the
            # corpus again on the SAME instance and re-read the fingerprint.
            # Covers determinism within the runtime; the cross-instance form is
            # tautological for these cases (see BackendCase.restart_parity).
            again = _embed(backend, PARITY_TEXTS)
            if case.restart_exact:
                assert np.array_equal(vecs, again)
            else:
                np.testing.assert_allclose(vecs, again, atol=_APPROX_ATOL)
            assert backend.model_fingerprint() == fingerprint
            return

        ref_factory = case.parity_ref or case.make
        reference = ref_factory(request)
        reference.start()
        try:
            ref_vecs = _embed(reference, PARITY_TEXTS)
            ref_fingerprint = reference.model_fingerprint()
        finally:
            reference.stop()

        if case.parity_ref is None or case.claims_reference_namespace:
            assert fingerprint == ref_fingerprint
        else:
            # A non-namespace-claiming runtime swap must NOT collide with the
            # reference's namespace (two spaces must never silently mix).
            assert fingerprint != ref_fingerprint

        if case.restart_exact and (case.parity_ref is None or case.claims_reference_namespace):
            assert np.array_equal(vecs, ref_vecs)
        else:
            np.testing.assert_allclose(vecs, ref_vecs, atol=_APPROX_ATOL)
