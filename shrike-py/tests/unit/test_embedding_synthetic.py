"""The in-process synthetic backend (#865): a deterministic, dependency-free
embedder for benchmarks and fast tests.

Gated on the `engine-synthetic` build feature — absent from the default/server
build (and the wheel), present after `scripts/build-native.sh --synthetic`. The
config-layer gate (a release build refusing `runtime: synthetic`) is covered in
test_profiles.py without needing any build.
"""

from __future__ import annotations

import math

import pytest
import shrike_native

from shrike.harness.engines.embedding.base import IMAGE, TEXT, EmbedderBackend
from shrike.harness.engines.embedding.synthetic import DEFAULT_SYNTHETIC_DIM, SyntheticBackend

pytestmark = pytest.mark.skipif(
    "engine-synthetic" not in shrike_native.build_features(),
    reason="engine-synthetic not compiled (build: scripts/build-native.sh --synthetic)",
)


def _unit(vec: list[float]) -> bool:
    return math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-5)


def test_satisfies_the_embedder_protocol():
    assert isinstance(SyntheticBackend(), EmbedderBackend)
    assert SyntheticBackend().modalities == frozenset({TEXT, IMAGE})


def test_text_vectors_are_deterministic_distinct_and_unit_norm():
    b = SyntheticBackend()
    b.start()
    try:
        vecs = b.embed_texts(["cat", "dog", "cat"])
        assert [len(v) for v in vecs] == [DEFAULT_SYNTHETIC_DIM] * 3
        assert vecs[0] == vecs[2], "identical input → identical vector"
        assert vecs[0] != vecs[1], "distinct inputs → distinct vectors"
        assert all(_unit(v) for v in vecs)
    finally:
        b.stop()


def test_images_embed_into_the_same_width():
    b = SyntheticBackend()
    b.start()
    try:
        vecs = b.embed_images([b"\x89PNG one", b"\x89PNG two"])
        assert [len(v) for v in vecs] == [DEFAULT_SYNTHETIC_DIM] * 2
        assert vecs[0] != vecs[1]
    finally:
        b.stop()


def test_dim_and_fingerprint_are_stable():
    b = SyntheticBackend(dim=64)
    b.start()
    try:
        assert b.embedding_dim() == 64
        assert b.model_fingerprint() == "synthetic:v1:dim=64"
        assert len(b.embed_texts(["x"])[0]) == 64
    finally:
        b.stop()


def test_native_embedder_composes_for_the_kernel_slot():
    b = SyntheticBackend()
    b.start()
    try:
        assert b.native_embedder() is not None
    finally:
        b.stop()


def test_embedding_before_start_raises():
    b = SyntheticBackend()
    with pytest.raises(RuntimeError, match="not running"):
        b.embed_texts(["x"])


def test_text_only_backend_still_reports_text_modality():
    b = SyntheticBackend(modalities=frozenset({TEXT}))
    assert b.modalities == frozenset({TEXT})
