# NOTE (#278 cutover): the Python-engine degradation + cross-engine file-compat
# tests retired with the Python engines; the native engine is the only path.
"""Native index engine (#273): protocol conformance + on-disk round-trip.

The behavioural gate is the main suite (test_index.py / test_tools_search.py),
which runs entirely on the native engine since the #278 cutover. This file
pins what only a dedicated test can: the Rust engine satisfies the IndexEngine
protocol, and an on-disk index written by one instance loads and searches
identically under a fresh one (file persistence end to end through the facade).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from shrike.embedding_base import IndexEngine
from shrike.index import NoteEmbedInput, VectorIndex

requires_shrike_native = pytest.mark.skipif(
    importlib.util.find_spec("shrike_native") is None,
    reason="shrike_native extension not installed (scripts/build-native.sh)",
)

NDIM = 8


def _embedder() -> MagicMock:
    svc = MagicMock()
    rng = np.random.default_rng(7)
    vocab: dict[str, list[float]] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t not in vocab:
                v = rng.normal(size=NDIM)
                vocab[t] = [float(x) for x in v / np.linalg.norm(v)]
            out.append(vocab[t])
        return out

    svc.embed_texts.side_effect = embed
    svc.modalities = frozenset({"text"})
    return svc


@requires_shrike_native
class TestNativeEngine:
    def test_satisfies_protocol(self) -> None:
        from shrike.index_engine import NativeIndexEngine

        assert isinstance(NativeIndexEngine(), IndexEngine)

    def test_on_disk_index_round_trips(self, tmp_path: Path) -> None:
        emb = _embedder()
        idx = VectorIndex(tmp_path / "i", backend=emb)
        idx.rebuild(
            [NoteEmbedInput(i, t) for i, t in ((1, "alpha"), (2, "beta"), (3, "gamma"))],
            col_mod=5,
            model_id="m",
        )
        baseline = idx.search(["alpha"], top_k=3)

        reloaded = VectorIndex(tmp_path / "i", backend=emb)
        assert reloaded.size == 3
        assert reloaded.col_mod == 5
        assert reloaded.check_drift(5, "m") is False  # no rebuild on reload
        got = reloaded.search(["alpha"], top_k=3)
        assert [h["note_id"] for h in got[0]] == [h["note_id"] for h in baseline[0]]
        for a, b in zip(got[0], baseline[0], strict=True):
            assert a["distance"] == pytest.approx(b["distance"], abs=1e-5)
