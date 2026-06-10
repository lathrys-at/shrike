"""Native index engine (#273): protocol conformance + cross-engine interop.

The behavioural parity gate is the existing suite run with SHRIKE_NATIVE_INDEX=1
(test_index.py / test_tools_search.py pass unmodified — CI's gated native lane).
This file pins what only a dedicated test can: the Rust engine satisfies the
IndexEngine protocol, and an on-disk index written by either engine loads and
searches identically under the other (the #272 compat verdict, end to end
through the facade).
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

    def test_python_written_index_loads_under_native(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        emb = _embedder()
        idx = VectorIndex(tmp_path / "i", backend=emb)
        idx.rebuild(
            [NoteEmbedInput(i, t) for i, t in ((1, "alpha"), (2, "beta"), (3, "gamma"))],
            col_mod=5,
            model_id="m",
        )
        baseline = idx.search(["alpha"], top_k=3)

        monkeypatch.setenv("SHRIKE_NATIVE_INDEX", "1")
        reloaded = VectorIndex(tmp_path / "i", backend=emb)
        from shrike.index_engine import NativeIndexEngine

        assert isinstance(reloaded._engine, NativeIndexEngine)
        assert reloaded.size == 3
        assert reloaded.col_mod == 5
        assert reloaded.check_drift(5, "m") is False  # no rebuild on engine switch
        got = reloaded.search(["alpha"], top_k=3)
        assert [h["note_id"] for h in got[0]] == [h["note_id"] for h in baseline[0]]
        for a, b in zip(got[0], baseline[0], strict=True):
            assert a["distance"] == pytest.approx(b["distance"], abs=1e-5)

    def test_native_written_index_loads_under_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        emb = _embedder()
        monkeypatch.setenv("SHRIKE_NATIVE_INDEX", "1")
        idx = VectorIndex(tmp_path / "i", backend=emb)
        idx.rebuild(
            [NoteEmbedInput(i, t) for i, t in ((1, "alpha"), (2, "beta"))],
            col_mod=9,
            model_id="m",
        )
        baseline = idx.search(["beta"], top_k=2)

        monkeypatch.delenv("SHRIKE_NATIVE_INDEX")
        from shrike.index_engine import UsearchIndexEngine

        reloaded = VectorIndex(tmp_path / "i", backend=emb)
        assert isinstance(reloaded._engine, UsearchIndexEngine)
        assert reloaded.size == 2
        assert reloaded.check_drift(9, "m") is False
        got = reloaded.search(["beta"], top_k=2)
        assert [h["note_id"] for h in got[0]] == [h["note_id"] for h in baseline[0]]

    def test_missing_extension_degrades_to_python_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shrike import index_engine

        monkeypatch.setenv("SHRIKE_NATIVE_INDEX", "1")
        monkeypatch.setattr(
            index_engine,
            "NativeIndexEngine",
            MagicMock(side_effect=ImportError("not installed")),
        )
        engine = index_engine.make_index_engine()
        assert isinstance(engine, index_engine.UsearchIndexEngine)
