"""The NativeIndexEngine *binding* surface (#355 port from test_index_engine.py).

The engine behaviours (dedup, phantom-hit filter, persistence, calibration)
are pinned crate-side in shrike-index; what only a Python test can pin is the
binding marshaling the search/action paths rely on — the parallel-array
ranking shape, the candidate-keys restore divergence, and that two engine
instances share no state.
"""

from __future__ import annotations

import math

import pytest

shrike_native = pytest.importorskip("shrike_native")

NDIM = 8


def _vec(seed: int) -> list[float]:
    raw = [math.sin(seed * 31.0 + i * 7.0) for i in range(NDIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _engine():
    return shrike_native.NativeIndexEngine(["text", "image"])


class TestBindingShape:
    def test_instance_per_space_no_shared_state(self) -> None:
        a, b = _engine(), _engine()
        a.add("text", [1], [_vec(1)])
        assert a.size() == 1
        assert b.size() == 0

    def test_empty_index_search_yields_no_phantom_hit(self) -> None:
        # An empty USearch index can return a phantom (key 0, distance 0)
        # match; the engine filters it, so a created-but-unpopulated
        # sub-index contributes nothing to the per-query map.
        engine = _engine()
        engine.ensure("text", NDIM)
        assert engine.search_by_modality([_vec(0)], 5) == [{}]

    def test_rankings_are_parallel_id_distance_arrays(self) -> None:
        # The raw binding returns {modality: (ids, distances)} per query —
        # the exact shape KernelIndexView.search and the kernel actions zip.
        engine = _engine()
        engine.add("text", [1, 2], [_vec(1), _vec(2)])
        per_query = engine.search_by_modality([_vec(1)], 5, ["text"])[0]
        ids, distances = per_query["text"]
        assert ids[0] == 1
        assert distances[0] == pytest.approx(0.0, abs=1e-5)
        assert len(ids) == len(distances) == 2

    def test_multi_key_dedup_is_min_distance_per_note(self) -> None:
        engine = _engine()
        engine.add("text", [1], [_vec(1)])
        engine.add("image", [1, 1], [_vec(1), _vec(2)])
        ids, distances = engine.search_by_modality([_vec(1)], 5)[0]["image"]
        assert list(ids) == [1]
        assert distances[0] == pytest.approx(0.0, abs=1e-5)

    def test_remove_returns_text_index_count(self) -> None:
        engine = _engine()
        engine.add("text", [1, 2], [_vec(1), _vec(2)])
        engine.add("image", [1, 1], [_vec(3), _vec(4)])
        # 2 notes leave the text index; the image vectors go too, uncounted.
        assert engine.remove([1, 2, 999]) == 2
        assert engine.size() == 0

    def test_modalities_filter_narrows_search(self) -> None:
        engine = _engine()
        engine.add("text", [1], [_vec(1)])
        engine.add("image", [1], [_vec(2)])
        assert set(engine.search_by_modality([_vec(1)], 5, ["text"])[0]) == {"text"}


class TestBindingPersistence:
    def test_save_restore_round_trip_with_candidate_keys(self, tmp_path) -> None:
        engine = _engine()
        engine.add("text", [1, 2], [_vec(1), _vec(2)])
        engine.add("image", [1], [_vec(3)])
        engine.save(str(tmp_path))

        fresh = _engine()
        # The native restore reconstructs key maps from candidate_keys (the
        # fingerprint-sidecar note ids) — the documented #273 divergence.
        assert fresh.restore(str(tmp_path), [1, 2]) is True
        assert fresh.size() == 3
        assert fresh.ndim() == NDIM
        assert fresh.contains(1) and fresh.contains(2)
        assert dict(fresh.modality_sizes()) == {"text": 2, "image": 1}
        # modality_stats (#684): per-sub-index (size, ndim); both sub-indexes
        # restored at NDIM here, so each reports its own width.
        stats = {m: (size, ndim) for m, size, ndim in fresh.modality_stats()}
        assert stats == {"text": (2, NDIM), "image": (1, NDIM)}

    def test_restore_corrupt_file_clears_and_fails(self, tmp_path) -> None:
        (tmp_path / "index.usearch").write_bytes(b"not a usearch file")
        engine = _engine()
        assert engine.restore(str(tmp_path)) is False
        assert engine.size() == 0
        assert engine.ndim() is None
