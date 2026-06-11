"""USearch-backed index engine — the storage layer under ``VectorIndex`` (#267).

This module is the *engine* half of the engine/orchestrator split: everything
that used to live under ``VectorIndex._lock`` — the per-modality USearch
sub-indexes, max-sim-per-note dedup, the empty-index phantom-hit guard, file
persistence, and the #201b activation calibration. The orchestrator
(``VectorIndex`` in ``index.py``) keeps the state machine, drift/reconcile
policy, per-note hashing, background threads, and metadata persistence.

The :class:`~shrike.embedding_base.IndexEngine` protocol this implements is
frozen as the future FFI surface (#273): coarse, batched calls trafficking only
in i64 key arrays, f32 vectors, and small JSON-able stats. The constructor is
instance-per-space with no global state — a multi-space manager (#232) is just
"make N engines".
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shrike.embedding_base import IMAGE, TEXT

logger = logging.getLogger("shrike.index")

# Search over-fetches this multiple of top_k raw vectors before deduping to distinct notes — a
# note can contribute several vectors (text + images) under one key, so the raw top-k can dedup
# to fewer notes. A small factor covers the common few-images-per-note case; a note with more
# images than this could in principle still under-fill (a real but minor recall reduction vs a
# single-vector index for image-heavy notes) — a smarter fetch bound is a rank-fusion concern,
# #180.
SEARCH_OVERFETCH = 4

# Modalities that get their own on-disk sub-index, in load order (TEXT first — it's mandatory and
# keeps the original index.usearch filename). Extend this tuple when a new media modality lands.
_INDEX_MODALITIES = (TEXT, IMAGE)


def modality_file_name(modality: str) -> str:
    """On-disk file name for a modality's sub-index.

    TEXT keeps the original ``index.usearch`` name so a pre-#201a index loads unchanged as the
    text sub-index; other modalities get ``index.<modality>.usearch`` (``index.image.usearch``).
    """
    return "index.usearch" if modality == TEXT else f"index.{modality}.usearch"


def modality_file_paths(path: str | Path) -> list[Path]:
    """Every *known* modality's sub-index file under ``path`` (whether present or not).

    Used by the orchestrator's ``clear`` so a present-but-unloaded sub-index file (e.g. after a
    corrupt-restore early-return) is unlinked too, not just the modalities currently loaded.
    """
    return [Path(path) / modality_file_name(m) for m in _INDEX_MODALITIES]


class _NativeSubIndex:
    """A read-mostly view of one modality's sub-index inside the native engine.

    Emulates the slice of the usearch-python ``Index`` surface the orchestrator's
    frozen ``_indexes`` contract exposes (len / ``in`` / ``.keys`` / ``.get``), so
    the tests that poke ``VectorIndex._indexes`` directly behave identically
    against the native engine.
    """

    def __init__(self, rust: Any, modality: str) -> None:
        self._rust = rust
        self._modality = modality

    def __len__(self) -> int:
        return int(dict(self._rust.modality_sizes()).get(self._modality, 0))

    def __contains__(self, key: int) -> bool:
        return bool(self._rust.modality_contains(self._modality, int(key)))

    @property
    def keys(self) -> list[int]:
        """All keys, one per vector (multi keys repeat) — usearch's ``.keys`` view."""
        return list(self._rust.modality_keys(self._modality))

    def get(self, key: int) -> list[list[float]] | None:
        vectors: list[list[float]] | None = self._rust.modality_get(self._modality, int(key))
        return vectors


class _NativeIndexesView(MutableMapping[str, Any]):
    """The native engine's ``_indexes`` map emulation (modality → sub-index view).

    Assignment (``idx._indexes = {TEXT: Index(...)}``) copies the given
    usearch-python indexes' vectors into the native engine — the handful of
    tests that simulate layouts this way keep working unchanged.
    """

    def __init__(self, rust: Any) -> None:
        self._rust = rust

    def __getitem__(self, modality: str) -> _NativeSubIndex:
        if modality not in self._rust.modality_names():
            raise KeyError(modality)
        return _NativeSubIndex(self._rust, modality)

    def __setitem__(self, modality: str, py_index: Any) -> None:
        self._rust.drop_modality(modality)
        ndim = int(py_index.ndim)
        self._rust.ensure(modality, ndim)
        distinct = sorted({int(k) for k in py_index.keys})
        for key in distinct:
            import numpy as _np

            vecs = _np.atleast_2d(_np.asarray(py_index.get(key), dtype=_np.float32))
            self._rust.add(modality, [key] * vecs.shape[0], vecs.tolist())

    def __delitem__(self, modality: str) -> None:
        if modality not in self._rust.modality_names():
            raise KeyError(modality)
        self._rust.drop_modality(modality)

    def __iter__(self) -> Any:
        return iter(self._rust.modality_names())

    def __len__(self) -> int:
        return len(self._rust.modality_names())


class NativeIndexEngine:
    """The Rust :class:`~shrike.embedding_base.IndexEngine` (#273), adapted.

    A thin marshaling adapter over ``shrike_native.NativeIndexEngine`` (the
    `shrike-index` crate): numpy arrays in, the protocol's dict shapes out. All
    storage, dedup, persistence, and calibration run crate-side with the GIL
    released; this class is ordinary patchable Python, per the facade rule.
    """

    def __init__(self) -> None:
        import shrike_native

        self._rust = shrike_native.NativeIndexEngine(list(_INDEX_MODALITIES))

    # The orchestrator's frozen `_indexes` contract, emulated (see the views).
    @property
    def _indexes(self) -> _NativeIndexesView:
        return _NativeIndexesView(self._rust)

    @_indexes.setter
    def _indexes(self, value: dict[str, Any]) -> None:
        self._rust.clear()
        view = _NativeIndexesView(self._rust)
        for modality, py_index in value.items():
            view[modality] = py_index

    @property
    def size(self) -> int:
        return int(self._rust.size())

    @property
    def ndim(self) -> int | None:
        # Explicit conversion: in an env without the extension (CI's lint job),
        # shrike_native resolves to Any and a bare return trips no-any-return.
        value = self._rust.ndim()
        return None if value is None else int(value)

    def modality_sizes(self) -> dict[str, int]:
        return dict(self._rust.modality_sizes())

    def ensure(self, modality: str, ndim: int) -> None:
        self._rust.ensure(modality, int(ndim))

    def clear(self) -> None:
        self._rust.clear()

    def restore(self, path: str, candidate_keys: Sequence[int] | None = None) -> bool:
        """Load sub-index files; ``candidate_keys`` (the hashes-sidecar note ids)
        reconstruct the per-key map the Rust binding can't enumerate. A
        multimodal index restored without (complete) candidates returns False —
        the standard one-time drift rebuild, never a silently wrong key map."""
        keys = [int(k) for k in candidate_keys] if candidate_keys is not None else None
        return bool(self._rust.restore(str(path), keys))

    def save(self, path: str) -> None:
        self._rust.save(str(path))

    def add(self, modality: str, keys: Any, vectors: Any) -> None:
        keys_list = [int(k) for k in np.asarray(keys, dtype=np.int64).tolist()]
        if not keys_list:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        self._rust.add(modality, keys_list, vecs.tolist())

    def remove(self, keys: Any) -> int:
        keys_list = [int(k) for k in np.asarray(keys, dtype=np.int64).tolist()]
        if not keys_list:
            return 0
        return int(self._rust.remove(keys_list))

    def search_by_modality(
        self,
        query_vectors: Any,
        k: int,
        *,
        modalities: Sequence[str] | None = None,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        q = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        raw = self._rust.search_by_modality(
            q.tolist(), int(k), list(modalities) if modalities is not None else None
        )
        out: list[dict[str, list[dict[str, Any]]]] = []
        for per_query in raw:
            ranking: dict[str, list[dict[str, Any]]] = {}
            for modality, (ids, distances) in per_query.items():
                ranking[modality] = [
                    {"note_id": int(nid), "distance": float(dist)}
                    for nid, dist in zip(ids, distances, strict=True)
                ]
            out.append(ranking)
        return out

    def contains(self, key: int) -> bool:
        return bool(self._rust.contains(int(key)))

    def keys(self) -> list[int]:
        return list(self._rust.keys())

    def get(self, key: int) -> Any:
        vecs = self._rust.get(int(key))
        return None if vecs is None else np.asarray(vecs, dtype=np.float32)

    def calibrate_activation(
        self, sample_size: int, k: int, min_count: int
    ) -> dict[str, dict[str, float]]:
        stats = self._rust.calibrate_activation(int(sample_size), int(k), int(min_count))
        return {m: {"n": n, "mean": mean, "std": std} for m, n, mean, std in stats}


def make_index_engine() -> NativeIndexEngine:
    """Build the index engine — native, unconditionally, since the #278
    cutover (the SHRIKE_NATIVE_INDEX bake flag and the Python usearch engine
    retired with it; shrike-native is a required dependency)."""
    return NativeIndexEngine()
