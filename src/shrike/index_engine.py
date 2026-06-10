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
import os
import threading
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


class UsearchIndexEngine:
    """In-process :class:`~shrike.embedding_base.IndexEngine` over usearch-python.

    Thread-safe (an internal lock), but compound sequences (remove-then-add of an
    upsert batch) are the orchestrator's to make atomic — it holds its own lock
    around them, exactly as the pre-split monolith did.
    """

    def __init__(self) -> None:
        # One USearch sub-index per modality (TEXT, IMAGE), each keyed by note_id. Split because a
        # USearch hit reveals only the note_id, not which vector matched — separate indexes are the
        # only way to rank a query per modality (see search_by_modality).
        self._indexes: dict[str, Any] = {}
        self._ndim: int | None = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Total vectors across every modality sub-index (text + images)."""
        return sum(len(idx) for idx in self._indexes.values())

    @property
    def ndim(self) -> int | None:
        return self._ndim

    def modality_sizes(self) -> dict[str, int]:
        """Vector count per *loaded* modality (a created-but-empty sub-index counts, at 0)."""
        return {m: len(idx) for m, idx in self._indexes.items()}

    def ensure(self, modality: str, ndim: int) -> None:
        """Create the (empty) sub-index for a modality if it doesn't exist yet."""
        with self._lock:
            self._ensure(modality, ndim)

    def _ensure(self, modality: str, ndim: int) -> Any:
        existing = self._indexes.get(modality)
        if existing is not None:
            return existing

        from usearch.index import Index

        # "cos" is scale-invariant — it divides by vector norms per distance
        # computation, so unnormalized embeddings rank and score identically to
        # normalized ones (verified on usearch 2.25.3). That makes llama-server's
        # --embd-normalize moot here; normalization is deliberately *not* a typed
        # setting. If the metric ever changes to one that isn't scale-invariant
        # (l2sq, ip), normalization becomes vector-affecting and must be typed +
        # folded into model_id like pooling. All modality sub-indexes share one
        # ndim — text and image vectors live in the same CLIP space.
        # multi=True: a note maps to several vectors under one note_id key — its single text vector
        # in the text index, one vector per image in the image index — so remove(note_id) drops them
        # all and search dedups multi-hits back to one result per note.
        idx = Index(ndim=ndim, metric="cos", dtype="f32", multi=True)
        self._indexes[modality] = idx
        self._ndim = ndim
        logger.info("Created new %s sub-index: %d dims", modality, ndim)
        return idx

    def clear(self) -> None:
        """Drop every in-memory sub-index (file deletion is the orchestrator's)."""
        with self._lock:
            self._indexes = {}
            self._ndim = None

    def restore(self, path: str, candidate_keys: Sequence[int] | None = None) -> bool:
        """Load the per-modality sub-index files under ``path``.

        The TEXT sub-index (index.usearch) restores a pre-#201a file directly; every other
        modality's file is optional (a text-only collection simply has none). A corrupt *present*
        file clears everything and returns False — so the caller forces a full rebuild — rather
        than silently serving from a half-loaded index. ``candidate_keys`` (the orchestrator's
        hashes-sidecar note ids) is unused here — the Python binding enumerates keys natively —
        but is part of the engine surface because the Rust binding can't (#273).
        """
        from usearch.index import Index

        base = Path(path)
        loaded: dict[str, Any] = {}
        for modality in _INDEX_MODALITIES:
            file = base / modality_file_name(modality)
            if not file.exists():
                continue
            try:
                loaded[modality] = Index.restore(str(file))
            except Exception as e:
                logger.warning("Failed to load %s index from %s: %s", modality, file, e)
                with self._lock:
                    self._indexes = {}
                    self._ndim = None
                return False
        with self._lock:
            self._indexes = loaded
            text = loaded.get(TEXT)
            self._ndim = int(text.ndim) if text is not None else None
        return True

    def save(self, path: str) -> None:
        """Persist every loaded modality sub-index under ``path``.

        Also deletes a sub-index file for a modality no longer loaded (e.g. a CLIP→text-only
        backend switch rebuilt without images) so it isn't reloaded as a phantom on restart.
        """
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for modality, idx in self._indexes.items():
                idx.save(str(base / modality_file_name(modality)))
            for modality in _INDEX_MODALITIES:
                if modality not in self._indexes:
                    stale = base / modality_file_name(modality)
                    if stale.exists():
                        stale.unlink()

    def add(self, modality: str, keys: Any, vectors: Any) -> None:
        """Add vectors under i64 keys to one modality's sub-index (created on first use).

        Pure add — replace semantics (drop a note's stale vectors first) are the caller's,
        via :meth:`remove`.
        """
        keys_arr = np.asarray(keys, dtype=np.int64)
        vec_arr = np.asarray(vectors, dtype=np.float32)
        if keys_arr.size == 0:
            return
        with self._lock:
            idx = self._ensure(modality, int(vec_arr.shape[1]))
            idx.add(keys_arr, vec_arr)

    def remove(self, keys: Any) -> int:
        """Remove every vector for the given keys across all sub-indexes.

        Returns the count removed from the *text* index — one text vector per note, so that's the
        number of notes actually present (image removals are incidental). Batch remove ignores
        keys not in an index — no per-key membership check needed.
        """
        keys_arr = np.asarray(keys, dtype=np.int64)
        if keys_arr.size == 0 or not self._indexes:
            return 0
        with self._lock:
            removed = 0
            for modality, idx in self._indexes.items():
                count = int(idx.remove(keys_arr))
                if modality == TEXT:
                    removed = count
            return removed

    @staticmethod
    def _dedup_per_note(
        raw: Any, n_queries: int, top_k: int, index_size: int
    ) -> list[list[dict[str, Any]]]:
        """Reduce one sub-index's batched matches to top_k distinct notes per query (max-sim).

        ``raw`` is what ``Index.search`` returns — ``Matches`` for one query, ``BatchMatches``
        (indexable per query) for several. Matches come back nearest-first, so the first time a
        note_id appears is its best (smallest) distance — its other vectors under that key (a note's
        several images) are dropped. That min-distance-per-note *is* the max-sim-over-items
        aggregation. Truncate to top_k distinct notes.
        """
        per_query = [raw] if n_queries == 1 else [raw[i] for i in range(n_queries)]
        out: list[list[dict[str, Any]]] = []
        for matches in per_query:
            result_list: list[dict[str, Any]] = []
            seen: set[int] = set()
            for key, dist in zip(matches.keys, matches.distances, strict=True):
                nid = int(key)
                if nid in seen:
                    continue
                # An empty USearch index can return a phantom (key 0, distance 0) match — drop it.
                if nid == 0 and float(dist) == 0.0 and index_size == 0:
                    continue
                seen.add(nid)
                result_list.append({"note_id": nid, "distance": float(dist)})
                if len(result_list) >= top_k:
                    break
            out.append(result_list)
        return out

    def search_by_modality(
        self,
        query_vectors: Any,
        k: int,
        *,
        modalities: Sequence[str] | None = None,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Rank notes per modality against each query vector (max-sim per note, best-first).

        Returns one ``{modality: [{note_id, distance}, ...]}`` map per query; modalities with an
        empty (or unloaded) sub-index are omitted. ``modalities`` narrows the search (the
        text-neighbour path passes ``(TEXT,)`` so it never pays for an image search); ``None``
        searches every loaded sub-index.
        """
        q = np.asarray(query_vectors, dtype=np.float32)
        n_queries = int(q.shape[0]) if q.ndim > 1 else 1
        fetch = max(k * SEARCH_OVERFETCH, k)

        rankings: list[dict[str, list[dict[str, Any]]]] = [{} for _ in range(n_queries)]
        with self._lock:
            for modality, idx in self._indexes.items():
                if modalities is not None and modality not in modalities:
                    continue
                if len(idx) == 0:
                    continue
                # One batched search over all queries — usearch parallelises across them
                # internally, versus a Python loop of single-query searches.
                raw = idx.search(q, fetch)
                per_query = self._dedup_per_note(raw, n_queries, k, len(idx))
                for i, ranking in enumerate(per_query):
                    if ranking:
                        rankings[i][modality] = ranking
        return rankings

    def contains(self, key: int) -> bool:
        """Whether a note is indexed (every indexed note has a text vector)."""
        text_idx = self._indexes.get(TEXT)
        if text_idx is None:
            return False
        return key in text_idx

    def keys(self) -> list[int]:
        """The distinct note ids in the text sub-index."""
        text_idx = self._indexes.get(TEXT)
        if text_idx is None:
            return []
        return sorted({int(k) for k in text_idx.keys})

    def get(self, key: int) -> Any:
        """A note's stored text vector(s) (2D for a multi-vector key), or None if absent."""
        text_idx = self._indexes.get(TEXT)
        if text_idx is None:
            return None
        return text_idx.get(key)

    def calibrate_activation(
        self, sample_size: int, k: int, min_count: int
    ) -> dict[str, dict[str, float]]:
        """Per-(non-text-)modality best-match ``{n, mean, std}`` for the activation gate (#201b).

        Samples up to ``sample_size`` text vectors *already in the text sub-index* (no re-embed) as
        pseudo-queries, searches each non-text modality with each, and records the best
        (min-distance) match **excluding the pseudo-query's own note** (a note's own image isn't a
        match). Sampling is deterministic (seeded) so the calibration is stable across runs over
        the same index. A modality with fewer than ``min_count`` non-self best-matches (or an
        empty sub-index) gets no stats → gate disabled. Returns ``{}`` when there is nothing to
        calibrate (no text vectors or no non-text modality).
        """
        with self._lock:
            text_idx = self._indexes.get(TEXT)
            non_text = {m: idx for m, idx in self._indexes.items() if m != TEXT and len(idx) > 0}
            if text_idx is None or len(text_idx) == 0 or not non_text:
                return {}

            all_keys = np.array([int(key) for key in text_idx.keys], dtype=np.int64)
            rng = np.random.default_rng(0)  # deterministic → calibration is stable across runs
            sample = (
                rng.choice(all_keys, size=sample_size, replace=False)
                if len(all_keys) > sample_size
                else all_keys
            )

            # Pull the sampled text vectors straight from the index (one per note in the text idx).
            query_rows: list[np.ndarray] = []
            query_ids: list[int] = []
            for sampled in sample:
                kid = int(sampled)
                vec = text_idx.get(kid)
                if vec is None:
                    continue
                query_rows.append(np.atleast_2d(np.asarray(vec, dtype=np.float32))[0])
                query_ids.append(kid)
            if not query_rows:
                return {}
            query_array = np.array(query_rows, dtype=np.float32)

            stats: dict[str, dict[str, float]] = {}
            for modality, idx in non_text.items():
                # k>1 so a pseudo-query whose own image is the nearest hit still has a non-self hit.
                raw = idx.search(query_array, min(k, len(idx)))
                per_query = (
                    [raw] if len(query_ids) == 1 else [raw[i] for i in range(len(query_ids))]
                )
                best_sims: list[float] = []
                for qi, matches in enumerate(per_query):
                    self_id = query_ids[qi]
                    for key, dist in zip(matches.keys, matches.distances, strict=True):
                        if int(key) == self_id:
                            continue  # exclude the pseudo-query's own note (self-match)
                        best_sims.append(1.0 - float(dist))
                        break  # nearest non-self hit = this pseudo-query's best match
                if len(best_sims) >= min_count:
                    arr = np.array(best_sims, dtype=np.float64)
                    stats[modality] = {
                        "n": float(len(best_sims)),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                    }
            return stats


# ── Native engine (#273) ─────────────────────────────────────────────────────


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
        return self._rust.ndim()

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


def native_index_requested() -> bool:
    """Whether the operator opted into the native index engine (#273 bake flag).

    Selection is environment-driven for the bake (``SHRIKE_NATIVE_INDEX=1``);
    full config/CLI plumbing arrives with the default flip.
    """
    return os.environ.get("SHRIKE_NATIVE_INDEX", "").lower() in ("1", "true", "yes")


def make_index_engine() -> UsearchIndexEngine | NativeIndexEngine:
    """Build the configured engine: native when requested and installed, else Python.

    A requested-but-missing native extension degrades to the Python engine with
    a warning (never a boot failure) — the index is a derived cache; degrading
    is always safe.
    """
    if native_index_requested():
        try:
            return NativeIndexEngine()
        except ImportError:
            logger.warning(
                "SHRIKE_NATIVE_INDEX set but the shrike-native extension is not "
                "installed; using the Python index engine."
            )
    return UsearchIndexEngine()
