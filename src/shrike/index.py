"""Vector index for semantic note search.

Wraps a USearch HNSW index with cosine similarity. Note IDs (int64) are
used directly as index keys so lookups stays O(1) and no separate ID
mapping is needed.

The index is persisted to disk and can be rebuilt from scratch if the
file is missing or corrupted. Dimensions are detected from the first
embedding and stored in metadata alongside the index.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import hashlib
import json
import logging
import threading
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# NoteEmbedInput / ImageResolver / ImageExists moved to embedding_base (#266) — the
# boundary types live with the protocol seam. Re-exported here so existing import
# sites (tests included) keep working unchanged.
from shrike.embedding_base import (
    IMAGE,
    TEXT,
    ImageExists,
    ImageResolver,
    IndexEngine,
    NoteEmbedInput,
)

# The engine/orchestrator split (#267): the storage layer (USearch sub-indexes, dedup,
# persistence, calibration) lives in index_engine; this module keeps the orchestration
# (state machine, drift/reconcile policy, note hashing, background threads, IndexSaver).
# SEARCH_OVERFETCH is re-exported for compatibility.
from shrike.index_engine import (
    SEARCH_OVERFETCH,
    make_index_engine,
    modality_file_paths,
)

if TYPE_CHECKING:
    from shrike.embedding_base import EmbedderBackend

__all__ = [
    "SEARCH_OVERFETCH",
    "ImageExists",
    "ImageResolver",
    "IndexEngine",
    "IndexSaver",
    "IndexState",
    "NoteEmbedInput",
    "VectorIndex",
    "activation_floor",
]

logger = logging.getLogger("shrike.index")

BATCH_SIZE = 64

# On-disk index schema version, stamped into index.meta.json as "schema". v1 (implicit — the marker
# absent) is the pre-#201a single-index layout: a 3c index stored a note's text + image vectors
# mixed under one note_id key in one file, which can't be split per modality after the fact. v2
# (this module) is the per-modality layout — a separate USearch sub-index file per modality. A
# text-only v1 index loads losslessly as the v2 text index (its file *is* index.usearch, holding
# only text vectors), so text-only users never rebuild on upgrade; an image-capable backend meeting
# a v1 index can't unmix it, so it rebuilds once (see check_drift / reconcile).
INDEX_SCHEMA_VERSION = 2

# Calibration searches fetch this many neighbours per pseudo-query — k>1 so a pseudo-query whose
# own image is the nearest hit still has a non-self hit to record.
_CALIB_K = 5

# Activation-gate calibration (#201b): a non-text modality's ranking only contributes to search
# fusion when its best match for a query meaningfully beats that modality's *typical* best match,
# estimated offline by sampling text vectors already in the index as pseudo-queries. CALIB_SAMPLE
# pseudo-queries are used; a modality needs at least CALIB_MIN non-self best-matches to get stats
# (otherwise the gate stays off for it — a tiny media collection isn't worth calibrating).
CALIB_SAMPLE = 256
CALIB_MIN = 30


def activation_floor(stats: dict[str, float] | None, margin: float) -> float | None:
    """Similarity a modality's best match must exceed to activate: ``mean + margin·std``.

    ``stats`` is one modality's calibrated ``{n, mean, std}`` (or ``None`` when uncalibrated — then
    there is no floor and the gate is disabled, i.e. the modality always contributes). Pure: no
    index or embedding state, so it is unit-testable in isolation and shared by the gate in
    ``tools.py`` (#201b).
    """
    if not stats:
        return None
    return stats["mean"] + margin * stats["std"]


def _hash_text(text: str) -> str:
    """Stable fingerprint of a note's embedding text.

    Used to tell, on drift, which notes' embedding text actually changed so a
    reconcile re-embeds only those. Must be process-stable (so not ``hash()``);
    BLAKE2b-64 is fast and collision-safe enough for change detection — a
    collision would at worst skip re-embedding one changed note (a stale vector),
    which the next ``model_id`` change or explicit rebuild corrects.
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


# Defaults for the debounced index flush (see IndexSaver). Persistence is
# otherwise shutdown- and rebuild-only; the flush bounds how much incremental
# work an ungraceful exit (SIGKILL, crash, power loss) discards. Correctness
# never depends on it — a col_mod mismatch on next startup still forces a full
# rebuild — but flushing shortly after edits lets a then-idle server survive a
# hard kill without re-embedding the whole collection.
DEFAULT_SAVE_DELAY = 60.0  # seconds of idle since the last change before a flush
DEFAULT_SAVE_THRESHOLD = 100  # unsaved changes that force an immediate flush


class IndexState(enum.Enum):
    READY = "ready"
    BUILDING = "building"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class VectorIndex:
    """HNSW vector index backed by USearch, stored on disk.

    The index file lives at ``{path}/index.usearch`` with a sidecar
    ``{path}/index.meta.json`` tracking the embedding dimensions and
    collection modification timestamp for drift detection.
    """

    def __init__(
        self,
        path: str | Path,
        backend: EmbedderBackend | None = None,
    ) -> None:
        self._dir = Path(path)
        self._index_path = self._dir / "index.usearch"
        self._meta_path = self._dir / "index.meta.json"
        self._hashes_path = self._dir / "index.hashes.json"
        self._embedding = backend
        self._read_image: ImageResolver | None = None
        self._image_exists_fn: ImageExists | None = None
        # The storage engine (#267): per-modality USearch sub-indexes, dedup, file persistence,
        # calibration. The orchestrator (this class) holds policy and delegates storage to it.
        # Native (shrike-index, #273) — unconditional since the #278 cutover.
        self._engine = make_index_engine()
        self._col_mod: int | None = None
        self._model_id: str | None = None
        # On-disk schema of the loaded index (INDEX_SCHEMA_VERSION for a freshly-built one). A v1
        # index loaded by an image-capable backend can't be split per modality → drift-rebuild.
        self._schema = INDEX_SCHEMA_VERSION
        # Per-(non-text-)modality best-match {n, mean, std}, calibrated offline from the index, for
        # the activation gate (#201b). Empty until calibrated (text-only collections stay empty).
        self._activation_stats: dict[str, dict[str, float]] = {}
        # Whether activation calibration has run for a multimodal index — persisted (the meta key's
        # presence) so a collection that legitimately produced *no* stats (too few media notes)
        # isn't re-sampled every boot. Distinct from `_activation_stats` being empty.
        self._calibration_attempted = False
        # note_id -> embedding-text hash for the vectors currently in the index.
        # Drives incremental reconcile (re-embed only changed notes). ``None``
        # means "no per-note state" (old index / never built) — reconcile then
        # falls back to a full rebuild. Maintained by add/remove/rebuild.
        self._note_hashes: dict[int, str] | None = None
        self._state = IndexState.UNAVAILABLE if backend is None else IndexState.READY
        self._build_progress: tuple[int, int] = (0, 0)
        self._build_error: str | None = None
        self._build_thread: threading.Thread | None = None
        self._dirty = 0  # incremental add/remove ops since the last save
        self._lock = threading.Lock()
        self._load()

    @property
    def state(self) -> IndexState:
        return self._state

    @property
    def available(self) -> bool:
        return (
            self._state == IndexState.READY
            and TEXT in self._indexes
            and self._embedding is not None
        )

    @property
    def size(self) -> int:
        """Total vectors across every modality sub-index (text + images)."""
        return self._engine.size

    @property
    def _indexes(self) -> MutableMapping[str, Any]:
        """The engine's live per-modality sub-index map (a view, for the native engine).

        Deliberately exposed (and assignable) as the pre-split attribute: a handful of unit
        tests simulate on-disk layouts by poking it directly, and that surface is part of the
        frozen ``VectorIndex`` contract. Production code goes through the engine API.
        """
        return self._engine._indexes

    @_indexes.setter
    def _indexes(self, value: dict[str, Any]) -> None:
        self._engine._indexes = value

    @property
    def ndim(self) -> int | None:
        return self._engine.ndim

    @property
    def pending_changes(self) -> int:
        """Unsaved incremental add/remove operations since the last save."""
        return self._dirty

    @property
    def col_mod(self) -> int | None:
        return self._col_mod

    @col_mod.setter
    def col_mod(self, value: int | None) -> None:
        self._col_mod = value

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def activation_stats(self) -> dict[str, dict[str, float]]:
        """Per-(non-text-)modality best-match ``{n, mean, std}`` for the activation gate (#201b).

        Empty for a text-only or uncalibrated index, in which case the gate is disabled (every
        modality always contributes). Read by ``tools.py`` via :func:`activation_floor`.
        """
        return self._activation_stats

    def set_backend(self, backend: EmbedderBackend | None) -> None:
        """Attach or detach the embedding backend at runtime.

        Detaching (``None``) marks the index ``UNAVAILABLE`` — the on-disk
        vectors are kept, but search/add can't run without an embedder.
        Attaching flips ``UNAVAILABLE`` back to ``READY`` so search can resume;
        a ``BUILDING`` or ``ERROR`` state is left alone (a rebuild, if needed,
        is the caller's job once the backend is up).
        """
        self._embedding = backend
        if backend is None:
            self._state = IndexState.UNAVAILABLE
        elif self._state == IndexState.UNAVAILABLE:
            self._state = IndexState.READY

    def set_image_resolver(
        self, resolver: ImageResolver | None, exists: ImageExists | None = None
    ) -> None:
        """Attach the media resolver the index uses to read/locate images for embedding.

        ``resolver(filename) -> bytes | None`` (``None`` = missing/unreadable → skipped) reads the
        bytes for embedding. ``exists(filename) -> bool`` is a *cheap* presence check (a stat, no
        byte read) folded into the per-note fingerprint, so the hash reflects which images actually
        resolve — that's what makes a later-stored image (note authored before the media landed)
        re-embed on reconcile instead of being skipped forever. The server closes both over
        ``CollectionWrapper.media_dir`` (lock-free). When ``exists`` is omitted, presence falls
        back to the byte read; with neither resolver, an image-capable backend embeds text only.
        """
        self._read_image = resolver
        self._image_exists_fn = exists

    def _fused_text_handles(self) -> tuple[Any, Any] | None:
        """The (native embedder, native engine) pair for the fused FFI paths (#274).

        Non-None only when *both* sides are native — the engine is the Rust
        NativeIndexEngine and the attached backend is the onnx-rs OnnxBackend —
        and the backend is text-only. Then embed→add and embed→search compose
        inside one GIL-released native call and the vectors never cross the FFI.
        Any other combination uses the regular per-side paths unchanged.
        """
        from shrike.index_engine import NativeIndexEngine

        if not isinstance(self._engine, NativeIndexEngine):
            return None
        backend = self._embedding
        native = getattr(backend, "_native_engine", None)
        if native is None or self._embeds_images():
            return None
        try:
            import shrike_native
        except ImportError:
            return None
        if not isinstance(native, shrike_native.OnnxTextEmbedder):
            return None
        return native, self._engine._rust

    def _embeds_images(self) -> bool:
        """True when the attached backend embeds images (so the index must be multi-vector)."""
        return self._embedding is not None and IMAGE in self._embedding.modalities

    def _image_exists(self, name: str) -> bool:
        """Whether an image resolves (cheaply) — drives both the hash and what gets embedded."""
        if self._image_exists_fn is not None:
            return self._image_exists_fn(name)
        if self._read_image is not None:
            return self._read_image(name) is not None
        return True  # no resolver attached → can't check; assume present (direct-hash unit tests)

    def _note_hash(self, text: str, image_names: Sequence[str]) -> str:
        """Per-note change fingerprint. Folds image filenames in **only** when the backend embeds
        images *and the image resolves* — so the hash matches what ``add`` actually embedded, and a
        full rebuild over the same notes lands on the identical state (the reconcile==rebuild
        invariant holds even for a note whose image is stored after it). A text-only backend's hash
        is byte-identical to the pre-3c text-only scheme (no spurious re-embed on upgrade). Anki
        content-addresses media (a filename is a stable content identity), so hashing the names of
        the *present* images detects add/remove/swap/late-arrival cheaply (a stat, no byte read).
        """
        if self._embeds_images():
            present = sorted(n for n in image_names if self._image_exists(n))
            if present:
                return _hash_text(f"{text}\x1f" + "\x1f".join(present))
        return _hash_text(text)

    @property
    def build_progress(self) -> tuple[int, int]:
        return self._build_progress

    def _load(self) -> None:
        """Load existing per-modality sub-indexes from disk, if present."""
        if not self._index_path.exists() or not self._meta_path.exists():
            logger.debug("No existing index at %s", self._dir)
            return

        try:
            meta = json.loads(self._meta_path.read_text())
            _ = meta["ndim"]  # ndim is mandatory — a meta without it is corrupt
            self._col_mod = meta.get("col_mod")
            self._model_id = meta.get("model_id")
            # marker absent → pre-#201a (v1) single-index layout
            self._schema = meta.get("schema", 1)
            # absent on a pre-#201b index → {} → gate disabled until ensure_calibrated/next rebuild.
            # The key's *presence* (even as {}) records that calibration already ran (one-shot).
            self._activation_stats = meta.get("activation", {})
            self._calibration_attempted = "activation" in meta
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt index metadata at %s: %s", self._meta_path, e)
            return

        # Per-note hashes are an optional sidecar: a missing or corrupt file (or
        # an index built before this existed) leaves _note_hashes None, so the
        # next reconcile safely falls back to a full rebuild. Loaded before the
        # engine restore because the native engine reconstructs its per-key map
        # from these ids (candidate_keys below).
        if self._hashes_path.exists():
            try:
                self._note_hashes = {
                    int(k): v for k, v in json.loads(self._hashes_path.read_text()).items()
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Corrupt index hashes at %s: %s", self._hashes_path, e)

        # The engine restores whatever modality files are present (a corrupt present file clears
        # everything, so the next check_drift forces a full rebuild rather than silently serving
        # from a half-loaded index) and re-derives ndim from the loaded text index.
        candidates = list(self._note_hashes) if self._note_hashes is not None else None
        if not self._engine.restore(str(self._dir), candidates):
            self._note_hashes = None  # the rebuild repopulates both together
            return

        logger.info(
            "Loaded vector index: %d vectors, %d dims (schema v%d)",
            self.size,
            self.ndim,
            self._schema,
        )

    def materialize_empty(self, ndim: int, col_mod: int, model_id: str | None) -> None:
        """Create an empty, ready index for an empty collection.

        An empty collection's index is trivially complete and current, so create
        the (zero-vector) USearch index eagerly at the model's dimension and stamp
        the current ``col_mod``/``model_id`` — rather than leaving ``_index`` None.
        That flips ``available`` to True so notes upserted *later in the same
        session* are indexed incrementally via the upsert path (which gates on
        ``available``), instead of being silently skipped until a restart (#148).

        No-op if an index already exists (drift on a non-empty-to-empty change is
        the reconcile path's job, not this one).
        """
        if self._indexes:
            return
        with self._lock:
            self._engine.ensure(TEXT, ndim)
            self._dir.mkdir(parents=True, exist_ok=True)
            if self._note_hashes is None:
                self._note_hashes = {}
            self._col_mod = col_mod
            self._model_id = model_id
            self._state = IndexState.READY
        self.save()
        logger.info("Materialized empty index (%d dims, col_mod=%d)", ndim, col_mod)

    def add(self, inputs: Sequence[NoteEmbedInput]) -> int:
        """Embed notes and add their vectors. Returns the count of notes added.

        Each note contributes a text vector and — when the backend embeds images and a resolver is
        attached — one vector per resolvable image, all under the ``note_id`` key. A note already
        in the index is replaced: its existing vectors (all of them, multi remove-by-key) are
        dropped before the fresh set is added. Image bytes are read lazily, only here.
        """
        if not self._embedding:
            raise RuntimeError("No embedding service available")
        if not inputs:
            return 0

        # Local (narrowed) resolver: non-None only when the backend embeds images and one is set.
        read_image = self._read_image if self._embeds_images() else None
        fused = self._fused_text_handles()
        added = 0
        for i in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[i : i + BATCH_SIZE]

            if fused is not None:
                # Fused native path (#274): embed→remove→add composes inside one
                # GIL-released call; the vectors never cross the FFI. The embed
                # chunk honours the backend's probed batch-safety ceiling.
                import shrike_native

                native_embedder, native_engine = fused
                chunk = getattr(self._embedding, "_effective_batch", lambda n: 1)(len(batch))
                with self._lock:
                    shrike_native.fused_add_text(
                        native_embedder,
                        native_engine,
                        TEXT,
                        [int(inp.note_id) for inp in batch],
                        [inp.text for inp in batch],
                        chunk,
                    )
                    self._dir.mkdir(parents=True, exist_ok=True)
                    if self._note_hashes is not None:
                        for inp in batch:
                            self._note_hashes[int(inp.note_id)] = self._note_hash(
                                inp.text, inp.image_names
                            )
                added += len(batch)
                continue

            note_ids = [inp.note_id for inp in batch]
            note_ids_arr = np.array(note_ids, dtype=np.int64)
            text_vecs = self._embedding.embed_texts([inp.text for inp in batch])
            text_array = np.array(text_vecs, dtype=np.float32)

            img_vecs: list[list[float]] = []
            img_keys: list[int] = []
            if read_image is not None:
                img_bytes: list[bytes] = []
                for inp in batch:
                    for name in inp.image_names:
                        data = read_image(name)
                        if data:
                            img_bytes.append(data)
                            img_keys.append(inp.note_id)
                if img_bytes:
                    img_vecs = self._embedding.embed_images(img_bytes)  # type: ignore[attr-defined]
                    # Image and text vectors share the CLIP space; guard the (latent) case where a
                    # backend's image dim differs from its text dim — np.array would raise a cryptic
                    # ragged-array error and fail the whole (background) rebuild.
                    if img_vecs and len(img_vecs[0]) != len(text_vecs[0]):
                        raise ValueError(
                            f"image embedding dim {len(img_vecs[0])} != text dim "
                            f"{len(text_vecs[0])}; backend image/text spaces must match"
                        )

            with self._lock:
                # Replace semantics: drop every existing vector for these notes first (text and
                # any stale image vectors — a re-add may have changed or dropped a note's images,
                # and the remove is unconditional so a note that lost all its images doesn't keep
                # orphaned ones), then add the fresh set.
                self._engine.remove(note_ids_arr)
                self._engine.add(TEXT, note_ids_arr, text_array)
                if img_vecs:
                    self._engine.add(
                        IMAGE,
                        np.array(img_keys, dtype=np.int64),
                        np.array(img_vecs, dtype=np.float32),
                    )
                self._dir.mkdir(parents=True, exist_ok=True)
                if self._note_hashes is not None:
                    for inp in batch:
                        self._note_hashes[int(inp.note_id)] = self._note_hash(
                            inp.text, inp.image_names
                        )
            added += len(batch)

        self._dirty += added
        logger.debug("Added %d notes to index (total vectors: %d)", added, self.size)
        return added

    def remove(self, note_ids: list[int]) -> int:
        """Remove every vector (text + images) for the given note IDs across all sub-indexes.

        Returns the count removed from the *text* index — one text vector per note, so that's the
        number of notes actually present (image removals are incidental).
        """
        if not self._indexes or not note_ids:
            return 0

        ids_arr = np.array(note_ids, dtype=np.int64)
        with self._lock:
            # Batch remove returns the count actually removed and ignores ids not in the index —
            # no per-id membership check needed. The engine removes from every modality so a
            # note's images go with its text.
            removed = self._engine.remove(ids_arr)
            if self._note_hashes is not None:
                for nid in note_ids:
                    self._note_hashes.pop(int(nid), None)

        self._dirty += removed
        logger.debug("Removed %d notes from index (total vectors: %d)", removed, self.size)
        return removed

    def search(
        self,
        texts: list[str],
        top_k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """Embed query texts and return nearest **text** neighbors (text-similarity).

        Returns one result list per query text, each a dict with ``note_id`` (int) and ``distance``
        (float, 0 = identical). Backed by the text sub-index only — this is the text-neighbour path
        (upsert similar-note suggestions); per-modality retrieval is :meth:`search_by_modality`.
        """
        if not self.available or not texts:
            return [[] for _ in texts]

        fused = self._fused_text_handles()
        if fused is not None:
            # No orchestrator lock here: the fused call embeds inside it, and
            # holding the lock across an embed would serialize searches against
            # adds for the embed's whole duration (the non-fused path embeds
            # outside the lock for the same reason). The engine's internal lock
            # keeps the index read consistent.
            rankings = self._fused_search(fused, texts, top_k, modalities=(TEXT,))
            return [r.get(TEXT, []) for r in rankings]

        vectors = self._embedding.embed_texts(texts)  # type: ignore[union-attr]
        query_array = np.array(vectors, dtype=np.float32)

        with self._lock:
            # The engine reads its sub-index map under its own lock, so a background rebuild
            # clearing the engine between the lock-free `available` check / embed_texts
            # round-trip above and here degrades to empty results, never a KeyError.
            rankings = self._engine.search_by_modality(query_array, top_k, modalities=(TEXT,))
        return [r.get(TEXT, []) for r in rankings]

    def _fused_search(
        self,
        handles: tuple[Any, Any],
        texts: list[str],
        top_k: int,
        *,
        modalities: tuple[str, ...] | None = None,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Embed + rank in one GIL-released native call (#274); engine-locked internally."""
        import shrike_native

        native_embedder, native_engine = handles
        raw = shrike_native.fused_search_text(
            native_embedder,
            native_engine,
            texts,
            top_k,
            list(modalities) if modalities is not None else None,
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

    def search_by_modality(
        self,
        texts: list[str],
        top_k: int = 10,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Embed each query once and rank notes **per modality** against it.

        Returns one entry per query: a ``{modality: [{note_id, distance}, ...]}`` map, each list a
        max-sim-per-note ranking (best-first) over that modality's sub-index. The query is always
        text — searching the image sub-index compares the text query vector against image vectors in
        the shared CLIP space (cross-modal retrieval). A text query that matches a card's *image*
        thus surfaces in ``ranking[i]["image"]``; the modality gap (image distances sit a constant
        offset above text ones) is harmless here because the caller feeds each modality into RRF as
        its own rank-based signal (#180/#201). Modalities with an empty sub-index are omitted.
        """
        if not self.available or not texts:
            return [{} for _ in texts]

        fused = self._fused_text_handles()
        if fused is not None:
            # Lock-free like search() above — see the comment there.
            return self._fused_search(fused, texts, top_k)

        vectors = self._embedding.embed_texts(texts)  # type: ignore[union-attr]
        query_array = np.array(vectors, dtype=np.float32)

        with self._lock:
            return self._engine.search_by_modality(query_array, top_k)

    def _calibrate_activation(self) -> None:
        """Recompute per-(non-text-)modality best-match statistics for the activation gate (#201b).

        Samples up to ``CALIB_SAMPLE`` text vectors *already in the text sub-index* (no re-embed) as
        pseudo-queries, searches each non-text modality with each, and records the best
        (min-distance) match **excluding the pseudo-query's own note** (a note's own image isn't a
        match). The ``{n, mean, std}`` of those best-match similarities per modality is what the
        gate compares a real query's best match against. Sampling is deterministic (seeded) so the
        calibration is stable across runs over the same index. A modality with fewer than
        ``CALIB_MIN`` non-self best-matches (or an empty sub-index) gets no stats → gate disabled.
        """
        with self._lock:
            sizes = self._engine.modality_sizes()
            has_text = sizes.get(TEXT, 0) > 0
            has_non_text = any(m != TEXT and n > 0 for m, n in sizes.items())
            if not has_text or not has_non_text:
                self._activation_stats = {}
                return  # text-only / empty → nothing to calibrate; don't mark attempted
            # A genuine multimodal calibration ran (even if it ends up below CALIB_MIN): record it
            # so ensure_calibrated treats this as one-shot rather than re-sampling every boot.
            self._calibration_attempted = True
            self._activation_stats = self._engine.calibrate_activation(
                CALIB_SAMPLE, _CALIB_K, CALIB_MIN
            )
        logger.info("Calibrated activation stats: %s", self._activation_stats or "none")

    def ensure_calibrated(self) -> None:
        """Calibrate the activation gate if a non-text modality has vectors but no stats yet.

        A no-op when already calibrated or when there's no non-text modality (text-only). Lets a
        pre-#201b index — loaded without ``activation`` in its meta — turn the gate on without
        waiting for the next drift rebuild. Cheap (a sample of HNSW searches); safe to call at boot.
        """
        sizes = self._engine.modality_sizes()
        has_non_text = any(m != TEXT and n > 0 for m, n in sizes.items())
        if not has_non_text or self._calibration_attempted:
            return
        self._calibrate_activation()
        if self._calibration_attempted:  # persist the attempt (stats or the {} marker) → one-shot
            self.save()

    def contains(self, note_id: int) -> bool:
        """Check if a note ID is in the index (every indexed note has a text vector)."""
        return self._engine.contains(note_id)

    def save(self) -> None:
        """Persist every modality sub-index and the shared metadata to disk."""
        if not self._indexes or self.ndim is None:
            return

        self._dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"ndim": self.ndim, "schema": INDEX_SCHEMA_VERSION}
        if self._col_mod is not None:
            meta["col_mod"] = self._col_mod
        if self._model_id is not None:
            meta["model_id"] = self._model_id
        if self._calibration_attempted:  # write the key (stats, or {} marker) once calibration ran
            meta["activation"] = self._activation_stats
        # Whole save under the lock so concurrent callers (a debounced save on a
        # worker thread, the signal-handler save, /embedding/stop, a manual
        # /index/save) can't interleave a sub-index write with the metadata write
        # and leave a torn meta.json. The files are small; the lock is brief.
        with self._lock:
            # The engine writes the sub-index files (and deletes a stale one for a modality no
            # longer present — e.g. a CLIP→text-only backend switch rebuilt without images — so
            # it isn't reloaded as a phantom on restart).
            self._engine.save(str(self._dir))
            self._meta_path.write_text(json.dumps(meta))
            if self._note_hashes is not None:
                self._hashes_path.write_text(json.dumps(self._note_hashes))
            self._schema = INDEX_SCHEMA_VERSION
            self._dirty = 0
        logger.debug("Saved vector index: %d vectors to %s", self.size, self._dir)

    def clear(self) -> None:
        """Remove all vectors and delete the index files."""
        with self._lock:
            self._engine.clear()
            self._model_id = None
            self._note_hashes = None
        # Unlink every *known* modality's file (like the engine's save()), not just the ones
        # currently loaded — a present-but-unloaded sub-index file (e.g. a corrupt-restore
        # early-return in _load left the engine empty) would otherwise survive and reload as a
        # phantom on the next startup. modality_file_paths covers index.usearch (TEXT) too.
        paths = modality_file_paths(self._dir) + [self._meta_path, self._hashes_path]
        for path in paths:
            if path.exists():
                path.unlink()
        logger.info("Cleared vector index at %s", self._dir)

    def check_drift(self, current_col_mod: int, model_id: str | None = None) -> bool:
        """Compare collection mod timestamp and embedding model against stored values.

        Returns True if the index is stale and a rebuild is needed. A change in
        the embedding model (``model_id``) invalidates every vector — the old
        embeddings live in a different space — so it forces a full rebuild.
        """
        if not self._indexes:
            logger.info("No index loaded, rebuild needed")
            return True

        if self._col_mod is None:
            logger.info("No col_mod in index metadata, rebuild needed")
            return True

        if model_id is not None and self._model_id != model_id:
            logger.info(
                "Embedding model changed: stored model_id=%s, current=%s; rebuild needed",
                self._model_id,
                model_id,
            )
            return True

        if self._embeds_images() and self._schema < INDEX_SCHEMA_VERSION:
            # An image-capable backend attached to a pre-#201a (v1) index: a v1 CLIP index stored a
            # note's text + image vectors mixed under one key in one file, which can't be split per
            # modality after the fact — rebuild once into the v2 per-modality layout. (A v1
            # *text-only* index never reaches here: a text-only backend doesn't embed images.)
            logger.info(
                "Backend embeds images but index predates per-modality layout; rebuild needed"
            )
            return True

        if self._col_mod != current_col_mod:
            logger.info(
                "Index drift detected: stored col_mod=%d, current=%d",
                self._col_mod,
                current_col_mod,
            )
            return True

        logger.info("Index up to date (col_mod=%d)", current_col_mod)
        return False

    def rebuild(
        self,
        inputs: Sequence[NoteEmbedInput],
        col_mod: int,
        *,
        model_id: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Full rebuild: clear the index and re-embed all notes (text + images)."""
        total = len(inputs)
        self._state = IndexState.BUILDING
        self._build_progress = (0, total)
        self._build_error = None

        logger.info("Starting full index rebuild: %d notes", total)

        try:
            with self._lock:
                self._engine.clear()
                self._schema = INDEX_SCHEMA_VERSION  # rebuilds always land on the current layout
                # Start fresh — add() repopulates per-note hashes as it embeds.
                self._note_hashes = {}

            indexed = 0
            for i in range(0, total, BATCH_SIZE):
                batch = inputs[i : i + BATCH_SIZE]
                self.add(batch)
                indexed += len(batch)
                self._build_progress = (indexed, total)
                if on_progress:
                    on_progress(indexed, total)

            self._col_mod = col_mod
            if model_id is not None:
                self._model_id = model_id
            self._calibrate_activation()  # stats land in the same meta write as the vectors
            self.save()
            self._state = IndexState.READY
            logger.info("Index rebuild complete: %d vectors, %d dims", self.size, self.ndim or 0)
        except Exception as e:
            self._state = IndexState.ERROR
            self._build_error = str(e)
            logger.error("Index rebuild failed: %s", e)
            raise

    def reconcile(
        self,
        inputs: Sequence[NoteEmbedInput],
        col_mod: int,
        *,
        model_id: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Incrementally bring the index up to date on drift.

        Re-embeds only the notes whose embedding fingerprint changed (text, or — for an
        image-capable backend — the note's image set) plus new notes, removes deleted ones, and
        leaves every unchanged vector in place. The end state is identical to a full rebuild over
        the same notes.

        Falls back to ``rebuild`` when there's no prior per-note hash state (an index built before
        this existed, or never built), no index is loaded, the model changed (every vector moves to
        a different space), or an image-capable backend meets a pre-#201a (single-index) layout that
        can't be split per modality.
        """
        old = self._note_hashes
        model_changed = model_id is not None and self._model_id != model_id
        needs_restructure = self._embeds_images() and self._schema < INDEX_SCHEMA_VERSION
        if old is None or not self._indexes or model_changed or needs_restructure:
            self.rebuild(inputs, col_mod, model_id=model_id, on_progress=on_progress)
            return

        new_hashes = {
            int(inp.note_id): self._note_hash(inp.text, inp.image_names) for inp in inputs
        }
        input_by_id = {int(inp.note_id): inp for inp in inputs}
        to_embed = [nid for nid, h in new_hashes.items() if old.get(nid) != h]
        to_remove = [nid for nid in old if nid not in new_hashes]

        if not to_embed and not to_remove:
            # A non-embedding edit (tags/deck/template) bumped col.mod without
            # changing any note's embedding fingerprint: just advance the watermark.
            # The modality distributions are unchanged, so only calibrate if it never has been (a
            # pre-#201b index reaching us via a metadata-only drift, which would otherwise persist a
            # fresh col_mod with no stats and leave the gate off for the session).
            self._col_mod = col_mod
            if not self._calibration_attempted:
                self._calibrate_activation()
            self.save()
            logger.info("Index reconcile: no embedding changes (col_mod=%d)", col_mod)
            return

        logger.info(
            "Index reconcile: re-embed %d, remove %d (of %d notes; a full rebuild would embed %d)",
            len(to_embed),
            len(to_remove),
            len(new_hashes),
            len(new_hashes),
        )
        self._state = IndexState.BUILDING
        self._build_progress = (0, len(to_embed))
        self._build_error = None
        try:
            if to_remove:
                self.remove(to_remove)
            if to_embed:
                self.add([input_by_id[nid] for nid in to_embed])
            # add/remove already maintain hashes incrementally; set the full
            # current state as the authoritative record before saving.
            self._note_hashes = new_hashes
            self._col_mod = col_mod
            if model_id is not None:
                self._model_id = model_id
            self._calibrate_activation()  # the changed set may have shifted a modality distribution
            self.save()
            self._state = IndexState.READY
            logger.info("Index reconcile complete: %d vectors", self.size)
        except Exception as e:
            self._state = IndexState.ERROR
            self._build_error = str(e)
            logger.error("Index reconcile failed: %s", e)
            raise

    def reconcile_in_background(
        self,
        inputs: Sequence[NoteEmbedInput],
        col_mod: int,
        *,
        model_id: str | None = None,
    ) -> None:
        """Start an incremental reconcile in a background thread."""
        if self._state == IndexState.BUILDING:
            logger.warning("Rebuild already in progress, skipping")
            return

        def _run() -> None:
            with contextlib.suppress(Exception):
                self.reconcile(inputs, col_mod, model_id=model_id)

        self._build_thread = threading.Thread(target=_run, name="index-reconcile", daemon=True)
        self._build_thread.start()
        logger.info("Background index reconcile started (%d notes)", len(inputs))

    def rebuild_in_background(
        self,
        inputs: Sequence[NoteEmbedInput],
        col_mod: int,
        *,
        model_id: str | None = None,
    ) -> None:
        """Start a full rebuild in a background thread."""
        if self._state == IndexState.BUILDING:
            logger.warning("Rebuild already in progress, skipping")
            return

        def _run() -> None:
            # rebuild() already logs the failure and records IndexState.ERROR;
            # suppress here only to keep the daemon thread from dumping a
            # traceback for an error that's already been handled and surfaced.
            with contextlib.suppress(Exception):
                self.rebuild(inputs, col_mod, model_id=model_id)

        self._build_thread = threading.Thread(target=_run, name="index-rebuild", daemon=True)
        self._build_thread.start()
        logger.info("Background index rebuild started (%d notes)", len(inputs))

    def status(self) -> dict[str, Any]:
        """Return index status for diagnostics."""
        info: dict[str, Any] = {
            "state": self._state.value,
            "available": self.available,
            "size": self.size,
            "ndim": self.ndim,
            "path": str(self._dir),
        }
        if self._col_mod is not None:
            info["col_mod"] = self._col_mod
        if self._model_id is not None:
            info["model_id"] = self._model_id
        if self._activation_stats:
            info["activation"] = self._activation_stats
        if self._state == IndexState.BUILDING:
            indexed, total = self._build_progress
            info["progress"] = {"indexed": indexed, "total": total}
        if self._state == IndexState.ERROR and self._build_error:
            info["error"] = self._build_error
        return info


class IndexSaver:
    """Debounced, off-thread persistence for a :class:`VectorIndex`.

    Incremental edits don't each hit the disk. Callers invoke
    :meth:`request_save` after every change, and the index is flushed:

    - ``delay`` seconds after the *last* change (idle debounce) — so a handful
      of edits followed by a pause are persisted promptly without forcing a
      write on every single edit, and
    - immediately once ``threshold`` unsaved changes accumulate (a burst cap),
      so sustained churn that never goes idle can't grow the unsaved delta —
      and the matching re-embed window on a crash — without bound.

    Lives on the event loop (uses ``loop.call_later``), but the disk write runs
    in a worker thread so the loop never blocks on I/O. ``request_save`` and the
    timer callback must run on the loop thread; ``aclose`` is the shutdown path.
    """

    def __init__(
        self,
        index: VectorIndex,
        *,
        delay: float = DEFAULT_SAVE_DELAY,
        threshold: int = DEFAULT_SAVE_THRESHOLD,
    ) -> None:
        self._index = index
        self._delay = delay
        self._threshold = threshold
        self._handle: asyncio.TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None

    def request_save(self) -> None:
        """Note that the index changed; (re)arm the debounce, flush on burst.

        Cheap and non-blocking — safe to call after every incremental edit.
        """
        # Restart the idle timer: the flush fires `delay` after the last change.
        self._arm()
        # Burst cap: don't let unsaved changes pile up unbounded.
        if self._index.pending_changes >= self._threshold:
            self._flush_now()

    def _arm(self) -> None:
        self._cancel_timer()
        loop = asyncio.get_running_loop()
        self._handle = loop.call_later(self._delay, self._on_timer)

    def _on_timer(self) -> None:
        self._handle = None
        self._flush_now()

    def _flush_now(self) -> None:
        if self._index.pending_changes <= 0:
            return
        # Coalesce: if a save is already in flight, let it finish — any changes
        # made meanwhile keep pending_changes > 0 and will re-arm via the next
        # request_save (or the still-armed idle timer).
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_save())

    async def _run_save(self) -> None:
        try:
            await asyncio.to_thread(self._index.save)
        except Exception:
            logger.warning("Debounced index save failed", exc_info=True)

    def _cancel_timer(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    async def aclose(self) -> None:
        """Cancel any pending timer and flush now if dirty (graceful shutdown).

        A no-op write when nothing is pending — the on-disk index (and its
        ``col_mod``) is already current, so reload needs no rebuild.
        """
        self._cancel_timer()
        if self._task is not None and not self._task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        if self._index.pending_changes > 0:
            await asyncio.to_thread(self._index.save)
