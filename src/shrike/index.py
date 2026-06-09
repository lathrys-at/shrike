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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from shrike.embedding_base import IMAGE, TEXT

if TYPE_CHECKING:
    from shrike.embedding_base import EmbedderBackend

logger = logging.getLogger("shrike.index")

BATCH_SIZE = 64
# Search over-fetches this multiple of top_k raw vectors before deduping to distinct notes — a
# note can contribute several vectors (text + images) under one key, so the raw top-k can dedup
# to fewer notes. A small factor covers the common few-images-per-note case; a note with more
# images than this could in principle still under-fill (a real but minor recall reduction vs a
# single-vector index for image-heavy notes) — a smarter fetch bound is a rank-fusion concern,
# #180.
SEARCH_OVERFETCH = 4

# On-disk index schema version, stamped into index.meta.json as "schema". v1 (implicit — the marker
# absent) is the pre-#201a single-index layout: a 3c index stored a note's text + image vectors
# mixed under one note_id key in one file, which can't be split per modality after the fact. v2
# (this module) is the per-modality layout — a separate USearch sub-index file per modality. A
# text-only v1 index loads losslessly as the v2 text index (its file *is* index.usearch, holding
# only text vectors), so text-only users never rebuild on upgrade; an image-capable backend meeting
# a v1 index can't unmix it, so it rebuilds once (see check_drift / reconcile).
INDEX_SCHEMA_VERSION = 2

# Modalities that get their own on-disk sub-index, in load order (TEXT first — it's mandatory and
# keeps the original index.usearch filename). Extend this tuple when a new media modality lands.
_INDEX_MODALITIES = (TEXT, IMAGE)

# An image resolver maps a media filename to its bytes (None = missing/unreadable → skipped).
ImageResolver = Callable[[str], "bytes | None"]
# A cheap presence check for a media filename (a stat, not a byte read) — folded into the hash.
ImageExists = Callable[[str], bool]


@dataclass(frozen=True)
class NoteEmbedInput:
    """One note's embedding inputs: its normalized text plus any image filenames.

    Produced by ``CollectionWrapper.note_embed_inputs`` (a cheap DB + regex pass, no file reads);
    the index turns it into a text vector and — for an image-capable backend — one vector per
    resolvable image, all stored under the ``note_id`` key.
    """

    note_id: int
    text: str
    image_names: list[str] = field(default_factory=list)


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
        # One USearch sub-index per modality (TEXT, IMAGE), each keyed by note_id. Split because a
        # USearch hit reveals only the note_id, not which vector matched — separate indexes are the
        # only way to rank a query per modality (see search_by_modality). The TEXT index persists as
        # index.usearch (back-compat with the pre-#201a single index); IMAGE as index.image.usearch.
        self._indexes: dict[str, Any] = {}
        self._ndim: int | None = None
        self._col_mod: int | None = None
        self._model_id: str | None = None
        # On-disk schema of the loaded index (INDEX_SCHEMA_VERSION for a freshly-built one). A v1
        # index loaded by an image-capable backend can't be split per modality → drift-rebuild.
        self._schema = INDEX_SCHEMA_VERSION
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
        return sum(len(idx) for idx in self._indexes.values())

    def _modality_path(self, modality: str) -> Path:
        """On-disk file for a modality's sub-index.

        TEXT keeps the original ``index.usearch`` name so a pre-#201a index loads unchanged as the
        text sub-index; other modalities get ``index.<modality>.usearch`` (``index.image.usearch``).
        """
        return self._index_path if modality == TEXT else self._dir / f"index.{modality}.usearch"

    @property
    def ndim(self) -> int | None:
        return self._ndim

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
            self._ndim = meta["ndim"]
            self._col_mod = meta.get("col_mod")
            self._model_id = meta.get("model_id")
            # marker absent → pre-#201a (v1) single-index layout
            self._schema = meta.get("schema", 1)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt index metadata at %s: %s", self._meta_path, e)
            return

        from usearch.index import Index

        # The TEXT sub-index (index.usearch) is mandatory — a pre-#201a file restores here directly.
        # Every other modality's file is optional: a text-only collection simply has none. A corrupt
        # *present* file clears everything (so the next check_drift forces a full rebuild) rather
        # than silently serving from a half-loaded index.
        for modality in _INDEX_MODALITIES:
            path = self._modality_path(modality)
            if not path.exists():
                continue
            try:
                self._indexes[modality] = Index.restore(str(path))
            except Exception as e:
                logger.warning("Failed to load %s index from %s: %s", modality, path, e)
                self._indexes = {}
                self._ndim = None
                return

        logger.info(
            "Loaded vector index: %d vectors, %d dims (schema v%d)",
            self.size,
            self._ndim,
            self._schema,
        )

        # Per-note hashes are an optional sidecar: a missing or corrupt file (or
        # an index built before this existed) leaves _note_hashes None, so the
        # next reconcile safely falls back to a full rebuild.
        if self._hashes_path.exists():
            try:
                self._note_hashes = {
                    int(k): v for k, v in json.loads(self._hashes_path.read_text()).items()
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Corrupt index hashes at %s: %s", self._hashes_path, e)

    def _ensure_index(self, ndim: int, modality: str) -> Any:
        """Create the USearch sub-index for a modality if it doesn't exist yet."""
        existing = self._indexes.get(modality)
        if existing is not None:
            return existing

        from usearch.index import Index

        self._ndim = ndim
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
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created new %s sub-index: %d dims", modality, ndim)
        return idx

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
            self._ensure_index(ndim, TEXT)
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
        added = 0
        for i in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[i : i + BATCH_SIZE]

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
                ndim = text_array.shape[1]
                tidx = self._ensure_index(ndim, TEXT)
                tidx.remove(note_ids_arr)
                tidx.add(note_ids_arr, text_array)
                # Drop any stale image vectors for these notes first — a re-add may have changed or
                # dropped a note's images — then add the fresh set. The remove is unconditional so a
                # note that lost all its images doesn't keep orphaned image vectors.
                existing_img = self._indexes.get(IMAGE)
                if existing_img is not None:
                    existing_img.remove(note_ids_arr)
                if img_vecs:
                    iidx = self._ensure_index(ndim, IMAGE)
                    iidx.add(
                        np.array(img_keys, dtype=np.int64),
                        np.array(img_vecs, dtype=np.float32),
                    )
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
            # no per-id membership check needed. Remove from every modality so a note's images go
            # with its text.
            removed = 0
            for modality, idx in self._indexes.items():
                count = int(idx.remove(ids_arr))
                if modality == TEXT:
                    removed = count
            if self._note_hashes is not None:
                for nid in note_ids:
                    self._note_hashes.pop(int(nid), None)

        self._dirty += removed
        logger.debug("Removed %d notes from index (total vectors: %d)", removed, self.size)
        return removed

    def _dedup_per_note(
        self, raw: Any, n_queries: int, top_k: int, index_size: int
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

        vectors = self._embedding.embed_texts(texts)  # type: ignore[union-attr]
        query_array = np.array(vectors, dtype=np.float32)

        # Over-fetch so the dedup-to-distinct-notes (a note can match via several of its vectors)
        # still yields up to top_k notes; usearch caps k at the index size internally.
        fetch = max(top_k * SEARCH_OVERFETCH, top_k)

        text_idx = self._indexes[TEXT]
        with self._lock:
            # One batched search over all queries — usearch parallelises across them internally,
            # versus a Python loop of single-query searches.
            raw = text_idx.search(query_array, fetch)
            return self._dedup_per_note(raw, len(texts), top_k, len(text_idx))

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

        vectors = self._embedding.embed_texts(texts)  # type: ignore[union-attr]
        query_array = np.array(vectors, dtype=np.float32)
        fetch = max(top_k * SEARCH_OVERFETCH, top_k)

        rankings: list[dict[str, list[dict[str, Any]]]] = [{} for _ in texts]
        with self._lock:
            for modality, idx in self._indexes.items():
                if len(idx) == 0:
                    continue
                raw = idx.search(query_array, fetch)
                per_query = self._dedup_per_note(raw, len(texts), top_k, len(idx))
                for i, ranking in enumerate(per_query):
                    if ranking:
                        rankings[i][modality] = ranking
        return rankings

    def contains(self, note_id: int) -> bool:
        """Check if a note ID is in the index (every indexed note has a text vector)."""
        text_idx = self._indexes.get(TEXT)
        if text_idx is None:
            return False
        return note_id in text_idx

    def save(self) -> None:
        """Persist every modality sub-index and the shared metadata to disk."""
        if not self._indexes or self._ndim is None:
            return

        self._dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"ndim": self._ndim, "schema": INDEX_SCHEMA_VERSION}
        if self._col_mod is not None:
            meta["col_mod"] = self._col_mod
        if self._model_id is not None:
            meta["model_id"] = self._model_id
        # Whole save under the lock so concurrent callers (a debounced save on a
        # worker thread, the signal-handler save, /embedding/stop, a manual
        # /index/save) can't interleave a sub-index write with the metadata write
        # and leave a torn meta.json. The files are small; the lock is brief.
        with self._lock:
            for modality, idx in self._indexes.items():
                idx.save(str(self._modality_path(modality)))
            # Delete a sub-index file for a modality no longer present (e.g. a CLIP→text-only
            # backend switch rebuilt without images) so it isn't reloaded as a phantom on restart.
            for modality in _INDEX_MODALITIES:
                if modality not in self._indexes:
                    stale = self._modality_path(modality)
                    if stale.exists():
                        stale.unlink()
            self._meta_path.write_text(json.dumps(meta))
            if self._note_hashes is not None:
                self._hashes_path.write_text(json.dumps(self._note_hashes))
            self._schema = INDEX_SCHEMA_VERSION
            self._dirty = 0
        logger.debug("Saved vector index: %d vectors to %s", self.size, self._dir)

    def clear(self) -> None:
        """Remove all vectors and delete the index files."""
        with self._lock:
            modalities = list(self._indexes)
            self._indexes = {}
            self._ndim = None
            self._model_id = None
            self._note_hashes = None
        paths = [self._modality_path(m) for m in modalities]
        paths += [self._index_path, self._meta_path, self._hashes_path]
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
                self._indexes = {}
                self._ndim = None
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
            self.save()
            self._state = IndexState.READY
            logger.info("Index rebuild complete: %d vectors, %d dims", self.size, self._ndim or 0)
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
            self._col_mod = col_mod
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
            "ndim": self._ndim,
            "path": str(self._dir),
        }
        if self._col_mod is not None:
            info["col_mod"] = self._col_mod
        if self._model_id is not None:
            info["model_id"] = self._model_id
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
