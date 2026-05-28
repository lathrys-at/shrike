"""Vector index for semantic note search.

Wraps a USearch HNSW index with cosine similarity. Note IDs (int64) are
used directly as index keys so lookups stays O(1) and no separate ID
mapping is needed.

The index is persisted to disk and can be rebuilt from scratch if the
file is missing or corrupted. Dimensions are detected from the first
embedding and stored in metadata alongside the index.
"""

from __future__ import annotations

import contextlib
import enum
import json
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shrike.embedding import EmbeddingService

logger = logging.getLogger("shrike.index")

BATCH_SIZE = 64


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
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._dir = Path(path)
        self._index_path = self._dir / "index.usearch"
        self._meta_path = self._dir / "index.meta.json"
        self._embedding = embedding_service
        self._index: Any | None = None
        self._ndim: int | None = None
        self._col_mod: int | None = None
        self._state = IndexState.UNAVAILABLE if embedding_service is None else IndexState.READY
        self._build_progress: tuple[int, int] = (0, 0)
        self._build_error: str | None = None
        self._build_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._load()

    @property
    def state(self) -> IndexState:
        return self._state

    @property
    def available(self) -> bool:
        return (
            self._state == IndexState.READY
            and self._index is not None
            and self._embedding is not None
        )

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return len(self._index)

    @property
    def ndim(self) -> int | None:
        return self._ndim

    @property
    def col_mod(self) -> int | None:
        return self._col_mod

    @col_mod.setter
    def col_mod(self, value: int | None) -> None:
        self._col_mod = value

    @property
    def build_progress(self) -> tuple[int, int]:
        return self._build_progress

    def _load(self) -> None:
        """Load an existing index from disk, if present."""
        if not self._index_path.exists() or not self._meta_path.exists():
            logger.debug("No existing index at %s", self._dir)
            return

        try:
            meta = json.loads(self._meta_path.read_text())
            self._ndim = meta["ndim"]
            self._col_mod = meta.get("col_mod")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt index metadata at %s: %s", self._meta_path, e)
            return

        try:
            from usearch.index import Index

            self._index = Index.restore(str(self._index_path))
            logger.info(
                "Loaded vector index: %d vectors, %d dims",
                len(self._index),  # type: ignore[arg-type]
                self._ndim,
            )
        except Exception as e:
            logger.warning("Failed to load index from %s: %s", self._index_path, e)
            self._index = None
            self._ndim = None

    def _ensure_index(self, ndim: int) -> Any:
        """Create the USearch index if it doesn't exist yet."""
        if self._index is not None:
            return self._index

        from usearch.index import Index

        self._ndim = ndim
        self._index = Index(ndim=ndim, metric="cos", dtype="f32")
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created new vector index: %d dims", ndim)
        return self._index

    def add(self, note_ids: Sequence[int], texts: Sequence[str]) -> int:
        """Embed texts and add them to the index. Returns count added."""
        if not self._embedding:
            raise RuntimeError("No embedding service available")
        if not note_ids:
            return 0
        if len(note_ids) != len(texts):
            raise ValueError("note_ids and texts must have the same length")

        added = 0
        for i in range(0, len(texts), BATCH_SIZE):
            batch_ids = note_ids[i : i + BATCH_SIZE]
            batch_texts = texts[i : i + BATCH_SIZE]

            vectors = self._embedding.embed(list(batch_texts))
            vecs_array = np.array(vectors, dtype=np.float32)
            keys_array = np.array(batch_ids, dtype=np.int64)

            with self._lock:
                idx = self._ensure_index(vecs_array.shape[1])

                for key in batch_ids:
                    if key in idx:
                        idx.remove(key)

                idx.add(keys_array, vecs_array)
            added += len(batch_ids)

        logger.debug("Added %d vectors to index (total: %d)", added, self.size)
        return added

    def remove(self, note_ids: list[int]) -> int:
        """Remove vectors by note ID. Returns count removed."""
        if self._index is None or not note_ids:
            return 0

        removed = 0
        with self._lock:
            for nid in note_ids:
                if nid in self._index:
                    self._index.remove(nid)
                    removed += 1

        logger.debug("Removed %d vectors from index (total: %d)", removed, self.size)
        return removed

    def search(
        self,
        texts: list[str],
        top_k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """Embed query texts and return nearest neighbors.

        Returns one result list per query text. Each result is a dict
        with ``note_id`` (int) and ``distance`` (float, 0 = identical).
        """
        if not self.available or not texts:
            return [[] for _ in texts]

        vectors = self._embedding.embed(texts)  # type: ignore[union-attr]
        query_array = np.array(vectors, dtype=np.float32)

        assert self._index is not None
        results: list[list[dict[str, Any]]] = []
        with self._lock:
            for vec in query_array:
                matches = self._index.search(vec, top_k)
                result_list: list[dict[str, Any]] = []
                for key, dist in zip(matches.keys, matches.distances, strict=True):
                    if int(key) == 0 and float(dist) == 0.0 and self.size == 0:
                        continue
                    result_list.append({"note_id": int(key), "distance": float(dist)})
                results.append(result_list)

        return results

    def contains(self, note_id: int) -> bool:
        """Check if a note ID is in the index."""
        if self._index is None:
            return False
        return note_id in self._index

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if self._index is None or self._ndim is None:
            return

        self._dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._index.save(str(self._index_path))
        meta: dict[str, Any] = {"ndim": self._ndim}
        if self._col_mod is not None:
            meta["col_mod"] = self._col_mod
        self._meta_path.write_text(json.dumps(meta))
        logger.debug("Saved vector index: %d vectors to %s", self.size, self._index_path)

    def clear(self) -> None:
        """Remove all vectors and delete the index files."""
        with self._lock:
            self._index = None
            self._ndim = None
        if self._index_path.exists():
            self._index_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        logger.info("Cleared vector index at %s", self._dir)

    def check_drift(self, current_col_mod: int) -> bool:
        """Compare collection mod timestamp against stored value.

        Returns True if the index is stale and a rebuild is needed.
        """
        if self._index is None:
            logger.info("No index loaded, rebuild needed")
            return True

        if self._col_mod is None:
            logger.info("No col_mod in index metadata, rebuild needed")
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
        note_ids: Sequence[int],
        texts: Sequence[str],
        col_mod: int,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Full rebuild: clear the index and re-embed all notes."""
        total = len(note_ids)
        self._state = IndexState.BUILDING
        self._build_progress = (0, total)
        self._build_error = None

        logger.info("Starting full index rebuild: %d notes", total)

        try:
            with self._lock:
                self._index = None
                self._ndim = None

            indexed = 0
            for i in range(0, total, BATCH_SIZE):
                batch_ids = note_ids[i : i + BATCH_SIZE]
                batch_texts = texts[i : i + BATCH_SIZE]
                self.add(batch_ids, batch_texts)
                indexed += len(batch_ids)
                self._build_progress = (indexed, total)
                if on_progress:
                    on_progress(indexed, total)

            self._col_mod = col_mod
            self.save()
            self._state = IndexState.READY
            logger.info("Index rebuild complete: %d vectors, %d dims", self.size, self._ndim or 0)
        except Exception as e:
            self._state = IndexState.ERROR
            self._build_error = str(e)
            logger.error("Index rebuild failed: %s", e)
            raise

    def rebuild_in_background(
        self,
        note_ids: Sequence[int],
        texts: Sequence[str],
        col_mod: int,
    ) -> None:
        """Start a full rebuild in a background thread."""
        if self._state == IndexState.BUILDING:
            logger.warning("Rebuild already in progress, skipping")
            return

        def _run() -> None:
            with contextlib.suppress(Exception):
                self.rebuild(note_ids, texts, col_mod)

        self._build_thread = threading.Thread(target=_run, name="index-rebuild", daemon=True)
        self._build_thread.start()
        logger.info("Background index rebuild started (%d notes)", len(note_ids))

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
        if self._state == IndexState.BUILDING:
            indexed, total = self._build_progress
            info["progress"] = {"indexed": indexed, "total": total}
        if self._state == IndexState.ERROR and self._build_error:
            info["error"] = self._build_error
        return info
