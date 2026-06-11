"""The kernel-mode server core (#332 S3d-2): AsyncKernel + harness services.

This replaces ``ShrikeKernel`` for the HTTP host: the kernel (Rust) owns the
collection, the index orchestration, and the derived ingest; this module is
the *assembly* — the harness thread running the kernel's executor, the
embedding runtime attached as a registered service (#342), the derived-store
build driver, and the operational verbs behind the custom routes. Every verb
is a coroutine on the host loop (the kernel's ops are loop-driven awaitables;
only genuinely blocking work — a model load — hops to a thread).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from typing import Any

import shrike_native

from shrike.collection import CollectionWrapper, collect_derived_rows
from shrike.derived import DerivedTextStore
from shrike.embedding import EmbeddingRuntime
from shrike.embedding_base import EmbedderBackend
from shrike.kernel import KernelConfigError

logger = logging.getLogger("shrike.kernel")


class KernelIndexView:
    """The search-facing slice of the old ``VectorIndex``, live over the kernel.

    The actions' search path needs: availability/state/progress, the engine
    handle, query embedding (host-side, via the runtime's backend), and the
    activation stats — all of which the kernel owns now. This view reads them
    live (``index_status_json``) instead of holding facade copies.
    """

    def __init__(self, kernel: Any, runtime: EmbeddingRuntime) -> None:
        self._kernel = kernel
        self._runtime = runtime
        self._engine_handle = kernel.engine_handle()
        # Facade-shaped engine access: the search action extracts the native
        # handle as `index._engine._rust`, so mirror that exact attribute.
        self._engine = SimpleNamespace(_rust=self._engine_handle)

    def _status(self) -> dict[str, Any]:
        return json.loads(self._kernel.index_status_json())  # type: ignore[no-any-return]

    @property
    def state_name(self) -> str:
        return str(self._status()["state"])

    @property
    def state(self) -> Any:
        """The facade's ``IndexState`` enum, for the search action's gating."""
        from shrike.index import IndexState

        name = self.state_name
        if self._runtime.backend is None and name == "ready":
            return IndexState.UNAVAILABLE
        return IndexState(name)

    @property
    def available(self) -> bool:
        return self.state_name == "ready" and self._runtime.backend is not None

    @property
    def size(self) -> int:
        return int(self._status()["size"])

    @property
    def build_progress(self) -> tuple[int, int]:
        progress = self._status()["progress"]
        return (int(progress["indexed"]), int(progress["total"]))

    @property
    def activation_stats(self) -> dict[str, dict[str, float]]:
        return dict(self._status().get("activation") or {})

    @property
    def engine(self) -> Any:
        """The Arc-shared native engine handle (the vectors the kernel maintains)."""
        return self._engine_handle

    def embed_queries(self, texts: list[str]) -> list[list[float]] | None:
        backend = self._runtime.backend
        if backend is None or not texts:
            return None
        return backend.embed_texts(texts)

    def search(self, texts: list[str], top_k: int = 10) -> list[list[dict[str, Any]]]:
        """Nearest **text** neighbors per query (the upsert-neighbors path):
        one list per text of ``{note_id, distance}`` dicts — the old facade's
        ``search`` shape, over the kernel's engine."""
        vectors = self.embed_queries(texts)
        if vectors is None:
            return [[] for _ in texts]
        rankings = self._engine_handle.search_by_modality(vectors, top_k, ["text"])
        out: list[list[dict[str, Any]]] = []
        for per_query in rankings:
            ids, distances = per_query.get("text", ([], []))
            out.append(
                [
                    {"note_id": int(nid), "distance": float(dist)}
                    for nid, dist in zip(ids, distances, strict=True)
                ]
            )
        return out


class Harness:
    """Assembled kernel-mode server core: one ``AsyncKernel`` + the services
    the harness registers on it, plus the operational verbs the routes call."""

    def __init__(
        self,
        *,
        kernel: Any,
        executor: Any,
        wrapper: CollectionWrapper,
        runtime: EmbeddingRuntime,
        derived: DerivedTextStore,
        media_read: Any,
        media_exists: Any,
    ) -> None:
        self.kernel = kernel
        self._executor = executor
        self.wrapper = wrapper
        self.runtime = runtime
        self.derived = derived
        self._media_read = media_read
        self._media_exists = media_exists
        self.index_view = KernelIndexView(kernel, runtime)

    @classmethod
    async def assemble(
        cls,
        *,
        collection_path: str,
        cache_dir: str,
        runtime: EmbeddingRuntime,
        derived: DerivedTextStore,
        cooperative: bool,
        hold_seconds: float,
        media_read: Any,
        media_exists: Any,
    ) -> Harness:
        """Open the kernel on the running loop with a dedicated harness thread
        driving its executor (the one serialization domain for everything)."""
        executor = shrike_native.WorkerExecutor()
        threading.Thread(target=executor.worker_loop, name="shrike-collection", daemon=True).start()
        kernel = await shrike_native.async_kernel_open(collection_path, cache_dir, executor)
        wrapper = CollectionWrapper.over_kernel(
            kernel, collection_path, cooperative=cooperative, hold_seconds=hold_seconds
        )
        return cls(
            kernel=kernel,
            executor=executor,
            wrapper=wrapper,
            runtime=runtime,
            derived=derived,
            media_read=media_read,
            media_exists=media_exists,
        )

    # -- boot ------------------------------------------------------------------

    async def boot(self, *, start_embedding: bool) -> None:
        """One-shot boot orchestration on the loop: log the collection shape,
        start + attach embedding (degrading on failure), reconcile index drift
        in the background, build the derived store on drift, and install the
        cooperative re-acquire hook."""
        summary = (await self.wrapper.get_collection_info(["summary"], []))["summary"]
        logger.info(
            "Collection ready: %d notes, %d decks, %d note types",
            summary["notes"],
            summary["decks"],
            summary["note_types"],
        )

        if start_embedding:
            try:
                await self.start_embedding({})
            except (KernelConfigError, FileNotFoundError, RuntimeError) as e:
                # Degrade — boot without embedding rather than killing the server.
                logger.error("Failed to start embedding service: %s", e)
        elif self.runtime.model:
            logger.info("Embedding service disabled at boot (--no-embedding); model configured")

        # The derived-text store builds whether or not a backend is configured.
        await self._maybe_build_derived()

        if self.wrapper.cooperative:
            self.wrapper.set_acquire_hook(self._on_reacquire(asyncio.get_running_loop()))
            # Release now so a freshly-booted, never-touched idle daemon doesn't
            # hold the lock; the first request re-acquires on demand.
            await self.kernel.release()
            self.wrapper._open_flag = False

    def _on_reacquire(self, loop: asyncio.AbstractEventLoop) -> Any:
        """The cooperative re-acquire hook: runs on the executor inside the
        re-opening job, so it only *schedules* the drift work onto the loop
        (cheap col_mod checks + background rebuilds — never blocking the job)."""

        def hook(core: Any) -> None:
            col_mod = core.col_mod()
            loop.call_soon_threadsafe(self._spawn_reacquire_tasks, col_mod)

        return hook

    def _spawn_reacquire_tasks(self, col_mod: int) -> None:
        if self.derived.check_drift(col_mod):
            task = asyncio.ensure_future(self._rebuild_derived())
            task.add_done_callback(_log_task_failure)
        reindex = asyncio.ensure_future(self._drive_reindex())
        reindex.add_done_callback(_log_task_failure)

    async def _drive_reindex(self) -> None:
        if await self.kernel.reindex_if_needed():
            logger.info("Collection changed while idle; index reconciled")

    async def _maybe_build_derived(self) -> None:
        """Cheap col_mod probe; full text read only on real drift."""
        col_mod = await self.wrapper.col_mod()
        if self.derived.check_drift(col_mod):
            await self._rebuild_derived()

    async def _rebuild_derived(self) -> None:
        rows, dmod = await self.wrapper.run(collect_derived_rows)
        self.derived.build_in_background(rows, dmod)
        logger.info("Derived-text store drift; building in background (%d rows)", len(rows))

    # -- status ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """The core status block — everything in ``/status`` minus host concerns."""
        # health() may probe llama-server over HTTP — off the loop.
        embedding = await asyncio.to_thread(self.runtime.health)
        return {
            "embedding": embedding,
            "index": self._index_status(),
            "derived": self.derived.status(),
            "locking": "cooperative" if self.wrapper.cooperative else "permanent",
            "collection_held": self.wrapper.is_open,
        }

    def _index_status(self) -> dict[str, Any]:
        """The kernel's index status in the facade's diagnostic shape (state,
        available, size/ndim/path, stamps, activation; progress only while
        building, error only on failure — the wire's IndexStatus contract)."""
        raw = json.loads(self.kernel.index_status_json())
        state = raw["state"]
        if self.runtime.backend is None and state == "ready":
            state = "unavailable"
        status: dict[str, Any] = {
            "state": state,
            "available": state == "ready" and self.runtime.backend is not None,
            "size": int(raw.get("size", 0)),
            "ndim": raw.get("ndim"),
        }
        if raw.get("col_mod") is not None:
            status["col_mod"] = raw["col_mod"]
        if raw.get("model_id") is not None:
            status["model_id"] = raw["model_id"]
        if raw.get("activation"):
            status["activation"] = raw["activation"]
        if state == "building":
            progress = raw.get("progress") or {}
            status["progress"] = {
                "indexed": int(progress.get("indexed", 0)),
                "total": int(progress.get("total", 0)),
            }
        if state == "error" and raw.get("error"):
            status["error"] = str(raw["error"])
        return status

    # -- index ops -----------------------------------------------------------------

    async def rebuild_index(self) -> dict[str, Any]:
        """Full index rebuild (the ``POST /index/rebuild`` semantics)."""
        if self.runtime.backend is None:
            raise KernelConfigError("Embedding service is not running")
        raw = json.loads(self.kernel.index_status_json())
        if raw["state"] == "building":
            progress = raw.get("progress") or {}
            return {"status": "already_building", "progress": progress}

        total_notes = await self.wrapper.run(lambda c: len(c.find_notes("")))
        if total_notes == 0:
            await self.kernel.rebuild_index()
            return {"status": "complete", "size": 0}
        task = asyncio.ensure_future(self.kernel.rebuild_index())
        task.add_done_callback(_log_task_failure)
        return {"status": "started", "total": total_notes}

    async def save_index(self) -> dict[str, Any]:
        """Flush the index now (the ``POST /index/save`` semantics)."""
        raw = json.loads(self.kernel.index_status_json())
        if raw["state"] == "building":
            return {"status": "building", "progress": raw.get("progress") or {}}
        if raw.get("ndim") is None:
            return {"status": "empty"}
        await asyncio.to_thread(self.kernel.save_index)
        return {"status": "saved", "size": int(raw.get("size", 0)), "pending": 0}

    # -- embedding ops ---------------------------------------------------------------

    async def start_embedding(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Start + attach the embedding service (``POST /embedding/start``)."""
        if self.runtime.running:
            return {
                "status": "already_running",
                "embedding": await asyncio.to_thread(self.runtime.health),
            }
        try:
            backend = await asyncio.to_thread(lambda: self.runtime.start(**overrides))
        except (ValueError, ImportError) as e:
            raise KernelConfigError(str(e)) from e
        self._attach(backend)
        task = asyncio.ensure_future(self._drive_boot_reindex())
        task.add_done_callback(_log_task_failure)
        return {
            "status": "started",
            "embedding": await asyncio.to_thread(self.runtime.health),
            "index": self._index_status(),
        }

    def _attach(self, backend: EmbedderBackend) -> None:
        embedder = shrike_native.PyEmbedder.capture(backend)
        self.kernel.attach_embedder(embedder, self._media_read, self._media_exists)

    async def _drive_boot_reindex(self) -> None:
        if await self.kernel.reindex_if_needed():
            logger.info("Index reconciled after embedding start")

    async def stop_embedding(self) -> dict[str, Any]:
        """Detach + stop the embedding service (``POST /embedding/stop``)."""
        if not self.runtime.running:
            return {"status": "not_running"}
        self.kernel.detach_embedder()  # flushes the index, marks unavailable
        await asyncio.to_thread(self.runtime.stop)
        return {"status": "stopped", "index": self._index_status()}

    # -- lifecycle ----------------------------------------------------------------

    async def reload(self) -> dict[str, Any]:
        """Close and re-open the collection; re-check drift (``POST /reload``)."""
        await self.wrapper.reopen()
        col_mod = await self.wrapper.col_mod()
        await self._maybe_build_derived()
        rebuilding = False
        if self.runtime.backend is not None:
            rebuilding = await self.kernel.reindex_if_needed()
        return {"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding}

    async def close(self) -> None:
        """Tear down: derived, embedding, then the kernel (flushes the index)."""
        self.derived.close()
        await asyncio.to_thread(self.runtime.stop)
        self.wrapper.close()
        await self.kernel.close()
        self._executor.shutdown()


def _log_task_failure(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background kernel task failed: %s", exc)
