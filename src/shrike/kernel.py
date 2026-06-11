"""ShrikeKernel — the transport-free, synchronous core (#275, implements #224).

``server.py`` is the HTTP *host*: FastMCP transport, routes, the Host/Origin
guard, signal handlers, the ServerLock, and the asyncio event loop. Everything
underneath — the collection, the vector index orchestrator, the derived-text
store, the embedding runtime, and the operational verbs behind the custom
routes — lives here, with **no** transport or event-loop coupling: every kernel
method is a plain blocking function. The HTTP host calls them via
``asyncio.to_thread``; the future embedded host (#224) calls them from its own
threads; stretch slice 2 (#279) re-homes the whole thing in Rust.

Concurrency goes through the :class:`Scheduler` port — exactly three
primitives, derived from auditing every concurrency touchpoint:

- ``run_on_collection``: FIFO-serialized onto the one collection-owning thread
  (the #224 invariant: the collection is opened *on* its owning thread and no
  live handle crosses).
- ``spawn_compute``: background work (index rebuild/reconcile threads, the #98
  derived-store build).
- ``call_later``: debounce/idle timers (IndexSaver, cooperative-lock release).

The Python implementations are **transitional** (the epic's stretch slice 2
internalizes scheduling into the Rust kernel). Today the index/derived
background builders and IndexSaver keep their own (asyncio/thread) mechanics —
they already match the port's semantics, their tests pin them, and they retire
with the rest of the Python scheduling at slice 2 — while the kernel's own
collection access routes through the port. Re-entrancy rule (#224): a
collection job must never wait on another collection job — that's a deadlock on
the single worker; native compute routes out through its own bindings, never
back through the queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from shrike_native import CollectionCore

from shrike.collection import CollectionWrapper, collect_derived_rows, collect_embed_inputs
from shrike.embedding_base import EmbedderBackend, NoteEmbedInput
from shrike.index import IndexSaver, IndexState, VectorIndex

if TYPE_CHECKING:
    from shrike.derived import DerivedTextStore
    from shrike.embedding import EmbeddingRuntime

logger = logging.getLogger("shrike.server")

T = TypeVar("T")


class CancelHandle(Protocol):
    def cancel(self) -> None: ...


@runtime_checkable
class Scheduler(Protocol):
    """The pluggable threading runtime under the kernel (#224's port).

    The HTTP host implements it over asyncio + the collection worker thread
    (behaviour unchanged); an embedded host implements it over its own threads.
    """

    def run_on_collection(self, fn: Callable[[CollectionCore], T]) -> T:
        """Run ``fn(core)`` on the collection-owning thread, blocking for the result."""
        ...

    def spawn_compute(self, name: str, fn: Callable[[], None]) -> None:
        """Run background work (a rebuild, a store build) off the caller's thread."""
        ...

    def call_later(self, delay: float, fn: Callable[[], None]) -> CancelHandle:
        """Arm a debounce/idle timer."""
        ...


class _TimerHandle:
    def __init__(self, timer: threading.Timer) -> None:
        self._timer = timer

    def cancel(self) -> None:
        self._timer.cancel()


class WorkerScheduler:
    """The HTTP host's Scheduler: the wrapper's single worker thread + plain timers.

    ``run_on_collection`` is ``CollectionWrapper.run_sync`` — the same
    FIFO-serialized, reopen-if-released path every collection op takes — plus
    the cooperative idle-release re-arm that ``wrapper.run`` performs, scheduled
    onto the host loop when one is bound (``bind_loop`` is called by the route
    layer, which always runs on the loop).
    """

    def __init__(self, wrapper: CollectionWrapper) -> None:
        self._wrapper = wrapper
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def run_on_collection(self, fn: Callable[[CollectionCore], T]) -> T:
        try:
            return self._wrapper.run_sync(fn)
        finally:
            # Mirror wrapper.run's finally: in cooperative mode, (re)arm the
            # idle-release timer on the host loop so the collection lock is
            # still surrendered after kernel-driven ops.
            if self._wrapper.cooperative and self._loop is not None:
                self._loop.call_soon_threadsafe(self._wrapper._schedule_release, self._loop)

    def spawn_compute(self, name: str, fn: Callable[[], None]) -> None:
        threading.Thread(target=fn, name=name, daemon=True).start()

    def call_later(self, delay: float, fn: Callable[[], None]) -> CancelHandle:
        timer = threading.Timer(delay, fn)
        timer.daemon = True
        timer.start()
        return _TimerHandle(timer)


class KernelConfigError(Exception):
    """A caller-actionable configuration error (the HTTP host maps it to a 400)."""


# The collection-thread collectors live in shrike.collection
# (collect_embed_inputs / collect_derived_rows) — they read through the native
# core and are shared with the boot path.
_collect_for_rebuild = collect_embed_inputs
_collect_derived_rows = collect_derived_rows


def _maybe_rebuild(
    index: VectorIndex,
    model_id: str,
    col_mod: int,
    inputs: list[NoteEmbedInput],
    embedding: EmbedderBackend,
) -> bool:
    """Reconcile the index in the background if it drifted or the model changed.

    Drift (an external ``col.mod`` bump) reconciles incrementally — re-embedding
    only the notes whose text changed — rather than re-embedding the whole
    collection; ``reconcile`` itself falls back to a full rebuild when the model
    changed or there's no prior per-note state. Returns True if work was started
    (drift detected and the collection is non-empty), False otherwise.

    When the collection is *empty* there is nothing to embed, but we still
    materialize an empty, ready index at the model's dimension so notes added
    later in the session are indexed incrementally instead of being skipped until
    a restart (#148).
    """
    if index.check_drift(col_mod, model_id):
        if inputs:
            index.reconcile_in_background(inputs, col_mod, model_id=model_id)
            return True  # the background reconcile recalibrates the activation gate at its tail
        ndim = embedding.embedding_dim()
        if ndim is not None:
            index.materialize_empty(ndim, col_mod, model_id)
        else:
            logger.info("Collection is empty and embedding dim unknown, skipping index rebuild")
    # No background rebuild was started (the index loaded clean, or the collection is empty). Make
    # sure a clean index that predates the activation gate (#201b) gets calibrated now rather than
    # waiting for the next drift; a no-op when it's already calibrated or has no non-text modality.
    # We're already off the event loop here (callers run kernel methods via asyncio.to_thread).
    index.ensure_calibrated()
    return False


# ── the kernel ───────────────────────────────────────────────────────────────


class ShrikeKernel:
    """The synchronous, transport-free core: collection + index + derived store
    + embedding runtime, with the operational verbs behind the host's custom
    routes. Every method blocks; hosts choose how to schedule the calls."""

    def __init__(
        self,
        *,
        wrapper: CollectionWrapper,
        index: VectorIndex,
        saver: IndexSaver,
        derived: DerivedTextStore,
        runtime: EmbeddingRuntime,
        scheduler: Scheduler,
    ) -> None:
        self.wrapper = wrapper
        self.index = index
        self.saver = saver
        self.derived = derived
        self.runtime = runtime
        self.scheduler = scheduler

    # -- boot ----------------------------------------------------------------

    def boot(self, *, start_embedding: bool) -> None:
        """One-shot boot orchestration, called by the host once the components
        are assembled and before it starts serving: log the collection shape,
        start the embedding service (degrading to no-embedding on failure),
        reconcile index drift, build the derived store on drift, and install
        the cooperative re-acquire hook.
        """
        summary = self.scheduler.run_on_collection(
            lambda c: json.loads(c.collection_info(["summary"], []))["summary"]
        )
        logger.info(
            "Collection ready: %d notes, %d decks, %d note types",
            summary["notes"],
            summary["decks"],
            summary["note_types"],
        )

        if start_embedding:
            try:
                svc = self.runtime.start()
            except (FileNotFoundError, RuntimeError, ValueError, ImportError) as e:
                # ImportError: the onnx backend was selected without the optional
                # 'onnx' extra installed. Like a missing model file, degrade — boot
                # without embedding rather than killing the server (the
                # /embedding/start handler returns 400 for the same case).
                logger.error("Failed to start embedding service: %s", e)
            else:
                model_id = svc.model_fingerprint()
                inputs, col_mod = self.scheduler.run_on_collection(_collect_for_rebuild)
                _maybe_rebuild(self.index, model_id, col_mod, inputs, svc)
        elif self.runtime.model:
            logger.info("Embedding service disabled at boot (--no-embedding); model configured")

        # The derived-text store (FTS5 trigram sidecar) is independent of the
        # embedding index — it builds whether or not a backend is configured.
        # Cheap col_mod probe first; only read all field text on real drift
        # (first build, or an external edit), so a clean reload does no full read.
        logger.info("Derived-text store: %s", self.derived.status())
        d_col_mod = self.scheduler.run_on_collection(lambda c: c.col_mod())
        if self.derived.check_drift(d_col_mod):
            rows, dmod = self.scheduler.run_on_collection(_collect_derived_rows)
            self.derived.build_in_background(rows, dmod)
            logger.info("Derived-text store drift; building in background (%d rows)", len(rows))

        if self.wrapper.cooperative:
            # On every re-acquire after an idle release, re-check drift (below).
            self.wrapper.set_acquire_hook(self._on_reacquire)
            # Release now so a freshly-booted, never-touched idle daemon doesn't
            # hold the lock; the first request re-acquires on demand.
            self.wrapper.release_now()

    def _on_reacquire(self, core: CollectionCore) -> None:
        """Cooperative-lock re-acquire hook: if the collection changed on disk
        while the lock was released (Anki, sync, import), rebuild the derived
        store and reconcile the index. Cheap col_mod-only checks; texts are read
        under the lock and embedded off-lock only on real drift. Runs on the
        worker thread."""
        # The derived store is independent of the embedder — rebuild it on
        # drift even with no embedding service (a cheap text-only build).
        if self.derived.check_drift(core.col_mod()):
            rows, dmod = _collect_derived_rows(core)
            logger.info(
                "Collection changed while idle; rebuilding derived store (%d rows)", len(rows)
            )
            self.derived.build_in_background(rows, dmod)
        if not self.index.available or not self.index.check_drift(core.col_mod()):
            return
        svc = self.runtime.service
        if svc is None or not svc.running:
            return
        inputs, changed_mod = _collect_for_rebuild(core)
        logger.info("Collection changed while idle (col_mod=%d); rebuilding index", changed_mod)
        self.index.rebuild_in_background(inputs, changed_mod, model_id=svc.model_fingerprint())

    # -- status ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The core status block — everything in ``/status`` minus host concerns
        (running/pid/url/uptime/log paths, which the host layers on)."""
        return {
            # health() may probe llama-server over HTTP; hosts call status() off
            # their request path (the HTTP host uses asyncio.to_thread).
            "embedding": self.runtime.health(),
            "index": self.index.status(),
            "derived": self.derived.status(),
            "locking": "cooperative" if self.wrapper.cooperative else "permanent",
            "collection_held": self.wrapper.is_open,
        }

    # -- index ops ---------------------------------------------------------------

    def rebuild_index(self) -> dict[str, Any]:
        """Full index rebuild (the ``POST /index/rebuild`` semantics)."""
        svc = self.runtime.service
        if svc is None or not svc.running:
            raise KernelConfigError("Embedding service is not running")

        if self.index.state == IndexState.BUILDING:
            indexed, total = self.index.build_progress
            return {
                "status": "already_building",
                "progress": {"indexed": indexed, "total": total},
            }

        model_id = svc.model_fingerprint()
        inputs, col_mod = self.scheduler.run_on_collection(_collect_for_rebuild)
        if not inputs:
            self.index.rebuild([], col_mod, model_id=model_id)
            return {"status": "complete", "size": 0}

        self.index.rebuild_in_background(inputs, col_mod, model_id=model_id)
        return {"status": "started", "total": len(inputs)}

    def save_index(self) -> dict[str, Any]:
        """Flush the in-memory index now (the ``POST /index/save`` semantics)."""
        # Refuse mid-rebuild: a save here would persist a partial index with a
        # stale col_mod, and rebuild() saves at its own end anyway.
        if self.index.state == IndexState.BUILDING:
            indexed, total = self.index.build_progress
            return {"status": "building", "progress": {"indexed": indexed, "total": total}}
        if self.index.ndim is None:
            return {"status": "empty"}

        pending = self.index.pending_changes
        self.index.save()
        return {"status": "saved", "size": self.index.size, "pending": pending}

    # -- embedding ops -----------------------------------------------------------

    def start_embedding(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Start the embedding service (the ``POST /embedding/start`` semantics).

        Raises :class:`KernelConfigError` for caller-actionable config problems
        (unknown backend, no model, missing optional dependency); lets runtime
        failures (``FileNotFoundError``/``RuntimeError``) propagate for the host
        to map to its 500-equivalent.
        """
        if self.runtime.running:
            return {"status": "already_running", "embedding": self.runtime.health()}

        try:
            svc = self.runtime.start(**overrides)
        except (ValueError, ImportError) as e:
            raise KernelConfigError(str(e)) from e

        model_id = svc.model_fingerprint()
        inputs, col_mod = self.scheduler.run_on_collection(_collect_for_rebuild)
        _maybe_rebuild(self.index, model_id, col_mod, inputs, svc)

        return {
            "status": "started",
            "embedding": self.runtime.health(),
            "index": self.index.status(),
        }

    def stop_embedding(self) -> dict[str, Any]:
        """Stop the embedding service (the ``POST /embedding/stop`` semantics)."""
        if not self.runtime.running:
            return {"status": "not_running"}
        # Persist current vectors before tearing down the embedder.
        self.index.save()
        self.runtime.stop()
        return {"status": "stopped", "index": self.index.status()}

    # -- lifecycle ----------------------------------------------------------------

    def reload(self) -> dict[str, Any]:
        """Close and re-open the collection; re-check drift (the ``POST /reload`` semantics)."""
        self.scheduler.run_on_collection(self.wrapper._do_reopen)
        col_mod = self.scheduler.run_on_collection(lambda c: c.col_mod())

        # The derived-text store is independent of the embedder — rebuild it on drift regardless
        # (cheap text-only build).
        if self.derived.check_drift(col_mod):
            rows, dmod = self.scheduler.run_on_collection(_collect_derived_rows)
            self.derived.build_in_background(rows, dmod)

        # Re-check index drift against the re-opened collection. Without a running
        # embedder we can't rebuild (the index stays unavailable); just report.
        rebuilding = False
        svc = self.runtime.service
        if svc is not None and svc.running:
            model_id = svc.model_fingerprint()
            inputs, new_col_mod = self.scheduler.run_on_collection(_collect_for_rebuild)
            rebuilding = _maybe_rebuild(self.index, model_id, new_col_mod, inputs, svc)

        return {"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding}

    def close(self) -> None:
        """Tear down the core (the host flushes its loop-bound saver first)."""
        self.derived.close()
        self.runtime.stop()
        self.wrapper.close()
