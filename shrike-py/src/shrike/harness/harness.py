"""The kernel-mode server core: AsyncKernel + harness services.

The kernel (Rust) owns the collection, the index orchestration, and the derived
ingest; this module is the *assembly* — the harness thread running the kernel's
executor, the embedding runtime attached as a registered service, the
derived-store build driver, and the operational verbs behind the custom routes.
Every verb is a coroutine on the host loop (the kernel's ops are loop-driven
awaitables; only genuinely blocking work — a model load — hops to a thread).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shrike_native

from shrike.harness import cache_layout
from shrike.harness.collection import BOOT_COLLECTION_KEY, CollectionWrapper
from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine
from shrike.harness.engines.embedding.base import EmbedderBackend
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime
from shrike.harness.index import (
    ACTIVATION_MARGIN,  # the cross-space floor margin default
)
from shrike.harness.profiles import MODALITIES
from shrike.harness.registry import Registry
from shrike.observability.metrics import metrics

logger = logging.getLogger("shrike.kernel")


class KernelConfigError(Exception):
    """A caller-actionable configuration error (the HTTP host maps it to a 400)."""


# The kernel source string each recognition purpose lands under — the routing
# key the harness passes to `attach_recognizer_with`, also the `/status` map
# key.
RECOGNITION_OCR = "ocr"
RECOGNITION_DESCRIBE = "vlm"
# ASR isn't wired into the kernel yet, but the coverage matrix names the source
# it WILL land under so an attached ASR engine lights up text→audio honestly
# once it's integrated. (Today nothing populates this key, so audio stays
# `unavailable` — the honest state.)
RECOGNITION_ASR = "asr"

# Which recognition source derives TEXT from which TARGET modality into the text
# vector space. An attached, ready engine on one of these sources is what makes
# its target reachable `via_derived_text` from a text query: OCR text and VLM
# prose both land in the text space for images; an ASR transcript for audio.
_DERIVED_TEXT_SOURCES: dict[str, frozenset[str]] = {
    "image": frozenset({RECOGNITION_OCR, RECOGNITION_DESCRIBE}),
    "audio": frozenset({RECOGNITION_ASR}),
}


def _coverage_matrix(
    served: frozenset[str],
    ready_recognizers: frozenset[str],
    spaces: Sequence[frozenset[str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the cross-modal coverage matrix: query-modality →
    target-modality → ``native`` | ``via_derived_text`` | ``unavailable``.

    ``served`` is the UNION of modalities the live embedding spaces embed — it
    governs reachability (whether a query modality is embeddable at all, and
    whether a text space exists to derive into). ``spaces`` is the PER-SPACE
    breakdown: one ``frozenset`` per live embedding space, used for the
    ``native`` cell, because *native* means **one single space embeds BOTH**
    modalities — two DISJOINT single-modality spaces (a dedicated text space + a
    separate image-only space) must NOT read text↔image native off the union.
    ``spaces=None`` treats ``served`` as one implicit space (the N≤1 / single-
    backend case), so the cell reduces to ``q in served and t in served``.

    ``ready_recognizers`` is the set of recognition sources currently attached
    AND ready (``ocr``/``vlm``/``asr`` — an errored or unattached engine doesn't
    derive anything, so it doesn't light a cell).

    Per pair (query ``q``, target ``t``):

    - ``native`` when SOME single live space embeds BOTH ``q`` and ``t``. A
      joint CLIP/omni space → text↔image native; a dedicated-text + separate-CLIP
      deployment → text↔image native *via the CLIP space* (it embeds both),
      while text↔text is native via the dedicated text space; two disjoint
      single-modality spaces are native only on their own diagonal.
    - ``via_derived_text`` when ``q`` can reach the TEXT space (``q in served``
      and ``"text" in served``) and a ready recognizer derives text from ``t``
      into that space. Strictly weaker than native; native wins when both hold.
    - ``unavailable`` otherwise.

    Degrades sanely: embedding down → ``served`` empty (``spaces`` empty/None) →
    every cell ``unavailable``; a text-only space with no recognizers → only
    text→text is ``native``, every media target ``unavailable``.
    """
    # The per-space sets for the `native` cell. None → the union is one implicit
    # space (N≤1 / single-backend → the union check).
    per_space: tuple[frozenset[str], ...] = tuple(spaces) if spaces is not None else (served,)

    def native(q: str, t: str) -> bool:
        return any(q in s and t in s for s in per_space)

    text_space_up = "text" in served
    matrix: dict[str, dict[str, str]] = {}
    for q in MODALITIES:
        row: dict[str, str] = {}
        for t in MODALITIES:
            if native(q, t):
                row[t] = "native"
            elif (
                q in served
                and text_space_up
                and ready_recognizers & _DERIVED_TEXT_SOURCES.get(t, frozenset())
            ):
                # q reaches the text space and t derives text into it.
                row[t] = "via_derived_text"
            else:
                row[t] = "unavailable"
        matrix[q] = row
    return matrix


@dataclass
class _RecognitionEngine:
    """One attached recognition engine's harness-side state: the backend kind
    (`apple`/`describe-remote`), the lifecycle state (the
    `RecognitionEngineStatus` enum — `ready`/`error` today, the schema also
    carries `unavailable`/`building` for future use), and its fingerprint when
    known. Per-purpose so `/status` reports one row per engine."""

    backend: str
    state: str = "ready"
    fingerprint: str | None = None


class DedupStatsRecorder:
    """Rolling dedup best-match statistics: one sample per search query group —
    the best SEMANTIC cosine, or a no-match tick. The calibration feedstock for
    the dedup threshold; loop-confined (the actions record on the event loop),
    so no lock. Process-lifetime only — a restart starts fresh (durable
    accumulation is a later refinement)."""

    BUCKETS = 20

    def __init__(self) -> None:
        self.samples = 0
        self.no_match = 0
        self.buckets = [0] * self.BUCKETS

    def record(self, best: float | None) -> None:
        self.samples += 1
        if best is None:
            self.no_match += 1
            return
        index = min(int(best * self.BUCKETS), self.BUCKETS - 1)
        self.buckets[max(index, 0)] += 1

    def snapshot(self) -> dict[str, Any] | None:
        if self.samples == 0:
            return None
        return {
            "samples": self.samples,
            "no_match": self.no_match,
            "buckets": list(self.buckets),
        }


def _assume_normalized(backend: Any) -> bool:
    """Whether ``backend`` guarantees unit-length output (``EmbedderBackend.
    assume_normalized``, default ``False``).

    Drives the attach-time opt-out: when set, the kernel's boundary normalize
    wrap is skipped. That single wrap covers the kernel's one ``EmbedService``
    embedder, which serves BOTH stored embeds and in-kernel query embedding, so
    the skip keeps stored and query vectors consistent under the index's
    inner-product metric. A backend predating the property is treated as
    non-unit."""
    return bool(getattr(backend, "assume_normalized", False))


class KernelIndexView:
    """The search-facing index view, live over the kernel.

    The actions' search path needs availability/state/progress, the engine
    handle, and the activation stats — all of which the kernel owns now. This
    view reads them live (``index_status_json``) instead of holding facade
    copies.
    """

    def __init__(self, kernel: Any, runtime: EmbeddingRuntime) -> None:
        self._kernel = kernel
        self._runtime = runtime
        self._engine_handle = kernel.engine_handle()

    def _status(self) -> dict[str, Any]:
        return json.loads(self._kernel.index_status_json())  # type: ignore[no-any-return]

    @property
    def state_name(self) -> str:
        return str(self._status()["state"])

    @property
    def state(self) -> Any:
        """The facade's ``IndexState`` enum, for the search action's gating."""
        from shrike.harness.index import IndexState

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


class Harness:
    """Assembled kernel-mode server core: one ``AsyncKernel`` + the services
    the harness registers on it, plus the operational verbs the routes call."""

    def __init__(
        self,
        *,
        kernel: Any,
        wrapper: CollectionWrapper,
        runtime: EmbeddingRuntime,
        derived: DerivedTextStore,
        media_read: Any,
        media_exists: Any,
        owns_runtime: bool = True,
        secondary_runtimes: Sequence[EmbeddingRuntime] | None = None,
        cross_space_floor_margin: float = ACTIVATION_MARGIN,
        shared_llama_manager: Any = None,
        collection_key: str = BOOT_COLLECTION_KEY,
    ) -> None:
        self.kernel = kernel
        self.wrapper = wrapper
        # The value of the ``collection`` label on every per-collection metric
        # this harness emits. "default" for the boot collection; a routed
        # harness gets its registry profile name, so multi-collection gauges
        # (lock_held, collection_size, index size/state) keep one series per
        # collection instead of all colliding on "default".
        self._metrics_key = collection_key
        self.runtime = runtime
        # The shared llama.cpp ROUTER manager: when N remote/no-endpoint
        # embedder spaces share ONE managed server (managed.llama_server.
        # models_dir), this is the single `LlamaServerManager.router(...)` they
        # all talk to over loopback — spawned once, owned here, stopped only by
        # the owner on close(). None = no router (the N=1 / single-managed /
        # endpoint / onnx cases — every shape but the shared router). It is
        # started at the top of start_embedding, BEFORE any router-managed
        # remote backend (whose connectivity-proof embed needs it listening).
        self._shared_llama_manager = shared_llama_manager
        # The cross-space image floor margin: the precision/recall dial folded
        # into the secondary floor's `mean + margin·std` at calibration. Default
        # 1.0 (ACTIVATION_MARGIN); resolved harness-side from
        # `search.cross_space_fusion.margin` / the env twin.
        self.cross_space_floor_margin = cross_space_floor_margin
        # Additional embedding spaces: the SECONDARY runtimes, attached to their
        # own kernel embed-space keys alongside the primary. Empty in the N=1
        # case (the default), so single-space behaviour.
        self.secondary_runtimes: list[EmbeddingRuntime] = list(secondary_runtimes or [])
        self.derived = derived
        self._media_read = media_read
        self._media_exists = media_exists
        # Whether this harness OWNS the embedding runtime's lifecycle. In
        # multi-collection mode one runtime (the llama-server subprocess / onnx
        # session) is SHARED across the per-collection harnesses — only the
        # owner stops it on close(), so a routed collection's teardown never
        # kills the shared embedder out from under the others. Single-collection
        # mode (the default) owns its runtime.
        self.owns_runtime = owns_runtime
        self.index_view = KernelIndexView(kernel, runtime)
        self.dedup_stats = DedupStatsRecorder()
        # Background maintenance tasks: tracked so close() drains them (no
        # destroyed-pending teardown) and tests can settle deterministically.
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        # The readiness MARKER tasks (`_settle_and_mark_ready`) are tracked
        # apart from the maintenance work they wait on. A marker awaits
        # settle_background(), which gathers the OTHER bg tasks — so a marker
        # must never gather a sibling marker, or two concurrent re-acquires
        # deadlock (M1 awaits M2 awaiting M1, neither sets _ready). Markers
        # wait only on maintenance; close() still drains them via _bg_tasks.
        self._settle_markers: set[asyncio.Task[Any]] = set()
        # The readiness barrier (#850): boot / reload / cooperative re-acquire
        # run their index+derived maintenance to quiescence and set this Event;
        # the data plane and tests await it instead of polling status + sleeping.
        # A GENERATION counter makes it re-entrant — a /reload or re-acquire
        # mid-flight bumps the generation, so a stale stage's "ready" can never
        # strand the gate at the newer generation (the ready→not-ready→ready
        # transition the doc warns about).
        self._ready = asyncio.Event()
        self._generation = 0
        # Recognition: per-purpose engine state, keyed by the kernel source
        # string (`"ocr"`/`"vlm"`). Empty = nothing attached (a distinct,
        # representable state from "attached but errored"). Each row carries its
        # backend kind, state, and (when known) fingerprint for per-engine
        # `/status`. One shared background sweep task drives every attached
        # purpose (the kernel's `recognize_pending` sweeps all of them per
        # batch); tracked so close() drains it.
        self._recognition_engines: dict[str, _RecognitionEngine] = {}
        self._recognition_task: asyncio.Task[Any] | None = None

    @classmethod
    async def assemble(
        cls,
        *,
        collection_path: str,
        cache_dir: str,
        runtime: EmbeddingRuntime,
        derived: DerivedTextStore | None = None,
        derived_engine_factory: Callable[[Path], NativeDerivedEngine] = NativeDerivedEngine,
        cooperative: bool,
        hold_seconds: float,
        media_read: Any,
        media_exists: Any,
        index_save_delay: float | None = None,
        index_save_threshold: int | None = None,
        owns_runtime: bool = True,
        secondary_runtimes: Sequence[EmbeddingRuntime] | None = None,
        cross_space_floor_margin: float = ACTIVATION_MARGIN,
        shared_llama_manager: Any = None,
        collection_key: str = BOOT_COLLECTION_KEY,
    ) -> Harness:
        """Open the kernel on the running loop. Scheduling is the kernel's own
        (the owned tokio runtime spawns the collection actor); the harness
        assembles services and awaits completions. The ``index_save_*`` tuning
        reaches the kernel's debounced saver; ``None`` keeps the built-in
        defaults. ``owns_runtime`` is False for a per-collection harness sharing
        one embedding runtime — then close() leaves the shared runtime running.
        ``secondary_runtimes`` are the additional embedding spaces; empty (the
        default) is the N=1 case.

        The host ``DerivedTextStore`` (the ``/status`` read surface) is built
        **here, AFTER the kernel opens the collection**, at the same
        ``cache_layout.derived_db_path`` the kernel's ``DerivedEngine`` opened —
        not by a caller before assembly. The path-derived namespace
        canonicalizes the collection path, and that canonicalization differs by
        whether the file EXISTS at computation time (an existing file → realpath,
        which folds a symlinked prefix like macOS ``/var/folders`` →
        ``/private/var/...``; an absent file → a lexical abspath that does NOT).
        A fresh collection computed before open hashed under the abspath
        namespace while the kernel — which creates the file during open — hashed
        under the realpath one, so the host ``/status`` read an EMPTY store while
        the kernel's search store held the rows. Building after open means the
        file exists for both, so both realpath to the SAME
        ``derived/<namespace>/shrike.db``. A pre-built ``derived`` may still be
        injected (the test seam); production passes none and lets assembly
        resolve the kernel-authoritative path."""
        kernel = await shrike_native.async_kernel_open(
            collection_path,
            cache_dir,
            save_delay=index_save_delay,
            save_threshold=index_save_threshold,
        )
        if derived is None:
            # Resolve AFTER open (the collection file now exists, so the host's
            # canonicalization matches the kernel's). Same computation the
            # kernel used internally, so they open one shared shrike.db.
            derived_path = cache_layout.derived_db_path(cache_dir, collection_path)
            derived = DerivedTextStore(
                path=Path(derived_path), engine_factory=derived_engine_factory
            )
        wrapper = CollectionWrapper.over_kernel(
            kernel,
            collection_path,
            cooperative=cooperative,
            hold_seconds=hold_seconds,
            collection_key=collection_key,
        )
        return cls(
            kernel=kernel,
            wrapper=wrapper,
            runtime=runtime,
            derived=derived,
            media_read=media_read,
            media_exists=media_exists,
            owns_runtime=owns_runtime,
            secondary_runtimes=secondary_runtimes,
            cross_space_floor_margin=cross_space_floor_margin,
            shared_llama_manager=shared_llama_manager,
            collection_key=collection_key,
        )

    # -- boot ------------------------------------------------------------------

    async def boot(self, *, start_embedding: bool) -> None:
        """One-shot boot orchestration on the loop, run as ORDERED stages with a
        single re-entrant readiness barrier (#850): log the collection shape,
        start + attach embedding (degrading on failure), kick off the index
        drift reconcile + the derived-store build, then drive them to quiescence
        and open the data plane (``await_ready``). Returns only once ready, so
        ``await harness.boot()`` is the deterministic boot await (no status poll
        + sleep). Finally installs the cooperative re-acquire hook."""
        generation = self._begin_generation()
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

        # Drive the boot maintenance to quiescence (the kernel serializes the
        # index reconcile + derived rebuild through its ingest actor), then open
        # the data plane. The ordering that closed the #828/#650 race family is
        # the kernel's (Theme A); the barrier is the deterministic ready signal.
        await self._settle_and_mark_ready(generation)

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
        # A re-acquire is a new readiness generation: the data plane goes
        # not-ready until the re-driven index/derived maintenance settles. The
        # marker task awaits the spawned rebuilds (settle_background excludes it)
        # then re-opens the gate at this generation.
        generation = self._begin_generation()
        if self.derived.check_drift(col_mod):
            self._spawn_bg(self._rebuild_derived())
        self._spawn_bg(self._drive_reindex())
        self._spawn_marker(generation)

    async def _reconcile_index(self) -> bool:
        """Run ``kernel.reindex_if_needed()`` under the reconcile metrics.

        The single instrumented choke point for every drift reconcile — idle
        re-acquire, boot/embedding-start, and ``/reload`` — so none of them is a
        metrics blind spot. Returns whether the index changed; callers own the
        log line and any secondary-floor recalibration.
        """
        started = time.perf_counter()
        result = "ok"
        try:
            changed = bool(await self.kernel.reindex_if_needed())
            if not changed:
                result = "noop"
            return changed
        except Exception:
            result = "error"
            raise
        finally:
            metrics.index_operations.labels(self._metrics_key, "vector", "reconcile", result).inc()
            metrics.index_operation_duration.labels(
                self._metrics_key, "vector", "reconcile", result
            ).observe(time.perf_counter() - started)
            self._update_index_metrics()

    async def _drive_reindex(self) -> None:
        if await self._reconcile_index():
            logger.info("Collection changed while idle; index reconciled")
            await self._recalibrate_secondary_floors()

    async def _maybe_build_derived(self) -> None:
        """Cheap col_mod probe; the rebuild itself is fire-and-forget — boot
        and /reload must not block on the FTS5 build (the store reports BUILDING
        and searches fall back until ready). The claim in _rebuild_derived
        dedupes double-fires."""
        col_mod = await self.wrapper.col_mod()
        if self.derived.check_drift(col_mod):
            self._spawn_bg(self._rebuild_derived())

    def _spawn_bg(self, coro: Any) -> asyncio.Task[Any]:
        """Spawn tracked background maintenance: logged on failure, discarded
        from the set on completion, drained by close()."""
        task: asyncio.Task[Any] = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(_log_task_failure)
        return task

    def _spawn_marker(self, generation: int) -> asyncio.Task[Any]:
        """Spawn a readiness marker for `generation` and register it as a marker
        so settle_background() never awaits it (a marker waits on maintenance,
        not on other markers — see settle_background)."""
        task = self._spawn_bg(self._settle_and_mark_ready(generation))
        self._settle_markers.add(task)
        task.add_done_callback(self._settle_markers.discard)
        return task

    async def settle_background(self) -> None:
        """Await in-flight background maintenance (drift rebuilds, reindex
        drivers) — deterministic boots for tests and operational verbs. Excludes
        every readiness MARKER task (`_settle_and_mark_ready`), not just the
        caller: a marker gathers only the maintenance it waits on, so two
        concurrent re-acquires can't deadlock on each other's markers.

        Excludes the recognition sweep too. Readiness is "the index/derived
        maintenance has settled"; a recognition sweep (OCR/ASR, many seconds) is
        NOT that maintenance, and gating readiness on it would park the data plane
        behind a re-acquire/reload that overlapped a live sweep. The sweep's own
        writes still serialize through the kernel ingest actor; close() drains it
        directly (via `_bg_tasks`)."""
        while True:
            pending = [
                t
                for t in self._bg_tasks
                if t not in self._settle_markers and t is not self._recognition_task
            ]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

    def _begin_generation(self) -> int:
        """Open a new readiness generation: the data plane goes not-ready until
        this generation's maintenance settles. Returns the generation token."""
        self._generation += 1
        self._ready.clear()
        return self._generation

    async def _settle_and_mark_ready(self, generation: int) -> None:
        """Drain this generation's boot maintenance — the background index/
        derived rebuilds AND the kernel ingest queue — then open the data plane,
        but ONLY if a newer generation hasn't superseded it (re-entrancy)."""
        await self.settle_background()
        await self.kernel.settle()
        if generation == self._generation:
            self._ready.set()

    async def await_ready(self) -> None:
        """Block until the current generation's boot maintenance has settled —
        the one deterministic readiness await the data plane and tests use
        instead of polling derived/index status + sleeping."""
        await self._ready.wait()

    @property
    def is_ready(self) -> bool:
        """Whether the data plane is open (the current generation settled)."""
        return self._ready.is_set()

    async def _rebuild_derived(self) -> None:
        # Kernel-side rebuild: the kernel op collects + builds crate-side
        # (avoiding a whole-collection Rust→Python→Rust round-trip, ~150-250MB
        # transient at 100k notes); the store here only drives the status state
        # machine around the await.
        if not self.derived.claim_external_build():
            return
        started = time.perf_counter()
        col_mod: int | None = None
        try:
            rows, col_mod = await self.kernel.rebuild_derived()
            logger.info("Derived-text store rebuilt kernel-side (%d rows)", rows)
        except Exception:
            col_mod = None
            logger.exception("Kernel-side derived rebuild failed")
        finally:
            self.derived.settle_external_build(col_mod)
            result = "ok" if col_mod is not None else "error"
            metrics.index_operations.labels(self._metrics_key, "derived", "rebuild", result).inc()
            metrics.index_operation_duration.labels(
                self._metrics_key, "derived", "rebuild", result
            ).observe(time.perf_counter() - started)
            self._update_index_metrics()

    # -- status ------------------------------------------------------------------

    def collection_status(self) -> dict[str, Any]:
        """The cheap per-collection status fields for a multi-collection row:
        held (the cooperative-lock state), the index state, and the last col_mod
        the index saw. No embedding health probe (that's the default
        collection's full ``status()`` job) — this is a fan-out over every
        assembled collection, so it stays a pure in-memory read."""
        idx = self._index_status()
        return {
            "held": self.wrapper.is_open,
            "index_state": idx.get("state"),
            "col_mod": idx.get("col_mod"),
        }

    async def status(self) -> dict[str, Any]:
        """The core status block — everything in ``/status`` minus host concerns."""
        # Per-space embedding health: the primary runtime PLUS every secondary
        # space, not just the primary — a multi-space profile has more than one
        # embedder, each its own /status entry. health() may probe llama-server
        # over HTTP, so every space's probe rides to_thread.
        runtimes = [self.runtime, *self.secondary_runtimes]
        embedding_spaces = [await asyncio.to_thread(rt.health) for rt in runtimes]
        # ``embedding`` stays the PRIMARY space's health for back-compat (every
        # existing consumer reads it); ``embedding_spaces`` is the full list.
        embedding = embedding_spaces[0]
        status: dict[str, Any] = {
            "embedding": embedding,
            "embedding_spaces": embedding_spaces,
            "index": self._index_status(),
            "derived": self.derived.status(),
            "locking": "cooperative" if self.wrapper.cooperative else "permanent",
            "collection_held": self.wrapper.is_open,
        }
        if (dedup := self.dedup_stats.snapshot()) is not None:
            status["dedup"] = dedup
        # Degraded-writer signal: non-zero means the sole ingest-drain writer
        # caught a panic (likely a poisoned lock), skipped that work, and
        # survived — the affected notes are un-indexed until a reconcile heals
        # them. Emitted only when non-zero, so a healthy server's shape is
        # unchanged.
        if (drain_panics := self.kernel.ingest_drain_panics()) > 0:
            status["ingest_drain_panics"] = drain_panics
        # Per-engine recognition status: a map keyed by source — one row per
        # attached purpose, each {state, backend, fingerprint}. An empty map is
        # "nothing attached" (distinct from an attached-but-errored engine).
        status["recognition"] = {
            source: {
                "state": eng.state,
                "backend": eng.backend,
                "fingerprint": eng.fingerprint,
            }
            for source, eng in self._recognition_engines.items()
        }
        # The cross-modal coverage matrix: for each (query, target) modality
        # pair, how the target is reachable — `native` (one live space embeds
        # both), `via_derived_text` (a ready recognizer derives text from the
        # target into the text space), or `unavailable`. Derived from the live
        # embedding spaces and the attached, ready recognizers.
        # All-`unavailable` when embedding is down, so the shape is stable for
        # clients.
        # Per-space modality sets: the primary runtime + every live secondary
        # space. `native` is computed per-space (one space embeds both
        # modalities), while `served` (their union) governs reachability. Empty
        # when embedding is down → every cell unavailable.
        per_space: list[frozenset[str]] = [
            frozenset(rt.backend.modalities)
            for rt in (self.runtime, *self.secondary_runtimes)
            if rt.backend is not None and rt.backend.running
        ]
        served = frozenset().union(*per_space) if per_space else frozenset()
        ready_recognizers = frozenset(
            source for source, eng in self._recognition_engines.items() if eng.state == "ready"
        )
        status["coverage"] = _coverage_matrix(served, ready_recognizers, per_space)
        metrics.update_index(
            "vector",
            str(status["index"]["state"]),
            int(status["index"].get("size", 0)),
            collection=self._metrics_key,
        )
        metrics.update_index(
            "derived",
            str(status["derived"]["state"]),
            int(status["derived"].get("size", 0)),
            collection=self._metrics_key,
        )
        metrics.lock_held.labels(self._metrics_key).set(self.wrapper.is_open)
        return status

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
        # Per-modality sub-index breakdown: the kernel reports each sub-index's
        # own size/ndim (the aggregate above collapses them). Pass it straight
        # through to IndexModalityStat (text-first, kernel-ordered).
        if raw.get("modalities"):
            status["modalities"] = [
                {
                    "modality": m["modality"],
                    "size": int(m.get("size", 0)),
                    "ndim": m.get("ndim"),
                }
                for m in raw["modalities"]
            ]
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

    def _update_index_metrics(self) -> None:
        """Refresh gauges from in-memory/native snapshots; never opens the collection."""
        index = self._index_status()
        metrics.update_index(
            "vector",
            str(index.get("state", "unavailable")),
            index["size"],
            collection=self._metrics_key,
        )
        derived = self.derived.status()
        metrics.update_index(
            "derived",
            str(derived.get("state", "unavailable")),
            int(derived.get("size", 0)),
            collection=self._metrics_key,
        )
        metrics.lock_held.labels(self._metrics_key).set(self.wrapper.is_open)

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
        metrics.collection_size.labels(self._metrics_key).set(total_notes)
        if total_notes == 0:
            started = time.perf_counter()
            await self.kernel.rebuild_index()
            await self._recalibrate_secondary_floors()
            metrics.index_operations.labels(self._metrics_key, "vector", "rebuild", "ok").inc()
            metrics.index_operation_duration.labels(
                self._metrics_key, "vector", "rebuild", "ok"
            ).observe(time.perf_counter() - started)
            self._update_index_metrics()
            return {"status": "complete", "size": 0}
        self._spawn_bg(self._rebuild_then_calibrate())
        return {"status": "started", "total": total_notes}

    async def _rebuild_then_calibrate(self) -> None:
        """Full index rebuild, then recalibrate the secondary floors — the
        secondary image vectors are fresh, so the floor must be re-derived."""
        started = time.perf_counter()
        result = "ok"
        try:
            await self.kernel.rebuild_index()
            await self._recalibrate_secondary_floors()
        except Exception:
            result = "error"
            raise
        finally:
            metrics.index_operations.labels(self._metrics_key, "vector", "rebuild", result).inc()
            metrics.index_operation_duration.labels(
                self._metrics_key, "vector", "rebuild", result
            ).observe(time.perf_counter() - started)
            self._update_index_metrics()

    async def save_index(self) -> dict[str, Any]:
        """Flush the index now (the ``POST /index/save`` semantics)."""
        raw = json.loads(self.kernel.index_status_json())
        if raw["state"] == "building":
            return {"status": "building", "progress": raw.get("progress") or {}}
        if raw.get("ndim") is None:
            return {"status": "empty"}
        try:
            await asyncio.to_thread(self.kernel.save_index)
        finally:
            self._update_index_metrics()
        return {"status": "saved", "size": int(raw.get("size", 0)), "pending": 0}

    # -- embedding ops ---------------------------------------------------------------

    async def start_embedding(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Start + attach the embedding service (``POST /embedding/start``)."""
        if self.runtime.running:
            return {
                "status": "already_running",
                "embedding": await asyncio.to_thread(self.runtime.health),
            }
        # Spawn the shared llama.cpp router ONCE before any router-managed remote
        # backend: the primary + secondary remote spaces all talk to it over
        # loopback, and their connectivity-proof embed at start() needs it
        # already healthy. router /health is 200 before any model lazy-loads, so
        # this is a fast health-wait. Owned here; stopped only on owner close().
        await self._ensure_shared_router()
        try:
            backend = await asyncio.to_thread(lambda: self.runtime.start(**overrides))
        except (ValueError, ImportError) as e:
            raise KernelConfigError(str(e)) from e
        self._attach(backend)
        # Fan out the secondary spaces: each is its own kernel embed space,
        # attached by its content-fingerprint key. A secondary that fails to
        # start degrades ONLY its space — never the primary, which is already
        # attached and serving the index/search path.
        await self._attach_secondaries(overrides)
        self._spawn_bg(self._drive_boot_reindex())
        return {
            "status": "started",
            "embedding": await asyncio.to_thread(self.runtime.health),
            "index": self._index_status(),
        }

    async def _ensure_shared_router(self) -> None:
        """Spawn the shared llama.cpp router manager if one is configured and
        not yet running. Idempotent — the native manager's own ``start`` no-ops
        a live child, and we additionally guard so a repeat call doesn't block
        on the health-wait. Off the loop (spawn + health-wait blocks)."""
        mgr = self._shared_llama_manager
        if mgr is None:
            return
        if await asyncio.to_thread(mgr.running):
            return
        await asyncio.to_thread(mgr.start)

    async def _attach_secondaries(self, overrides: dict[str, Any]) -> None:
        """Start + attach every SECONDARY embedding space, best-effort: a space
        that fails to start (bad model, missing dep, unreachable endpoint) is
        logged and skipped — only that space degrades, the rest (and the
        primary) stay live. No-op in the N=1 case (no secondaries)."""
        for rt in self.secondary_runtimes:
            if rt.running:
                continue
            try:
                backend = await asyncio.to_thread(functools.partial(rt.start, **overrides))
            except (ValueError, ImportError, FileNotFoundError, RuntimeError, OSError) as e:
                logger.error(
                    "Secondary embedding space (%s) failed to start: %s — "
                    "that space is degraded; other spaces stay live",
                    rt.backend_kind,
                    e,
                )
                continue
            self._attach(backend, space_key=self._space_key_for(rt))

    @staticmethod
    def _space_key_for(rt: EmbeddingRuntime) -> str | None:
        """A secondary space's explicit key: the started backend's model
        fingerprint (the CONTENT fingerprint, reorder-stable). ``None`` lets the
        kernel fall back to the embedder's own fingerprint — identical, since
        the backend carries it; the explicit pass keeps the harness in control
        of the space identity."""
        backend = rt.backend
        if backend is None:
            return None
        fp = getattr(backend, "model_fingerprint", None)
        return fp() if callable(fp) else None

    async def attach_shared_embedder(self) -> None:
        """Attach an ALREADY-RUNNING shared embedding runtime's backend to this
        harness's kernel and reconcile its index (multi-collection).

        Unlike :meth:`start_embedding`, this never starts/stops the runtime —
        the runtime is owned and started elsewhere (the default harness). A
        routed per-collection harness calls this so semantic search works on it,
        then reconciles its own (namespaced) index against the shared model.
        No-op when no backend is running (lexical search still works)."""
        backend = self.runtime.backend
        if backend is None:
            return
        self._attach(backend)
        self._spawn_bg(self._drive_boot_reindex())

    def _attach(self, backend: EmbedderBackend, *, space_key: str | None = None) -> None:
        """Attach the backend to a kernel embed SPACE. A backend exposing
        ``native_embedder()`` (the onnx/clip/llama facades — every production
        backend) hands over a native composition — kernel embeds then never
        re-enter Python; a custom/test backend without one is captured behind
        the PyEmbedder dispatch seam. ``space_key`` pins the space identity;
        ``None`` (the primary / N=1 case) lets the kernel key off the embedder's
        own fingerprint."""
        native = getattr(backend, "native_embedder", None)
        embedder = native() if callable(native) else shrike_native.PyEmbedder.capture(backend)
        # A backend that guarantees unit output opts out of the kernel's boundary
        # normalize. The kernel's single EmbedService embedder serves BOTH stored
        # embeds and in-kernel query embedding, so the one skip keeps stored and
        # query vectors consistent.
        self.kernel.attach_embedder(
            embedder,
            self._media_read,
            self._media_exists,
            space_key=space_key,
            unsafe_assume_normalized=_assume_normalized(backend),
        )

    def attach_recognizer(self, backend: Any, purpose: str = RECOGNITION_OCR) -> None:
        """Attach an OCR/describe backend for a recognition ``purpose`` (the
        kernel source — ``"ocr"`` / ``"vlm"``), keyed by purpose. A native
        engine (``AppleVisionRecognizer`` in engine-apple builds,
        ``RemoteDescriber`` in engine-remote builds) goes to the kernel directly
        — recognition then never re-enters Python; any other object satisfying
        the RecognizerBackend contract (a blocking ``recognize(items)`` plus
        ``model_fingerprint()``) is captured behind the PyRecognizer dispatch
        seam. Must run on the event loop (both paths grab the running loop)."""
        if self._media_read is None or self._media_exists is None:
            raise KernelConfigError("recognition needs media access (media_read/media_exists)")
        native_cls = getattr(shrike_native, "AppleVisionRecognizer", None)
        describe_cls = getattr(shrike_native, "RemoteDescriber", None)
        is_native = (native_cls is not None and isinstance(backend, native_cls)) or (
            describe_cls is not None and isinstance(backend, describe_cls)
        )
        recognizer: Any = backend if is_native else shrike_native.Recognizer.capture(backend)
        self.kernel.attach_recognizer_with(
            purpose, recognizer, self._media_read, self._media_exists
        )

    def detach_recognizer(self, purpose: str | None = None) -> None:
        """Detach a recognition engine. ``purpose=None`` detaches OCR (the
        back-compat default); a source string detaches that purpose."""
        if purpose is None or purpose == RECOGNITION_OCR:
            self.kernel.detach_recognizer()
        else:
            self.kernel.detach_recognizer_for(purpose)

    async def recognition_sweep(
        self, batch_size: int = 8, max_batches: int | None = None
    ) -> dict[str, Any]:
        """Drive bounded recognition sweeps until nothing is pending.

        The drive-to-quiescence loop lives in the KERNEL now
        (``recognize_all_pending``): one FFI crossing instead of one per batch,
        with the no-progress STOP and the per-purpose abort decided in the
        runtime that owns both stores. Each internal batch keeps the bounded-
        batch yield so collection ops interleave. Returns the final report plus
        the run totals (``total_stored``, ``batches``)."""
        started = time.perf_counter()
        metrics.recognition_running.labels(self._metrics_key).set(1)
        result = "ok"
        try:
            report: dict[str, Any] = json.loads(
                await self.kernel.recognize_all_pending(batch_size, max_batches)
            )
        except Exception:
            result = "error"
            raise
        finally:
            metrics.recognition_running.labels(self._metrics_key).set(0)
            metrics.recognition_sweeps.labels(result).inc()
            metrics.recognition_duration.labels(result).observe(time.perf_counter() - started)
        total_stored = int(report.get("total_stored", 0))
        metrics.recognition_items.inc(total_stored)
        if total_stored:
            logger.info(
                "Recognition sweep stored %d item(s) over %d batch(es)",
                total_stored,
                int(report.get("batches", 0)),
            )
        return report

    def start_recognition(self, kind: str) -> None:
        """Construct + attach an OCR backend by kind and launch a background
        sweep. Degrades to an 'error' state row on a missing dependency (the
        engine isn't compiled in) or an unknown kind — never kills boot (the OCR
        purpose, the legacy ``--ocr-backend`` flow)."""
        from shrike.harness.engines.recognition import make_recognizer

        try:
            backend = make_recognizer(kind)
            fingerprint = backend.model_fingerprint()
            self.attach_recognizer(backend, RECOGNITION_OCR)
        except (ImportError, ValueError, KernelConfigError) as e:
            logger.error("Recognition backend %r unavailable: %s", kind, e)
            self._recognition_engines[RECOGNITION_OCR] = _RecognitionEngine(
                backend=kind, state="error"
            )
            return
        self._recognition_engines[RECOGNITION_OCR] = _RecognitionEngine(
            backend=kind, state="ready", fingerprint=fingerprint
        )
        logger.info("Recognition backend attached: %s (ocr)", kind)
        self._ensure_recognition_sweep()

    def start_recognition_describe(
        self,
        endpoint: str,
        *,
        model: str | None = None,
        api_key_env: str | None = None,
        mmproj: str | None = None,
    ) -> None:
        """Construct + attach the remote VLM describe engine for the
        ``describe`` purpose (image→prose into the text embedding space,
        vector-only) and launch the shared background sweep. Reports an 'error'
        state row when construction fails (a missing build feature / missing key
        env) OR when the endpoint is unreachable at attach — never kills boot.

        An unreachable endpoint still ATTACHES (the engine and its degenerate
        fingerprint): the sweep's chunk-Err-aborts contract leaves the backlog
        pending and a later sweep retries once the endpoint is up. We surface it
        as 'error' rather than 'ready' so a degraded engine is visible — rows
        minted under the degenerate fingerprint re-derive once on the next
        restart, when model_info resolves and the fingerprint sharpens."""
        from shrike.harness.engines.recognition import make_describe_recognizer

        try:
            backend, fingerprint, reachable = make_describe_recognizer(
                endpoint, model=model, api_key_env=api_key_env, mmproj=mmproj
            )
            self.attach_recognizer(backend, RECOGNITION_DESCRIBE)
        except (ImportError, ValueError, RuntimeError, KernelConfigError) as e:
            logger.error("Describe recognizer unavailable: %s", e)
            self._recognition_engines[RECOGNITION_DESCRIBE] = _RecognitionEngine(
                backend="describe-remote", state="error"
            )
            return
        # Attached either way; the row state reflects whether the endpoint
        # answered at attach (a closed port / DNS failure → 'error', visible).
        state = "ready" if reachable else "error"
        self._recognition_engines[RECOGNITION_DESCRIBE] = _RecognitionEngine(
            backend="describe-remote", state=state, fingerprint=fingerprint
        )
        if reachable:
            logger.info("Recognition backend attached: describe-remote (vlm)")
        else:
            logger.warning(
                "Recognition backend attached but endpoint %s is unreachable; "
                "reporting describe (vlm) as error until a sweep reaches it",
                endpoint,
            )
        self._ensure_recognition_sweep()

    def _ensure_recognition_sweep(self) -> None:
        """Launch the shared background sweep task if one isn't already running.
        The kernel's ``recognize_pending`` sweeps EVERY attached purpose per
        batch, so one driver serves all engines. Tracked: close() cancels-and-
        awaits it like every other maintenance task, so a mid-sweep teardown
        never leaves a destroyed-pending task."""
        if self._recognition_task is not None and not self._recognition_task.done():
            return
        self._recognition_task = self._spawn_bg(self._drive_recognition())

    async def _drive_recognition(self) -> None:
        """Background recognition: sweep to completion, off the request path.
        A failure marks every attached engine's state without disturbing the
        rest of the server."""
        try:
            await self.recognition_sweep()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Recognition sweep failed", exc_info=True)
            for eng in self._recognition_engines.values():
                eng.state = "error"

    def stop_recognition(self, purpose: str | None = None) -> None:
        """Detach a recognition engine (or all) and cancel the sweep. With
        ``purpose=None`` every engine is detached (the boot/teardown path);
        a source string detaches just that purpose. The shared sweep task is
        cancelled whenever no engine remains."""
        sources = (
            list(self._recognition_engines)
            if purpose is None
            else [purpose] * (purpose in self._recognition_engines)
        )
        for source in sources:
            with contextlib.suppress(Exception):
                self.detach_recognizer(source)
            self._recognition_engines.pop(source, None)
        if not self._recognition_engines:
            if self._recognition_task is not None and not self._recognition_task.done():
                self._recognition_task.cancel()
            self._recognition_task = None

    async def _drive_boot_reindex(self) -> None:
        if await self._reconcile_index():
            logger.info("Index reconciled after embedding start")
        # Recalibrate unconditionally: the embedder was just (re)attached, so the
        # secondary cross-space floors must be derived whether or not drift was
        # found.
        await self._recalibrate_secondary_floors()

    async def _recalibrate_secondary_floors(self) -> None:
        """Recompute the secondary cross-space image activation floors.

        A dedicated CLIP secondary is image-only, so the engine's own intra-
        modal calibration (text→image) finds no text vectors there; this
        harness-driven pass CLIP-text-embeds a sample of note texts through each
        secondary's own backend and fires them at its image vectors to derive
        the floor (``mean + margin·std``). Best effort: a failure leaves the
        prior floors and only weakens the floor admission to "no floor" (admit
        any non-empty space) — never breaks search. No-op in the N=1 case (the
        kernel returns an empty list when there are no secondaries)."""
        try:
            derived = await self.kernel.calibrate_secondary_floors(self.cross_space_floor_margin)
        except Exception as e:  # noqa: BLE001 — calibration is advisory; never fail search
            logger.warning("Secondary cross-space floor calibration failed: %s", e)
            return
        for space_key, floor in derived:
            if floor is None:
                logger.info(
                    "Cross-space floor for %s: uncalibrated (too few image notes)", space_key
                )
            else:
                logger.info("Cross-space image floor for %s: %.3f", space_key, floor)

    async def stop_embedding(self) -> dict[str, Any]:
        """Detach + stop the embedding service (``POST /embedding/stop``)."""
        if not self.runtime.running and not any(rt.running for rt in self.secondary_runtimes):
            return {"status": "not_running"}
        self.kernel.detach_embedder()  # clears every space, flushes, unavailable
        await asyncio.to_thread(self.runtime.stop)
        # Stop the secondary spaces' runtimes too — detach already cleared their
        # kernel slots; this releases their backend resources.
        for rt in self.secondary_runtimes:
            await asyncio.to_thread(rt.stop)
        # Stop the shared router too: `embedding stop` frees GPU/RAM, and the
        # router process is the resource the remote spaces held. start_embedding
        # re-spawns it (idempotent) on the next cycle. Owner-only (mirrors
        # close()): a routed harness never owns the runtime, so it must not kill
        # the shared router out from under the owner + siblings.
        if self.owns_runtime and self._shared_llama_manager is not None:
            await asyncio.to_thread(self._shared_llama_manager.stop)
        return {"status": "stopped", "index": self._index_status()}

    # -- lifecycle ----------------------------------------------------------------

    async def reload(self) -> dict[str, Any]:
        """Close and re-open the collection; re-check drift (``POST /reload``).
        A new readiness generation: the data plane goes not-ready until the
        re-driven maintenance settles, so an in-flight reload can't strand the
        gate (the generation counter)."""
        generation = self._begin_generation()
        await self.wrapper.reopen()
        col_mod = await self.wrapper.col_mod()
        await self._maybe_build_derived()
        rebuilding = False
        if self.runtime.backend is not None:
            rebuilding = await self._reconcile_index()
            if rebuilding:
                # The reconcile re-embedded secondary image vectors, so the
                # cross-space image floor must be re-derived — every other
                # reindex path (_drive_reindex/_rebuild_then_calibrate/
                # _drive_boot_reindex) recalibrates, and /reload must too.
                # No-op at N=1 (the kernel returns an empty list with no secondaries).
                await self._recalibrate_secondary_floors()
        await self._settle_and_mark_ready(generation)
        return {"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding}

    async def close(self) -> None:
        """Tear down: background tasks, derived, embedding, then the kernel
        (flushes the index)."""
        # Cancel-and-drain the tracked maintenance tasks: a rebuild mid-flight
        # detaches kernel-side (the op completes; detach never aborts), but the
        # Python task must not be destroyed pending.
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
        self.derived.close()
        if self.owns_runtime:
            await asyncio.to_thread(self.runtime.stop)
            # Stop the secondary spaces' runtimes too — owned alongside the
            # primary; a shared harness leaves them to the owner.
            for rt in self.secondary_runtimes:
                await asyncio.to_thread(rt.stop)
            # Stop the shared llama.cpp router LAST: the router-managed remote
            # backends (now stopped) talked to it, so the process they depend on
            # outlives them by exactly this teardown. Owner-only — a shared
            # routed harness never owns the runtime, so it never reaches here and
            # never kills the router out from under siblings.
            if self._shared_llama_manager is not None:
                await asyncio.to_thread(self._shared_llama_manager.stop)
        self.wrapper.close()
        # kernel.close drains the collection actor: nothing in flight when this
        # returns — the interpreter-teardown guard.
        await self.kernel.close()


def _log_task_failure(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background kernel task failed: %s", exc)


# ── multi-collection routing ────────────────────────────────────────────────


class RoutingError(Exception):
    """A caller-actionable routing error: the selector named an unknown
    profile, or no collection could be resolved (no selector and no active
    default). The HTTP host maps it to a 400 / ToolInputError — never a bug."""


@dataclass(frozen=True)
class HarnessParams:
    """The collection-independent assembly parameters every per-collection
    harness shares. The shared base ``cache_dir`` (NOT per-collection —
    isolation comes from per-artifact namespacing: the index namespace, the
    derived store via :func:`cache_layout.derived_db_path`), the shared
    embedding runtime, media resolvers, cooperative-lock settings, and the
    index-flush tuning. One of these is built once at boot; the manager
    stamps a per-collection ``cache_dir``-derived derived store onto each."""

    cache_dir: str
    runtime: EmbeddingRuntime
    media_read: Any
    media_exists: Any
    cooperative: bool
    hold_seconds: float
    index_save_delay: float | None = None
    index_save_threshold: int | None = None
    cross_space_floor_margin: float = ACTIVATION_MARGIN


class CollectionManager:
    """Routes per-call to a per-collection :class:`Harness`, lazily assembled.

    One daemon, many collections: each registered/active collection gets its
    own ``AsyncKernel`` + namespaced index + per-collection derived store,
    assembled on first route and governed by its own cooperative-lock
    idle-release — only the collection(s) mid-operation hold a lock. The N
    kernels share one process-global tokio runtime and one embedding runtime
    (the llama-server subprocess / onnx session is expensive; one is shared,
    attached to each kernel's embed slot).

    Selection is **per-call and stateless** (consistent with ``stateless_http``):
    a selector resolves name → path via a **live, re-readable** registry view
    (re-read on demand so a ``profile create`` in one session routes in the same
    session), defaulting to the registry's active profile; no server-side
    mutable "current collection". The registry's stored path is routed THROUGH
    :mod:`shrike.harness.cache_layout` so canonicalization is the equalizer —
    never a raw-abspath-derived index path.

    The **default collection** (the one the daemon booted with, ``--collection``)
    is seeded as a ready harness so single-collection behavior — boot embedding,
    the operational routes, the recognition sweep — is unchanged; it owns the
    shared embedding runtime's lifecycle. Routed (non-default) collections
    attach the same shared backend to their own kernel and never stop it on
    teardown.
    """

    # The registry key for the daemon's boot collection when it isn't a
    # registered profile — so `--collection` always has a routable handle even
    # with an empty registry (the single-collection default).
    DEFAULT_KEY = "<default>"

    def __init__(
        self,
        *,
        params: HarnessParams,
        default_harness: Harness,
        default_collection_path: str,
        config_path: str | os.PathLike[str] | None,
    ) -> None:
        self._params = params
        # name -> Harness. The default collection is keyed by its registry
        # profile name if it matches one, else DEFAULT_KEY.
        self._harnesses: dict[str, Harness] = {}
        self._default_path = os.path.abspath(default_collection_path)
        self._config_path = config_path
        # One assembly lock PER key so concurrent first-routes to the SAME
        # collection don't double-assemble (two anki opens on one file), while
        # routes to DIFFERENT collections still assemble in parallel.
        self._assembly_locks: dict[str, asyncio.Lock] = {}
        # Seed the default harness under the name the registry knows it by (so
        # `profile list`'s default and `--collection` resolve to one harness),
        # falling back to DEFAULT_KEY.
        self._default_key = self._key_for_path(self._default_path) or self.DEFAULT_KEY
        self._harnesses[self._default_key] = default_harness

    # -- the live registry view -----------------------------------------------

    def registry(self) -> Registry:
        """Re-read the registry from config every call — the live view.

        Cheap (a small YAML read) and authoritative: the config file is the
        source of truth the ``shrike profile`` CLI writes, so re-reading is how
        a register-then-route in one session works without a server-side mutable
        cache that ``stateless_http`` forbids. A missing/unreadable config
        yields an empty registry (the single-collection default), never an
        error.
        """
        if self._config_path is None:
            return Registry()
        try:
            from shrike.cli.config import load_config

            return Registry.from_config(load_config(Path(self._config_path)))
        except Exception:  # noqa: BLE001 — routing degrades to the default, never crashes
            logger.debug("registry re-read failed; routing to the default collection only")
            return Registry()

    def _key_for_path(self, path: str, registry: Registry | None = None) -> str | None:
        """The registry profile name whose collection path matches ``path``
        (path-based, canonicalized through cache_layout so two spellings of one
        file match), or None if unregistered.

        ``registry`` lets a caller pass an already-read snapshot so a fan-out
        (``status_rows``) shares one config read instead of re-reading per
        lookup; None re-reads the live view (the construction-time default-key
        resolution, which runs once)."""
        reg = registry if registry is not None else self.registry()
        target = cache_layout.index_namespace(path)
        for profile in reg.profiles:
            if cache_layout.index_namespace(profile.path) == target:
                return profile.name
        return None

    # -- selector resolution --------------------------------------------------

    def resolve(self, selector: str | None) -> tuple[str, str]:
        """Resolve a selector to ``(key, collection_path)``.

        ``selector`` is a registry profile name (or None). None falls back to
        the registry's active default; with no default set and no selector, the
        daemon's boot collection is the implicit default (so a single-collection
        daemon needs no registry at all). An explicit selector that names no
        registered profile is a :class:`RoutingError`; a None selector with
        neither a default nor a boot collection is the "no default set" error.
        """
        registry = self.registry()
        if selector is not None:
            profile = registry.get(selector)
            if profile is None:
                raise RoutingError(
                    f"unknown collection {selector!r} — register it with "
                    "`shrike profile create`, or omit the selector for the default"
                )
            return (profile.name, os.path.abspath(os.path.expanduser(profile.path)))
        # No selector: the registry default, else the daemon's boot collection.
        default = registry.resolve_default()
        if default is not None:
            return (default.name, os.path.abspath(os.path.expanduser(default.path)))
        return (self._default_key, self._default_path)

    # -- routing --------------------------------------------------------------

    async def harness_for(self, selector: str | None) -> Harness:
        """Route ``selector`` to its harness, assembling on first use.

        The default collection is always already assembled; any other resolves
        via the live registry and is lazily assembled (its own kernel +
        namespaced index + per-collection derived store), attaching the shared
        embedding backend so search works on it too. Concurrent first-routes to
        the same collection serialize on a per-key lock.
        """
        key, path = self.resolve(selector)
        existing = self._harnesses.get(key)
        if existing is not None:
            return existing

        lock = self._assembly_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check under the lock: a concurrent route may have assembled it.
            existing = self._harnesses.get(key)
            if existing is not None:
                return existing
            harness = await self._assemble(key, path)
            self._harnesses[key] = harness
            return harness

    async def _assemble(self, key: str, path: str) -> Harness:
        """Assemble a non-default per-collection harness: its own kernel +
        namespaced index + per-collection derived store, sharing the base
        cache dir and the embedding runtime."""
        logger.info("Routing: assembling collection %r at %s", key, path)
        # The per-collection derived store (<cache_dir>/derived/<ns>/shrike.db)
        # is built by assemble AFTER the kernel opens this collection, so the
        # host namespace canonicalizes the now-existing file identically to the
        # kernel's DerivedEngine — never pre-computed here from a maybe-absent
        # path. The base cache_dir is SHARED (so the index namespace under it is
        # preserved for the single-collection user).
        harness = await Harness.assemble(
            collection_path=path,
            cache_dir=self._params.cache_dir,
            runtime=self._params.runtime,
            cooperative=self._params.cooperative,
            hold_seconds=self._params.hold_seconds,
            media_read=self._params.media_read,
            media_exists=self._params.media_exists,
            index_save_delay=self._params.index_save_delay,
            index_save_threshold=self._params.index_save_threshold,
            cross_space_floor_margin=self._params.cross_space_floor_margin,
            owns_runtime=False,  # the shared runtime is owned by the default harness
            collection_key=key,
        )
        # Boot WITHOUT starting embedding (the shared runtime is already
        # started by the default harness); attach its backend so search works,
        # then reconcile drift. A routed collection in a cooperative daemon
        # releases its lock after the idle window like any other.
        await harness.boot(start_embedding=False)
        await harness.attach_shared_embedder()
        return harness

    async def resolve_bundle(self, selector: str | None) -> Any:
        """The per-call action bundle for ``selector`` — the resolver
        ``register_tools`` is handed.

        Routes to the selector's harness (lazily assembling it) and hands back
        its ``CollectionBundle`` (wrapper + index view + kernel + derived +
        dedup recorder) — exactly the handles an action operates on. A
        :class:`RoutingError` (unknown selector) propagates to the action
        layer, which maps it to a clean ``ToolInputError``."""
        from shrike.api.actions import CollectionBundle

        harness = await self.harness_for(selector)
        return CollectionBundle(
            wrapper=harness.wrapper,
            index=harness.index_view,
            derived=harness.derived,
            kernel=harness.kernel,
            dedup_stats=harness.dedup_stats,
        )

    def status_rows(self) -> list[dict[str, Any]]:
        """One status row per KNOWN collection: the daemon's boot/default
        collection plus every registered profile, deduped by path-namespace (a
        registered profile that IS the boot collection shows once). An assembled
        collection contributes its live held/index/col_mod; a
        registered-but-never-routed one is ``active=False`` with no index
        figures yet (nothing has been opened to read them from).

        The top-level ``/status`` embedding/index/derived fields describe the
        DEFAULT collection (which the operational routes act on); these rows are
        the per-collection view tool calls route across."""
        # One config read for the whole fan-out (the live registry is shared by
        # every lookup below — `resolve()`'s per-routed-call read is the hot
        # path, but a status sweep needn't re-read four times).
        registry = self.registry()
        # The routing default a bare call resolves to (registry default, else
        # the boot collection) — inlined from `resolve(None)` to reuse the one
        # snapshot. Marked `is_default` on whichever row matches it.
        default = registry.resolve_default()
        default_path = (
            os.path.abspath(os.path.expanduser(default.path))
            if default is not None
            else self._default_path
        )
        default_ns = cache_layout.index_namespace(default_path)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add(name: str, path: str, registered: bool) -> None:
            ns = cache_layout.index_namespace(path)
            if ns in seen:
                return
            seen.add(ns)
            row: dict[str, Any] = {
                "name": name,
                "path": path,
                "registered": registered,
                "is_default": ns == default_ns,
                "active": False,
                # Uniform shape: an unassembled collection has no live figures
                # (nothing opened to read them from), so these stay None.
                "held": None,
                "index_state": None,
                "col_mod": None,
            }
            # An assembled harness (keyed by name) contributes live figures.
            harness = self._harnesses.get(name)
            if harness is not None:
                row["active"] = True
                row.update(harness.collection_status())
            rows.append(row)

        # The daemon's boot collection ALWAYS gets a row (it's assembled and
        # physically open), named by the registry if it matches a profile.
        boot_profile_name = self._key_for_path(self._default_path, registry)
        _add(
            name=boot_profile_name or self.DEFAULT_KEY,
            path=self._default_path,
            registered=boot_profile_name is not None,
        )
        # Then every registered profile (deduped by namespace against the boot
        # collection — a registered boot collection shows once, by its name).
        for profile in registry.profiles:
            _add(
                name=profile.name,
                path=os.path.abspath(os.path.expanduser(profile.path)),
                registered=True,
            )
        return rows

    # -- status + lifecycle ---------------------------------------------------

    def active_keys(self) -> list[str]:
        """The currently-assembled collection keys (those with a live harness)."""
        return list(self._harnesses)

    def active_harnesses(self) -> list[Harness]:
        """Every currently-assembled harness (for teardown / status fan-out)."""
        return list(self._harnesses.values())

    def get_assembled(self, key: str) -> Harness | None:
        """The harness for an already-assembled key, or None (no assembly)."""
        return self._harnesses.get(key)

    @property
    def default_harness(self) -> Harness:
        """The daemon's boot collection harness — the operational routes
        (embedding start/stop, index rebuild/save, recognition) act on it, and
        it owns the shared embedding runtime."""
        return self._harnesses[self._default_key]

    async def close(self) -> None:
        """Tear down every assembled harness (the default last, so it stops the
        shared runtime after the routed ones have detached)."""
        for key, harness in list(self._harnesses.items()):
            if key == self._default_key:
                continue
            with contextlib.suppress(Exception):
                await harness.close()
        with contextlib.suppress(Exception):
            await self.default_harness.close()
