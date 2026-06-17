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
import contextlib
import functools
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shrike_native

from shrike import cache_layout
from shrike.actions import ACTIVATION_MARGIN
from shrike.collection import CollectionWrapper
from shrike.derived import DerivedTextStore, NativeDerivedEngine
from shrike.embedding import EmbeddingRuntime
from shrike.embedding_base import EmbedderBackend
from shrike.profiles import MODALITIES
from shrike.registry import Registry

logger = logging.getLogger("shrike.kernel")

# Query-embedding LRU size (#181): repeated/backspace-retyped queries (and the
# Enter-after-pause commit) reuse the vector instead of re-embedding. Keyed by
# backend identity, so a model swap never serves a stale-space vector.
EMBED_CACHE_SIZE = 128


class KernelConfigError(Exception):
    """A caller-actionable configuration error (the HTTP host maps it to a 400)."""


# The kernel source string each recognition purpose lands under (#485) — the
# routing key the harness passes to `attach_recognizer_with`, also the `/status`
# map key. OCR's is unchanged (`"ocr"`).
RECOGNITION_OCR = "ocr"
RECOGNITION_DESCRIBE = "vlm"
# ASR isn't wired into the kernel yet (#485 gates it), but the coverage matrix
# names the source it WILL land under so an attached ASR engine lights up
# text→audio honestly once it's integrated. (Today nothing populates this key,
# so audio stays `unavailable` — the honest state.)
RECOGNITION_ASR = "asr"

# Which recognition source derives TEXT from which TARGET modality into the text
# vector space (#235). An attached, ready engine on one of these sources is what
# makes its target reachable `via_derived_text` from a text query: OCR text and
# VLM prose both land in the text space for images; an ASR transcript for audio.
_DERIVED_TEXT_SOURCES: dict[str, frozenset[str]] = {
    "image": frozenset({RECOGNITION_OCR, RECOGNITION_DESCRIBE}),
    "audio": frozenset({RECOGNITION_ASR}),
}


def _coverage_matrix(
    served: frozenset[str],
    ready_recognizers: frozenset[str],
    spaces: Sequence[frozenset[str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the cross-modal coverage matrix (#235): query-modality →
    target-modality → ``native`` | ``via_derived_text`` | ``unavailable``.

    ``served`` is the UNION of modalities the live embedding spaces embed — it
    governs reachability (whether a query modality is embeddable at all, and
    whether a text space exists to derive into). ``spaces`` is the PER-SPACE
    breakdown (#229/#235): one ``frozenset`` per live embedding space, used for
    the ``native`` cell, because *native* means **one single space embeds BOTH**
    modalities — two DISJOINT single-modality spaces (a dedicated text space + a
    separate image-only space) must NOT read text↔image native off the union.
    ``spaces=None`` treats ``served`` as one implicit space (the N≤1 / single-
    backend case), so the cell reduces to ``q in served and t in served`` — the
    pre-#235 behaviour, byte-identical.

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
    # space (N≤1 / single-backend → byte-identical to the pre-#235 union check).
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
    """One attached recognition engine's harness-side state (#485): the backend
    kind (`apple`/`describe-remote`), the lifecycle state (the
    `RecognitionEngineStatus` enum — `ready`/`error` today, the schema also
    carries `unavailable`/`building` for future use), and its fingerprint when
    known. Per-purpose so `/status` reports one row per engine."""

    backend: str
    state: str = "ready"
    fingerprint: str | None = None


class DedupStatsRecorder:
    """Rolling dedup best-match statistics (#207): one sample per upsert
    draft note — the best SEMANTIC neighbor cosine, or a no-match tick.
    The calibration feedstock for the dedup threshold; loop-confined (the
    actions record on the event loop), so no lock. Process-lifetime only —
    a restart starts fresh (durable accumulation is a later refinement)."""

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


class KernelIndexView:
    """The search-facing index view, live over the kernel.

    The actions' search path needs: availability/state/progress, the engine
    handle, query embedding (host-side, via the runtime's backend), and the
    activation stats — all of which the kernel owns now. This view reads them
    live (``index_status_json``) instead of holding facade copies.
    """

    def __init__(self, kernel: Any, runtime: EmbeddingRuntime) -> None:
        self._kernel = kernel
        self._runtime = runtime
        self._engine_handle = kernel.engine_handle()
        self._embed_cache: OrderedDict[tuple[int, str], list[float]] = OrderedDict()

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
        # LRU per (backend identity, query). Accessed from to_thread workers
        # since #445 (no longer loop-confined): safe under the GIL — each dict
        # op is atomic, and a race costs at worst a redundant embed or an
        # off-by-one eviction, never corruption.
        base = id(backend)
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[str] = []
        missing_at: list[int] = []
        for i, text in enumerate(texts):
            cached = self._embed_cache.get((base, text))
            if cached is not None:
                self._embed_cache.move_to_end((base, text))
                out[i] = cached
            else:
                missing.append(text)
                missing_at.append(i)
        if missing:
            fresh = backend.embed_texts(missing)
            for i, text, vector in zip(missing_at, missing, fresh, strict=True):
                out[i] = vector
                self._embed_cache[(base, text)] = vector
            while len(self._embed_cache) > EMBED_CACHE_SIZE:
                self._embed_cache.popitem(last=False)
        return [v for v in out if v is not None]

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
        wrapper: CollectionWrapper,
        runtime: EmbeddingRuntime,
        derived: DerivedTextStore,
        media_read: Any,
        media_exists: Any,
        owns_runtime: bool = True,
        secondary_runtimes: Sequence[EmbeddingRuntime] | None = None,
        cross_space_floor_margin: float = ACTIVATION_MARGIN,
        shared_llama_manager: Any = None,
    ) -> None:
        self.kernel = kernel
        self.wrapper = wrapper
        self.runtime = runtime
        # The shared llama.cpp ROUTER manager (#567): when N remote/no-endpoint
        # embedder spaces share ONE managed server (managed.llama_server.
        # models_dir), this is the single `LlamaServerManager.router(...)` they
        # all talk to over loopback — spawned once, owned here, stopped only by
        # the owner on close(). None = no router (the N=1 / single-managed /
        # endpoint / onnx cases — every shape but the shared router). It is
        # started at the top of start_embedding, BEFORE any router-managed
        # remote backend (whose connectivity-proof embed needs it listening).
        self._shared_llama_manager = shared_llama_manager
        # The cross-space image floor margin (#580): the precision/recall dial
        # folded into the secondary floor's `mean + margin·std` at calibration.
        # Default 1.0 (ACTIVATION_MARGIN) = today's behaviour; resolved harness-
        # side from `search.cross_space_fusion.margin` / the env twin.
        self.cross_space_floor_margin = cross_space_floor_margin
        # Additional embedding spaces (#233): the SECONDARY runtimes, attached
        # to their own kernel embed-space keys alongside the primary. Empty in
        # the N=1 case (the default), so single-space behaviour — and every
        # on-disk artifact + the index/search path, which consume only the
        # PRIMARY space this PR — is byte-identical. The fan-out (an N-per-space
        # index + cross-space fusion) is PR-B/C; here the secondaries are
        # attached + reconciled but the index path still reads the primary.
        self.secondary_runtimes: list[EmbeddingRuntime] = list(secondary_runtimes or [])
        self.derived = derived
        self._media_read = media_read
        self._media_exists = media_exists
        # Whether this harness OWNS the embedding runtime's lifecycle (#68).
        # In multi-collection mode one runtime (the llama-server subprocess /
        # onnx session) is SHARED across the per-collection harnesses — only
        # the owner stops it on close(), so a routed collection's teardown
        # never kills the shared embedder out from under the others. Single-
        # collection mode (the default) owns its runtime, unchanged.
        self.owns_runtime = owns_runtime
        self.index_view = KernelIndexView(kernel, runtime)
        self.dedup_stats = DedupStatsRecorder()
        # Background maintenance tasks (#471): tracked so close() drains them
        # (no destroyed-pending teardown) and tests can settle deterministically.
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        # Recognition (#228/#221/#485): per-purpose engine state, keyed by the
        # kernel source string (`"ocr"`/`"vlm"`). Empty = nothing attached (a
        # distinct, representable state from "attached but errored"). Each row
        # carries its backend kind, state, and (when known) fingerprint for
        # per-engine `/status`. One shared background sweep task drives every
        # attached purpose (the kernel's `recognize_pending` sweeps all of them
        # per batch); tracked so close() drains it.
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
    ) -> Harness:
        """Open the kernel on the running loop. Scheduling is the kernel's
        own (#374 — the owned tokio runtime spawns the collection actor);
        the harness assembles services and awaits completions. The
        ``index_save_*`` tuning reaches the kernel's debounced saver
        (#355 item 2); ``None`` keeps the built-in defaults. ``owns_runtime``
        is False for a per-collection harness sharing one embedding runtime
        (#68) — then close() leaves the shared runtime running.
        ``secondary_runtimes`` are the additional embedding spaces (#233);
        empty (the default) is the byte-identical N=1 case.

        The host ``DerivedTextStore`` (the ``/status`` read surface) is built
        **here, AFTER the kernel opens the collection** (#562), at the same
        ``cache_layout.derived_db_path`` the kernel's ``DerivedEngine`` opened —
        not by a caller before assembly. The path-derived namespace
        canonicalizes the collection path, and that canonicalization differs by
        whether the file EXISTS at computation time (an existing file → realpath,
        which folds a symlinked prefix like macOS ``/var/folders`` →
        ``/private/var/...``; an absent file → a lexical abspath that does NOT).
        A fresh collection computed before open (the old caller order) hashed
        under the abspath namespace while the kernel — which creates the file
        during open — hashed under the realpath one, so the host ``/status`` read
        an EMPTY store while the kernel's search store held the rows. Building
        after open means the file exists for both, so both realpath to the SAME
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
            # canonicalization matches the kernel's — #562). Same computation the
            # kernel used internally, so they open one shared shrike.db.
            derived_path = cache_layout.derived_db_path(cache_dir, collection_path)
            derived = DerivedTextStore(
                path=Path(derived_path), engine_factory=derived_engine_factory
            )
        wrapper = CollectionWrapper.over_kernel(
            kernel, collection_path, cooperative=cooperative, hold_seconds=hold_seconds
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
            self._spawn_bg(self._rebuild_derived())
        self._spawn_bg(self._drive_reindex())

    async def _drive_reindex(self) -> None:
        if await self.kernel.reindex_if_needed():
            logger.info("Collection changed while idle; index reconciled")
            await self._recalibrate_secondary_floors()

    async def _maybe_build_derived(self) -> None:
        """Cheap col_mod probe; the rebuild itself is fire-and-forget — boot
        and /reload must not block on the FTS5 build (#451 review: the old
        thread path never did; the store reports BUILDING and searches fall
        back until ready). The claim in _rebuild_derived dedupes double-fires."""
        col_mod = await self.wrapper.col_mod()
        if self.derived.check_drift(col_mod):
            self._spawn_bg(self._rebuild_derived())

    def _spawn_bg(self, coro: Any) -> asyncio.Task[Any]:
        """Spawn tracked background maintenance (#471): logged on failure,
        discarded from the set on completion, drained by close()."""
        task: asyncio.Task[Any] = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(_log_task_failure)
        return task

    async def settle_background(self) -> None:
        """Await in-flight background maintenance (drift rebuilds, reindex
        drivers) — deterministic boots for tests and operational verbs."""
        while self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    async def _rebuild_derived(self) -> None:
        # Kernel-side rebuild (#445): the field rows used to round-trip the
        # whole collection Rust→Python→Rust (~150-250MB transient at 100k
        # notes). The kernel op collects + builds crate-side; the store here
        # only drives the status state machine around the await.
        if not self.derived.claim_external_build():
            return
        col_mod: int | None = None
        try:
            rows, col_mod = await self.kernel.rebuild_derived()
            logger.info("Derived-text store rebuilt kernel-side (%d rows)", rows)
        except Exception:
            col_mod = None
            logger.exception("Kernel-side derived rebuild failed")
        finally:
            self.derived.settle_external_build(col_mod)

    # -- status ------------------------------------------------------------------

    def collection_status(self) -> dict[str, Any]:
        """The cheap per-collection status fields for a multi-collection row
        (#68): held (the cooperative-lock state), the index state, and the last
        col_mod the index saw. No embedding health probe (that's the default
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
        # Per-space embedding health (#681): the primary runtime PLUS every
        # secondary space (#233), not just the primary — a multi-space profile
        # has more than one embedder, each its own /status entry. health() may
        # probe llama-server over HTTP, so every space's probe rides to_thread.
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
        # Per-engine recognition status (#485): a map keyed by source — one row
        # per attached purpose, each {state, backend, fingerprint}. An empty map
        # is "nothing attached" (distinct from an attached-but-errored engine).
        status["recognition"] = {
            source: {
                "state": eng.state,
                "backend": eng.backend,
                "fingerprint": eng.fingerprint,
            }
            for source, eng in self._recognition_engines.items()
        }
        # The cross-modal coverage matrix (#498/#235): for each (query, target)
        # modality pair, how the target is reachable — `native` (one live space
        # embeds both), `via_derived_text` (a ready recognizer derives text from
        # the target into the text space), or `unavailable`. Derived from the
        # live embedding spaces (one today; the union is forward-compatible) and
        # the attached, ready recognizers. All-`unavailable` when embedding is
        # down, so the shape is stable for clients.
        # Per-space modality sets (#235): the primary runtime + every live
        # secondary space. `native` is computed per-space (one space embeds
        # both modalities), while `served` (their union) governs reachability.
        # Empty when embedding is down → every cell unavailable.
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
        # Per-modality sub-index breakdown (#684): the kernel reports each
        # sub-index's own size/ndim (the aggregate above collapses them). Pass it
        # straight through to IndexModalityStat (text-first, kernel-ordered).
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
            await self._recalibrate_secondary_floors()
            return {"status": "complete", "size": 0}
        self._spawn_bg(self._rebuild_then_calibrate())
        return {"status": "started", "total": total_notes}

    async def _rebuild_then_calibrate(self) -> None:
        """Full index rebuild, then recalibrate the secondary floors (#576) —
        the secondary image vectors are fresh, so the floor must be re-derived."""
        await self.kernel.rebuild_index()
        await self._recalibrate_secondary_floors()

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
        # Spawn the shared llama.cpp router ONCE before any router-managed remote
        # backend (#567): the primary + secondary remote spaces all talk to it
        # over loopback, and their connectivity-proof embed at start() needs it
        # already healthy. router /health is 200 before any model lazy-loads, so
        # this is a fast health-wait. Owned here; stopped only on owner close().
        await self._ensure_shared_router()
        try:
            backend = await asyncio.to_thread(lambda: self.runtime.start(**overrides))
        except (ValueError, ImportError) as e:
            raise KernelConfigError(str(e)) from e
        self._attach(backend)
        # Fan out the secondary spaces (#233): each is its own kernel embed
        # space, attached by its content-fingerprint key. A secondary that
        # fails to start degrades ONLY its space — never the primary, which is
        # already attached and serving the index/search path this PR.
        await self._attach_secondaries(overrides)
        self._spawn_bg(self._drive_boot_reindex())
        return {
            "status": "started",
            "embedding": await asyncio.to_thread(self.runtime.health),
            "index": self._index_status(),
        }

    async def _ensure_shared_router(self) -> None:
        """Spawn the shared llama.cpp router manager if one is configured and
        not yet running (#567). Idempotent — the native manager's own ``start``
        no-ops a live child, and we additionally guard so a repeat call doesn't
        block on the health-wait. Off the loop (spawn + health-wait blocks)."""
        mgr = self._shared_llama_manager
        if mgr is None:
            return
        if await asyncio.to_thread(mgr.running):
            return
        await asyncio.to_thread(mgr.start)

    async def _attach_secondaries(self, overrides: dict[str, Any]) -> None:
        """Start + attach every SECONDARY embedding space (#233), best-effort:
        a space that fails to start (bad model, missing dep, unreachable
        endpoint) is logged and skipped — only that space degrades, the rest
        (and the primary) stay live. No-op in the N=1 case (no secondaries)."""
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
        """A secondary space's explicit key (#233): the started backend's model
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
        harness's kernel and reconcile its index (#68 multi-collection).

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
        """Attach the backend to a kernel embed SPACE (#342/#233). A backend
        exposing ``native_embedder()`` (the onnx/clip/llama facades — every
        production backend) hands over a native composition — kernel embeds
        then never re-enter Python; a custom/test backend without one is
        captured behind the PyEmbedder dispatch seam. ``space_key`` pins the
        space identity (#233); ``None`` (the primary / N=1 case) lets the kernel
        key off the embedder's own fingerprint — byte-identical to the
        single-slot attach."""
        native = getattr(backend, "native_embedder", None)
        embedder = native() if callable(native) else shrike_native.PyEmbedder.capture(backend)
        self.kernel.attach_embedder(
            embedder, self._media_read, self._media_exists, space_key=space_key
        )

    def attach_recognizer(self, backend: Any, purpose: str = RECOGNITION_OCR) -> None:
        """Attach an OCR/describe backend (#228/#485) for a recognition
        ``purpose`` (the kernel source — ``"ocr"`` / ``"vlm"``) — the second
        #342 slot, now keyed by purpose. A native engine
        (``AppleVisionRecognizer`` in engine-apple builds, ``RemoteDescriber``
        in engine-remote builds) goes to the kernel directly — recognition
        then never re-enters Python; any other object satisfying the
        RecognizerBackend contract (a blocking ``recognize(items)`` plus
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
        back-compat default); a source string detaches that purpose (#485)."""
        if purpose is None or purpose == RECOGNITION_OCR:
            self.kernel.detach_recognizer()
        else:
            self.kernel.detach_recognizer_for(purpose)

    async def recognition_sweep(
        self, batch_size: int = 8, max_batches: int | None = None
    ) -> dict[str, Any]:
        """Drive bounded recognition sweeps until nothing is pending (#228).

        Each kernel call recognizes at most ``batch_size`` images then yields
        the executor, so collection ops interleave; the harness runs this as
        a background task (the reindex discipline). Returns the final report
        plus the total stored across the run."""
        total_stored = 0
        batches = 0
        while True:
            report: dict[str, Any] = json.loads(await self.kernel.recognize_pending(batch_size))
            total_stored += int(report.get("stored", 0))
            batches += 1
            # ``batches`` is the count of kernel sweep calls this run made — the
            # observable that distinguishes the no-progress STOP (returns after
            # one no-progress batch) from a livelock REGRESSION (would re-take
            # the same window every call). Surfaced so a test can assert the
            # driver stopped by logic, not by a wall-clock timeout (#525).
            if report.get("status") != "ran" or int(report.get("remaining", 0)) == 0:
                report["total_stored"] = total_stored
                report["batches"] = batches
                if total_stored:
                    logger.info(
                        "Recognition sweep stored %d item(s) over %d batch(es)",
                        total_stored,
                        batches,
                    )
                return report
            if int(report.get("recognized", 0)) == 0:
                # No progress: nothing in this batch was drainable (an
                # unreadable prefix of the pending order — skipped items stay
                # pending, so the next batch would re-take the same window
                # forever). Stop here; the next sweep trigger (boot, /reload,
                # cooperative re-acquire) retries when the read may have
                # healed. Keyed on recognized == 0, NOT stored == 0: a batch
                # that recognized items but gated them all out did real work
                # (the reads drained, recognition ran) and must not stop the
                # sweep.
                report["total_stored"] = total_stored
                report["batches"] = batches
                logger.warning(
                    "Recognition sweep stopped on a no-progress batch "
                    "(%d item(s) still pending, unreadable)",
                    int(report.get("remaining", 0)),
                )
                return report
            if max_batches is not None and batches >= max_batches:
                report["total_stored"] = total_stored
                report["batches"] = batches
                return report

    def start_recognition(self, kind: str) -> None:
        """Construct + attach an OCR backend by kind (#221) and launch a
        background sweep. Degrades to an 'error' state row on a missing
        dependency (the engine isn't compiled in) or an unknown kind — never
        kills boot. Byte-identical OCR path to before #485 (the OCR purpose,
        the legacy ``--ocr-backend`` flow)."""
        from shrike.recognition import make_recognizer

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
        """Construct + attach the remote VLM describe engine (#433/#485) for the
        ``describe`` purpose (image→prose into the text embedding space,
        vector-only) and launch the shared background sweep. Reports an 'error'
        state row when construction fails (a missing build feature / missing key
        env) OR when the endpoint is unreachable at attach — never kills boot.

        An unreachable endpoint still ATTACHES (the engine and its degenerate
        fingerprint): the sweep's chunk-Err-aborts contract leaves the backlog
        pending and a later sweep retries once the endpoint is up. We surface it
        as 'error' rather than 'ready' so a degraded engine is visible (#485) —
        rows minted under the degenerate fingerprint re-derive once on the next
        restart, when model_info resolves and the fingerprint sharpens."""
        from shrike.recognition import make_describe_recognizer

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
        """Launch the shared background sweep task if one isn't already running
        (#485). The kernel's ``recognize_pending`` sweeps EVERY attached purpose
        per batch, so one driver serves all engines. Tracked (#471): close()
        cancels-and-awaits it like every other maintenance task, so a mid-sweep
        teardown never leaves a destroyed-pending task."""
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
        a source string detaches just that purpose (#485). The shared sweep
        task is cancelled whenever no engine remains."""
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
        if await self.kernel.reindex_if_needed():
            logger.info("Index reconciled after embedding start")
        await self._recalibrate_secondary_floors()

    async def _recalibrate_secondary_floors(self) -> None:
        """Recompute the secondary cross-space image activation floors (#576).

        A dedicated CLIP secondary is image-only, so the engine's own #201b
        calibration (text→image) finds no text vectors there; this harness-driven
        pass CLIP-text-embeds a sample of note texts through each secondary's own
        backend and fires them at its image vectors to derive the floor
        (``mean + margin·std``, the #580 margin dial). Best effort: a failure
        leaves the prior floors and only weakens the floor admission to "no
        floor" (admit any non-empty space) — never breaks search. No-op in the
        N=1 case (the kernel returns an empty list when there are no
        secondaries)."""
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
        # Stop the secondary spaces' runtimes too (#233) — detach already
        # cleared their kernel slots; this releases their backend resources.
        for rt in self.secondary_runtimes:
            await asyncio.to_thread(rt.stop)
        # Stop the shared router too (#567): `embedding stop` frees GPU/RAM, and
        # the router process is the resource the remote spaces held. start_
        # embedding re-spawns it (idempotent) on the next cycle. Owner-only
        # (mirrors close()): a routed (#68) harness never owns the runtime, so it
        # must not kill the shared router out from under the owner + siblings.
        if self.owns_runtime and self._shared_llama_manager is not None:
            await asyncio.to_thread(self._shared_llama_manager.stop)
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
            if rebuilding:
                # The reconcile re-embedded secondary image vectors, so the
                # cross-space image floor (#576/#580) must be re-derived — every
                # other reindex path (_drive_reindex/_rebuild_then_calibrate/
                # _drive_boot_reindex) recalibrates; /reload was the outlier (#596).
                # No-op at N=1 (the kernel returns an empty list with no secondaries).
                await self._recalibrate_secondary_floors()
        return {"status": "reloaded", "col_mod": col_mod, "rebuilding": rebuilding}

    async def close(self) -> None:
        """Tear down: background tasks, derived, embedding, then the kernel
        (flushes the index)."""
        # Cancel-and-drain the tracked maintenance tasks (#471): a rebuild
        # mid-flight detaches kernel-side (the op completes; detach never
        # aborts), but the Python task must not be destroyed pending.
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
        self.derived.close()
        if self.owns_runtime:
            await asyncio.to_thread(self.runtime.stop)
            # Stop the secondary spaces' runtimes too (#233) — owned alongside
            # the primary; a shared (#68) harness leaves them to the owner.
            for rt in self.secondary_runtimes:
                await asyncio.to_thread(rt.stop)
            # Stop the shared llama.cpp router LAST (#567): the router-managed
            # remote backends (now stopped) talked to it, so the process they
            # depend on outlives them by exactly this teardown. Owner-only —
            # a shared (#68) routed harness never owns the runtime, so it never
            # reaches here and never kills the router out from under siblings.
            if self._shared_llama_manager is not None:
                await asyncio.to_thread(self._shared_llama_manager.stop)
        self.wrapper.close()
        # kernel.close drains the collection actor (#374): nothing in flight
        # when this returns — the interpreter-teardown guard.
        await self.kernel.close()


def _log_task_failure(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background kernel task failed: %s", exc)


# ── multi-collection routing (#68) ──────────────────────────────────────────


class RoutingError(Exception):
    """A caller-actionable routing error: the selector named an unknown
    profile, or no collection could be resolved (no selector and no active
    default). The HTTP host maps it to a 400 / ToolInputError — never a bug."""


@dataclass(frozen=True)
class HarnessParams:
    """The collection-independent assembly parameters every per-collection
    harness shares (#68). The shared base ``cache_dir`` (NOT per-collection —
    isolation comes from per-artifact namespacing: the index via #67, the
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
    """Routes per-call to a per-collection :class:`Harness`, lazily assembled
    (#68 — the multi-collection capstone).

    One daemon, many collections: each registered/active collection gets its
    own ``AsyncKernel`` + namespaced index (#67) + per-collection derived store,
    assembled on first route and governed by its own cooperative-lock
    idle-release (#64) — only the collection(s) mid-operation hold a lock. The
    N kernels share one process-global tokio runtime (verified) and one
    embedding runtime (the llama-server subprocess / onnx session is expensive;
    one is shared, attached to each kernel's embed slot).

    Selection is **per-call and stateless** (consistent with ``stateless_http``):
    a selector resolves name → path via a **live, re-readable** registry view
    (contract #2 — re-read on demand so a ``profile create`` in one session routes
    in the same session), defaulting to the registry's active profile; no
    server-side mutable "current collection". The registry's stored path is
    routed THROUGH :mod:`shrike.cache_layout` (contract #1) so canonicalization
    is the equalizer — never a raw-abspath-derived index path.

    The **default collection** (the one the daemon booted with, ``--collection``)
    is seeded as a ready harness so today's single-collection behavior — boot
    embedding, the operational routes, the recognition sweep — is unchanged; it
    owns the shared embedding runtime's lifecycle. Routed (non-default)
    collections attach the same shared backend to their own kernel and never
    stop it on teardown.
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

    # -- the live registry view (contract #2) --------------------------------

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
        # The per-collection derived store (#547, <cache_dir>/derived/<ns>/shrike.db)
        # is built by assemble AFTER the kernel opens this collection (#562), so
        # the host namespace canonicalizes the now-existing file identically to
        # the kernel's DerivedEngine — never pre-computed here from a maybe-absent
        # path. The base cache_dir is SHARED (so #67's index namespace under it is
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
        )
        # Boot WITHOUT starting embedding (the shared runtime is already
        # started by the default harness); attach its backend so search works,
        # then reconcile drift. A routed collection in a cooperative daemon
        # releases its lock after the idle window like any other.
        await harness.boot(start_embedding=False)
        await harness.attach_shared_embedder()
        return harness

    async def resolve_bundle(self, selector: str | None) -> Any:
        """The per-call action bundle for ``selector`` (#68 — the resolver
        ``register_tools`` is handed).

        Routes to the selector's harness (lazily assembling it) and hands back
        its ``CollectionBundle`` (wrapper + index view + kernel + derived +
        dedup recorder) — exactly the handles an action operates on. A
        :class:`RoutingError` (unknown selector) propagates to the action
        layer, which maps it to a clean ``ToolInputError``."""
        from shrike.actions import CollectionBundle

        harness = await self.harness_for(selector)
        return CollectionBundle(
            wrapper=harness.wrapper,
            index=harness.index_view,
            derived=harness.derived,
            kernel=harness.kernel,
            dedup_stats=harness.dedup_stats,
        )

    def status_rows(self) -> list[dict[str, Any]]:
        """One status row per KNOWN collection (#68 S3): the daemon's boot/
        default collection plus every registered profile, deduped by
        path-namespace (a registered profile that IS the boot collection shows
        once). An assembled collection contributes its live held/index/col_mod;
        a registered-but-never-routed one is ``active=False`` with no index
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
