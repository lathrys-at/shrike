"""The gold workloads the perf harness times.

A workload is one operation, repeated against a booted harness over a fixed
corpus. The data-plane workloads drive the **actions API** directly
(``booted.call(<action>, ...)``) — the transport-neutral maintained serving path
(write + index + derived + watermark, search fusion), off the FastMCP transport.
We benchmark the system, not Anki and not the wire adapter.

Each read/write op comes in two shapes that mirror the batching axis:

- ``search-{batch,seq}`` — one ``search_notes`` call with N queries (one batched
  query-embed) vs N calls of one query (N embeds).
- ``upsert-{batch,seq}`` — one ``upsert_notes`` call with N notes (one
  transaction) vs N calls of one note (one journal fsync each).
- ``delete-{batch,seq}`` — one ``delete_notes`` call with N ids vs N calls of one
  id (one maintained delete op each).

Plus ``rebuild`` (the O(collection) index rebuild), ``reconcile`` (out-of-band
drift recovery), and ``churn`` (sustained insert/delete/search — the steady-state
regime that fragments the FTS5 index, which a build-then-search run never sees;
its *response* p50 drift across the run is the index-maintenance signal #938 reads).
A true *ingest* (cold import of a synthetic package) is measured
by ``driver.measure_ingest`` — it owns its own boot lifecycle, so it isn't a
``Workload`` here. The heavier *sync* and OCR sweeps are tracked as their own
issues.

Two-phase timing: a write returns once committed with index/derived maintenance
*enqueued*, so timing only ``run_one`` would miss the drain. The write workloads
expose a ``settle`` coroutine — a second, separately-timed phase that drives the
kernel to quiescence — so the harness reports both the time-to-response and the
time-to-completion (and their per-iteration total). Read workloads (search,
rebuild) have no asynchronous tail and report ``response`` only.

``mutates`` marks a workload that changes the collection
(upsert/delete/reconcile); the runner orders read-only workloads first so a
single boot stays representative. Running MORE THAN ONE mutating workload in a
single invocation compounds the collection state across them, so for a clean
absolute number run one mutating workload per invocation (comparisons across runs
stay valid regardless — the mutation is deterministic and identical on both
sides).

A workload MAY define an optional ``prepare(booted, iteration)`` coroutine: per-
iteration setup run UNTIMED before each timed ``run_one`` (see ``driver.Workload``).
"""

from __future__ import annotations

import random

from shrike.schemas import NoteInput
from tests.manual.perf.corpus import choose, n_topics
from tests.manual.perf.driver import Booted, Workload


def _query(rng: random.Random, topics: int) -> str:
    """A short synthetic query: 1-4 terms drawn from one random deck's vocabulary,
    so it hits that deck (common terms in the draw still match broadly across decks,
    domain terms match ~that one) — the realistic mix a real search produces."""
    topic = rng.randrange(topics)
    return " ".join(choose(rng, rng.randint(1, 4), topic))


#: N — the operations each data-plane workload performs per timed iteration: the
#: search queries, the upserted/deleted notes, the reconcile drift. The complement
#: to the runner's M repeats (a workload runs N ops × M repeats), scaled uniformly
#: across the workloads by ``run.py --ops``. ``rebuild`` is the one exception — an
#: O(collection) pass with no per-op N. 100 is a representative add/edit-session
#: size and the upsert/delete batch cap.
DEFAULT_OPS = 100


# ``search-batch`` issues all N queries in ONE call. At the default N that is past
# the action's wire-level 50-query cap, but the harness drives the action *impl*
# directly (below FastMCP's arg validation), so the cap never applies and the impl
# — the path being benchmarked — sees the whole batch.
class _SearchWorkload:
    """Drive the real ``search_notes`` action over a fixed bank of synthetic
    queries — the headline read path (embed the query, fan out, fuse). Read-only,
    so no settle phase. Subclasses choose batched vs sequential issue. ``run_one``
    reports the number of queries issued (the work done), not the matches returned,
    so ``items`` is the per-iteration query count like the other workloads."""

    mutates = False

    def __init__(self, *, count: int = DEFAULT_OPS, limit: int = 20, corpus_size: int = 0) -> None:
        self._count = count
        self._limit = limit
        self._topics = n_topics(corpus_size)
        self._batches: list[list[str]] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        # Precompute every iteration's queries (untimed) so the string formatting
        # never lands in the timed region.
        rng = random.Random(0x5EED)
        self._batches = [
            [_query(rng, self._topics) for _ in range(self._count)] for _ in range(iterations)
        ]

    async def _search(self, booted: Booted, queries: list[str]) -> None:
        await booted.call(
            "search_notes", queries=queries, ids=[], limit=self._limit, tags=[], exclude_ids=[]
        )


class SearchBatchWorkload(_SearchWorkload):
    """One ``search_notes`` call carrying all ``count`` queries — one batched
    query-embed for the whole set."""

    name = "search-batch"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        await self._search(booted, self._batches[iteration])
        return self._count


class SearchSeqWorkload(_SearchWorkload):
    """``count`` separate ``search_notes`` calls of one query each — the delta
    from ``search-batch`` is the per-call assembly + per-query embed batching
    elides."""

    name = "search-seq"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        for q in self._batches[iteration]:
            await self._search(booted, [q])
        return self._count


class RebuildWorkload:
    """Rebuild the whole vector index — the O(collection) maintenance path a perf
    audit watches for full-collection scans. Synchronous to completion (boot/
    rebuild paths refresh synchronously), so no settle phase."""

    name = "rebuild"
    mutates = False

    def __init__(self, *, count: int = DEFAULT_OPS, corpus_size: int = 0) -> None:
        # A rebuild is one O(collection) pass; the per-op N (``--ops``) doesn't
        # apply. The uniform ``count`` keyword lets the runner build every workload
        # identically — here it's accepted and discarded.
        del count

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        await booted.harness.kernel.rebuild_index()
        return 1  # one full-collection rebuild


def _upsert_note(iteration: int, j: int) -> NoteInput:
    # A fresh deck/tag so upserted notes never collide with the corpus; the text
    # is deterministic per (iteration, j) and the front is globally unique.
    rng = random.Random((iteration << 20) ^ j)
    body = " ".join(choose(rng, 12))
    return NoteInput(
        deck="Perf::Upsert",
        note_type="Basic",
        tags=["perf-upsert"],
        fields={"Front": f"upsert {iteration}-{j}", "Back": body},
    )


class _UpsertWorkload:
    """Upsert ``count`` fresh notes against the loaded corpus through the
    maintained ``upsert_notes`` action. The write returns once committed with the
    index/derived maintenance enqueued, so a ``settle`` phase times the drain to
    quiescence separately.

    Each iteration ADDS ``count`` notes, so the collection grows across the run
    (the recorded ``corpus_size`` is the *starting* size); identical deterministic
    growth on both sides keeps comparisons valid. Run ``upsert-batch`` and
    ``upsert-seq`` in SEPARATE invocations (a fresh boot each) for an
    apples-to-apples batch-vs-sequential comparison."""

    mutates = True

    def __init__(self, *, count: int = DEFAULT_OPS, corpus_size: int = 0) -> None:
        self._count = count
        self._batches: list[list[NoteInput]] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        # Precompute every iteration's notes (untimed) so neither the random text
        # generation nor the NoteInput construction lands in the timed region.
        self._batches = [
            [_upsert_note(i, j) for j in range(self._count)] for i in range(iterations)
        ]

    async def settle(self, booted: Booted, iteration: int) -> None:
        await booted.harness.kernel.settle()


class UpsertBatchWorkload(_UpsertWorkload):
    """One ``upsert_notes`` call of ``count`` notes — the batched write path (one
    transaction). Pairs with ``upsert-seq`` to show the batching win."""

    name = "upsert-batch"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        await booted.call("upsert_notes", notes=self._batches[iteration])
        return self._count


class UpsertSeqWorkload(_UpsertWorkload):
    """``count`` separate ``upsert_notes`` calls of one note each — the delta from
    ``upsert-batch`` is the per-call/per-transaction overhead batching elides (one
    journal fsync per note vs one for the whole batch)."""

    name = "upsert-seq"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        for note in self._batches[iteration]:
            await booted.call("upsert_notes", notes=[note])
        return self._count


def _delete_note(j: int) -> NoteInput:
    rng = random.Random(0xDE1E7E ^ j)
    return NoteInput(
        deck="Perf::Delete",
        note_type="Basic",
        tags=["perf-delete"],
        fields={"Front": f"delete {j}", "Back": " ".join(choose(rng, 8))},
    )


class _DeleteWorkload:
    """Delete notes by id through the maintained ``delete_notes`` action. Setup
    ingests a disposable pool (iterations × count fresh notes, so deletes never
    touch the corpus) through the SAME maintained path and settles, so the pool is
    fully indexed before timing — each delete then drops real vectors/fingerprints,
    not a no-op. Each iteration deletes its own disjoint slice; a ``settle`` phase
    times any drain after the delete returns."""

    mutates = True

    def __init__(self, *, count: int = DEFAULT_OPS, corpus_size: int = 0) -> None:
        self._count = count
        self._ids: list[int] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        pool = [_delete_note(j) for j in range(iterations * self._count)]
        resp = await booted.call("upsert_notes", notes=pool)
        self._ids = [r.id for r in resp.results if r.status in ("created", "updated")]
        # Drain so the pool is indexed before timing — the timed delete then drops
        # real sidecar state rather than touching never-indexed notes.
        await booted.harness.kernel.settle()

    async def settle(self, booted: Booted, iteration: int) -> None:
        await booted.harness.kernel.settle()

    def _slice(self, iteration: int) -> list[int]:
        return self._ids[iteration * self._count : (iteration + 1) * self._count]


class DeleteBatchWorkload(_DeleteWorkload):
    """One ``delete_notes`` call removing ``count`` ids — the batched delete path
    (one maintained op)."""

    name = "delete-batch"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        ids = self._slice(iteration)
        await booted.call("delete_notes", ids=ids)
        return len(ids)


class DeleteSeqWorkload(_DeleteWorkload):
    """``count`` separate ``delete_notes`` calls of one id each — the delta from
    ``delete-batch`` is the per-call/per-op overhead (one maintained delete +
    watermark advance per id vs one for the whole batch)."""

    name = "delete-seq"

    async def run_one(self, booted: Booted, iteration: int) -> int:
        ids = self._slice(iteration)
        for note_id in ids:
            await booted.call("delete_notes", ids=[note_id])
        return len(ids)


def _reconcile_note(iteration: int, j: int) -> dict:
    # A fresh note in its own deck/tag, deterministic per (iteration, j), so each
    # iteration drifts a disjoint set the reconcile sees as new.
    rng = random.Random((iteration << 24) ^ (j ^ 0x5EC0))
    body = " ".join(choose(rng, 12))
    return {
        "deck": "Perf::Reconcile",
        "note_type": "Basic",
        "tags": ["perf-reconcile"],
        "fields": {"Front": f"reconcile {iteration}-{j}", "Back": body},
    }


class ReconcileWorkload:
    """Out-of-band drift recovery — the reconcile path, NOT an in-band write.

    Each iteration first (UNTIMED, in ``prepare``) edits the collection from under
    the server: it writes ``count`` fresh notes STRAIGHT through the collection
    actor (``wrapper.upsert_notes`` bypasses the kernel index path), so ``col.mod``
    moves while the index watermark stays stale — exactly the drift a GUI edit, a
    sync, or a restore leaves behind. The reconcile cost is dominated by that
    changed set (its re-embed) on top of a full-collection fingerprint scan. The
    TIMED ``run_one`` then runs ``reindex_if_needed`` (the *response* phase: detect
    drift, diff fingerprints, enqueue the re-embed of the changed set), and the
    ``settle`` phase drains it to quiescence — so the harness reports both
    detect-and-enqueue and drain-to-done.

    Writing through the same actor makes the new ``col.mod`` live to the reconcile,
    so no reopen is needed; reopen is a collection-reacquire cost, not a reconcile
    cost. Each iteration ADDS ``count`` notes, so the collection grows across the
    run (the recorded ``corpus_size`` is the *starting* size); identical
    deterministic growth on both sides keeps comparisons valid."""

    name = "reconcile"
    mutates = True

    def __init__(self, *, count: int = DEFAULT_OPS, corpus_size: int = 0) -> None:
        self._count = count
        self._batches: list[list[dict]] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        # Precompute the drift notes per iteration (untimed) — the prepare write is
        # untimed too, but this keeps the random text generation out of the loop.
        self._batches = [
            [_reconcile_note(i, j) for j in range(self._count)] for i in range(iterations)
        ]

    async def prepare(self, booted: Booted, iteration: int) -> None:
        # Out-of-band: write through the collection only, leaving the index stale.
        await booted.harness.wrapper.upsert_notes(self._batches[iteration])

    async def run_one(self, booted: Booted, iteration: int) -> int:
        # Detect drift + enqueue the re-embed of the changed set (the settle phase
        # drains it).
        await booted.harness.kernel.reindex_if_needed()
        return self._count

    async def settle(self, booted: Booted, iteration: int) -> None:
        await booted.harness.kernel.settle()


def _churn_note(iteration: int, j: int) -> NoteInput:
    # Deterministic per (iteration, j), in its own deck/tag so churn never collides
    # with the loaded corpus.
    rng = random.Random((iteration << 22) ^ (j ^ 0xC40))
    body = " ".join(choose(rng, 12))
    return NoteInput(
        deck="Perf::Churn",
        note_type="Basic",
        tags=["perf-churn"],
        fields={"Front": f"churn {iteration}-{j}", "Back": body},
    )


class ChurnWorkload:
    """Sustained mixed insert/delete/search — the steady-state regime index
    maintenance exists for, and which a build-then-search benchmark cannot see.

    Each iteration (UNTIMED ``prepare``) churns the FTS5 index: upsert ``count``
    fresh notes through the maintained ``upsert_notes`` path, delete the PREVIOUS
    iteration's ``count`` notes, then settle to quiescence. Across the run, inserts
    add level-0 segments and deletes write delete-keys into new b-trees, so
    fragmentation accumulates the way real use accumulates it — the live churned set
    stays ~``count``, but the index does NOT, absent compaction. The TIMED
    ``run_one`` then searches a fixed query bank, so the *response* p50 across the
    run is the search-latency DRIFT as the index fragments: flat with the folded
    merge (#938), climbing without it. Run the same boot with and without compaction
    for the A/B — the divergence is the maintenance win."""

    name = "churn"
    mutates = True

    def __init__(self, *, count: int = DEFAULT_OPS, corpus_size: int = 0) -> None:
        self._count = count
        self._topics = n_topics(corpus_size)
        self._upserts: list[list[NoteInput]] = []
        self._queries: list[str] = []
        self._ids: list[list[int]] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        # Per-iteration churn notes (untimed text gen) + a fixed query bank sized to
        # the corpus topics (like the search workloads) so the searches actually hit.
        self._upserts = [[_churn_note(i, j) for j in range(self._count)] for i in range(iterations)]
        rng = random.Random(0xC0FFEE)
        self._queries = [_query(rng, self._topics) for _ in range(min(max(self._count, 1), 20))]
        self._ids = [[] for _ in range(iterations)]

    async def prepare(self, booted: Booted, iteration: int) -> None:
        resp = await booted.call("upsert_notes", notes=self._upserts[iteration])
        self._ids[iteration] = [r.id for r in resp.results if r.status in ("created", "updated")]
        if iteration > 0 and self._ids[iteration - 1]:
            await booted.call("delete_notes", ids=self._ids[iteration - 1])
        # Drain so the inserts AND the delete-keys are committed to the index before
        # the timed search — the fragmentation is real, not still pending.
        await booted.harness.kernel.settle()

    async def run_one(self, booted: Booted, iteration: int) -> int:
        await booted.call(
            "search_notes", queries=self._queries, ids=[], limit=20, tags=[], exclude_ids=[]
        )
        return len(self._queries)


#: name → workload class.
WORKLOADS = {
    w.name: w
    for w in (
        SearchBatchWorkload,
        SearchSeqWorkload,
        RebuildWorkload,
        UpsertBatchWorkload,
        UpsertSeqWorkload,
        DeleteBatchWorkload,
        DeleteSeqWorkload,
        ReconcileWorkload,
        ChurnWorkload,
    )
}


def build_workload(name: str, *, ops: int = DEFAULT_OPS, corpus_size: int = 0) -> Workload:
    """Instantiate workload ``name`` with ``ops`` operations per iteration — the N
    the runner scales via ``--ops``. ``corpus_size`` lets the search workload size
    its query topics to the corpus (one deck per ~500 notes); the other workloads
    accept and ignore it (uniform construction, like ``count`` for ``rebuild``)."""
    return WORKLOADS[name](count=ops, corpus_size=corpus_size)
