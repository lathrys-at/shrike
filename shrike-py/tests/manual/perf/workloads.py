"""The gold workloads the perf harness times.

A workload is one operation, repeated against a booted harness over a fixed
corpus: search, rebuild, upsert-batch, upsert-seq, delete, reconcile. The workflows
that need a heavier scenario harness remain a focused follow-up: a true *ingest*
(importing a synthetic .apkg/.colpkg via the package-import path — NOT the upsert
action), *sync*, and OCR sweeps.

``mutates`` marks a workload that changes the collection
(upsert-batch/upsert-seq/delete/reconcile); the runner orders read-only workloads
first so a single boot stays representative. Running MORE THAN ONE mutating workload
in a single invocation compounds the collection state across them, so for a clean
absolute number run one mutating workload per invocation (comparisons across runs
stay valid regardless — the mutation is deterministic and identical on both sides).

A workload MAY define an optional ``prepare(booted, iteration)`` coroutine: per-
iteration setup run UNTIMED before each timed ``run_one`` (see ``driver.Workload``).
"""

from __future__ import annotations

import random

from tests.manual.perf.corpus import VOCAB
from tests.manual.perf.driver import Booted


class SearchWorkload:
    """Drive the real ``search_notes`` action over a fixed bank of synthetic
    queries — the headline read path (embed the query, fan out, fuse)."""

    name = "search"
    mutates = False

    def __init__(self, *, n_queries: int = 64, limit: int = 20) -> None:
        self._limit = limit
        rng = random.Random(0x5EED)
        self._queries = [
            " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 4))) for _ in range(n_queries)
        ]

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        resp = await booted.search(self._queries[iteration % len(self._queries)], limit=self._limit)
        groups = resp.get("results", [])
        return sum(len(g.get("matches", [])) for g in groups)


class RebuildWorkload:
    """Rebuild the whole vector index — the O(collection) maintenance path a perf
    audit watches for full-collection scans."""

    name = "rebuild"
    mutates = False

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        await booted.harness.kernel.rebuild_index()
        return 1  # one full-collection rebuild


# The fresh-note count each upsert iteration writes against the loaded corpus —
# the upsert action measures "add N notes to a {500,5k,50k} collection", not
# "build a collection". 100 is a representative add-session size and a typical
# max batch.
_UPSERT_COUNT = 100


def _upsert_note(iteration: int, j: int) -> dict:
    # A fresh deck/tag so upserted notes never collide with the corpus; the text
    # is deterministic per (iteration, j).
    rng = random.Random((iteration << 20) ^ j)
    body = " ".join(rng.choice(VOCAB) for _ in range(12))
    return {
        "deck": "Perf::Upsert",
        "note_type": "Basic",
        "tags": ["perf-upsert"],
        "fields": {"Front": f"upsert {iteration}-{j}", "Back": body},
    }


class UpsertBatchWorkload:
    """Upsert ``count`` fresh notes against the loaded corpus in ONE ``upsert_notes``
    call — the batched write path (one transaction). Pairs with ``upsert-seq`` to
    show the batching win.

    Each iteration ADDS ``count`` notes, so the collection grows across the run
    (the recorded ``corpus_size`` is the *starting* size); identical deterministic
    growth on both sides keeps comparisons valid. Run ``upsert-batch`` and
    ``upsert-seq`` in SEPARATE invocations (a fresh boot each) for an
    apples-to-apples batch-vs-sequential comparison."""

    name = "upsert-batch"
    mutates = True

    def __init__(self, *, count: int = _UPSERT_COUNT) -> None:
        self._count = count

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        notes = [_upsert_note(iteration, j) for j in range(self._count)]
        await booted.harness.wrapper.upsert_notes(notes)
        return len(notes)


class UpsertSeqWorkload:
    """Upsert ``count`` fresh notes against the loaded corpus as ``count`` SEPARATE
    ``upsert_notes`` calls — one upsert request (one transaction) per note. The
    delta from ``upsert-batch`` is the per-call/per-transaction overhead batching
    elides (one journal fsync per note vs one for the whole batch)."""

    name = "upsert-seq"
    mutates = True

    def __init__(self, *, count: int = _UPSERT_COUNT) -> None:
        self._count = count

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        for j in range(self._count):
            await booted.harness.wrapper.upsert_notes([_upsert_note(iteration, j)])
        return self._count


class DeleteWorkload:
    """Bulk-delete a batch of notes by id — the delete write path. Setup ingests a
    disposable pool (iterations × batch fresh notes, so deletes never touch the
    corpus); each timed iteration deletes its own slice."""

    name = "delete"
    mutates = True

    def __init__(self, *, batch: int = 200) -> None:
        self._batch = batch
        self._ids: list[int] = []

    async def setup(self, booted: Booted, iterations: int) -> None:
        pool = [self._note(j) for j in range(iterations * self._batch)]
        results = await booted.harness.wrapper.upsert_notes(pool)
        self._ids = [r["id"] for r in results if r.get("status") in ("created", "updated")]

    async def run_one(self, booted: Booted, iteration: int) -> int:
        ids = self._ids[iteration * self._batch : (iteration + 1) * self._batch]
        await booted.harness.wrapper.delete_notes(ids)
        return len(ids)

    def _note(self, j: int) -> dict:
        rng = random.Random(0xDE1E7E ^ j)
        return {
            "deck": "Perf::Delete",
            "note_type": "Basic",
            "tags": ["perf-delete"],
            "fields": {
                "Front": f"delete {j}",
                "Back": " ".join(rng.choice(VOCAB) for _ in range(8)),
            },
        }


# How many notes are edited under the server before each reconcile. The reconcile
# cost is dominated by the changed set (re-embed) on top of a full-collection
# fingerprint scan; 100 is a representative sync/edit burst.
_RECONCILE_DRIFT = 100


def _reconcile_note(iteration: int, j: int) -> dict:
    # A fresh note in its own deck/tag, deterministic per (iteration, j), so each
    # iteration drifts a disjoint set the reconcile sees as new.
    rng = random.Random((iteration << 24) ^ (j ^ 0x5EC0))
    body = " ".join(rng.choice(VOCAB) for _ in range(12))
    return {
        "deck": "Perf::Reconcile",
        "note_type": "Basic",
        "tags": ["perf-reconcile"],
        "fields": {"Front": f"reconcile {iteration}-{j}", "Back": body},
    }


class ReconcileWorkload:
    """Out-of-band drift recovery — the reconcile path, NOT an in-band write.

    Each iteration first (UNTIMED, in ``prepare``) edits the collection from under
    the server: it writes ``drift`` fresh notes STRAIGHT through the collection
    actor (``wrapper.upsert_notes`` bypasses the kernel index path), so ``col.mod``
    moves while the index watermark stays stale — exactly the drift a GUI edit, a
    sync, or a restore leaves behind. The TIMED ``run_one`` then runs
    ``reindex_if_needed`` + ``settle``: the kernel sees ``col.mod`` moved, diffs the
    per-note fingerprints, and re-embeds/re-indexes only the changed notes. The cost
    tracks the changed set (plus a full-collection fingerprint scan), not the upsert
    write — so the drift write is kept out of the timed region.

    Writing through the same actor makes the new ``col.mod`` live to the reconcile,
    so no reopen is needed; reopen is a collection-reacquire cost, not a reconcile
    cost. Each iteration ADDS ``drift`` notes, so the collection grows across the run
    (the recorded ``corpus_size`` is the *starting* size); identical deterministic
    growth on both sides keeps comparisons valid."""

    name = "reconcile"
    mutates = True

    def __init__(self, *, drift: int = _RECONCILE_DRIFT) -> None:
        self._drift = drift

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def prepare(self, booted: Booted, iteration: int) -> None:
        # Out-of-band: write through the collection only, leaving the index stale.
        notes = [_reconcile_note(iteration, j) for j in range(self._drift)]
        await booted.harness.wrapper.upsert_notes(notes)

    async def run_one(self, booted: Booted, iteration: int) -> int:
        # Reconcile to quiescence: detect drift, diff fingerprints, re-embed changed.
        await booted.harness.kernel.reindex_if_needed()
        await booted.harness.kernel.settle()
        return self._drift


#: name → workload class (instantiated with defaults by the runner).
WORKLOADS = {
    w.name: w
    for w in (
        SearchWorkload,
        RebuildWorkload,
        UpsertBatchWorkload,
        UpsertSeqWorkload,
        DeleteWorkload,
        ReconcileWorkload,
    )
}
