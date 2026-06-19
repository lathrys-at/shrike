"""The gold workloads the perf harness times.

A workload is one operation, repeated against a booted harness over a fixed
corpus: search, rebuild, ingest, reconcile, delete. The remaining gold workflows
(sync, OCR sweeps) need their own scenario / recognizer setup and are a focused
follow-up.

``mutates`` marks a workload that changes the collection (ingest/reconcile/delete);
the runner orders read-only workloads first so a single boot stays representative.
Running MORE THAN ONE mutating workload in a single invocation compounds the
collection state across them, so for a clean absolute number run one mutating
workload per invocation (comparisons across runs stay valid regardless — the
mutation is deterministic and identical on both sides).
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


class IngestWorkload:
    """Upsert a batch of fresh notes through the real write path — the bulk-write
    path (one transaction, embed + index the new notes).

    Note: each timed iteration ADDS a batch, so the collection grows across the
    run — the recorded ``corpus_size`` condition is the *starting* size, not the
    size during the later iterations. Comparisons stay valid (identical
    deterministic growth on both sides); the absolute number is "ingest into a
    growing collection seeded at corpus_size", not "into a fixed corpus_size"."""

    name = "ingest"
    mutates = True

    def __init__(self, *, batch: int = 200) -> None:
        self._batch = batch

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        notes = [self._note(iteration, j) for j in range(self._batch)]
        await booted.harness.wrapper.upsert_notes(notes)
        return len(notes)

    def _note(self, iteration: int, j: int) -> dict:
        # A fresh deck/tag so ingested notes never collide with the corpus; the
        # text is deterministic per (iteration, j).
        rng = random.Random((iteration << 20) ^ j)
        body = " ".join(rng.choice(VOCAB) for _ in range(12))
        return {
            "deck": "Perf::Ingest",
            "note_type": "Basic",
            "tags": ["perf-ingest"],
            "fields": {"Front": f"ingest {iteration}-{j}", "Back": body},
        }


class ReconcileWorkload:
    """Upsert one note (creating drift) then reconcile the index — the incremental
    reindex path, i.e. the interactive add-one-note-and-be-searchable cost, NOT a
    full rebuild. Times the upsert + the reconcile together (the drift must exist
    to be reconciled)."""

    name = "reconcile"
    mutates = True

    async def setup(self, booted: Booted, iterations: int) -> None:
        return None

    async def run_one(self, booted: Booted, iteration: int) -> int:
        rng = random.Random(0x9EC0 ^ iteration)
        note = {
            "deck": "Perf::Reconcile",
            "note_type": "Basic",
            "tags": ["perf-reconcile"],
            "fields": {
                "Front": f"reconcile {iteration}",
                "Back": " ".join(rng.choice(VOCAB) for _ in range(10)),
            },
        }
        await booted.harness.wrapper.upsert_notes([note])
        await booted.harness.kernel.reindex_if_needed()
        return 1  # one note's incremental reconcile


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


#: name → workload class (instantiated with defaults by the runner).
WORKLOADS = {
    w.name: w
    for w in (
        SearchWorkload,
        RebuildWorkload,
        IngestWorkload,
        ReconcileWorkload,
        DeleteWorkload,
    )
}
