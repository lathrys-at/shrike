"""Manual smoke for the perf runner: boot perf-stub over a tiny corpus and
time each workload through the real harness.

Needs the synthetic embedder (``engine-synthetic``); skips on a lean build. Manual
lane — off the per-PR critical path (it boots a kernel + driver threads)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import shrike_native

from tests.manual.perf.corpus import CorpusSpec, build_corpus
from tests.manual.perf.driver import boot_from_profile, measure, measure_ingest, run_async
from tests.manual.perf.workloads import (
    DeleteWorkload,
    RebuildWorkload,
    ReconcileWorkload,
    SearchWorkload,
    UpsertBatchWorkload,
    UpsertSeqWorkload,
)

pytestmark = pytest.mark.skipif(
    "engine-synthetic" not in shrike_native.build_features(),
    reason="engine-synthetic not compiled (build: scripts/build-native.sh --synthetic)",
)

_PROFILE = Path(__file__).resolve().parent / "profiles" / "perf-stub.yml"


@pytest.fixture(scope="session")
def _driven() -> Iterator[None]:
    """Install + park the kernel's committed driver threads for the session — the
    kernel runtime has no lazy fallback (mirrors tests/native/conftest)."""
    from shrike.platform.driven_runtime import DrivenRuntime

    if getattr(shrike_native, "_shrike_test_driven", False):
        yield
        return
    shrike_native._shrike_test_driven = True
    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    try:
        yield
    finally:
        runtime.shutdown()
        shrike_native._shrike_test_driven = False


async def _measure(tmp_path: Path, workload, *, repeats: int, warmup: int):
    corpus = build_corpus(CorpusSpec(notes=24, variant="text"), tmp_path / "corpus")
    booted = await boot_from_profile(_PROFILE, corpus.anki2_path, tmp_path / "cache")
    try:
        return await measure(workload, booted, repeats=repeats, warmup=warmup)
    finally:
        await booted.close()


def test_search_workload_produces_a_distribution(_driven, tmp_path):
    res = run_async(_measure(tmp_path, SearchWorkload(n_queries=4, limit=5), repeats=2, warmup=1))
    assert res.workload == "search"
    assert res.distribution.n == 2  # the post-warmup repeats
    assert res.distribution.p50_ms >= 0.0


def test_rebuild_workload_runs(_driven, tmp_path):
    res = run_async(_measure(tmp_path, RebuildWorkload(), repeats=2, warmup=0))
    assert res.workload == "rebuild"
    assert res.distribution.n == 2


def test_upsert_batch_reports_items(_driven, tmp_path):
    res = run_async(_measure(tmp_path, UpsertBatchWorkload(count=8), repeats=2, warmup=0))
    assert res.workload == "upsert-batch"
    assert res.items == 8


def test_upsert_seq_writes_each_note_separately(_driven, tmp_path):
    res = run_async(_measure(tmp_path, UpsertSeqWorkload(count=8), repeats=2, warmup=0))
    assert res.workload == "upsert-seq"
    assert res.items == 8


def test_delete_workload_deletes_its_own_pool_slice(_driven, tmp_path):
    # setup upserts iterations*batch notes; each iteration deletes one batch slice.
    res = run_async(_measure(tmp_path, DeleteWorkload(batch=5), repeats=2, warmup=0))
    assert res.workload == "delete"
    assert res.items == 5


def test_reconcile_workload_recovers_out_of_band_drift(_driven, tmp_path):
    # prepare() drifts `drift` notes out-of-band each iteration; the timed run_one
    # reconciles. Two timed iterations -> two reconciles, each over `drift` notes.
    res = run_async(_measure(tmp_path, ReconcileWorkload(drift=6), repeats=2, warmup=0))
    assert res.workload == "reconcile"
    assert res.distribution.n == 2
    assert res.items == 6


def test_ingest_workload_imports_a_cold_package(_driven, tmp_path):
    # measure_ingest exports the corpus to a package, then imports it into a fresh
    # empty collection per iteration (its own boot lifecycle, not a shared boot).
    corpus = build_corpus(CorpusSpec(notes=12, variant="text"), tmp_path / "corpus")
    res = run_async(measure_ingest(_PROFILE, corpus, tmp_path / "ingest", repeats=2, warmup=0))
    assert res.workload == "ingest"
    assert res.distribution.n == 2
    assert res.items == 12  # all 12 notes imported as new
