"""Manual smoke for the perf runner: boot perf-stub over a tiny corpus and
time each workload through the real harness.

Needs the synthetic embedder (``engine-synthetic``); skips on a lean build. Manual
lane — off the per-PR critical path (it boots a kernel + driver threads)."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest
import shrike_native

from tests.manual.perf.corpus import CorpusSpec, build_corpus
from tests.manual.perf.driver import boot_from_profile, measure, measure_ingest, run_async
from tests.manual.perf.workloads import (
    DeleteBatchWorkload,
    DeleteSeqWorkload,
    RebuildWorkload,
    ReconcileWorkload,
    SearchBatchWorkload,
    SearchSeqWorkload,
    UpsertBatchWorkload,
    UpsertSeqWorkload,
    build_workload,
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


def test_search_batch_produces_a_response_only_distribution(_driven, tmp_path):
    res = run_async(_measure(tmp_path, SearchBatchWorkload(count=4, limit=5), repeats=2, warmup=1))
    assert res.workload == "search-batch"
    assert res.distribution.n == 2  # the post-warmup repeats
    assert res.distribution.p50_ms >= 0.0
    assert set(res.phases) == {"response"}  # a read path has no settle phase
    assert res.items == 4  # queries issued (work done), not matches returned


def test_search_seq_issues_one_call_per_query(_driven, tmp_path):
    res = run_async(_measure(tmp_path, SearchSeqWorkload(count=4, limit=5), repeats=2, warmup=0))
    assert res.workload == "search-seq"
    assert res.distribution.n == 2
    assert res.items == 4  # one query per call, count calls -> count queries


def test_rebuild_workload_runs(_driven, tmp_path):
    res = run_async(_measure(tmp_path, RebuildWorkload(), repeats=2, warmup=0))
    assert res.workload == "rebuild"
    assert res.distribution.n == 2
    assert set(res.phases) == {"response"}


def test_upsert_batch_reports_items_and_times_both_phases(_driven, tmp_path):
    res = run_async(_measure(tmp_path, UpsertBatchWorkload(count=8), repeats=2, warmup=0))
    assert res.workload == "upsert-batch"
    assert res.items == 8
    # A write is timed in two phases: the action return, then the drain to settle,
    # plus their per-iteration total.
    assert set(res.phases) == {"response", "settle", "total"}
    assert res.phases["total"].p50_ms >= res.phases["response"].p50_ms


def test_upsert_seq_writes_each_note_separately(_driven, tmp_path):
    res = run_async(_measure(tmp_path, UpsertSeqWorkload(count=8), repeats=2, warmup=0))
    assert res.workload == "upsert-seq"
    assert res.items == 8
    assert "settle" in res.phases


def test_delete_batch_deletes_its_own_pool_slice(_driven, tmp_path):
    # setup upserts iterations*count notes (maintained + settled); each iteration
    # deletes one disjoint slice.
    res = run_async(_measure(tmp_path, DeleteBatchWorkload(count=5), repeats=2, warmup=0))
    assert res.workload == "delete-batch"
    assert res.items == 5
    assert set(res.phases) == {"response", "settle", "total"}


def test_delete_seq_deletes_one_id_per_call(_driven, tmp_path):
    res = run_async(_measure(tmp_path, DeleteSeqWorkload(count=5), repeats=2, warmup=0))
    assert res.workload == "delete-seq"
    assert res.items == 5


def test_reconcile_workload_recovers_out_of_band_drift(_driven, tmp_path):
    # prepare() drifts `count` notes out-of-band each iteration; the timed run_one
    # reconciles. Two timed iterations -> two reconciles, each over `count` notes.
    res = run_async(_measure(tmp_path, ReconcileWorkload(count=6), repeats=2, warmup=0))
    assert res.workload == "reconcile"
    assert res.distribution.n == 2
    assert res.items == 6
    assert set(res.phases) == {"response", "settle", "total"}


def test_ingest_workload_imports_a_cold_package(_driven, tmp_path):
    # measure_ingest exports the corpus to a package, then imports it into a fresh
    # empty collection per iteration (its own boot lifecycle, not a shared boot).
    corpus = build_corpus(CorpusSpec(notes=12, variant="text"), tmp_path / "corpus")
    res = run_async(measure_ingest(_PROFILE, corpus, tmp_path / "ingest", repeats=2, warmup=0))
    assert res.workload == "ingest"
    assert res.distribution.n == 2
    assert res.items == 12  # all 12 notes imported as new


def test_build_workload_scales_ops_uniformly_excepting_rebuild() -> None:
    # --ops N is applied uniformly as each workload's per-iteration count.
    for name in ("search-batch", "search-seq", "upsert-batch", "delete-seq", "reconcile"):
        assert getattr(build_workload(name, ops=7), "_count") == 7  # noqa: B009
    # rebuild is the exception — an O(collection) pass with no per-op N.
    assert build_workload("rebuild", ops=7).name == "rebuild"


def _profile_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {"profile": None, "profile_path": None}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_profile_path_requires_exactly_one_selector() -> None:
    from tests.manual.perf.run import _resolve_profile_path

    parser = argparse.ArgumentParser()
    # parser.error exits the process; neither and both selectors are rejected.
    with pytest.raises(SystemExit):
        _resolve_profile_path(_profile_args(), parser)
    with pytest.raises(SystemExit):
        _resolve_profile_path(_profile_args(profile="stub", profile_path=Path("/x.yml")), parser)
    with pytest.raises(SystemExit):
        _resolve_profile_path(_profile_args(profile_path=Path("/no/such/profile.yml")), parser)
    # A built-in resolves to its checked-in YAML; the stem is the run/condition label.
    resolved = _resolve_profile_path(_profile_args(profile="stub"), parser)
    assert resolved.is_file()
    assert resolved.stem == "perf-stub"


def test_uses_synthetic_reads_the_profile_embedders() -> None:
    from tests.manual.perf.run import _uses_synthetic

    profiles = Path(__file__).resolve().parent / "profiles"
    assert _uses_synthetic(profiles / "perf-stub.yml") is True
    assert _uses_synthetic(profiles / "perf-real.yml") is False
