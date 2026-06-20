"""Run the perf harness: build/boot a corpus under a profile, time the
selected workloads, emit a result artifact (and optionally diff a baseline).

    # kernel-isolation run (needs scripts/build-native.sh --release --synthetic):
    python shrike-py/tests/manual/perf/run.py --profile stub --size 500 \
        --variant text --workloads search,rebuild

    # end-to-end run (needs --release + the onnx/CLIP models in the model cache):
    python shrike-py/tests/manual/perf/run.py --profile real --size 5000 \
        --variant text+image --workloads search

Build the extension OPTIMIZED (`--release`, i.e. `-c opt`) — the default fastbuild
is meaningless for perf. The run records whether the build was optimized and warns
on a debug one; the baseline diff refuses to compare a debug run with a release one.

Off the per-PR critical path; run by name. Results land under .cache/perf/runs/
(gitignored). Comparison/gating against a stored baseline is the next child (#869);
this emits the comparable artifact and a diff on request.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running from a bare checkout: put the repo's shrike-py/ (for `tests.*`)
# and shrike-py/src (for `shrike.*`) on sys.path.
_PKG_ROOT = Path(__file__).resolve().parents[3]
for _p in (_PKG_ROOT, _PKG_ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tests.manual.perf.compare import compare, render_comparison  # noqa: E402
from tests.manual.perf.corpus import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    STANDARD_SIZES,
    VARIANTS,
    CorpusSpec,
    ensure_corpus,
)
from tests.manual.perf.driver import boot_from_profile, measure, run_async  # noqa: E402
from tests.manual.perf.result import (  # noqa: E402
    Conditions,
    RunResult,
    WorkloadResult,
    render_table,
)
from tests.manual.perf.workloads import WORKLOADS  # noqa: E402

_RUNS_DIR = DEFAULT_CACHE_ROOT.parent / "runs"


def _isolated_working_copy(corpus_anki2: Path, corpus_media: Path, run_dir: Path) -> Path:
    """A throwaway copy of the corpus to boot over, so a mutating workload
    (upsert/delete) never pollutes the cached corpus. The collection file is copied;
    the media dir is symlinked — read-only reuse that relies on the invariant that
    the workloads write no media. A future media-writing workload (image upsert /
    store_media) MUST copy the media dir instead, or it would write through the
    symlink into the cached corpus."""
    run_dir.mkdir(parents=True, exist_ok=True)
    working = run_dir / "collection.anki2"
    shutil.copy2(corpus_anki2, working)
    if corpus_media.is_dir():
        link = run_dir / "collection.media"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(corpus_media)
    return working


async def _run_workloads(
    profile_path: Path, working: Path, cache_dir: Path, workloads: list, repeats: int, warmup: int
) -> list[WorkloadResult]:
    booted = await boot_from_profile(profile_path, working, cache_dir)
    try:
        return [await measure(w, booted, repeats=repeats, warmup=warmup) for w in workloads]
    finally:
        await booted.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("stub", "real"), required=True, help="Embedder mode.")
    parser.add_argument("--size", type=int, default=STANDARD_SIZES[0], help="Corpus note count.")
    parser.add_argument("--variant", choices=VARIANTS, default="text", help="Corpus modality.")
    parser.add_argument(
        "--workloads",
        default="search,rebuild,upsert-batch",
        help=f"Subset of {sorted(WORKLOADS)} (default: search,rebuild,upsert-batch).",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per workload.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations discarded.")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="A stored result to diff this run against."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the result JSON (default: under runs/).",
    )
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Reserved for the profiler-attach seam (#866); not wired yet.",
    )
    args = parser.parse_args()

    if args.instrument:
        print("note: --instrument is the #866 seam and is not wired yet; running clean-timed.")

    import shrike_native

    if "debug-assertions" in shrike_native.build_features():
        print(
            "WARNING: benchmarking an UNOPTIMIZED (debug) shrike-core — the numbers are "
            "not representative. Rebuild with `scripts/build-native.sh --release"
            + (" --synthetic" if args.profile == "stub" else "")
            + "` for real results."
        )

    names = [n.strip() for n in args.workloads.split(",") if n.strip()]
    unknown = [n for n in names if n not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s) {unknown}; choices: {sorted(WORKLOADS)}")
    # Read-only workloads first so one boot stays representative (a mutator grows
    # the collection, which would skew a later read).
    workloads = sorted((WORKLOADS[n]() for n in names), key=lambda w: getattr(w, "mutates", False))

    spec = CorpusSpec(notes=args.size, variant=args.variant)
    print(f"Ensuring corpus: {args.size} notes ({args.variant}) ...")
    corpus = ensure_corpus(spec)

    profile_path = Path(__file__).resolve().parent / "profiles" / f"perf-{args.profile}.yml"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = (
        _RUNS_DIR / f"perf-{args.profile}-{args.variant.replace('+', '_')}-{args.size}-{stamp}"
    )
    working = _isolated_working_copy(corpus.anki2_path, corpus.media_dir, run_dir)

    print(f"Booting perf-{args.profile} over {working} ...")
    # The kernel runs a harness-driven runtime with no lazy fallback: install +
    # park the committed driver threads before any kernel op, tear down after.
    from shrike.platform.driven_runtime import DrivenRuntime

    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    try:
        results = run_async(
            _run_workloads(
                profile_path, working, run_dir / "cache", workloads, args.repeats, args.warmup
            )
        )
    finally:
        runtime.shutdown()

    run = RunResult(
        conditions=Conditions.capture(
            profile=f"perf-{args.profile}",
            corpus_size=args.size,
            corpus_variant=args.variant,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        results=results,
        timestamp=datetime.now(UTC).isoformat(),
    )

    out = args.out or (run_dir / "result.json")
    run.save(out)
    print()
    print(render_table(run))
    print(f"\nwrote {out}")

    if args.baseline:
        baseline = RunResult.load(args.baseline)
        try:
            cmp = compare(baseline, run)
        except ValueError as err:
            print(f"\ncannot diff baseline: {err}")
            return 0
        print("\n# vs baseline")
        print(render_comparison(cmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
