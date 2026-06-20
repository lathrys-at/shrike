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
(gitignored). Perf is checked by hand: this emits the comparable artifact and an
on-request `--baseline` diff against a prior run — there is no automated gate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from tests.manual.perf.driver import (  # noqa: E402
    INGEST,
    boot_from_profile,
    measure,
    measure_ingest,
    run_async,
)
from tests.manual.perf.instrument import RUN_DIR_ENV, flame_path, pyspy_command  # noqa: E402
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
    profile_path: Path,
    corpus: Any,
    run_dir: Path,
    names: list[str],
    repeats: int,
    warmup: int,
    with_media: bool,
) -> list[WorkloadResult]:
    results: list[WorkloadResult] = []
    registry = [n for n in names if n != INGEST]
    if registry:
        # Read-only workloads first so the single shared boot stays representative.
        workloads = sorted(
            (WORKLOADS[n]() for n in registry), key=lambda w: getattr(w, "mutates", False)
        )
        # An isolated working copy so a mutating workload never pollutes the cached
        # corpus (ingest makes its own copies; skipped when only ingest is requested).
        working = _isolated_working_copy(corpus.anki2_path, corpus.media_dir, run_dir)
        print(f"Booting {profile_path.stem} over {working} ...")
        booted = await boot_from_profile(profile_path, working, run_dir / "cache")
        try:
            for w in workloads:
                results.append(await measure(w, booted, repeats=repeats, warmup=warmup))
        finally:
            # Close before ingest opens its own kernels — the driven runtime holds
            # one kernel at a time.
            await booted.close()
    if INGEST in names:
        results.append(
            await measure_ingest(
                profile_path,
                corpus,
                run_dir / "ingest",
                repeats=repeats,
                warmup=warmup,
                with_media=with_media,
            )
        )
    return results


def _profile_under_pyspy(
    args: argparse.Namespace, names: list[str], parser: argparse.ArgumentParser
) -> int:
    """Re-exec the run under ``py-spy record --native`` to capture a flamegraph
    spanning the Python harness AND the Rust kernel. ONE workload per run keeps the
    attribution clean; the inner run shares this run's output dir (via env) so the
    flamegraph and result.json land together."""
    if len(names) != 1:
        parser.error("--instrument profiles ONE workload per run; pass a single --workloads")
    if args.out is not None:
        parser.error("--out is ignored under --instrument; result.json lands beside the flamegraph")
    if shutil.which("py-spy") is None:
        parser.error(
            "--instrument needs py-spy on PATH (`pip install py-spy`); see docs/dev/testing.md"
        )
    workload = names[0]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = (
        _RUNS_DIR
        / f"perf-{args.profile}-{args.variant.replace('+', '_')}-{args.size}-{workload}-{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = pyspy_command(
        Path(__file__).resolve(),
        run_dir,
        profile=args.profile,
        size=args.size,
        variant=args.variant,
        workload=workload,
        repeats=args.repeats,
        warmup=args.warmup,
        baseline=args.baseline,
    )
    flame = flame_path(run_dir, workload)
    print(f"Instrumenting '{workload}' under py-spy --native -> {flame}")
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, env={**os.environ, RUN_DIR_ENV: str(run_dir)})
    if rc == 0:
        print(f"\nwrote {flame}")
    else:
        # Attaching usually needs root, especially on macOS; preserve PATH/venv.
        print(f"\npy-spy exited {rc}. Attaching often needs root — retry under sudo:")
        print(f"  sudo --preserve-env=PATH,VIRTUAL_ENV {' '.join(cmd)}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("stub", "real"), required=True, help="Embedder mode.")
    parser.add_argument("--size", type=int, default=STANDARD_SIZES[0], help="Corpus note count.")
    parser.add_argument("--variant", choices=VARIANTS, default="text", help="Corpus modality.")
    parser.add_argument(
        "--workloads",
        default="search,rebuild,upsert-batch",
        help=f"Subset of {sorted({*WORKLOADS, INGEST})} (default: search,rebuild,upsert-batch). "
        "'ingest' is the cold package-import scenario (its own boot per sample; heavy "
        "— use a small --repeats).",
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
        help="Profile ONE workload under py-spy --native: a flamegraph (Python + Rust) "
        "next to the result. Needs py-spy + a `--frame-pointers` build; often needs sudo. "
        "See docs/dev/testing.md.",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.workloads.split(",") if n.strip()]
    known = {*WORKLOADS, INGEST}
    unknown = [n for n in names if n not in known]
    if unknown:
        parser.error(f"unknown workload(s) {unknown}; choices: {sorted(known)}")

    # The profiling run re-execs the inner run under py-spy (which carries no
    # --instrument); the inner run lands here without re-entering this branch.
    if args.instrument and not os.environ.get(RUN_DIR_ENV):
        return _profile_under_pyspy(args, names, parser)

    import shrike_native

    if "debug-assertions" in shrike_native.build_features():
        print(
            "WARNING: benchmarking an UNOPTIMIZED (debug) shrike-core — the numbers are "
            "not representative. Rebuild with `scripts/build-native.sh --release"
            + (" --synthetic" if args.profile == "stub" else "")
            + "` for real results."
        )

    spec = CorpusSpec(notes=args.size, variant=args.variant)
    print(f"Ensuring corpus: {args.size} notes ({args.variant}) ...")
    corpus = ensure_corpus(spec)

    profile_path = Path(__file__).resolve().parent / "profiles" / f"perf-{args.profile}.yml"
    # Under --instrument the outer run picks the dir and passes it down, so the
    # flamegraph and this run's result.json land together.
    env_run_dir = os.environ.get(RUN_DIR_ENV)
    if env_run_dir:
        run_dir = Path(env_run_dir)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        run_dir = (
            _RUNS_DIR / f"perf-{args.profile}-{args.variant.replace('+', '_')}-{args.size}-{stamp}"
        )

    # The kernel runs a harness-driven runtime with no lazy fallback: install +
    # park the committed driver threads before any kernel op, tear down after.
    from shrike.platform.driven_runtime import DrivenRuntime

    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    try:
        results = run_async(
            _run_workloads(
                profile_path,
                corpus,
                run_dir,
                names,
                args.repeats,
                args.warmup,
                with_media=(args.variant == "text+image"),
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
