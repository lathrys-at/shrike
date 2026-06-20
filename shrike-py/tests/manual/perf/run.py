"""Run the perf harness: build/boot a corpus under a profile, time the
selected workloads, emit a result artifact (and optionally diff a baseline).

    # kernel-isolation run (needs scripts/build-native.sh --release --synthetic):
    python shrike-py/tests/manual/perf/run.py --profile stub --size 500 \
        --variant text --workloads search-batch,rebuild

    # end-to-end run (needs --release + the onnx/CLIP models in the model cache):
    python shrike-py/tests/manual/perf/run.py --profile real --size 5000 \
        --variant text+image --workloads search-batch

Build the extension OPTIMIZED (`--release`, i.e. `-c opt`) — the default fastbuild
is meaningless for perf. The run records whether the build was optimized and warns
on a debug one; the baseline diff refuses to compare a debug run with a release one.

Off the per-PR critical path; run by name. Results land under .cache/perf/runs/
(gitignored). Perf is checked by hand: this emits the comparable artifact and an
on-request `--baseline` diff against a prior run — there is no automated gate.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running from a bare checkout: put the repo's shrike-py/ (for `tests.*`)
# and shrike-py/src (for `shrike.*`) on sys.path.
_PKG_ROOT = Path(__file__).resolve().parents[3]
for _p in (_PKG_ROOT, _PKG_ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shrike.platform.log import FILE_DATE_FORMAT, FILE_FORMAT  # noqa: E402
from tests.manual.perf.compare import (  # noqa: E402
    compare,
    render_comparison,
    render_markdown_comparison,
)
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
from tests.manual.perf.instrument import (  # noqa: E402
    DEFAULT_INSTRUMENT,
    INSTRUMENTERS,
    RUN_DIR_ENV,
    artifact_path,
    instrument_binary,
    instrument_command,
    pyspy_native_supported,
    validate_instrument_request,
)
from tests.manual.perf.result import (  # noqa: E402
    Conditions,
    RunResult,
    WorkloadResult,
    render_markdown_table,
    render_table,
)
from tests.manual.perf.workloads import DEFAULT_OPS, WORKLOADS, build_workload  # noqa: E402

_RUNS_DIR = DEFAULT_CACHE_ROOT.parent / "runs"


def _install_log_buffer() -> io.StringIO:
    """Route all logging into an in-memory buffer instead of the terminal, so log
    writes (the per-call INFO lines, native tracing) never perturb the timed
    iterations. Cleared of any prior handlers, so nothing reaches stdout/stderr
    during the run; flushed to ``run.log`` at the end (see :func:`_flush_log_buffer`)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return buf


def _flush_log_buffer(buf: io.StringIO, path: Path) -> None:
    """Write the captured log buffer to ``path`` (the run's ``run.log``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue())


# A fixed-width in-place progress bar, refreshed once per completed iteration.
_PROGRESS_WIDTH = 24


def _progress_printer(label: str, warmup: int) -> Callable[[int, int, bool], None]:
    """A pytest-style in-place progress line for one workload's iterations, so a
    long run shows forward motion instead of an apparent hang. Goes to stdout
    (harness UI), separate from the captured log buffer; the per-tick write sits
    between timed iterations, never inside a measurement.

    Warmup is not counted: while ``warming`` the line reads ``warming up...``; once
    the timed repeats begin it becomes a ``done/total`` bar over the repeats (the
    driver passes the repeat count as ``total``), so the bar measures only the
    iterations that land in the result. The warmup notice is shown up front too —
    before the first (possibly slow) warmup iteration finishes."""

    def render_warming() -> None:
        sys.stdout.write(f"\r  {label:<16} warming up...")
        sys.stdout.flush()

    if warmup > 0:
        render_warming()

    def tick(done: int, total: int, warming: bool) -> None:
        if warming:
            render_warming()
            return
        filled = int(_PROGRESS_WIDTH * done / total) if total else _PROGRESS_WIDTH
        bar = "#" * filled + "-" * (_PROGRESS_WIDTH - filled)
        sys.stdout.write(f"\r  {label:<16} [{bar}] {done:>3}/{total}")
        sys.stdout.flush()

    return tick


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
    ops: int,
    with_media: bool,
) -> list[WorkloadResult]:
    results: list[WorkloadResult] = []
    registry = [n for n in names if n != INGEST]
    if registry:
        # Read-only workloads first so the single shared boot stays representative.
        workloads = sorted(
            (build_workload(n, ops=ops) for n in registry),
            key=lambda w: getattr(w, "mutates", False),
        )
        # An isolated working copy so a mutating workload never pollutes the cached
        # corpus (ingest makes its own copies; skipped when only ingest is requested).
        working = _isolated_working_copy(corpus.anki2_path, corpus.media_dir, run_dir)
        print(f"Booting {profile_path.stem} over {working} ...")
        booted = await boot_from_profile(profile_path, working, run_dir / "cache")
        try:
            for w in workloads:
                res = await measure(
                    w,
                    booted,
                    repeats=repeats,
                    warmup=warmup,
                    on_tick=_progress_printer(w.name, warmup),
                )
                print()  # close the completed bar's line; the full table prints at the end
                results.append(res)
        finally:
            # Close before ingest opens its own kernels — the driven runtime holds
            # one kernel at a time.
            await booted.close()
    if INGEST in names:
        res = await measure_ingest(
            profile_path,
            corpus,
            run_dir / "ingest",
            repeats=repeats,
            warmup=warmup,
            with_media=with_media,
            on_tick=_progress_printer(INGEST, warmup),
        )
        print()  # close the completed bar's line; the full table prints at the end
        results.append(res)
    return results


def _resolve_profile_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    """The profile YAML this run boots from. ``--profile {stub,real}`` selects a
    built-in; ``--profile-path PATH`` overrides it with any custom profile, so a
    non-stub run isn't pinned to the two built-ins. Exactly one is required."""
    if args.profile_path is not None:
        if args.profile is not None:
            parser.error("pass --profile OR --profile-path, not both")
        path = Path(args.profile_path).expanduser()
        if not path.is_file():
            parser.error(f"--profile-path {path} does not exist")
        return path.resolve()
    if args.profile is None:
        parser.error("one of --profile {stub,real} or --profile-path PATH is required")
    return Path(__file__).resolve().parent / "profiles" / f"perf-{args.profile}.yml"


def _uses_synthetic(profile_path: Path) -> bool:
    """True when the profile's embedders are the synthetic (no-model) backend — so
    its ``settle``/``total`` figures reflect kernel/IO orchestration drain only, not
    real embedding/index cost. The runner warns on those phases under this profile."""
    import yaml

    data = yaml.safe_load(profile_path.read_text()) or {}
    return any(
        isinstance(e, dict) and e.get("runtime") == "synthetic" for e in data.get("embedders") or []
    )


#: How to install each instrumenter, surfaced when its binary is missing from PATH.
_INSTALL_HINT = {
    "py-spy": "pip install py-spy",
    "samply": "cargo install samply",
    "xctrace": "install Xcode (provides `xcrun xctrace`)",
}


def _profile_under_instrumenter(
    args: argparse.Namespace,
    profile_path: Path,
    names: list[str],
    parser: argparse.ArgumentParser,
) -> int:
    """Re-exec the run under the selected profiler (``--instrument``) to capture a
    flamegraph/trace. ONE workload per run keeps the attribution clean; the inner
    run shares this run's output dir (via env) so the profile artifact and
    result.json land together.

    py-spy ``--native`` is the cross-boundary view (Python + Rust in one
    flamegraph); samply/xctrace give deeper Rust detail but opaque Python frames.
    See :mod:`tests.manual.perf.instrument`."""
    # The request shape (one workload, no --out, xctrace-on-macOS, …) is validated
    # purely in main(); here only the impure tool-on-PATH probe remains.
    tool = args.instrument
    binary = instrument_binary(tool)
    if shutil.which(binary) is None:
        parser.error(
            f"--instrument={tool} needs `{binary}` on PATH "
            f"({_INSTALL_HINT[tool]}); see docs/dev/testing.md"
        )
    workload = names[0]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = (
        _RUNS_DIR
        / f"{profile_path.stem}-{args.variant.replace('+', '_')}-{args.size}-{workload}-{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if tool == "py-spy" and not pyspy_native_supported(sys.platform):
        print(
            "NOTE: py-spy has no native unwinding on this platform — the flamegraph is "
            "Python-only (the Rust kernel shows as opaque native frames). For Rust detail "
            "use --instrument=samply or =xctrace; for the merged Python+Rust view, run "
            "py-spy --native on Linux."
        )
    cmd = instrument_command(
        tool,
        Path(__file__).resolve(),
        run_dir,
        platform=sys.platform,
        extra_args=args.instrument_arg,
        profile=args.profile,
        profile_path=args.profile_path,
        size=args.size,
        variant=args.variant,
        workload=workload,
        repeats=args.repeats,
        warmup=args.warmup,
        ops=args.ops,
        baseline=args.baseline,
    )
    artifact = artifact_path(run_dir, workload, tool)
    print(f"Instrumenting '{workload}' under {tool} -> {artifact}")
    print(f"  $ {shlex.join(cmd)}")
    rc = subprocess.call(cmd, env={**os.environ, RUN_DIR_ENV: str(run_dir)})
    if rc == 0:
        print(f"\nwrote {artifact}")
        if tool == "samply":
            print(f"  view it: samply load {artifact}")
        elif tool == "xctrace":
            print(f"  open it: open {artifact}")
    elif tool == "py-spy":
        # py-spy attaches via the OS process-inspection API, which usually needs
        # root (especially on macOS); preserve PATH/venv so it finds the interpreter.
        print(f"\npy-spy exited {rc}. Attaching often needs root — retry under sudo:")
        print(f"  sudo --preserve-env=PATH,VIRTUAL_ENV {shlex.join(cmd)}")
    else:
        # samply/xctrace launch the target themselves and need no elevation.
        print(f"\n{tool} exited {rc}. Re-run with an extra --instrument-arg to diagnose.")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("stub", "real"),
        default=None,
        help="Built-in embedder profile (perf-{stub,real}.yml). Required unless "
        "--profile-path is given.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="A custom profile YAML to boot from instead of a built-in --profile — "
        "any embedder set, same path-free schema as perf-{stub,real}.yml.",
    )
    parser.add_argument("--size", type=int, default=STANDARD_SIZES[0], help="Corpus note count.")
    parser.add_argument("--variant", choices=VARIANTS, default="text", help="Corpus modality.")
    parser.add_argument(
        "--workloads",
        default="search-batch,rebuild,upsert-batch",
        help=f"Subset of {sorted({*WORKLOADS, INGEST})} "
        "(default: search-batch,rebuild,upsert-batch). 'ingest' is the cold "
        "package-import scenario (its own boot per sample; heavy — use a small --repeats).",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per workload.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations discarded.")
    parser.add_argument(
        "--ops",
        type=int,
        default=DEFAULT_OPS,
        help="N: operations per workload iteration (search queries, upsert/delete "
        f"notes, reconcile drift); complements --repeats (N ops x M repeats). "
        f"Default {DEFAULT_OPS}. 'rebuild' ignores it (one O(collection) pass).",
    )
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
        "--output-format",
        choices=("plain", "table"),
        default="plain",
        help="Terminal rendering of the result + baseline diff: 'plain' fixed-width "
        "text (default), or 'table' a GitHub-flavored markdown table to paste into a "
        "comment. The result JSON is written regardless.",
    )
    parser.add_argument(
        "--instrument",
        nargs="?",
        const=DEFAULT_INSTRUMENT,
        default=None,
        choices=INSTRUMENTERS,
        help="Profile ONE workload under a sampling profiler, writing its artifact next "
        "to the result. Bare --instrument uses py-spy (Python+Rust flamegraph on Linux; "
        "Python-only on macOS). --instrument=samply / =xctrace give deeper Rust detail "
        "with opaque Python frames (xctrace is macOS-only). Needs the tool on PATH + a "
        "`--frame-pointers` build; py-spy attach often needs sudo. See docs/dev/testing.md.",
    )
    parser.add_argument(
        "--instrument-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra arg passed verbatim to the --instrument tool (repeatable; one token "
        "each, so a valued flag is two: --instrument-arg=--rate --instrument-arg=2000).",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.workloads.split(",") if n.strip()]
    known = {*WORKLOADS, INGEST}
    unknown = [n for n in names if n not in known]
    if unknown:
        parser.error(f"unknown workload(s) {unknown}; choices: {sorted(known)}")

    instrument_error = validate_instrument_request(
        tool=args.instrument,
        instrument_args=args.instrument_arg,
        workloads=names,
        out_given=args.out is not None,
        platform=sys.platform,
    )
    if instrument_error:
        parser.error(instrument_error)

    profile_path = _resolve_profile_path(args, parser)

    # The profiling run re-execs the inner run under the selected profiler (which
    # carries no --instrument); the inner run lands here without re-entering this
    # branch (the shared run dir in RUN_DIR_ENV marks it).
    if args.instrument and not os.environ.get(RUN_DIR_ENV):
        return _profile_under_instrumenter(args, profile_path, names, parser)

    import shrike_native

    synthetic = _uses_synthetic(profile_path)
    if "debug-assertions" in shrike_native.build_features():
        print(
            "WARNING: benchmarking an UNOPTIMIZED (debug) shrike-core — the numbers are "
            "not representative. Rebuild with `scripts/build-native.sh --release"
            + (" --synthetic" if synthetic else "")
            + "` for real results."
        )

    # The synthetic embedder has negligible cost, so the settle/total phases (the
    # index/derived drain a write enqueues) measure orchestration only, not the real
    # embed. Warn when a settling workload is selected under it.
    if synthetic and any(hasattr(WORKLOADS[n], "settle") for n in names if n in WORKLOADS):
        print(
            "NOTE: the synthetic embedder makes the 'settle' (and 'total') phases "
            "unrepresentative — they measure the kernel/IO/orchestration drain, not "
            "real embedding/index cost. Use --profile real (or a real-engine "
            "--profile-path) for representative settle/total figures; 'response' is "
            "unaffected."
        )

    # Under --instrument the outer run picks the dir and passes it down, so the
    # flamegraph and this run's result.json land together. Resolved before the log
    # buffer (and the corpus build) so run.log always has a home, even on a later
    # failure.
    env_run_dir = os.environ.get(RUN_DIR_ENV)
    if env_run_dir:
        run_dir = Path(env_run_dir)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        run_dir = (
            _RUNS_DIR / f"{profile_path.stem}-{args.variant.replace('+', '_')}-{args.size}-{stamp}"
        )

    # Capture logging into an in-memory buffer (off the terminal, so the per-call
    # INFO lines and native tracing don't perturb the timed iterations); flushed
    # to run.log in the finally below.
    log_buffer = _install_log_buffer()
    try:
        spec = CorpusSpec(notes=args.size, variant=args.variant)
        print(f"Ensuring corpus: {args.size} notes ({args.variant}) ...")
        corpus = ensure_corpus(spec)

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
                    args.ops,
                    with_media=(args.variant == "text+image"),
                )
            )
        finally:
            runtime.shutdown()

        run = RunResult(
            conditions=Conditions.capture(
                profile=profile_path.stem,
                corpus_size=args.size,
                corpus_variant=args.variant,
                repeats=args.repeats,
                warmup=args.warmup,
                ops=args.ops,
            ),
            results=results,
            timestamp=datetime.now(UTC).isoformat(),
        )

        markdown = args.output_format == "table"
        out = args.out or (run_dir / "result.json")
        run.save(out)
        print()
        print(render_markdown_table(run) if markdown else render_table(run))
        print(f"\nwrote {out}")

        if args.baseline:
            baseline = RunResult.load(args.baseline)
            try:
                cmp = compare(baseline, run)
            except ValueError as err:
                print(f"\ncannot diff baseline: {err}")
            else:
                print("\n## vs baseline" if markdown else "\n# vs baseline")
                print(render_markdown_comparison(cmp) if markdown else render_comparison(cmp))
    finally:
        log_path = run_dir / "run.log"
        _flush_log_buffer(log_buffer, log_path)
        print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
