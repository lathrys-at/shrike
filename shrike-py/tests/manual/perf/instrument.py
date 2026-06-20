"""Wire a perf run to a sampling profiler for a flamegraph/trace of one workload.

``run.py --instrument[=<tool>]`` re-execs the run under a profiler chosen from a
small registry, so a hotspot is attributable to a line whether it lives in the
harness glue or the kernel:

- **py-spy** (default) — the only profiler that merges Python-level frames AND
  native Rust frames into one flamegraph (``run.py -> search_notes ->
  kernel.search -> usearch...`` in a single view). Its native unwinding
  (``--native``) is **Linux/Windows-only**; on macOS py-spy rejects ``--native``
  outright, so there the flamegraph is Python-only (the Rust kernel shows as
  opaque native leaves).
- **samply** — a sampling profiler with deep native (Rust) detail, pleasant on
  macOS-arm64, but it renders the Python side as opaque CPython interpreter
  frames. Reach for it when the hotspot is known-Rust.
- **xctrace** — Apple Instruments' Time Profiler (macOS only). Native detail like
  samply, opaque Python frames; the ``.trace`` opens in Instruments.app.

The cross-boundary view is py-spy's alone; the rationale and the rejected
alternatives are in ``docs/dev/decisions.md``.

This module builds those invocations. It is kept pure (no native, no harness
import, the platform passed in rather than read) so it unit-tests on the per-PR
lane, off the manual native lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: How the outer (profiling) run hands the inner run their shared output dir, so
#: the profile artifact and the result.json land together. The inner run reads it
#: in place of minting a fresh timestamped dir.
RUN_DIR_ENV = "SHRIKE_PERF_RUN_DIR"

#: The profilers ``--instrument`` selects between, and the default when
#: ``--instrument`` is given bare.
INSTRUMENTERS: tuple[str, ...] = ("py-spy", "samply", "xctrace")
DEFAULT_INSTRUMENT = "py-spy"

#: The on-PATH executable each instrumenter drives (xctrace rides ``xcrun``).
_BINARY = {"py-spy": "py-spy", "samply": "samply", "xctrace": "xcrun"}

#: Each tool's artifact: a filename prefix and extension. py-spy writes an SVG
#: flamegraph; samply a Firefox-profiler JSON; xctrace an Instruments ``.trace``
#: bundle. Named per tool so the three formats never collide in one run dir.
_ARTIFACT = {
    "py-spy": ("flame", "svg"),
    "samply": ("profile", "json"),
    "xctrace": ("profile", "trace"),
}


def instrument_binary(tool: str) -> str:
    """The executable that must be on PATH for ``tool`` (the ``shutil.which`` probe)."""
    return _BINARY[tool]


def artifact_path(run_dir: Path, workload: str, tool: str) -> Path:
    """Where ``tool``'s profile artifact for ``workload`` lands — beside the run's
    result.json, named per tool so the three output formats don't collide."""
    prefix, ext = _ARTIFACT[tool]
    return run_dir / f"{prefix}-{workload}.{ext}"


def pyspy_native_supported(platform: str) -> bool:
    """Whether py-spy can unwind native (Rust) frames on ``platform`` (a
    ``sys.platform`` string). py-spy implements native unwinding on Linux and
    Windows only; macOS (``darwin``) and the BSDs have none, so ``--native`` there
    is rejected outright and the flamegraph is Python-only."""
    return platform.startswith("linux") or platform.startswith("win")


def validate_instrument_request(
    *,
    tool: str | None,
    instrument_args: list[str],
    workloads: list[str],
    out_given: bool,
    platform: str,
) -> str | None:
    """Validate an ``--instrument`` request purely — returns an error message if it
    is malformed, else ``None``. Covers everything decidable without touching the
    host: the arg-without-a-tool mistake, the one-workload-per-run rule, the
    ``--out`` clash, and xctrace's macOS-only constraint. PATH availability of the
    tool is impure and checked by the caller, not here."""
    if instrument_args and tool is None:
        return "--instrument-arg needs --instrument (no tool to pass it to)"
    if tool is None:
        return None
    if len(workloads) != 1:
        return "--instrument profiles ONE workload per run; pass a single --workloads"
    if out_given:
        return "--out is ignored under --instrument; result.json lands beside the artifact"
    if tool == "xctrace" and not platform.startswith("darwin"):
        return "--instrument=xctrace is macOS-only (Apple Instruments); use py-spy or samply"
    return None


def inner_run_argv(
    run_py: Path,
    *,
    profile: str | None,
    profile_path: Path | None,
    size: int,
    variant: str,
    workload: str,
    repeats: int,
    warmup: int,
    ops: int,
    baseline: Path | None,
) -> list[str]:
    """The ``python run.py ...`` argv the profiler launches: ONE workload (one per
    run keeps the attribution clean) and NO ``--instrument`` (so it doesn't
    re-exec). Driven by ``sys.executable`` so the venv's interpreter — and thus the
    staged native extension — is the one profiled. The profile selection is
    reproduced as it was given: ``--profile-path`` for a custom profile, else
    ``--profile``."""
    inner = [sys.executable, str(run_py)]
    if profile_path is not None:
        inner += ["--profile-path", str(profile_path)]
    else:
        inner += ["--profile", str(profile)]
    inner += [
        "--size", str(size),
        "--variant", variant,
        "--workloads", workload,
        "--repeats", str(repeats),
        "--warmup", str(warmup),
        "--ops", str(ops),
    ]  # fmt: skip
    if baseline is not None:
        inner += ["--baseline", str(baseline)]
    return inner


def instrument_command(
    tool: str,
    run_py: Path,
    run_dir: Path,
    *,
    platform: str,
    extra_args: list[str],
    profile: str | None = None,
    profile_path: Path | None = None,
    size: int,
    variant: str,
    workload: str,
    repeats: int,
    warmup: int,
    ops: int,
    baseline: Path | None,
) -> list[str]:
    """The full argv that profiles ONE workload under ``tool`` and writes its
    artifact into ``run_dir``. ``extra_args`` (from ``--instrument-arg``) are
    passed verbatim to the tool, after its built-in options and before the target
    command, so a caller can tune the run (sampling rate, time limit, …)."""
    if tool not in INSTRUMENTERS:
        raise ValueError(f"unknown instrumenter {tool!r}; choices: {list(INSTRUMENTERS)}")
    inner = inner_run_argv(
        run_py,
        profile=profile,
        profile_path=profile_path,
        size=size,
        variant=variant,
        workload=workload,
        repeats=repeats,
        warmup=warmup,
        ops=ops,
        baseline=baseline,
    )
    artifact = str(artifact_path(run_dir, workload, tool))
    if tool == "py-spy":
        return _pyspy_command(artifact, inner, extra_args, platform)
    if tool == "samply":
        return _samply_command(artifact, inner, extra_args)
    assert tool == "xctrace"  # membership checked above; the registry is exhaustive
    return _xctrace_command(artifact, inner, extra_args)


def _pyspy_command(artifact: str, inner: list[str], extra: list[str], platform: str) -> list[str]:
    cmd = ["py-spy", "record"]
    if pyspy_native_supported(platform):
        # --native is the cross-boundary requirement (Python + Rust in one
        # flamegraph). py-spy has no native unwinding on macOS/BSD, where passing
        # it aborts the run, so it is omitted there and the graph is Python-only.
        cmd.append("--native")
    cmd += ["--format", "flamegraph", "--output", artifact, *extra, "--", *inner]
    return cmd


def _samply_command(artifact: str, inner: list[str], extra: list[str]) -> list[str]:
    # --save-only writes the profile and exits, instead of launching samply's
    # blocking web-UI server; load it later with `samply load <artifact>`.
    return ["samply", "record", "--save-only", "--output", artifact, *extra, "--", *inner]


def _xctrace_command(artifact: str, inner: list[str], extra: list[str]) -> list[str]:
    # Apple Instruments' Time Profiler. --output must not already exist (run_dir is
    # fresh per run); --target-stdout - forwards the inner run's stdout so its
    # progress still prints; --launch runs the target command that follows it.
    return [
        "xcrun", "xctrace", "record",
        "--template", "Time Profiler",
        "--output", artifact,
        "--target-stdout", "-",
        *extra,
        "--launch", "--", *inner,
    ]  # fmt: skip
