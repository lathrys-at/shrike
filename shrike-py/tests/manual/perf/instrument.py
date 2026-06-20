"""Wire a perf run to a profiler for cross-boundary (Python + Rust) flamegraphs.

``run.py --instrument`` re-execs the run under **py-spy ``--native``** — the only
profiler that merges Python-level frames and native Rust frames into one
flamegraph, so a hotspot is attributable whether it lives in the harness glue or
the kernel (``run.py -> search_notes -> kernel.search -> usearch...`` in a single
view). The rationale and the rejected alternatives are in ``docs/dev/decisions.md``.

This module builds that invocation. It is kept pure (no native, no harness import)
so it unit-tests on the per-PR lane, off the manual native lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: How the outer (profiling) run hands the inner run their shared output dir, so
#: the flamegraph and the result.json land together. The inner run reads it in
#: place of minting a fresh timestamped dir.
RUN_DIR_ENV = "SHRIKE_PERF_RUN_DIR"


def flame_path(run_dir: Path, workload: str) -> Path:
    """Where the workload's flamegraph SVG lands — next to the run's result."""
    return run_dir / f"flame-{workload}.svg"


def pyspy_command(
    run_py: Path,
    run_dir: Path,
    *,
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
    """The ``py-spy record --native -- python run.py ...`` argv that profiles ONE
    workload and writes its flamegraph into ``run_dir``.

    The inner run carries a single ``--workloads`` (one workload per run keeps the
    attribution clean) and NO ``--instrument`` (so it doesn't re-exec). It is driven
    by ``sys.executable`` so the venv's interpreter — and thus the staged native
    extension — is the one profiled. The profile selection is reproduced as it was
    given: ``--profile-path`` when a custom profile was used, else ``--profile``."""
    inner = [sys.executable, str(run_py)]
    if profile_path is not None:
        inner += ["--profile-path", str(profile_path)]
    else:
        inner += ["--profile", str(profile)]
    inner += [
        "--size",
        str(size),
        "--variant",
        variant,
        "--workloads",
        workload,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--ops",
        str(ops),
    ]
    if baseline is not None:
        inner += ["--baseline", str(baseline)]
    return [
        "py-spy",
        "record",
        "--native",
        "--format",
        "flamegraph",
        "--output",
        str(flame_path(run_dir, workload)),
        "--",
        *inner,
    ]
