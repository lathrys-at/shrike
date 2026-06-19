"""The perf result artifact: a distribution AND the conditions it was taken under.

A latency number is meaningless without its context — the machine, the build,
the corpus, the embedder mode. :class:`RunResult` carries both, so a stored run
is a comparable artifact (see :mod:`compare`) and a cross-machine diff can refuse
itself rather than mislead. Artifacts serialize to JSON under ``.cache/perf/``
(gitignored); a committed baseline is a later concern (#869).
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tests.manual.perf.stats import Distribution


def _git(*args: str) -> str | None:
    """Best-effort ``git`` invocation; ``None`` when git or the repo is absent
    (a sandboxed build, a source tarball) — the conditions degrade, never throw."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass(frozen=True)
class Conditions:
    """The taken-under context that makes a run comparable. Two runs compare
    only when their *invariant* conditions match (see :func:`compatible_with`);
    ``repeats``/``warmup`` are recorded for provenance, not compatibility."""

    commit: str
    dirty: bool
    machine: str
    system: str
    python: str
    native_version: str
    optimized: bool
    profile: str
    corpus_size: int
    corpus_variant: str
    repeats: int
    warmup: int

    @classmethod
    def capture(
        cls,
        *,
        profile: str,
        corpus_size: int,
        corpus_variant: str,
        repeats: int,
        warmup: int,
    ) -> Conditions:
        """Snapshot the live environment. Imports the native extension lazily so
        this module stays importable (and unit-testable) without a built ``.so``."""
        import shrike_native

        features = shrike_native.build_features()
        return cls(
            commit=_git("rev-parse", "HEAD") or "unknown",
            dirty=bool(_git("status", "--porcelain")),
            machine=platform.machine(),
            system=platform.system(),
            python=platform.python_version(),
            native_version=shrike_native.version(),
            # An unoptimized extension reports `debug-assertions`; a `-c opt`
            # build doesn't. A debug-build latency must never be read as a real
            # number, so this is an invariant condition below.
            optimized="debug-assertions" not in features,
            profile=profile,
            corpus_size=corpus_size,
            corpus_variant=corpus_variant,
            repeats=repeats,
            warmup=warmup,
        )

    #: The fields that must match for two runs to be comparable — the machine,
    #: the build, and what was measured. ``commit``/``dirty`` are advisory (a
    #: diff across commits is the whole point); ``repeats``/``warmup`` are
    #: provenance. ``native_version`` guards a stale-vs-fresh extension.
    INVARIANT = (
        "machine",
        "system",
        "python",
        "native_version",
        "optimized",
        "profile",
        "corpus_size",
        "corpus_variant",
    )

    def compatible_with(self, other: Conditions) -> list[str]:
        """The invariant fields that DIFFER from ``other`` — empty means the two
        runs are comparable."""
        return [f for f in self.INVARIANT if getattr(self, f) != getattr(other, f)]


@dataclass(frozen=True)
class WorkloadResult:
    """One workload's measured distribution, plus how many items it processed
    (so a throughput view — items/sec — is derivable downstream)."""

    workload: str
    distribution: Distribution
    items: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "distribution": self.distribution.as_dict(),
            "items": self.items,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkloadResult:
        return cls(
            workload=raw["workload"],
            distribution=Distribution.from_dict(raw["distribution"]),
            items=raw["items"],
        )


@dataclass(frozen=True)
class RunResult:
    """A full perf run: every workload's distribution under one set of
    conditions, stamped with the wall-clock time it was taken."""

    conditions: Conditions
    results: list[WorkloadResult]
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "conditions": asdict(self.conditions),
                "results": [r.as_dict() for r in self.results],
                "timestamp": self.timestamp,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> RunResult:
        raw = json.loads(text)
        cond_fields = Conditions.__dataclass_fields__
        return cls(
            conditions=Conditions(**{k: raw["conditions"][k] for k in cond_fields}),
            results=[WorkloadResult.from_dict(r) for r in raw["results"]],
            timestamp=raw["timestamp"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> RunResult:
        return cls.from_json(path.read_text())


def render_table(run: RunResult) -> str:
    """A compact human-readable table of the run (the dogfooding artifact body),
    one row per workload with the latency percentiles."""
    lines = [
        f"# perf run — {run.conditions.profile} @ "
        f"{run.conditions.corpus_size} notes ({run.conditions.corpus_variant})",
        f"# {run.conditions.system}/{run.conditions.machine} "
        f"py{run.conditions.python} native={run.conditions.native_version} "
        f"build={'opt' if run.conditions.optimized else 'DEBUG'} "
        f"commit={run.conditions.commit[:12]}{'+dirty' if run.conditions.dirty else ''}",
        f"# repeats={run.conditions.repeats} warmup={run.conditions.warmup} @ {run.timestamp}",
        "",
        f"{'workload':<24}{'items':>8}{'p50 ms':>12}{'p90 ms':>12}{'p99 ms':>12}{'max ms':>12}",
    ]
    for r in run.results:
        d = r.distribution
        lines.append(
            f"{r.workload:<24}{r.items:>8}{d.p50_ms:>12.3f}{d.p90_ms:>12.3f}"
            f"{d.p99_ms:>12.3f}{d.max_ms:>12.3f}"
        )
    return "\n".join(lines)
