"""The perf result artifact: a distribution AND the conditions it was taken under.

A latency number is meaningless without its context — the machine, the build,
the corpus, the embedder mode. :class:`RunResult` carries both, so a stored run
is a comparable artifact (see :mod:`compare`) and a cross-machine diff can refuse
itself rather than mislead. Artifacts serialize to JSON under ``.cache/perf/``
(gitignored); a baseline is just a prior run's JSON the user keeps and diffs
against by hand — none is committed to the repo.
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
    ops: int

    @classmethod
    def capture(
        cls,
        *,
        profile: str,
        corpus_size: int,
        corpus_variant: str,
        repeats: int,
        warmup: int,
        ops: int,
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
            ops=ops,
        )

    #: The fields that must match for two runs to be comparable — the machine,
    #: the build, and what was measured. ``commit``/``dirty`` are advisory (a
    #: diff across commits is the whole point); ``repeats``/``warmup`` are
    #: provenance (sample count, not per-iteration work). ``ops`` IS invariant: a
    #: different N changes the work each iteration does, so the per-iteration
    #: latency isn't comparable across it. ``native_version`` guards a
    #: stale-vs-fresh extension.
    INVARIANT = (
        "machine",
        "system",
        "python",
        "native_version",
        "optimized",
        "profile",
        "corpus_size",
        "corpus_variant",
        "ops",
    )

    def differs_from(self, other: Conditions) -> list[str]:
        """The invariant fields that DIFFER from ``other`` — empty means the two
        runs are comparable."""
        return [f for f in self.INVARIANT if getattr(self, f) != getattr(other, f)]


#: The phases a workload can measure, in display order. ``response`` (the op's
#: time to return) is always present; a workload with an asynchronous tail
#: (writes draining the index/derived queue) adds ``settle`` (the drain to
#: quiescence) and ``total`` (response + settle, per iteration).
PHASE_ORDER = ("response", "settle", "total")


@dataclass(frozen=True)
class WorkloadResult:
    """One workload's measured latency distributions — one per timed *phase* —
    plus how many items it processed (so a throughput view — items/sec — is
    derivable downstream).

    ``phases`` always carries ``"response"``; a settling workload also carries
    ``"settle"`` and ``"total"`` (see :data:`PHASE_ORDER`)."""

    workload: str
    phases: dict[str, Distribution]
    items: int

    @property
    def distribution(self) -> Distribution:
        """The response-phase distribution — the op's time to return. For a
        non-settling workload (search, rebuild) this is the whole operation."""
        return self.phases["response"]

    def ordered_phases(self) -> list[tuple[str, Distribution]]:
        """The measured phases in canonical display order (see :data:`PHASE_ORDER`)."""
        return [(p, self.phases[p]) for p in PHASE_ORDER if p in self.phases]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkloadResult:
        return cls(
            workload=raw["workload"],
            phases={k: Distribution.from_dict(v) for k, v in raw["phases"].items()},
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
        # asdict() recurses through conditions + results + distributions (the
        # dataclass nesting), so serialization is uniform; from_json reconstructs
        # explicitly (dataclasses have no auto from-dict).
        return json.dumps(asdict(self), indent=2, sort_keys=True)

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


#: The table's column headers, shared by the plain and markdown renderers. The
#: trailing ``p50 (amortized) ms`` is the per-op cost (see :func:`_amortized_p50`).
_COLUMNS = (
    "workload",
    "phase",
    "items",
    "p50 ms",
    "p90 ms",
    "p99 ms",
    "max ms",
    "p50 drift ms",
    "p50 (amortized) ms",
)


def _amortized_p50(d: Distribution, items: int) -> str:
    """The per-op p50 cell: the phase p50 spread over the iteration's ops (so a
    batch row reads against a sequential row directly; ``rebuild`` with items=1
    amortizes to its own p50). ``-`` when the workload reported no items — the
    divide must not throw."""
    return f"{d.p50_ms / items:.3f}" if items else "-"


def _meta_lines(run: RunResult) -> tuple[str, str, str]:
    """The three provenance lines above the table: the run identity, the machine/
    build, and the sampling parameters. Shared verbatim (sans prefix) by both
    renderers."""
    c = run.conditions
    return (
        f"perf run — {c.profile} @ {c.corpus_size} notes ({c.corpus_variant})",
        f"{c.system}/{c.machine} py{c.python} native={c.native_version} "
        f"build={'opt' if c.optimized else 'DEBUG'} "
        f"commit={c.commit[:12]}{'+dirty' if c.dirty else ''}",
        f"repeats={c.repeats} warmup={c.warmup} ops={c.ops} @ {run.timestamp}",
    )


def render_table(run: RunResult) -> str:
    """A compact fixed-width table of the run (the dogfooding artifact body), one
    row per workload *phase* with the latency percentiles. A settling workload
    spans three rows (response/settle/total); the workload name and item count
    print once, on its first row."""
    title, machine, sampling = _meta_lines(run)
    # The workload-name column widens to the longest name in this run (+2 gutter),
    # floored at the historical width — a long name (a scoped workload) would
    # otherwise overrun the adjacent phase column.
    name_w = max(18, max(len(_COLUMNS[0]), *(len(r.workload) for r in run.results)) + 2)
    lines = [
        f"# {title}",
        f"# {machine}",
        f"# {sampling}",
        "",
        f"{'workload':<{name_w}}{'phase':<10}{'items':>8}"
        f"{'p50 ms':>12}{'p90 ms':>12}{'p99 ms':>12}{'max ms':>12}{'p50 drift ms':>14}"
        f"{'p50 (amortized) ms':>20}",
    ]
    for r in run.results:
        for idx, (phase, d) in enumerate(r.ordered_phases()):
            name = r.workload if idx == 0 else ""
            items = str(r.items) if idx == 0 else ""
            amortized = _amortized_p50(d, r.items)
            lines.append(
                f"{name:<{name_w}}{phase:<10}{items:>8}{d.p50_ms:>12.3f}{d.p90_ms:>12.3f}"
                f"{d.p99_ms:>12.3f}{d.max_ms:>12.3f}{d.drift_ms:>14.3f}{amortized:>20}"
            )
    return "\n".join(lines)


def render_markdown_table(run: RunResult) -> str:
    """The same run as a GitHub-flavored markdown table — paste-ready into a PR or
    issue comment. The provenance lines render as a bold title over two monospace
    detail bullets (not ``#`` headings, which a comment would blow up to H1); the
    numeric columns are right-aligned."""
    title, machine, sampling = _meta_lines(run)
    head = f"| {' | '.join(_COLUMNS)} |"
    # workload/phase left-aligned, every numeric column right-aligned (`---:`).
    align = "| --- | --- | " + " | ".join(["---:"] * (len(_COLUMNS) - 2)) + " |"
    lines = [f"**{title}**", "", f"- `{machine}`", f"- `{sampling}`", "", head, align]
    for r in run.results:
        for idx, (phase, d) in enumerate(r.ordered_phases()):
            name = r.workload if idx == 0 else ""
            items = str(r.items) if idx == 0 else ""
            cells = [
                name,
                phase,
                items,
                f"{d.p50_ms:.3f}",
                f"{d.p90_ms:.3f}",
                f"{d.p99_ms:.3f}",
                f"{d.max_ms:.3f}",
                f"{d.drift_ms:.3f}",
                _amortized_p50(d, r.items),
            ]
            lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)
