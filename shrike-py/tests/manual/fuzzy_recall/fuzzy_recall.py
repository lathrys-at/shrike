"""The fuzzy-recall + as-you-type-latency frontier eval driver.

Builds (or reuses) the perf corpus, materializes a derived-text store from it,
generates a deterministic typo-query set with gold known by construction, then runs
each cap-policy arm and reports recall@10 / MRR aggregated per (query-length bucket
× typo-count × edit-type), plus — with ``--latency`` — the **single-query lexical
latency** frontier that the #977 as-you-type budget reads off.

Two axes with very different trust:

- **Recall** (the RELIABLE result) is the fuzzy signal in isolation
  (``search_fuzzy_batch``), deterministic (a function of the seeded corpus + query set
  + cap policy) and mode-independent — so it is correct on ANY machine and the cap
  policy is the only variable. The recall@10 grid is the eval's actual output.
- **Latency** (a HARNESS, not a result) is the SINGLE-QUERY lexical path (one substring
  + one fuzzy read per query — the per-keystroke cost, not a batch), reported as the
  p50/p90/p95/p99/max distribution against a **p95 ≤ 10ms** budget (:data:`BUDGET_MS`).
  It is wired with ``--latency`` but is **untrustworthy on a loaded/dev machine** —
  there it is noise. The AUTHORITATIVE single-query latency, and therefore the budget
  verdict + the cap pick + the tail-driver, come ONLY from the maintainer's clean-env
  run; the local output is a template that run fills in (loudly labelled untrusted).

The cap sweep is the 2D ``k × ceiling`` grid (floor pinned at 6) — see :func:`cap_grid`.
On a clean run the latency harness picks the curve with max recall@10 whose single-query
**p95** clears the budget (the #927 cap-default decision), and the **tail diagnostic**
(:func:`_render_slow_query_diagnostic`) names which query shapes (trigram count,
candidate-set size, typo count) drive the slow tail — so the worst-case bound (PR2) is
designed from real numbers, not a dev-box guess.

Run it directly::

    python shrike-py/tests/manual/fuzzy_recall/fuzzy_recall.py --notes 5000
    python shrike-py/tests/manual/fuzzy_recall/fuzzy_recall.py --notes 50000 --sample 1000 --latency
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Allow running from a bare checkout without an editable install (mirrors corpus.py).
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shrike_native  # noqa: E402

from shrike.harness.collection import CollectionWrapper  # noqa: E402
from tests.manual.fuzzy_recall import misspellings as misspellings_mod  # noqa: E402
from tests.manual.fuzzy_recall.typo_queries import (  # noqa: E402
    GoldResolver,
    TypoQuery,
    generate_queries,
    length_bucket,
)
from tests.manual.perf import corpus as corpus_mod  # noqa: E402
from tests.manual.perf import wordlist  # noqa: E402
from tests.manual.search_quality.metrics import (  # noqa: E402
    GradedGold,
    ReturnedCard,
    evaluate_query,
)

#: The schema version the derived store opens at (must match the facade / native).
_DERIVED_SCHEMA_VERSION = 2
#: recall@k / MRR are computed at this k (the eval's headline cut).
TOP_K = 10
#: The as-you-type single-query round-trip budget (#977): a lexical search must
#: clear this on a 50k corpus to populate a dropdown per keystroke. The frontier
#: picks the cap curve with max recall whose single-query p50 stays under it.
BUDGET_MS = 10.0


@dataclass(frozen=True)
class CapPolicy:
    """One cap-policy arm: a label and the ``(floor, k, ceiling)`` set on the engine
    before its queries run. The control is ``fixed-6`` (floor == ceiling == 6)."""

    label: str
    floor: int
    k: float
    ceiling: int


#: The fixed cap policy every arm is read against — the shipped default.
CONTROL: CapPolicy = CapPolicy("fixed-6 (control)", floor=6, k=2.7, ceiling=6)

#: The 2D cap grid (#927/#977): per-query rare-trigram budget = clamp(k·ln(n), floor,
#: ceiling). The frontier sweeps `k` (growth rate) × `ceiling` (the hard cap a long
#: query saturates at) with `floor` pinned at 6, so the recall/single-query-latency
#: knee is visible across the whole flank — the prior 3-point curve sweep sat on the
#: steep part and couldn't show where the budget cuts.
GRID_KS: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0)
GRID_CEILINGS: tuple[int, ...] = (10, 12, 16, 20, 24)
GRID_FLOOR: int = 6


def cap_grid(
    ks: tuple[float, ...] = GRID_KS,
    ceilings: tuple[int, ...] = GRID_CEILINGS,
    floor: int = GRID_FLOOR,
) -> tuple[CapPolicy, ...]:
    """The control followed by the full `k × ceiling` grid (floor pinned). A
    `(k, ceiling)` cell whose `k·ln(n)` never reaches `ceiling` over the query set
    still runs — it just behaves like a lower effective ceiling, which the frontier
    reads correctly. The control is first so it anchors the per-arm deltas."""
    grid = tuple(
        CapPolicy(f"k={k:g} ceil={ceil}", floor=floor, k=k, ceiling=ceil)
        for k in ks
        for ceil in ceilings
    )
    return (CONTROL, *grid)


#: The arms swept by default: the control + the 2D frontier grid.
DEFAULT_ARMS: tuple[CapPolicy, ...] = cap_grid()


@dataclass(frozen=True)
class SingleQueryLatency:
    """Single-query lexical-path latency over the query set, in milliseconds:
    each sample is ONE query's substring + fuzzy read (the as-you-type cost per
    keystroke), so the percentiles describe a single keystroke — not a batch.
    The user's budget is read at **p95** (`p95_ms`); the others bracket the tail."""

    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class QuerySample:
    """One query's measured latency plus the cheap shape features that might explain
    a slow tail — the diagnostic feedstock for "what shapes are the slow ones". All
    features are eval-side (no native DF/posting introspection): trigram count, typo
    count, and the candidate-set size the two lexical reads returned (the fuzzy posting
    scan + substring rows the assembly would hydrate). `cap_label` tags which arm."""

    query: str
    latency_ms: float
    n_trigrams: int
    typo_count: int
    candidates: int


@dataclass(frozen=True)
class ArmResult:
    """One arm's aggregated recall@10 / MRR over the query set, plus the per-slice
    breakdowns and the optional measured single-query latency + slow-query samples."""

    policy: CapPolicy
    recall_at_k: float
    mrr: float
    n_queries: int
    by_length: dict[str, tuple[float, float, int]]  # bucket -> (recall, mrr, n)
    by_typo_count: dict[int, tuple[float, float, int]]
    by_edit: dict[str, tuple[float, float, int]]
    latency: SingleQueryLatency | None = None
    #: Per-query latency + shape samples (only for the control arm — the tail
    #: diagnostic is per-corpus, not per-cap; capturing it once avoids 25× the noise).
    samples: list[QuerySample] | None = None


def _build_derived_store(
    rows: list[tuple[int, str, str, str]], path: Path
) -> shrike_native.DerivedTextEngine:
    """A derived-text store built from the corpus's ``(note_id, "field", ref,
    value)`` rows — the same rows the harness's rebuild would ingest, so the indexed
    text (and thus the fuzzy/substring behaviour) matches production."""
    engine = shrike_native.DerivedTextEngine(str(path), _DERIVED_SCHEMA_VERSION)
    engine.build(rows, 1, None)
    return engine


def _first_field_texts(rows: list[tuple[int, str, str, str]]) -> dict[int, str]:
    """Per note, the text of its FIRST field (``Front`` for Basic, ``Text`` for
    Cloze) — the corpus's plain-text field, the honest phrase source. The first row
    per note in ``derived_field_rows`` order is the first field; later rows (Back /
    Back Extra) may carry HTML or cloze markup, so they are not used as the phrase
    source (the gold substring probe needs a literal match against the index)."""
    out: dict[int, str] = {}
    for note_id, _source, _ref, value in rows:
        out.setdefault(note_id, value)
    return out


#: The substring-probe row limit when resolving gold. recall@k's denominator is the
#: WHOLE gold set, so a truncated probe would inflate recall (the cap is never the
#: true relevant set). Sized far above any honest gold count at the standard corpus
#: sizes (50k notes × ~2 text segments), and a phrase whose probe HITS the limit is
#: dropped as too common to be a useful fuzzy probe — never silently truncated.
_GOLD_SUBSTRING_LIMIT = 200_000


def _make_gold_resolver(engine: shrike_native.DerivedTextEngine) -> GoldResolver:
    """A resolver mapping a clean phrase to every note whose indexed text literally
    contains it, via the exact-substring path — the honest gold set (every matching
    note, not just the source). ``search_substring`` returns ``None`` for a
    sub-trigram phrase, mapping to an empty gold set (the query is dropped upstream);
    a phrase whose match count reaches the probe limit is also dropped (its gold
    would be truncated, which would inflate recall) by returning an empty set."""

    def resolve(phrase: str) -> list[int]:
        rows = engine.search_substring(phrase, _GOLD_SUBSTRING_LIMIT, None)
        if rows is None:
            return []
        # rows are (note_id, source, ref, snippet); a note can match on several
        # segments, so dedupe to note ids. A full probe (>= the limit) means gold
        # was truncated — drop the phrase rather than score against a partial set.
        if len(rows) >= _GOLD_SUBSTRING_LIMIT:
            return []
        return list({r[0] for r in rows})

    return resolve


def _percentile(ordered: list[float], q: float) -> float:
    """The ``q`` quantile (0..1) of an already-sorted list, by linear interpolation
    (the perf harness's convention, so the numbers read the same way)."""
    if not ordered:
        return 0.0
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


#: Single-query timing discards this many warmup samples (page cache + posting-list
#: priming) before the percentiles, so the first-touch cost never skews p50.
_LATENCY_WARMUP = 50


def _latency_summary(samples_ms: list[float]) -> SingleQueryLatency:
    """The percentile bracket of a single-query latency sample (the user's budget is
    read at p95; p50/p90/p99/max bracket the tail)."""
    ordered = sorted(samples_ms)
    return SingleQueryLatency(
        p50_ms=_percentile(ordered, 0.50),
        p90_ms=_percentile(ordered, 0.90),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
        max_ms=max(ordered) if ordered else 0.0,
    )


def _measure_single_query_latency(
    engine: shrike_native.DerivedTextEngine,
    queries: list[TypoQuery],
    *,
    capture_samples: bool = False,
) -> tuple[SingleQueryLatency, list[QuerySample]]:
    """Time the LEXICAL PATH per SINGLE query — one substring read + one fuzzy read
    per query, the as-you-type cost of a single keystroke (NOT a batch). This is the
    #977 budget probe: the kernel's fused search runs the two lexical lanes per query,
    so the single-query wall cost is their serial sum here (the kernel can overlap
    them on the compute pool — this is the conservative serial bound). The cap policy
    must already be set. Warmup samples are discarded so first-touch I/O never skews
    p50. With ``capture_samples`` it also records each query's latency + cheap shape
    features (trigram count, typo count, candidate-set size) for the tail diagnostic.
    Timing is noise-sensitive — read it only from a clean-environment run."""
    # Cap the warmup at half the set so a small query set still yields timed samples
    # (the standard --sample 500/1000 run is far above the constant either way).
    warmup = min(_LATENCY_WARMUP, len(queries) // 2)
    samples_ms: list[float] = []
    samples: list[QuerySample] = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        sub = engine.search_substring(q.query, TOP_K, None)
        fz = engine.search_fuzzy_batch([q.query], TOP_K, None)
        elapsed = (time.perf_counter() - start) * 1000.0
        if i < warmup:
            continue
        samples_ms.append(elapsed)
        if capture_samples:
            # Candidate-set size = the rows the two lanes returned (capped at TOP_K each
            # here; the eval reads the post-cap result, the honest as-you-type cost).
            candidates = (len(sub) if sub else 0) + (len(fz[0]) if fz else 0)
            samples.append(
                QuerySample(
                    query=q.query,
                    latency_ms=elapsed,
                    n_trigrams=q.n_trigrams,
                    typo_count=q.typo_count,
                    candidates=candidates,
                )
            )
    return _latency_summary(samples_ms), samples


def _evaluate_arm(
    engine: shrike_native.DerivedTextEngine,
    policy: CapPolicy,
    queries: list[TypoQuery],
    *,
    batch_size: int = 200,
    measure_latency: bool = False,
    capture_samples: bool = False,
) -> ArmResult:
    """Run one arm: set the cap policy, fuzzy-search the whole query set (batched, for
    recall), aggregate recall@10 / MRR overall and per slice, and — when
    ``measure_latency`` — time the SINGLE-QUERY lexical path (substring + fuzzy per
    query) for the as-you-type frontier. ``capture_samples`` additionally records the
    per-query latency + shape features for the tail diagnostic (set once, for the
    control arm). Recall is computed from the batched fuzzy read (deterministic,
    mode-independent); latency is a separate single-query pass. Timing is off by
    default — it belongs on a clean machine."""
    engine.set_fuzzy_cap_policy(policy.floor, policy.k, policy.ceiling)
    texts = [q.query for q in queries]

    results: list[list[tuple]] = []
    for i in range(0, len(texts), batch_size):
        results.extend(engine.search_fuzzy_batch(texts[i : i + batch_size], TOP_K, None))

    # Single-query latency is a SEPARATE pass (after the recall read, so the cap
    # policy and page cache are both warm) — the as-you-type per-keystroke cost.
    latency: SingleQueryLatency | None = None
    samples: list[QuerySample] | None = None
    if measure_latency:
        latency, captured = _measure_single_query_latency(
            engine, queries, capture_samples=capture_samples
        )
        samples = captured if capture_samples else None

    # Aggregate. recall@k / MRR come from the shared metric engine (open-world: an
    # un-graded fuzzy return is not a false positive — recall is all we measure).
    recalls: list[float] = []
    mrrs: list[float] = []
    by_length: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_typo: dict[int, list[tuple[float, float]]] = defaultdict(list)
    by_edit: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for q, rows in zip(queries, results, strict=True):
        returned = [ReturnedCard(note_id=r[0], rank=rank) for rank, r in enumerate(rows, start=1)]
        gold = GradedGold(
            grades={nid: 2 for nid in q.gold_ids},
            expected_signal="fuzzy",
            closed_world=False,
            top_k=TOP_K,
        )
        report = evaluate_query(q.query, "fuzzy", returned, gold)
        # recall_at_k / mrr are defined (gold always has grade>=2 here).
        r_at_k = report.recall_at_k or 0.0
        mrr = report.mrr or 0.0
        recalls.append(r_at_k)
        mrrs.append(mrr)
        by_length[length_bucket(q.n_trigrams)].append((r_at_k, mrr))
        by_typo[q.typo_count].append((r_at_k, mrr))
        for kind in q.edits:
            by_edit[str(kind)].append((r_at_k, mrr))

    def fold(pairs: list[tuple[float, float]]) -> tuple[float, float, int]:
        if not pairs:
            return (0.0, 0.0, 0)
        n = len(pairs)
        return (sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n, n)

    return ArmResult(
        policy=policy,
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        mrr=sum(mrrs) / len(mrrs) if mrrs else 0.0,
        n_queries=len(queries),
        by_length={k: fold(v) for k, v in by_length.items()},
        by_typo_count={k: fold(v) for k, v in by_typo.items()},
        by_edit={k: fold(v) for k, v in by_edit.items()},
        latency=latency,
        samples=samples,
    )


@dataclass(frozen=True)
class EvalRun:
    """A full eval run: the corpus spec, the query set, and one result per arm."""

    notes: int
    seed: int
    sample_size: int
    n_queries: int
    arms: list[ArmResult]


def run_eval(
    *,
    notes: int = 5000,
    seed: int = 0,
    sample_size: int = 500,
    arms: tuple[CapPolicy, ...] = DEFAULT_ARMS,
    measure_latency: bool = False,
    offline: bool = False,
    cache_root: Path | None = None,
) -> EvalRun:
    """Build/reuse the corpus, generate the query set once, and run every arm
    against it. Pure of timing unless ``measure_latency`` (off by default).
    ``cache_root`` overrides the corpus cache location (the default repo-root
    ``.cache``) — a sandboxed test points it at a writable temp dir."""
    if not offline:
        # Fetch the full vocabulary + the real misspellings list (both cached). A
        # full run uses them; offline falls back to the embedded sets.
        wordlist.ensure_wordlist()
        misspellings_mod.ensure_misspellings()
    misspell_map = misspellings_mod.load_misspellings()

    spec = corpus_mod.CorpusSpec(notes=notes, variant="text", seed=seed)
    built = corpus_mod.ensure_corpus(spec, cache_root=cache_root)

    wrapper = CollectionWrapper(str(built.anki2_path))
    try:
        note_ids = wrapper.run_sync(lambda c: c.find_notes(""))
        rows = wrapper.run_sync(lambda c: c.derived_field_rows(note_ids))
    finally:
        wrapper.close()

    with tempfile.TemporaryDirectory(prefix="fuzzy-recall-") as tmp:
        engine = _build_derived_store(rows, Path(tmp) / "shrike.db")
        try:
            first_field = _first_field_texts(rows)
            queries = generate_queries(
                first_field,
                _make_gold_resolver(engine),
                seed=seed,
                sample_size=sample_size,
                misspellings=misspell_map,
            )
            if not queries:
                raise RuntimeError("typo-query generation produced no queries — corpus too small?")
            # Capture the per-query tail diagnostic ONCE, on the first (control) arm —
            # the shape→latency relationship is a corpus property, not a per-cap one,
            # so one capture avoids 25× the sample noise.
            results = [
                _evaluate_arm(
                    engine,
                    p,
                    queries,
                    measure_latency=measure_latency,
                    capture_samples=(measure_latency and i == 0),
                )
                for i, p in enumerate(arms)
            ]
        finally:
            # Close the native handle before the temp dir is torn down, so the SQLite
            # file is released (not left open to GC) when the directory is removed.
            engine.close()

    return EvalRun(
        notes=notes,
        seed=seed,
        sample_size=sample_size,
        n_queries=len(queries),
        arms=results,
    )


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def render_results(run: EvalRun) -> str:
    """The RESULTS.md artifact: overall + per-bucket recall@10/MRR per arm, with the
    control (fixed-6) called out and the delta of each arm against it per bucket."""
    control = run.arms[0]
    lines: list[str] = []
    lines.append("# Fuzzy-recall eval results")
    lines.append("")
    lines.append(
        f"Corpus: **{run.notes} notes** (seed {run.seed}, text variant) · "
        f"**{run.n_queries}** typo queries (sample {run.sample_size}) · "
        f"recall@{TOP_K} primary, MRR secondary."
    )
    lines.append("")
    lines.append(
        "Gold by construction (a note's own clean text; gold = all notes containing "
        "the un-perturbed phrase). Recall is the fuzzy signal in isolation "
        "(`search_fuzzy_batch`), deterministic and mode-independent, so the cap policy "
        "is the only variable; latency (the frontier below) is the single-query lexical "
        "path. Control = **fixed-6** (the shipped default)."
    )
    lines.append("")

    # Headline table.
    lines.append("## Overall")
    lines.append("")
    lines.append("| arm | floor | k | ceiling | recall@10 | Δ vs control | MRR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for a in run.arms:
        delta = a.recall_at_k - control.recall_at_k
        dstr = "—" if a is control else f"{delta * 100:+.1f}"
        lines.append(
            f"| {a.policy.label} | {a.policy.floor} | {a.policy.k:g} | {a.policy.ceiling} | "
            f"{_fmt_pct(a.recall_at_k)} | {dstr} | {_fmt_pct(a.mrr)} |"
        )
    lines.append("")

    # Per query-length bucket (where the curve is meant to act).
    buckets = ["n<=6", "n7-11", "n12-17", "n18+"]
    lines.append("## recall@10 by query-length bucket (trigram count)")
    lines.append("")
    lines.append("The cap curve only acts on n > the floor, so the long-query buckets are where")
    lines.append("a growth arm must show a win and a floor-down arm must not lose.")
    lines.append("")
    header = "| arm | " + " | ".join(buckets) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in buckets) + "|")
    for a in run.arms:
        cells = []
        for b in buckets:
            rec, _mrr, n = a.by_length.get(b, (0.0, 0.0, 0))
            cells.append(f"{_fmt_pct(rec)} (n={n})" if n else "—")
        lines.append(f"| {a.policy.label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per typo count (degradation).
    lines.append("## recall@10 by typo count")
    lines.append("")
    counts = sorted({c for a in run.arms for c in a.by_typo_count})
    header = "| arm | " + " | ".join(f"{c} typo" for c in counts) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in counts) + "|")
    for a in run.arms:
        cells = []
        for c in counts:
            rec, _mrr, n = a.by_typo_count.get(c, (0.0, 0.0, 0))
            cells.append(f"{_fmt_pct(rec)} (n={n})" if n else "—")
        lines.append(f"| {a.policy.label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per edit type (secondary breakdown).
    lines.append("## recall@10 by edit type")
    lines.append("")
    edits = sorted({e for a in run.arms for e in a.by_edit})
    lines.append("| arm | " + " | ".join(edits) + " |")
    lines.append("|---|" + "|".join("---:" for _ in edits) + "|")
    for a in run.arms:
        cells = []
        for e in edits:
            rec, _mrr, n = a.by_edit.get(e, (0.0, 0.0, 0))
            cells.append(f"{_fmt_pct(rec)} (n={n})" if n else "—")
        lines.append(f"| {a.policy.label} | " + " | ".join(cells) + " |")
    lines.append("")

    if any(a.latency is not None for a in run.arms):
        lines.extend(_render_frontier(run, control))

    return "\n".join(lines)


def _render_frontier(run: EvalRun, control: ArmResult) -> list[str]:
    """The single-query latency HARNESS output (#977). IMPORTANT: latency is only
    trustworthy from a clean-environment run — on any loaded/dev machine these numbers
    are noise, so this whole section is a TEMPLATE the maintainer's clean run fills in,
    not a result. The budget is **p95 ≤ 10ms**; the `p95≤` verdict and the cap pick are
    computed mechanically from whatever latency was measured, so they only MEAN anything
    once the latency is real. The RELIABLE #977 result is the recall@10 grid above — that
    is deterministic and environment-independent; this section is not."""
    timed = [a for a in run.arms if a.latency is not None]
    lines: list[str] = []
    lines.append("## single-query latency harness — UNTRUSTED unless clean-env")
    lines.append("")
    lines.append(
        "> ⚠️ **Latency below is NOT a result unless this was a clean-environment run.** "
        "On a loaded/dev machine it is noise. The authoritative single-query latency (and "
        "therefore the budget verdict + cap pick) is the maintainer's clean-env run; until "
        "then read only the recall grid above. The budget is **p95 ≤ "
        f"{BUDGET_MS:g}ms** (#977)."
    )
    lines.append("")
    lines.append(
        f"Single-query latency = ONE query's substring + fuzzy read (the per-keystroke "
        f"cost) over {run.n_queries} queries after a {_LATENCY_WARMUP}-query warmup. The "
        "budget reads the p95 tail (what every keystroke feels), not the median; `p90≤`/"
        "`p99≤` bracket it. The serial sum is the conservative bound (the kernel can "
        "overlap the two lanes on the compute pool)."
    )
    lines.append("")
    lines.append(
        "| arm | recall@10 | Δ vs control | p50 ms | p90 ms | p95 ms | p99 ms | max ms | "
        "p90≤ | p95≤ | p99≤ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|")

    def mark(v: float) -> str:
        return "✓" if v <= BUDGET_MS else "✗"

    for a in timed:
        lat = a.latency
        assert lat is not None  # filtered above; for the type-checker
        delta = a.recall_at_k - control.recall_at_k
        dstr = "—" if a is control else f"{delta * 100:+.1f}"
        lines.append(
            f"| {a.policy.label} | {_fmt_pct(a.recall_at_k)} | {dstr} | "
            f"{lat.p50_ms:.3f} | {lat.p90_ms:.3f} | {lat.p95_ms:.3f} | {lat.p99_ms:.3f} | "
            f"{lat.max_ms:.3f} | {mark(lat.p90_ms)} | {mark(lat.p95_ms)} | {mark(lat.p99_ms)} |"
        )
    lines.append("")
    lines.extend(_render_picks(timed))
    if control.samples:
        lines.extend(_render_slow_query_diagnostic(control.samples))
    return lines


def _render_picks(timed: list[ArmResult]) -> list[str]:
    """The cap pick at the user's budget (p95 ≤ 10ms) — max recall@10 among arms whose p95
    clears the budget, ties broken by the lower p95. The p50 pick is surfaced too as the
    "if the bar were the median" contrast (it will be a looser cap), making explicit that
    the budget is the tail, not the median."""
    lines: list[str] = []

    def pick_for(pct: str, value: Callable[[SingleQueryLatency], float]) -> str:
        eligible = [a for a in timed if a.latency is not None and value(a.latency) <= BUDGET_MS]
        if not eligible:
            return (
                f"- **{pct} budget:** no arm clears {BUDGET_MS:g}ms at {pct} — the budget cut "
                "takes the cheapest arm, or the bound (PR2) has to bring the tail down first."
            )
        best = max(
            eligible,
            key=lambda a: (a.recall_at_k, -(value(a.latency) if a.latency else 0.0)),
        )
        assert best.latency is not None
        return (
            f"- **{pct} budget:** `{best.policy.label}` (floor={best.policy.floor}, "
            f"k={best.policy.k:g}, ceiling={best.policy.ceiling}) — recall@10 "
            f"{_fmt_pct(best.recall_at_k)} at {pct} {value(best.latency):.3f}ms."
        )

    lines.append(
        f"**Cap pick (max recall@10 within {BUDGET_MS:g}ms)** — mechanical from the "
        "latency above, so only valid on a clean-env run:"
    )
    lines.append(pick_for("p95", lambda lat: lat.p95_ms))  # the user's budget
    lines.append(pick_for("p50", lambda lat: lat.p50_ms))  # the looser median contrast
    lines.append("")
    return lines


#: How many of the slowest queries to surface in the tail diagnostic.
_SLOW_QUERY_SHOW = 15


def _render_slow_query_diagnostic(samples: list[QuerySample]) -> list[str]:
    """The TAIL DIAGNOSTIC (#977): what SHAPES are the slow single queries? Correlates
    per-query latency with the cheap eval-side shape features (trigram count, typo count,
    candidate-set size) so PR2 can pick the right bound (candidate cap / posting cap /
    time-budget early-exit). Captured once on the control arm.

    Native DF / posting-scan introspection is not exposed to the eval, so the rarest-
    trigram DF — the suspected fuzzy-tail driver (the posting SCAN) — is NOT measured
    here; candidate-set size is its observable proxy (a high-DF trigram yields a large
    posting → large candidate set). If candidate size doesn't explain the tail, a native
    per-query DF/scan-count hook is the follow-up."""
    lines: list[str] = []
    lines.append("### tail diagnostic — what shapes the slow single queries")
    lines.append("")
    lines.append(
        "> ⚠️ The slow/fast split is ordered by the UNTRUSTED local latency above, so the "
        "feature contrast is only meaningful on a clean-env run. The machinery is here so "
        "the maintainer's run names the driver; do not read a driver off dev-box noise."
    )
    lines.append("")

    n = len(samples)
    by_lat = sorted(samples, key=lambda s: s.latency_ms, reverse=True)
    slow = by_lat[: max(1, n // 10)]  # the slowest decile
    fast = by_lat[n - max(1, n // 10) :]  # the fastest decile

    def avg(rows: list[QuerySample], f: Callable[[QuerySample], float]) -> float:
        return sum(f(r) for r in rows) / len(rows) if rows else 0.0

    # The slow-vs-fast decile contrast: which feature moves with latency tells PR2 what
    # to bound. A big trigram/candidate gap → bound those; a flat gap → the tail is
    # elsewhere (native DF hook needed). Reliable only when the latency ordering is real.
    lines.append("Slowest vs fastest decile (mean feature value) — the gap names the driver:")
    lines.append("")
    lines.append("| feature | slowest 10% | fastest 10% | ratio |")
    lines.append("|---|---:|---:|---:|")
    for name, f in (
        ("latency ms", lambda s: s.latency_ms),
        ("n_trigrams", lambda s: float(s.n_trigrams)),
        ("candidates", lambda s: float(s.candidates)),
        ("typo_count", lambda s: float(s.typo_count)),
    ):
        s_avg, f_avg = avg(slow, f), avg(fast, f)
        ratio = f"{s_avg / f_avg:.1f}×" if f_avg else "—"
        lines.append(f"| {name} | {s_avg:.2f} | {f_avg:.2f} | {ratio} |")
    lines.append("")

    # The actual slow queries, so a human can eyeball the shape (all-common-word phrase?
    # one rare token? long phrase?) the aggregate can't show.
    lines.append(f"The {min(_SLOW_QUERY_SHOW, n)} slowest queries:")
    lines.append("")
    lines.append("| query | latency ms | n_trigrams | candidates | typos |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in by_lat[:_SLOW_QUERY_SHOW]:
        q = s.query if len(s.query) <= 40 else s.query[:37] + "..."
        lines.append(
            f"| `{q}` | {s.latency_ms:.3f} | {s.n_trigrams} | {s.candidates} | {s.typo_count} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notes", type=int, default=5000, help="Corpus size (5000 primary, 50000 scale)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Determinism seed.")
    parser.add_argument("--sample", type=int, default=500, help="Notes sampled for typo queries.")
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Also measure single-query lexical p50/p90/p95/p99/max per arm and emit "
        "the as-you-type frontier + the tail diagnostic (noisy on a loaded machine — "
        "for a clean-env run; budget is p95 <= 10ms).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the embedded fallback vocab/misspellings (no download).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "RESULTS.md",
        help="Where to write the results artifact.",
    )
    args = parser.parse_args()

    run = run_eval(
        notes=args.notes,
        seed=args.seed,
        sample_size=args.sample,
        measure_latency=args.latency,
        offline=args.offline,
    )
    report = render_results(run)
    args.out.write_text(report + "\n")
    print(report)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
