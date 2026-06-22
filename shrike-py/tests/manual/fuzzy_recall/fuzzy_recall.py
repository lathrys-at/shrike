"""The fuzzy-recall eval driver.

Builds (or reuses) the perf corpus, materializes a derived-text store from it,
generates a deterministic typo-query set with gold known by construction, then runs
each cap-policy arm and reports recall@10 / MRR aggregated per (query-length bucket
× typo-count × edit-type). The fuzzy signal is measured **in isolation** —
``search_fuzzy_batch`` directly, not the fused ``search_notes`` — so the cap policy
is the only thing moving between arms.

Recall@10 and MRR are deterministic (a function of the seeded corpus + query set +
the cap policy), so this is safe to run anywhere. The latency-delta probe is wired
but OFF by default (``--latency``): timing is noise-sensitive and belongs on a clean
machine, not a loaded dev box.

Run it directly::

    python shrike-py/tests/manual/fuzzy_recall/fuzzy_recall.py --notes 5000
    python shrike-py/tests/manual/fuzzy_recall/fuzzy_recall.py --notes 50000 --sample 1000

The arms swept by default are the floor sweep (fixed-6/5/4) and the log-growth curve
(k ∈ {2.0, 2.7, 4.0}, ceiling 12), all against the fixed-6 control.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import defaultdict
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


@dataclass(frozen=True)
class CapPolicy:
    """One cap-policy arm: a label and the ``(floor, k, ceiling)`` set on the engine
    before its queries run. The control is ``fixed-6`` (floor == ceiling == 6)."""

    label: str
    floor: int
    k: float
    ceiling: int


#: The arms swept by default. The floor sweep (down) asks whether 6 is even the
#: right constant; the log-growth curve (up) asks whether keeping more rare trigrams
#: on a long query recovers recall. fixed-6 is the control every arm is read against.
DEFAULT_ARMS: tuple[CapPolicy, ...] = (
    CapPolicy("fixed-6 (control)", floor=6, k=2.7, ceiling=6),
    CapPolicy("fixed-5", floor=5, k=2.7, ceiling=5),
    CapPolicy("fixed-4", floor=4, k=2.7, ceiling=4),
    CapPolicy("curve k=2.0", floor=6, k=2.0, ceiling=12),
    CapPolicy("curve k=2.7", floor=6, k=2.7, ceiling=12),
    CapPolicy("curve k=4.0", floor=6, k=4.0, ceiling=12),
)


@dataclass(frozen=True)
class ArmResult:
    """One arm's aggregated recall@10 / MRR over the query set, plus the per-slice
    breakdowns and the optional measured latency."""

    policy: CapPolicy
    recall_at_k: float
    mrr: float
    n_queries: int
    by_length: dict[str, tuple[float, float, int]]  # bucket -> (recall, mrr, n)
    by_typo_count: dict[int, tuple[float, float, int]]
    by_edit: dict[str, tuple[float, float, int]]
    latency_ms: float | None = None


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


def _evaluate_arm(
    engine: shrike_native.DerivedTextEngine,
    policy: CapPolicy,
    queries: list[TypoQuery],
    *,
    batch_size: int = 200,
    measure_latency: bool = False,
) -> ArmResult:
    """Run one arm: set the cap policy, fuzzy-search the whole query set (batched),
    and aggregate recall@10 / MRR overall and per slice. Optionally time the search
    (off by default — timing belongs on a clean machine)."""
    engine.set_fuzzy_cap_policy(policy.floor, policy.k, policy.ceiling)
    texts = [q.query for q in queries]

    latency_ms: float | None = None
    results: list[list[tuple]] = []
    start = time.perf_counter() if measure_latency else 0.0
    for i in range(0, len(texts), batch_size):
        results.extend(engine.search_fuzzy_batch(texts[i : i + batch_size], TOP_K, None))
    if measure_latency:
        latency_ms = (time.perf_counter() - start) * 1000.0

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
        latency_ms=latency_ms,
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
            results = [
                _evaluate_arm(engine, p, queries, measure_latency=measure_latency) for p in arms
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
        "the un-perturbed phrase). The fuzzy signal is measured in isolation "
        "(`search_fuzzy_batch`), so the cap policy is the only variable. Control = "
        "**fixed-6** (the shipped default)."
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

    if any(a.latency_ms is not None for a in run.arms):
        lines.append("## search latency (the whole query set, batched)")
        lines.append("")
        lines.append("> Timing is noise-sensitive — read it only from a clean-environment run.")
        lines.append("")
        lines.append("| arm | total ms | per query |")
        lines.append("|---|---:|---:|")
        for a in run.arms:
            if a.latency_ms is None:
                continue
            per = a.latency_ms / a.n_queries if a.n_queries else 0.0
            lines.append(f"| {a.policy.label} | {a.latency_ms:.1f} | {per:.3f} |")
        lines.append("")

    return "\n".join(lines)


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
        help="Also time each arm (noisy on a loaded machine — for a clean-env run).",
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
