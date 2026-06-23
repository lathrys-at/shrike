"""The search-quality metric engine.

A PURE function of ``(returned_matches_with_provenance, graded_gold)`` — no
server, no model, no I/O. That purity is the load-bearing property: the same
engine that pins the integration suite's pass/fail re-runs unchanged as a
parameter sweep (over ``threshold`` / ``ACTIVATION_MARGIN`` / weights /
``RRF_K``), because none of those knobs change the *shape* of a query result,
only which cards a server returns — and this module only ever consumes a
returned-vs-gold pair.

Grade scale (reconciled):
  0 = planted distractor (drives precision / false positives)
  1 = marginal (counts in nDCG only, never in the recall denominator)
  2 = relevant
  3 = canonical answer
The recall denominator is ``grade >= 2``; ``grade-1`` contributes graded gain
to nDCG but is neither a recall hit nor a false positive.

Closed-world grading: when a manifest is ``closed_world`` (the corpus is fully
graded), any returned card with no explicit grade is treated as grade-0 — a
false positive — so a query can be held to "return nothing off-topic".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class FailureKind(StrEnum):
    """How a query's result diverged from its gold — recall and precision
    regress independently, so the artifact must say which moved."""

    RECALL_MISS = "recall_miss"  # a grade>=2 gold card not in top-k
    PRECISION_FP = "precision_fp"  # a grade-0 card returned (incl. closed-world)
    OVER_RETURN = "over_return"  # a null-gold query returned a grade>=2-class hit
    DEGRADE_SILENT = "degrade_silent"  # a degraded response that didn't announce it


@dataclass(frozen=True)
class ReturnedCard:
    """One card a query returned, with the provenance the suite reasons over.

    ``signals`` is the set of fusion signal names on the match's provenance
    (``text`` / ``image`` / ``exact`` / ``fuzzy`` / ``tag``); ``has_substring``
    /``has_fuzzy`` mirror the per-signal annotations on ``SearchMatch``;
    ``score`` is the semantic cosine (``None`` for a lexical-only hit). The
    1-based ``rank`` is the card's position in the returned list.
    """

    note_id: int
    rank: int
    signals: frozenset[str] = frozenset()
    score: float | None = None
    has_substring: bool = False
    has_fuzzy: bool = False


@dataclass(frozen=True)
class Failure:
    """One tagged divergence, with the *why* (which signal should have / did)."""

    kind: FailureKind
    note_id: int | None
    detail: str
    expected_signal: str | None = None
    surfaced_signals: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QueryReport:
    """Per-query metrics + tagged failures over one returned-vs-gold pair."""

    query: str
    adversarial_class: str
    # Recall family (None when the query has no grade>=2 gold — a null/precision
    # query; recall is undefined, so it's excluded from recall aggregates).
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_k: float | None
    hit_at_k: float | None
    mrr: float | None
    ndcg_at_10: float | None
    # Precision family (always defined for a non-empty return).
    precision_at_k: float | None
    false_positive_rate: float | None  # grade-0 returned / total returned
    over_returned: bool  # a null-gold query surfaced a relevant-class hit
    exact_tier_pure: bool | None  # no grade-0 floated by the exact override
    returned_count: int
    failures: tuple[Failure, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class GradedGold:
    """The graded gold for one query: ``{note_id: grade}`` plus the query's
    intended winning signal (``None`` for a null-gold/precision query) and the
    ``closed_world`` flag that turns un-graded returns into grade-0."""

    grades: Mapping[int, int]
    expected_signal: str | None = None
    closed_world: bool = True
    top_k: int = 10

    def grade_of(self, note_id: int) -> int:
        """A returned card's grade — 0 under closed-world for an un-graded id,
        else ``None``-equivalent ignored (open-world: un-graded is unscored)."""
        if note_id in self.grades:
            return self.grades[note_id]
        return 0 if self.closed_world else -1  # -1 = "not graded, not a FP"

    @property
    def relevant_ids(self) -> frozenset[int]:
        return frozenset(nid for nid, g in self.grades.items() if g >= 2)

    @property
    def has_relevant(self) -> bool:
        return bool(self.relevant_ids)


# The canonical RRF constants (mirror shrike_kernel::fusion / search_fusion.py).
# Kept here so the golden-order recompute is a pure, dependency-free check the
# sweep can vary; the parity suite is what pins these against the kernel.
RRF_K = 60
RRF_WEIGHTS: dict[str, float] = {
    "text": 1.0,
    "image": 1.0,
    "tag": 1.0,
    "exact": 1.0,
    "fuzzy": 0.5,
}
PRIORITY_SIGNAL = "exact"


@dataclass(frozen=True)
class RankedCard:
    """A returned card with its per-signal 1-based ranks (from provenance)."""

    note_id: int
    signal_ranks: Mapping[str, int]


def rrf_order_from_ranks(
    cards: Sequence[RankedCard],
    *,
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
    priority_signal: str = PRIORITY_SIGNAL,
) -> list[int]:
    """Fused order recomputed from per-card per-signal ranks — the pure RRF.

    ``score = Σ_signal w_signal / (k + rank)``; a card carrying ``priority_signal``
    tiers above the rest; ties break on descending score then ascending note_id
    (byte-for-byte the kernel's ``rrf_fuse`` ordering)."""
    w = dict(RRF_WEIGHTS if weights is None else weights)
    rows = []
    for c in cards:
        score = sum(w.get(sig, 1.0) / (k + rank) for sig, rank in c.signal_ranks.items())
        tier = 0 if priority_signal in c.signal_ranks else 1
        rows.append((tier, -score, c.note_id))
    rows.sort()
    return [nid for _, _, nid in rows]


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(returned: Sequence[ReturnedCard], gold: GradedGold, k: int = 10) -> float:
    """Graded nDCG@k — the metric that catches a fusion reorder which keeps the
    answer in top-k but demotes it under a distractor. Gain = the card's grade
    (clamped to >=0); ideal = the corpus's best-k grades in descending order.

    Local 12-line implementation, no dependency. Returns 1.0 when there is no
    gradeable gold (vacuously perfect — a null-gold query's nDCG is not used).
    """
    gains = [max(0, gold.grade_of(c.note_id)) for c in returned[:k]]
    ideal = sorted((g for g in gold.grades.values() if g > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 1.0
    return _dcg(gains) / idcg


def evaluate_query(
    query: str,
    adversarial_class: str,
    returned: Sequence[ReturnedCard],
    gold: GradedGold,
    *,
    response_announced_degradation: bool = False,
    expects_degradation: bool = False,
) -> QueryReport:
    """The pure metric over one query's returned cards and its graded gold.

    ``expects_degradation`` + ``response_announced_degradation`` drive the
    DEGRADE_SILENT tag: a query that should run degraded (embedding down /
    sub-trigram / lexical mode) must carry the response's announcement
    (``message`` / ``completeness`` / a ``score is None``); a silent degrade is
    a failure even when the ranking happens to look fine.
    """
    ordered = sorted(returned, key=lambda c: c.rank)
    ids_in_order = [c.note_id for c in ordered]
    returned_count = len(ordered)
    failures: list[Failure] = []

    k = gold.top_k
    relevant = gold.relevant_ids

    # -- recall family (only defined when the query has grade>=2 gold) --------
    recall_at_1 = recall_at_5 = recall_at_k = hit_at_k = mrr = ndcg = None
    # recall@k normalizes by |relevant|, so a query whose gold phrase matches many
    # notes caps below 1.0 even on a perfect search — the absolute number is
    # deflated by gold-set size. The cross-arm DELTA (all arms share gold) is
    # unaffected, which is the decision input.
    if gold.has_relevant:
        top1 = set(ids_in_order[:1])
        top5 = set(ids_in_order[:5])
        topk = set(ids_in_order[:k])
        recall_at_1 = len(relevant & top1) / len(relevant)
        recall_at_5 = len(relevant & top5) / len(relevant)
        recall_at_k = len(relevant & topk) / len(relevant)
        hit_at_k = 1.0 if (relevant & topk) else 0.0
        mrr = 0.0
        for i, nid in enumerate(ids_in_order):
            if nid in relevant:
                mrr = 1.0 / (i + 1)
                break
        ndcg = ndcg_at(ordered, gold, k=10)
        for nid in relevant:
            if nid not in topk:
                failures.append(
                    Failure(
                        kind=FailureKind.RECALL_MISS,
                        note_id=nid,
                        detail=f"grade>=2 gold {nid} absent from top-{k}",
                        expected_signal=gold.expected_signal,
                    )
                )

    # -- precision family -----------------------------------------------------
    grade0_returned = [c for c in ordered if gold.grade_of(c.note_id) == 0]
    fpr = (len(grade0_returned) / returned_count) if returned_count else 0.0
    # P@k: relevant-class (grade>=2) fraction of what was actually returned.
    if returned_count:
        rel_returned = sum(1 for c in ordered[:k] if gold.grade_of(c.note_id) >= 2)
        precision_at_k = rel_returned / min(returned_count, k)
    else:
        precision_at_k = None

    for c in grade0_returned:
        failures.append(
            Failure(
                kind=FailureKind.PRECISION_FP,
                note_id=c.note_id,
                detail=f"grade-0 card {c.note_id} returned at rank {c.rank}",
                surfaced_signals=c.signals,
            )
        )

    # -- over-return (null-gold query must surface no relevant-class hit) ------
    # A genuine ∅-gold query: ANY return is junk the fusion floors should have
    # suppressed. (A query with only grade-0 gold is handled by FPR above.)
    over_returned = not gold.has_relevant and not gold.grades and bool(returned_count)
    if over_returned:
        failures.append(
            Failure(
                kind=FailureKind.OVER_RETURN,
                note_id=None,
                detail=f"null-gold query returned {returned_count} card(s)",
            )
        )

    # -- exact-tier purity (no grade-0 floated by the exact override) ---------
    exact_tier_pure: bool | None = None
    exact_tier = [c for c in ordered if "exact" in c.signals or c.has_substring]
    if exact_tier:
        exact_tier_pure = all(gold.grade_of(c.note_id) != 0 for c in exact_tier)

    # -- degradation announcement --------------------------------------------
    if expects_degradation and not response_announced_degradation:
        failures.append(
            Failure(
                kind=FailureKind.DEGRADE_SILENT,
                note_id=None,
                detail="response degraded without announcing it (message/completeness/score)",
            )
        )

    return QueryReport(
        query=query,
        adversarial_class=adversarial_class,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_k=recall_at_k,
        hit_at_k=hit_at_k,
        mrr=mrr,
        ndcg_at_10=ndcg,
        precision_at_k=precision_at_k,
        false_positive_rate=fpr,
        over_returned=over_returned,
        exact_tier_pure=exact_tier_pure,
        returned_count=returned_count,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class SuiteReport:
    """Aggregate over a run's per-query reports — the artifact the runner
    renders and the test asserts against."""

    queries: tuple[QueryReport, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(q.passed for q in self.queries)

    def _mean(self, attr: str, *, by_class: str | None = None) -> float | None:
        vals = [
            getattr(q, attr)
            for q in self.queries
            if getattr(q, attr) is not None
            and (by_class is None or q.adversarial_class == by_class)
        ]
        return (sum(vals) / len(vals)) if vals else None

    def mean_recall_at_k(self, by_class: str | None = None) -> float | None:
        return self._mean("recall_at_k", by_class=by_class)

    def mean_recall_at_1(self, by_class: str | None = None) -> float | None:
        return self._mean("recall_at_1", by_class=by_class)

    def mean_recall_at_5(self, by_class: str | None = None) -> float | None:
        return self._mean("recall_at_5", by_class=by_class)

    def mean_mrr(self, by_class: str | None = None) -> float | None:
        return self._mean("mrr", by_class=by_class)

    def mean_ndcg(self, by_class: str | None = None) -> float | None:
        return self._mean("ndcg_at_10", by_class=by_class)

    def mean_precision_at_k(self, by_class: str | None = None) -> float | None:
        return self._mean("precision_at_k", by_class=by_class)

    def failures_by_kind(self) -> dict[FailureKind, list[Failure]]:
        out: dict[FailureKind, list[Failure]] = {k: [] for k in FailureKind}
        for q in self.queries:
            for f in q.failures:
                out[f.kind].append(f)
        return out

    def classes(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for q in self.queries:
            seen.setdefault(q.adversarial_class, None)
        return tuple(seen)
