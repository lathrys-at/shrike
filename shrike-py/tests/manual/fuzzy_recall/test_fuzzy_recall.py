"""Fuzzy-recall eval entrypoint + generator unit tests.

The full eval (build the corpus, run every arm) is gated behind
``SHRIKE_FUZZY_RECALL=1`` and is a manual lane — it is heavy and its absolute
numbers are a tuning signal, not a pass/fail. The generator unit tests run
unconditionally (they are pure and fast, no corpus, offline) so the typo model and
gold-by-construction logic stay pinned.
"""

from __future__ import annotations

import os

import pytest

from tests.manual.fuzzy_recall import misspellings as misspellings_mod
from tests.manual.fuzzy_recall.typo_queries import (
    EditKind,
    generate_queries,
    length_bucket,
)

_GATE = "SHRIKE_FUZZY_RECALL"


def _gated() -> bool:
    return os.environ.get(_GATE) == "1"


# ── generator unit tests (pure, fast, always run) ───────────────────────────────


def test_length_buckets_straddle_the_floor_and_ceiling() -> None:
    # The buckets must split at the cap floor (6) and ceiling (12) so a global mean
    # never hides the curve's effect on long queries.
    assert length_bucket(2) == "n<=6"
    assert length_bucket(6) == "n<=6"
    assert length_bucket(7) == "n7-11"
    assert length_bucket(11) == "n7-11"
    assert length_bucket(12) == "n12-17"
    assert length_bucket(17) == "n12-17"
    assert length_bucket(18) == "n18+"


def test_generation_is_deterministic_and_gold_is_constructed() -> None:
    # The same (texts, seed) yields a byte-identical query set, and every query's
    # gold contains the source note (gold = all notes containing the clean phrase).
    texts = {
        100: "mitochondria is the powerhouse of the cell membrane",
        101: "cellular respiration releases energy from glucose molecules",
        102: "the membrane potential drives the action signal forward",
    }

    # A gold resolver over the same texts: every note whose text contains the phrase.
    def resolver(phrase: str) -> list[int]:
        p = phrase.lower()
        return [nid for nid, t in texts.items() if p in t.lower()]

    a = generate_queries(texts, resolver, seed=7, sample_size=3, misspellings={})
    b = generate_queries(texts, resolver, seed=7, sample_size=3, misspellings={})
    assert a == b, "generation must be byte-reproducible for a fixed seed"
    assert a, "the sample should produce queries"
    for q in a:
        assert q.source_note_id in q.gold_ids, "the source note is always in gold"
        assert q.query != q.clean_phrase, "every query carries an effective typo"
        assert q.edits, "every query records its applied edits"
        assert q.typo_count == len(q.edits)


def test_typo_count_strata_are_exercised() -> None:
    # Across a reasonable sample the generator must produce queries at >1 typo count
    # (the strata cycle 1/2/3), so the eval can show degradation behaviour.
    texts = {
        i: f"membrane potential gradient signal {i} releases stored energy molecules cleanly"
        for i in range(200, 230)
    }

    def resolver(phrase: str) -> list[int]:
        p = phrase.lower()
        return [nid for nid, t in texts.items() if p in t.lower()]

    qs = generate_queries(texts, resolver, seed=3, sample_size=30, misspellings={})
    counts = {q.typo_count for q in qs}
    assert counts & {2, 3}, f"expected multi-typo queries, got counts {counts}"


def test_real_misspelling_is_injected_when_the_word_is_present() -> None:
    # A phrase containing a known correct word ("receive") must be perturbable into
    # its real misspelling ("recieve"), tagged REAL_MISSPELLING.
    texts = {500: "students receive the separate environment government calendar handout"}
    misspell = misspellings_mod.load_misspellings()  # embedded fallback offline

    def resolver(phrase: str) -> list[int]:
        return [500] if phrase.lower() in texts[500].lower() else []

    # Force single-word phrases keyed on a known correction; iterate seeds until one
    # lands on a correctable word, then assert the misspelling fired.
    found = False
    for seed in range(50):
        qs = generate_queries(texts, resolver, seed=seed, sample_size=1, misspellings=misspell)
        for q in qs:
            if EditKind.REAL_MISSPELLING in q.edits:
                # The perturbed query holds a known misspelling of a corpus word.
                found = True
                break
        if found:
            break
    assert found, "a phrase with a known word should yield a real-misspelling query"


def test_misspellings_fallback_parses() -> None:
    # The embedded fallback is non-empty and well-formed (offline must still inject
    # some real misspellings).
    pairs = misspellings_mod.load_misspellings()
    assert pairs
    assert pairs.get("recieve") == "receive"
    assert all(w.isalpha() and r.isalpha() for w, r in pairs.items())


# ── the full eval (gated, manual) ───────────────────────────────────────────────


@pytest.mark.skipif(not _gated(), reason=f"set {_GATE}=1 to run the fuzzy-recall eval")
def test_fuzzy_recall_eval_runs_and_writes_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Run the eval at a small size (so the manual lane proves the wiring) and render
    the results artifact. Asserts only that the run produces queries and a result per
    arm — the recall NUMBERS are a tuning signal, not a pass/fail gate (the cap
    decision is a separate, eval-gated follow-up). The corpus cache and the artifact
    go under ``tmp_path`` so the lane runs hermetically (e.g. in a bazel sandbox)."""
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        DEFAULT_ARMS,
        render_results,
        run_eval,
    )

    # Small + offline by default so the gated test is fast and hermetic; a real run
    # uses the CLI with --notes 5000 and the downloaded assets.
    notes = int(os.environ.get("SHRIKE_FUZZY_RECALL_NOTES", "500"))
    sample = int(os.environ.get("SHRIKE_FUZZY_RECALL_SAMPLE", "150"))
    offline = os.environ.get("SHRIKE_FUZZY_RECALL_OFFLINE", "1") == "1"

    run = run_eval(
        notes=notes, seed=0, sample_size=sample, offline=offline, cache_root=tmp_path / "corpora"
    )
    assert run.n_queries > 0
    assert len(run.arms) == len(DEFAULT_ARMS)
    for arm in run.arms:
        assert 0.0 <= arm.recall_at_k <= 1.0
        assert 0.0 <= arm.mrr <= 1.0

    out = tmp_path / "RESULTS.md"
    out.write_text(render_results(run) + "\n")
    assert out.is_file()
