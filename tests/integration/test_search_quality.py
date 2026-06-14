"""The manual adversarial search-quality suite (#559, PR2) — real models.

MANUAL / LOCAL-ONLY, NEVER RUN IN CI. Three independent fences keep it off the
CI lanes (mirroring the embedding tests):

  1. **Bazel**: a ``pytest_test(tags=["manual"])`` target that is never named on
     a CI command line, AND excluded from the ``:integration`` glob — so
     ``bazel test //...`` skips it twice over (the load-bearing fence).
  2. **Marker + env**: ``pytest.mark.search_quality`` + ``requires_search_quality``
     skips every test unless ``SHRIKE_SEARCH_QUALITY=1`` (belt-and-suspenders —
     even if selection leaks, the tests SKIP, never FAIL).
  3. **Coverage**: ``scripts/coverage.sh`` runs ``-m "not embedding and not
     search_quality"``.

Run it explicitly::

    SHRIKE_SEARCH_QUALITY=1 pytest tests/integration/test_search_quality.py -m search_quality

It downloads a ~30-image Wikimedia Commons corpus (pinned URLs in
``eval/search_quality/resolved_urls.json``; bytes cached in the gitignored
``eval/search_quality/cache/``; attribution in ``ASSETS.md``) and runs real
CLIP + MiniLM. The recall/precision/cross-lingual/calibrated-gate *classes* land
in PR2b; PR2a proves the corpus loads, the Commons resolve→cache works, and the
real 2-space loop (a dedicated text space + a separate CLIP space, #229)
retrieves cross-modally through the REAL ``search_notes`` action.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.integration.conftest import requires_clip, requires_search_quality

pytestmark = [pytest.mark.integration, pytest.mark.search_quality, requires_search_quality]

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval" / "search_quality"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"
CACHE = EVAL_DIR / "cache"

# At least this many image-bearing notes so the activation gate calibrates on
# real CLIP (the kernel's CALIB_MIN). The corpus is sized to clear it.
MIN_IMAGE_NOTES = 30


def _model_cache_base() -> Path:
    base = os.environ.get("SHRIKE_TEST_MODEL_DIR")
    return Path(base) if base else (Path.home() / ".cache" / "shrike-test-models")


class TestCorpusManifest:
    """The corpus loads, is graded, and carries enough images to calibrate the
    gate — no models or network needed (pure manifest checks)."""

    def test_manifest_loads_and_is_graded(self) -> None:
        from tests.search_quality.manifest import load_manifest

        manifest = load_manifest(MANIFEST)
        # Open-world: per-query sparse gold (1 answer + a few hard-negatives among
        # 46 cards), so an un-graded return is a non-answer, not a false positive.
        assert manifest.closed_world is False
        assert manifest.cards, "the corpus has cards"
        assert manifest.queries, "the corpus has queries"
        # every query's gold references a real card id
        ids = {c.id for c in manifest.cards}
        for q in manifest.queries:
            for nid in q.gold.grades:
                assert nid in ids, f"query {q.q!r} grades a non-existent card {nid}"

    def test_at_least_thirty_image_bearing_notes(self) -> None:
        from tests.search_quality.manifest import load_manifest

        manifest = load_manifest(MANIFEST)
        image_cards = [c for c in manifest.cards if c.media]
        assert len(image_cards) >= MIN_IMAGE_NOTES, (
            f"need >= {MIN_IMAGE_NOTES} image-bearing notes to calibrate the gate; "
            f"have {len(image_cards)}"
        )

    def test_every_commons_image_is_pinned(self) -> None:
        # The committed pins make a replay reproducible without re-resolving.
        import json

        from tests.search_quality.manifest import load_manifest

        manifest = load_manifest(MANIFEST)
        pins = json.loads(RESOLVED.read_text())
        for card in manifest.cards:
            for media in card.media:
                if media.source == "commons":
                    assert media.handle in pins, (
                        f"image handle {media.handle!r} is not pinned in resolved_urls.json "
                        "(run scripts/eval_search_quality_corpus.py)"
                    )
                    assert pins[media.handle].startswith("http"), pins[media.handle]

    def test_adversarial_classes_are_present(self) -> None:
        # The corpus exercises the killer constructions, not just plain recall.
        from tests.search_quality.manifest import load_manifest

        manifest = load_manifest(MANIFEST)
        classes = {q.adversarial_class for q in manifest.queries}
        for required in (
            "modality_gap",
            "modality_gap_vs_token_share",  # portrait must beat shared-token text
            "gate_no_inject_portrait",  # text card must win, portrait must not inject
            "cross_lingual_exact",  # "manzana" via literal substring
            "over_return",  # null-gold precision probe
            "semantic_text",
        ):
            assert required in classes, f"corpus is missing the {required!r} adversarial class"


class TestCommonsResolution:
    """The Commons resolve→pin→cache machinery downloads real bytes (network);
    bytes land in the gitignored cache, never the repo."""

    def test_pinned_image_downloads_and_decodes(self) -> None:
        import io

        from PIL import Image

        from tests.search_quality.commons import CommonsCache

        cache = CommonsCache(RESOLVED, CACHE)
        pins = cache.load_pins()
        assert "heart" in pins, "the corpus is resolved (run the corpus script)"
        data = cache.fetch_bytes(pins["heart"])
        assert len(data) > 1000, "downloaded real image bytes"
        img = Image.open(io.BytesIO(data))
        assert img.size[0] > 0 and img.size[1] > 0, "the bytes decode to an image"


@requires_clip
class TestRealTwoSpaceSmoke:
    """The first real-model proof of the multi-space loop (#229/#232/#234): a
    dedicated text space (MiniLM) + a SEPARATE CLIP space, attached in-process,
    retrieve an image-bearing note CROSS-MODALLY through the REAL ``search_notes``
    action — the highest-value PR2 assertion, smoke-sized here (PR2b scales it to
    the full graded corpus). The image-only-meaning cards have answer-blind field
    text, so the ONLY path to them is the CLIP image vector."""

    @pytest.mark.asyncio
    async def test_text_query_retrieves_image_note_via_clip_space(self, tmp_path) -> None:
        from shrike.embedding_clip import ClipBackend
        from shrike.embedding_onnx import OnnxBackend
        from tests.integration.model_cache import (
            cached_clip_model_dir,
            cached_onnx_model_dir,
        )
        from tests.search_quality.commons import CommonsCache
        from tests.search_quality.inprocess import build_harness_real

        base = _model_cache_base()
        text = OnnxBackend(model=str(cached_onnx_model_dir(base)))
        clip = ClipBackend(model=str(cached_clip_model_dir(base)))
        text.start()
        clip.start()

        # A few distinctive image-only-meaning cards from the real corpus.
        cache = CommonsCache(RESOLVED, CACHE)
        pins = cache.load_pins()
        handles = ["heart", "cat", "apple", "guitar"]
        media = {f"{h}.png": cache.fetch_bytes(pins[h]) for h in handles}

        ip = await build_harness_real(tmp_path, text_backend=text, clip_backend=clip, media=media)
        try:
            assert ip.harness.kernel.embed_space_count() == 2, "two embedding spaces attached"
            notes = await ip.harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "AdversarialEval::Image",
                        "fields": {"Front": f"Review card {i}", "Back": f'<img src="{h}.png">'},
                    }
                    for i, h in enumerate(handles)
                ]
            )
            id_by_handle = dict(zip(handles, (n["id"] for n in notes), strict=True))
            await ip.finalize()
            assert ip.index_status()["size"] >= len(handles), "the image vectors indexed"

            # A text query whose ANSWER is an image surfaces the right card via
            # the CLIP space — the modality-gap payoff. The signal carries the
            # CLIP space key (#234: `image#<space>`), proving it came from the
            # secondary space's cross-space contribution, not the text space.
            matches = await ip.matches("a photograph of a cat", top_k=5, threshold=0.2)
            cat = next((m for m in matches if m["id"] == id_by_handle["cat"]), None)
            assert cat is not None, "the cat image card was retrieved by a text query"
            signals = {p["signal"] for p in cat["provenance"]}
            assert any(s.startswith("image") for s in signals), (
                f"retrieved via an image signal (the CLIP space), got {signals}"
            )
        finally:
            await ip.harness.close()
            text.stop()
            clip.stop()


# ── PR2b: the full real-model recall/precision suite over the graded corpus ──
#
# These build the WHOLE ≥30-image corpus once (a module-scoped fixture — the
# build + every query is one ~15-30s run on real CLIP + MiniLM) and assert the
# per-class thresholds. Thresholds are MEMBERSHIP/floor-based, never exact-slot,
# so they survive the int8 cosine wobble (#559 methodology). The numbers behind
# them are reproduced by `scripts/eval_search_quality.py` into RESULTS.md.


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """Build the real 2-space corpus + run every query ONCE for the module.

    Module-scoped so the ~15-30s real-model run is paid once and every
    threshold/characterization test reads the shared :class:`RunResult`."""
    import asyncio

    from tests.search_quality.runner import run_search_quality

    tmp = tmp_path_factory.mktemp("real_run")
    return asyncio.run(run_search_quality(tmp))


@requires_clip
class TestRealRecall:
    """Real cross-modal + semantic RECALL over the graded corpus. Floors are
    deliberately below the observed numbers (R@k=1.0, MRR=0.84) to absorb
    model/float wobble while still catching a real fusion regression."""

    def test_two_spaces_and_enough_images(self, real_run) -> None:
        assert real_run.space_count == 2, "a dedicated text space + a separate CLIP space"
        # >= 30 image-bearing notes is what lets the corpus exercise the gate on
        # real CLIP — count the manifest's image cards directly (the index `size`
        # conflates text + image vectors, so it can't prove the image half).
        image_cards = [c for c in real_run.manifest.cards if c.media]
        assert len(image_cards) >= 30, f"only {len(image_cards)} image-bearing notes"
        # The 2-space index holds both spaces' vectors → strictly more than the
        # text-only card count (a coarse "the image space actually indexed" check;
        # the gate-firing tests are the real backstop).
        assert real_run.index_size > len(real_run.manifest.cards), (
            f"index size {real_run.index_size} doesn't exceed the {len(real_run.manifest.cards)} "
            "cards — the image space may not have indexed"
        )

    def test_overall_recall_at_k_floor(self, real_run) -> None:
        rk = real_run.suite.mean_recall_at_k()
        assert rk is not None and rk >= 0.9, f"overall R@k {rk} below the 0.9 floor"

    def test_modality_gap_recall(self, real_run) -> None:
        # The core payoff: a text query retrieves an answer-blind image card.
        rk = real_run.suite.mean_recall_at_k(by_class="modality_gap")
        mrr = real_run.suite.mean_mrr(by_class="modality_gap")
        assert rk is not None and rk >= 0.9, f"modality_gap R@k {rk} < 0.9"
        assert mrr is not None and mrr >= 0.5, f"modality_gap MRR {mrr} < 0.5"

    def test_semantic_text_recall(self, real_run) -> None:
        # Plain text retrieval must not regress under the multi-space fusion.
        r1 = real_run.suite.mean_recall_at_1(by_class="semantic_text")
        assert r1 is not None and r1 >= 0.8, f"semantic_text R@1 {r1} < 0.8"

    def test_portrait_beats_shared_token_text_at_k(self, real_run) -> None:
        # "a portrait of Napoleon" must surface the IMAGE despite the Waterloo
        # text cards sharing the token "Napoleon" — at k (the modality gap can
        # cost rank-1 vs a lexically-strong text card, so floor on R@k not R@1).
        rk = real_run.suite.mean_recall_at_k(by_class="modality_gap_vs_token_share")
        assert rk is not None and rk >= 0.9, (
            f"portrait-vs-token-share R@k {rk} < 0.9 — the image isn't surfacing"
        )


@requires_clip
class TestRealPrecision:
    """Real PRECISION: the ∅-gold over-return probe and the planted grade-0
    hard-negatives. Open-world grading (the corpus is per-query sparse gold), so
    FPR counts only the explicit hard-negatives, never incidental non-answers."""

    def test_over_return_query_surfaces_only_weak_low_confidence_hits(self, real_run) -> None:
        # CHARACTERIZATION of the relative gate's known precision trade-off
        # (#234): a query that answers NOTHING ("purple elephant…") has a
        # near-zero text-space best, so the CLIP space's WEAK image cosines still
        # clear the relative gate (clip_best >= text_best ≈ 0) and a handful of
        # low-confidence image cards leak in. This is real and documented, not a
        # failure — but the leaks must be (a) only the weak image signal (never a
        # strong exact/text hit), and (b) all LOW confidence. So an ∅-gold query
        # never surfaces a *confident* false positive.
        q = next(qq for qq in real_run.manifest.queries if qq.adversarial_class == "over_return")
        returned = real_run.returns.get(q.q, [])
        OVER_RETURN_SCORE_CEILING = 0.45  # well below the threshold a real answer clears
        for m in returned:
            signals = {p["signal"] for p in m["provenance"]}
            assert not (signals & {"exact", "fuzzy"}), (
                f"the ∅-gold query produced a LEXICAL hit {m['id']} ({signals}) — junk fused"
            )
            score = m.get("score")
            assert score is None or score <= OVER_RETURN_SCORE_CEILING, (
                f"the ∅-gold query produced a CONFIDENT hit {m['id']} (score {score})"
            )

    def test_planted_hard_negative_fpr_is_bounded(self, real_run) -> None:
        # The portrait hard-negatives (grade-0) may leak in at a LOW rank via the
        # text space (the blind text + the entity-named filename), but the rate
        # must stay bounded — a characterization floor, not zero (see RESULTS.md
        # and the gate-no-inject test, which proves they never enter via CLIP).
        gate_queries = [
            q for q in real_run.suite.queries if q.adversarial_class == "gate_no_inject_portrait"
        ]
        assert gate_queries, "the gate-no-inject queries ran"
        for q in gate_queries:
            assert q.false_positive_rate is not None
            assert q.false_positive_rate <= 0.3, (
                f"hard-negative FPR {q.false_positive_rate} too high for {q.query!r}"
            )


@requires_clip
class TestRealActivationGate:
    """The relative cross-space activation gate (#234) on REAL CLIP cosines: it
    FIRES (the CLIP `image#clip` signal contributes) for clearly cross-modal
    queries where the image is the only path, and STAYS CLOSED (no portrait
    injected via CLIP) when the text space answers the query. This is the whole
    reason the corpus sources ≥30 real images."""

    # Queries where the answer is purely visual AND the filename doesn't leak the
    # subject — so the relative gate must open (CLIP beats the weak text signal).
    GATE_FIRES = [
        "a photograph of a domestic cat",
        "a photo of a green apple",
        "a sunflower with yellow petals",
        "the Mona Lisa painting by Leonardo da Vinci",
        "a photo of the Eiffel Tower in Paris",
    ]

    def test_gate_fires_for_strongly_cross_modal_queries(self, real_run) -> None:
        from tests.search_quality.runner import clip_fired

        # A MEMBERSHIP floor, not an all-or-nothing conjunction (matching the
        # suite's threshold philosophy): the gate must open for the large
        # majority of these strongly-visual queries — one non-fire from the int8
        # cosine wobble shouldn't fail the suite, but a broad gate regression
        # (most stop firing) must. Observed: all 5 fire.
        fired = [
            q for q in self.GATE_FIRES if any(clip_fired(m) for m in real_run.returns.get(q, []))
        ]
        assert len(fired) >= len(self.GATE_FIRES) - 1, (
            f"the relative gate opened for only {len(fired)}/{len(self.GATE_FIRES)} "
            f"strongly-visual queries — fired: {fired}"
        )

    def test_gate_does_not_inject_portrait_via_clip(self, real_run) -> None:
        from tests.search_quality.runner import clip_fired

        # For a query the TEXT answers (a Curie/Napoleon fact), the portrait
        # (grade-0 hard-negative) must NEVER surface via the CLIP image signal —
        # the relative gate stays closed for the vision space (the text space is
        # more confident). It may appear at a low rank via TEXT (a lexical leak),
        # but never `image#clip`. (Gold lives on the manifest query, not the
        # graded report — so iterate the manifest.)
        for q in real_run.manifest.queries:
            if q.adversarial_class != "gate_no_inject_portrait":
                continue
            hard_negatives = {nid for nid, g in q.gold.grades.items() if g == 0}
            for m in real_run.returns.get(q.q, []):
                if m["id"] in hard_negatives:
                    assert not clip_fired(m), (
                        f"portrait {m['id']} injected via CLIP for {q.q!r} — gate leaked"
                    )

    def test_gate_no_inject_text_card_wins_rank_one(self, real_run) -> None:
        # The fact query's rank-1 must be the correct TEXT card, not the portrait.
        for q in real_run.manifest.queries:
            if q.adversarial_class != "gate_no_inject_portrait":
                continue
            relevant = q.gold.relevant_ids
            top = real_run.returns.get(q.q, [])
            assert top, f"{q.q!r} returned nothing"
            assert top[0]["id"] in relevant, (
                f"rank-1 for {q.q!r} is {top[0]['id']}, not a relevant text card {relevant}"
            )


@requires_clip
class TestCrossLingualCharacterization:
    """Cross-lingual behaviour — characterized, not over-pinned (#559): the
    exact-match and image paths work regardless of model language, but SEMANTIC
    cross-lingual recall depends on the English-centric models, so it's
    documented (RESULTS.md), not hard-floored."""

    def test_exact_substring_works_cross_lingually(self, real_run) -> None:
        # "manzana" is a literal substring of the apple card's back text — the
        # exact signal recovers it regardless of the model's language.
        report = next(
            q for q in real_run.suite.queries if q.adversarial_class == "cross_lingual_exact"
        )
        assert report.recall_at_1 == 1.0, "the literal cross-lingual term must hit at rank 1"

    def test_cross_lingual_semantic_is_characterized(self, real_run) -> None:
        # A non-English semantic query ("une pomme rouge ou verte") — we DON'T
        # hard-floor recall (the model is English-centric), only assert the run
        # produced a defined metric so RESULTS.md can report where it reaches.
        report = next(
            q
            for q in real_run.suite.queries
            if q.adversarial_class == "cross_lingual_semantic_characterization"
        )
        # The card may surface via the image (the apple photo) or weak text — the
        # characterization records R@k; we only require the metric is computed.
        assert report.recall_at_k is not None, "cross-lingual semantic recall is measured"
