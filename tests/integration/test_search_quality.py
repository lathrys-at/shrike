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
        assert manifest.closed_world is True
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
        from tests.integration.model_cache import (
            cached_clip_model_dir,
            cached_onnx_model_dir,
        )
        from tests.search_quality.commons import CommonsCache
        from tests.search_quality.inprocess import build_harness_real
        from shrike.embedding_clip import ClipBackend
        from shrike.embedding_onnx import OnnxBackend

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
