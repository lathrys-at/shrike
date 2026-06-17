"""The real-model search-quality run engine (#559 PR2b).

ONE engine, two consumers (the plan's "single source of truth"):

  - ``tests/integration/test_search_quality.py`` asserts the resulting
    :class:`~tests.search_quality.metrics.SuiteReport` against per-class
    thresholds (the machine pass/fail);
  - ``scripts/eval_search_quality.py`` renders the same report to
    ``eval/search_quality/RESULTS.md`` (the dogfooding artifact).

It builds a **real** 2-space collection (a dedicated text space + a separate
CLIP image space, #229/#232/#234) from the manifest — downloading the pinned
Commons images through the licensing-clean ``CommonsCache`` — then drives every
manifest query through the REAL ``search_notes`` MCP action and feeds the
returned-with-provenance + the graded gold to the pure metric engine. No planted
vectors: real embeddings, real RRF fusion, the real relative cross-space gate.

The engine is offline after a first run: the images cache in the gitignored
``eval/search_quality/cache/`` and the models in the shared test-model cache.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.search_quality.commons import CommonsCache
from tests.search_quality.inprocess import InProcessSearch, build_harness_real, to_returned_cards
from tests.search_quality.manifest import Manifest, load_manifest
from tests.search_quality.metrics import QueryReport, SuiteReport, evaluate_query

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval" / "search_quality"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"
CACHE = EVAL_DIR / "cache"

# The cross-space CLIP signal name the kernel emits (#234: image#<space-key>).
CLIP_SIGNAL_PREFIX = "image#"
CLIP_SPACE_KEY = "clip"


def model_cache_base() -> Path:
    base = os.environ.get("SHRIKE_TEST_MODEL_DIR")
    return Path(base) if base else (Path.home() / ".cache" / "shrike-test-models")


_IMG_TOKEN = re.compile(r"\$IMG:([A-Za-z0-9_-]+)")


@dataclass
class RunResult:
    """One real-model run: the graded report + the per-query raw returns (for
    the gate/cross-lingual characterizations that read provenance directly)."""

    suite: SuiteReport
    manifest: Manifest
    # manifest-id ↔ live-anki-id, so a caller can map gold to returns.
    id_map: dict[int, int] = field(default_factory=dict)
    # query string → its returned matches (manifest-id-keyed, provenance intact).
    returns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    activation: dict[str, Any] | None = None
    index_size: int = 0
    space_count: int = 0


def _download_corpus_media(manifest: Manifest, cache: CommonsCache) -> dict[str, bytes]:
    """Download every pinned commons image into ``{handle}.png`` → bytes.

    A pin that fails to fetch (a deleted/renamed Commons file) is SKIPPED with a
    note, not fatal — the run proceeds with fewer images (and the gate-calibration
    floor test guards the >=30 count separately)."""
    pins = cache.load_pins()
    media: dict[str, bytes] = {}
    missing: list[str] = []
    for card in manifest.cards:
        for m in card.media:
            if m.source == "commons" and m.handle in pins and f"{m.handle}.png" not in media:
                try:
                    media[f"{m.handle}.png"] = cache.fetch_bytes(pins[m.handle])
                except Exception as e:  # noqa: BLE001 — a missing image is skipped, not fatal
                    missing.append(f"{m.handle} ({e})")
    if missing:
        print(f"  ! {len(missing)} image(s) unfetchable, skipped: {', '.join(missing[:5])}")
    return media


def _substitute_media(fields: dict[str, str]) -> dict[str, str]:
    """Replace ``$IMG:handle`` with the stored ``handle.png`` filename."""
    return {k: _IMG_TOKEN.sub(lambda mo: f"{mo.group(1)}.png", v) for k, v in fields.items()}


async def build_real_collection(
    tmp_path: Path,
    manifest: Manifest,
    *,
    cache: CommonsCache | None = None,
) -> tuple[InProcessSearch, dict[int, int], tuple[Any, Any]]:
    """Build a real 2-space (text + CLIP) collection from the manifest.

    Loads the cached MiniLM text backend + the cached CLIP backend, downloads
    the corpus images, upserts every card (``$IMG:handle`` → the stored
    filename), and finalizes (full rebuild of both spaces + their calibration).
    Returns the driver, the ``{manifest_id: anki_id}`` map, and the
    ``(text, clip)`` backends (the caller stops them at teardown)."""
    from shrike.embedding_clip import ClipBackend
    from shrike.embedding_onnx import OnnxBackend
    from tests.integration.model_cache import cached_clip_model_dir, cached_onnx_model_dir

    cache = cache or CommonsCache(RESOLVED, CACHE)
    media = _download_corpus_media(manifest, cache)

    base = model_cache_base()
    text = OnnxBackend(model=str(cached_onnx_model_dir(base)))
    clip = ClipBackend(model=str(cached_clip_model_dir(base)))
    text.start()
    clip.start()

    ip: InProcessSearch | None = None
    try:
        ip = await build_harness_real(
            tmp_path,
            text_backend=text,
            clip_backend=clip,
            media=media,
            clip_space_key=CLIP_SPACE_KEY,
        )
        id_map: dict[int, int] = {}
        for card in manifest.cards:
            fields = _substitute_media(dict(card.fields))
            result = await ip.harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": card.note_type,
                        "deck": card.deck,
                        "fields": fields,
                        "tags": list(card.tags),
                    }
                ]
            )
            id_map[card.id] = int(result[0]["id"])
        await ip.finalize()
    except BaseException:
        # A build failure must not leak the started ONNX/CLIP sessions or the
        # open Anki collection (it holds a file lock — a retry would then hit a
        # locked collection). Tear everything down before re-raising.
        if ip is not None:
            await ip.harness.close()
        text.stop()
        clip.stop()
        raise
    return ip, id_map, (text, clip)


async def run_search_quality(
    tmp_path: Path,
    *,
    manifest: Manifest | None = None,
    cache: CommonsCache | None = None,
) -> RunResult:
    """Build the real collection, run every query through the REAL
    ``search_notes`` action, and grade with the pure metric engine.

    Returns a :class:`RunResult` carrying the :class:`SuiteReport` (the machine
    pass/fail) AND the per-query raw returns (provenance intact) for the
    gate/cross-lingual characterizations. Closes the harness + stops the
    backends before returning."""
    manifest = manifest or load_manifest(MANIFEST)
    ip, id_map, backends = await build_real_collection(tmp_path, manifest, cache=cache)
    inv = {v: k for k, v in id_map.items()}

    reports: list[QueryReport] = []
    returns: dict[str, list[dict[str, Any]]] = {}
    try:
        status = ip.index_status()
        activation = status.get("activation")
        index_size = int(status.get("size", 0))
        space_count = ip.harness.kernel.embed_space_count()

        for q in manifest.queries:
            thr = q.threshold if q.threshold is not None else 0.5
            matches = await ip.matches(q.q, limit=q.top_k, threshold=thr)
            # Remap live anki ids back to manifest ids (gold is manifest-keyed).
            for m in matches:
                m["id"] = inv.get(m["id"], m["id"])
            returns[q.q] = matches
            reports.append(
                evaluate_query(q.q, q.adversarial_class, to_returned_cards(matches), q.gold)
            )
    finally:
        await ip.harness.close()
        for b in backends:
            b.stop()

    return RunResult(
        suite=SuiteReport(queries=tuple(reports)),
        manifest=manifest,
        id_map=id_map,
        returns=returns,
        activation=activation,
        index_size=index_size,
        space_count=space_count,
    )


def clip_fired(match: dict[str, Any]) -> bool:
    """True when a returned match carries a cross-space CLIP image signal
    (``image#clip``) — i.e. the relative activation gate opened for it."""
    return any(p["signal"].startswith(CLIP_SIGNAL_PREFIX) for p in match.get("provenance", []))
