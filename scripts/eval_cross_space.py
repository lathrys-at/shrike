#!/usr/bin/env python3
"""Cross-space retrieval eval — the mechanism gate for the multi-space epic (#231 / #229).

NOT production code. Distinct from `scripts/eval_multimodal.py` (the #193 single-shared-space eval).
This validates the *cross-space* MECHANISM: on a deployment that has NO single omni embedder — only
a dedicated TEXT space and a separate joint text<->image CLIP space (the canonical mobile config:
platform text-embedding API + a separate platform CLIP API) — is RRF-fusing a text query across the
two spaces SOUND? Multi-space is a *requirement* for those no-omni targets, so this is not a
build-vs-don't gate; it asks whether the fusion mechanism preserves text quality and delivers image
recall, and what params (RRF k, per-space rank caps, a cross-space activation gate) make it so.

Three conditions, one shared corpus (all notes — text-bearing and image-bearing — in ONE collection,
so every query competes against the full distractor set, the realistic shape):

  (a) text-only baseline   : a dedicated text embedder (ONNX MiniLM), ONE index of note text. THE BAR
                             (c) must not regress: text<->text quality of the dedicated space alone.
  (b) single CLIP pooled   : CLIP embeds note text AND images into ONE space, pooled into one cosine
                             ranking (the unified-space REFERENCE CEILING, where one happens to exist
                             — NOT the bar; the no-omni targets don't have it).
  (c) cross-space FUSED     : the dedicated MiniLM text space + the joint CLIP space (its text-tower
                             body vectors AND image vectors), each ranked in its OWN space, RRF-fused.
                             Why keep the separate text space: CLIP's text tower is weaker than a
                             dedicated text embedder, so we lean on the dedicated space for text and
                             only ADD the CLIP/image hits — without dragging text down. (+gate/+relgate
                             apply the cross-space activation gate, the #201b analogue.)

Each per-space ranker is per-vector max-over-items (a note's rank = its best vector for the query),
exactly as the kernel's `search_by_modality` aggregates today. Scope: TEXT + IMAGE only (audio/video
reach the text space via OCR/ASR-derived text, no joint audio space on these targets).

Models (the project's OWN native backends — NO torch / sentence-transformers; exactly what Shrike
ships): ONNX MiniLM (`OnnxBackend`, 384-dim text-only) + the small CLIP fixture (`ClipBackend`, int8
dual-encoder, 512-dim joint space). Both load from the shared test-model cache (`$SHRIKE_TEST_MODEL_DIR`
or `~/.cache/shrike-test-models`); the CLIP/MiniLM ONNX fixtures come from the test suite
(`tests/integration/model_cache.py`). Build the native extension first (`scripts/build-native.sh`);
`shrike_native.ClipEmbedder` is required. A strong-vision-space robustness run (jina-clip-v2) lives in
`scripts/eval_cross_space_jina.py`.

Image selection is pinned in `eval/multimodal/resolved_urls.json` (committed, reused verbatim from
the #193 eval) and bytes cached locally, so the numbers replay; `--refresh` re-resolves.

Run:
    python scripts/eval_cross_space.py [-k 5] [--rrf-k 60] [--rank-cap 10] [--gate-margin 2.0]
"""
# ruff: noqa: E501  (throwaway eval: prose docstrings + aligned print tables read better un-wrapped)

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval" / "multimodal"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = (
    EVAL_DIR / "resolved_urls.json"
)  # committed: pins image selection (shared with #193 eval)
CACHE = EVAL_DIR / "cache"  # gitignored: image bytes
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "shrike-cross-space-eval/0.1 (https://github.com/lathrys-at/shrike)"}

# Make `import shrike` work when run from a worktree without an editable install.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# -- Wikimedia Commons image resolution (pinned + retried) -------------------


def _get(url: str, *, params: dict | None = None, retries: int = 4) -> httpx.Response:
    """GET with raise_for_status and backoff on transient failures (429/5xx/network)."""
    last: Exception | None = None
    for i in range(retries):
        try:
            r = httpx.get(url, params=params, headers=UA, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 — retry transient API/network errors
            last = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _resolve_url(term: str, resolved: dict[str, str], refresh: bool) -> str | None:
    """The pinned Commons thumb URL for a term, resolving + recording it on first sight."""
    if not refresh and term in resolved:
        return resolved[term]
    r = _get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap|drawing {term}",
            "gsrnamespace": 6,
            "gsrlimit": 1,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 640,
        },
    )
    pages = r.json().get("query", {}).get("pages", {})
    url = next(
        (
            ii[0].get("thumburl") or ii[0].get("url")
            for p in pages.values()
            if (ii := p.get("imageinfo"))
        ),
        None,
    )
    if url:
        resolved[term] = url
    return url


def fetch_image(term: str, resolved: dict[str, str], refresh: bool) -> Any | None:
    """Pinned-URL Commons image as PIL, bytes-cached locally. None on failure (skipped)."""
    from PIL import Image

    try:
        url = _resolve_url(term, resolved, refresh)
        if not url:
            return None
        cache_file = CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.img"
        if cache_file.exists() and not refresh:
            data = cache_file.read_bytes()
        else:
            data = _get(url).content
            CACHE.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(data)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001 — best-effort; a failed image is skipped
        print(f"    ! {term!r}: {e}")
        return None


# -- per-space ranking with max-over-items aggregation -----------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows so the dot product is cosine (USearch `cos` is scale-invariant; we mirror
    that here, computing cosine directly so the eval has no USearch build dependency)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


class Space:
    """One vector space: a set of (note_id, vector) items, ranked per query by max-over-items.

    A note may contribute several vectors (its text vector and one per image, all under its note_id).
    A query's score for a note is the BEST (max cosine) over that note's vectors — exactly the
    kernel's per-modality `search_by_modality` aggregation. `rank(query)` returns the note_ids
    best-first (one entry per distinct note), the input shape `rrf_fuse` consumes.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []

    def add(self, note_id: int, vectors: np.ndarray) -> None:
        for v in np.atleast_2d(vectors):
            self._ids.append(note_id)
            self._vecs.append(v)

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self._ids, dtype=np.int64), _normalize(np.asarray(self._vecs))

    def rank(self, query: np.ndarray, cap: int | None = None) -> list[int]:
        """note_ids best-first for one query vector, max-over-items per note.

        ``cap`` truncates to the top-N candidates — a per-space RANK CAP. The kernel's index search
        is always bounded (it asks USearch for top-N, not the whole corpus), so fusing UN-capped
        full-corpus rankings would be degenerate: a signal that lists every note contributes a tiny
        RRF weight to all of them, and a note appearing across more signals wins regardless of where.
        A realistic cap is what lets RRF discriminate — and the cap is itself a tuning parameter.
        """
        ids, mat = self._matrix()
        q = _normalize(query.reshape(1, -1))[0]
        sims = mat @ q  # cosine per item
        best: dict[int, float] = {}
        for nid, s in zip(ids.tolist(), sims.tolist(), strict=True):
            if nid not in best or s > best[nid]:
                best[nid] = s
        ordered = [nid for nid, _ in sorted(best.items(), key=lambda kv: -kv[1])]
        return ordered[:cap] if cap else ordered

    def best_score(self, query: np.ndarray) -> float:
        """The single best cosine over all items — used for the cross-space activation probe."""
        _, mat = self._matrix()
        q = _normalize(query.reshape(1, -1))[0]
        return float((mat @ q).max())


# -- metrics -----------------------------------------------------------------


def metrics(ranked_per_query: list[list[int]], expected: list[int], k: int) -> tuple[float, ...]:
    """(R@1, R@k, MRR) over a list of best-first ranked note_id lists and the expected answers."""
    r1 = rk = mrr = 0.0
    for ranking, want in zip(ranked_per_query, expected, strict=True):
        top = ranking[:k]
        rank = top.index(want) + 1 if want in top else 0
        r1 += rank == 1
        rk += 1 <= rank <= k
        mrr += (1.0 / rank) if rank else 0.0
    n = len(expected)
    return r1 / n, rk / n, mrr / n


def fmt(label: str, m: tuple[float, ...], k: int) -> str:
    return f"  {label:<34} R@1={m[0]:5.2f}  R@{k}={m[1]:5.2f}  MRR={m[2]:5.2f}"


# -- the three conditions ----------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF dampening constant (default 60)")
    ap.add_argument(
        "--rank-cap",
        type=int,
        default=10,
        help="per-space top-N candidate cap fed into RRF (the kernel always bounds search; default 10)",
    )
    ap.add_argument(
        "--gate-margin",
        type=float,
        default=2.0,
        help=(
            "cross-space image-gate margin in std-devs: fire the CLIP-image signal only when its "
            "best cosine for the query clears mean+margin·std of its OFF-topic (text-target) "
            "best-cosine distribution — the #201b activation gate lifted across spaces (default 2.0)"
        ),
    )
    ap.add_argument("--refresh", action="store_true", help="re-resolve images (ignore pins/cache)")
    ap.add_argument(
        "--blind-image-bodies",
        action="store_true",
        help=(
            "Replace image notes' (leaky) label text with answer-independent filler, so an "
            "image-target query can ONLY be answered by the image vector — isolates the pure "
            "cross-space image signal from the semantic leak the labels carry (#193 caveat)."
        ),
    )
    args = ap.parse_args()

    os.environ.setdefault(
        "SHRIKE_TEST_MODEL_DIR", os.path.expanduser("~/.cache/shrike-test-models")
    )
    model_cache = Path(os.environ["SHRIKE_TEST_MODEL_DIR"])

    from shrike.search_fusion import rrf_fuse

    data = json.loads(MANIFEST.read_text())
    image_notes = data["image_notes"]
    text_notes = data["text_notes"]
    resolved: dict[str, str] = json.loads(RESOLVED.read_text()) if RESOLVED.exists() else {}

    # 1. Fetch the pinned Commons images.
    print(f"Fetching {len(image_notes)} Commons images (pinned URLs) ...")
    img_notes: list[dict[str, Any]] = []
    for n in image_notes:
        img = fetch_image(n["search_term"], resolved, args.refresh)
        if img is not None:
            img_notes.append({**n, "image": img})
            print(f"    + #{n['id']} {n['search_term']!r}")
    RESOLVED.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"  resolved {len(img_notes)}/{len(image_notes)} images")
    if len(img_notes) < 8:
        print("  too few images resolved to evaluate; aborting (need >= 8).")
        return

    # 2. Load the project's native backends.
    print("\nLoading backends (native ONNX — no torch) ...")
    from shrike.embedding_clip import ClipBackend
    from shrike.embedding_onnx import OnnxBackend

    txt_dir = _first_existing(model_cache, ("all-MiniLM-L6-v2-onnx-fp32", "all-MiniLM-L6-v2-onnx"))
    clip_dir = model_cache / "clip-vit-base-patch32-onnx"
    if txt_dir is None or not (clip_dir / "preprocessor_config.json").is_file():
        print(
            f"  MISSING models under {model_cache} — need a MiniLM ONNX dir and "
            f"clip-vit-base-patch32-onnx. Fetch via the test suite's model_cache. Aborting."
        )
        return
    t0 = time.time()
    text_be = OnnxBackend(model=str(txt_dir))
    text_be.start()
    clip_be = ClipBackend(model=str(clip_dir))
    clip_be.start()
    print(
        f"  text: {txt_dir.name} (dim {text_be.embedding_dim()})  "
        f"clip: {clip_dir.name} (dim {clip_be.embedding_dim()})  {time.time() - t0:.1f}s"
    )

    def text_embed(strings: list[str]) -> np.ndarray:
        return np.asarray(text_be.embed_texts(strings), dtype=np.float32)

    def clip_text(strings: list[str]) -> np.ndarray:
        return np.asarray(clip_be.embed_texts(strings), dtype=np.float32)

    def clip_image(images: list[Any]) -> np.ndarray:
        return np.asarray(clip_be.embed_images(images), dtype=np.float32)

    # 3. One shared corpus: ALL notes (text-bearing + image-bearing) compete in every condition.
    #    note "body" text is what the text embedder / CLIP text tower see; image notes also carry
    #    one image vector in the CLIP space. The corpus is the union, so an image-target query has
    #    the 12 text notes as distractors and vice-versa (the realistic mixed collection).
    all_text_bodies: dict[int, str] = {}
    for n in text_notes:
        all_text_bodies[n["id"]] = n["text"]
    for n in img_notes:
        # The label leaks the topic *semantically* even though it never names the answer (#193). With
        # --blind-image-bodies we swap in answer-independent filler so the ONLY route to an image
        # note is its image vector — the clean isolation of the cross-space image signal.
        all_text_bodies[n["id"]] = (
            f"study flashcard {n['id']}" if args.blind_image_bodies else n["text"]
        )

    corpus_ids = list(all_text_bodies.keys())
    bodies = [all_text_bodies[i] for i in corpus_ids]

    # Pre-embed everything once.
    print("\nEmbedding corpus (text bodies, CLIP text, CLIP images) ...")
    t0 = time.time()
    mini_body = {i: v for i, v in zip(corpus_ids, text_embed(bodies), strict=True)}
    clip_body = {i: v for i, v in zip(corpus_ids, clip_text(bodies), strict=True)}
    img_vecs = {n["id"]: clip_image([n["image"]])[0] for n in img_notes}
    print(
        f"  embedded {len(corpus_ids)} notes ({len(img_vecs)} with images) in {time.time() - t0:.1f}s"
    )

    # Query sets.
    text_q = {n["id"]: n["query"] for n in text_notes}
    image_q = {n["id"]: n["query"] for n in img_notes}

    mini_text_q = {i: v for i, v in zip(text_q, text_embed(list(text_q.values())), strict=True)}
    mini_image_q = {i: v for i, v in zip(image_q, text_embed(list(image_q.values())), strict=True)}
    clip_text_q = {i: v for i, v in zip(text_q, clip_text(list(text_q.values())), strict=True)}
    clip_image_q = {i: v for i, v in zip(image_q, clip_text(list(image_q.values())), strict=True)}

    # Three per-space rankers, each ranked in ITS OWN space (the key to dodging the CLIP modality
    # gap — #201a's per-modality sub-index split, lifted to the cross-space case):
    #   sp_mini       — the dedicated text embedder (MiniLM) over note bodies.
    #   sp_clip_text  — the CLIP text tower over note bodies.
    #   sp_clip_image — the CLIP image tower over image vectors ONLY (no text vectors to bury it).
    sp_mini = Space("mini_text")
    for i in corpus_ids:
        sp_mini.add(i, mini_body[i])
    sp_clip_text = Space("clip_text")
    for i in corpus_ids:
        sp_clip_text.add(i, clip_body[i])
    sp_clip_image = Space("clip_image")
    for i, v in img_vecs.items():
        sp_clip_image.add(i, v)

    # (b) single CLIP shared space: text bodies AND image vectors POOLED into ONE max-over-items
    #     cosine ranking — the literal one-omni-index shape (#532/#533). This is where the modality
    #     gap bites: a text-query→image-vector cosine (~0.32) loses rank to a text-query→text-body
    #     cosine (~0.6), so image vectors only ever surface additively, never rank-1.
    sp_clip_pooled = Space("clip_pooled")
    for i in corpus_ids:
        sp_clip_pooled.add(i, clip_body[i])
    for i, v in img_vecs.items():
        sp_clip_pooled.add(i, v)

    def fuse(rankings: dict[str, list[int]], weights: dict[str, float]) -> list[int]:
        hits = rrf_fuse(rankings, weights=weights, k=args.rrf_k)
        return [h.note_id for h in hits]

    cap = args.rank_cap

    # Run all conditions over one query set (one mini-query-vec + one clip-query-vec per id).
    def run(query_mini: dict, query_clip: dict) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {"a": [], "b": [], "c": [], "c_gate": [], "c_relgate": []}
        for qid in query_mini:
            qm, qc = query_mini[qid], query_clip[qid]
            mini_r = sp_mini.rank(qm, cap)
            ctext_r = sp_clip_text.rank(qc, cap)
            cimg_r = sp_clip_image.rank(qc, cap)
            pooled_r = sp_clip_pooled.rank(qc)  # (b) is a single ranking, no per-space cap needed

            # (a) text-only baseline.
            out["a"].append(mini_r)
            # (b) single CLIP shared (pooled) space.
            out["b"].append(pooled_r)
            # (c) cross-space FUSED: the dedicated text embedder + the CLIP-image space as SEPARATE
            #     RRF signals, each ranked in its own space. (We also fold the CLIP text tower in as
            #     a third signal — it costs nothing and is what the kernel would actually run.)
            out["c"].append(
                fuse(
                    {"text": mini_r, "clip_text": ctext_r, "clip_image": cimg_r},
                    {"text": 1.0, "clip_text": 1.0, "clip_image": 1.0},
                )
            )
            # (c+gate) the same, but the CLIP-image signal is GATED OUT unless its best cosine for
            #     this query clears the off-topic floor (the cross-space analogue of #201b). Stops a
            #     text-target query from ever having weak image vectors injected into its fusion.
            sigs = {"text": mini_r, "clip_text": ctext_r}
            wts = {"text": 1.0, "clip_text": 1.0}
            if sp_clip_image.best_score(qc) >= image_gate:
                sigs["clip_image"] = cimg_r
                wts["clip_image"] = 1.0
            out["c_gate"].append(fuse(sigs, wts))

            # (c+relgate) RELATIVE gate: fire the image signal only when its best cosine for this
            #     query beats the dedicated TEXT space's best for the same query — "is the image
            #     match more compelling than the text match?". Self-calibrating (no off-topic sample),
            #     and robust to a model's absolute cosine band, unlike the absolute mean+margin·std.
            rsigs = {"text": mini_r, "clip_text": ctext_r}
            rwts = {"text": 1.0, "clip_text": 1.0}
            if sp_clip_image.best_score(qc) >= sp_mini.best_score(qm):
                rsigs["clip_image"] = cimg_r
                rwts["clip_image"] = 1.0
            out["c_relgate"].append(fuse(rsigs, rwts))
        return out

    text_expect = list(text_q.keys())
    image_expect = list(image_q.keys())

    # The cross-space activation gate threshold (#201b analogue): mean + margin·std of the CLIP image
    # space's best cosine over OFF-topic (text-target) queries — i.e. how strongly image vectors fire
    # when the answer is NOT an image. Anything below this is "the image space didn't really match".
    off_scores = np.array([sp_clip_image.best_score(clip_text_q[i]) for i in text_q])
    image_gate = float(off_scores.mean() + args.gate_margin * off_scores.std())

    text_target = run(mini_text_q, clip_text_q)
    image_target = run(mini_image_q, clip_image_q)

    k = args.k
    label = " [BLIND image bodies]" if args.blind_image_bodies else " [leaky labels]"
    print(
        f"\n=== CROSS-SPACE EVAL (corpus={len(corpus_ids)} notes, rrf_k={args.rrf_k}, "
        f"rank_cap={cap}, image_gate={image_gate:.3f}){label} ==="
    )
    print(f"\n--- text-target queries ({len(text_expect)}: answer in a TEXT note) ---")
    print(fmt("(a) text-only [MiniLM]", metrics(text_target["a"], text_expect, k), k))
    print(fmt("(b) single CLIP pooled space", metrics(text_target["b"], text_expect, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(text_target["c"], text_expect, k), k))
    print(
        fmt(
            "(c+gate) cross-space + abs img gate", metrics(text_target["c_gate"], text_expect, k), k
        )
    )
    print(
        fmt("(c+relgate) + relative img gate", metrics(text_target["c_relgate"], text_expect, k), k)
    )

    print(f"\n--- image-target queries ({len(image_expect)}: answer in the IMAGE) ---")
    print(fmt("(a) text-only [MiniLM]", metrics(image_target["a"], image_expect, k), k))
    print(fmt("(b) single CLIP pooled space", metrics(image_target["b"], image_expect, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(image_target["c"], image_expect, k), k))
    print(
        fmt(
            "(c+gate) cross-space + abs img gate",
            metrics(image_target["c_gate"], image_expect, k),
            k,
        )
    )
    print(
        fmt(
            "(c+relgate) + relative img gate",
            metrics(image_target["c_relgate"], image_expect, k),
            k,
        )
    )

    # ---- isolating control: CLIP image-only retrieval (images-only corpus, no text vectors) ----
    # Proves the image signal IS present in the small CLIP (the #193 finding) — so any image-recall
    # collapse above is a RANKING/fusion artifact (the modality gap), not a dead image tower.
    iso = []
    for qid in image_q:
        ranking = sp_clip_image.rank(clip_image_q[qid])  # only image vectors live in this space
        iso.append(ranking)
    print("\n--- isolating control: CLIP IMAGE-ONLY space (image vectors only) ---")
    print(fmt("image-only retrieval", metrics(iso, image_expect, k), k))

    # ---- activation diagnostic: does the CLIP image space fire spuriously on text-target queries? ----
    # The #201b intra-modal gate floors a non-text modality. Across SPACES the analogue is: does the
    # CLIP image space's best score for a text-target query (right answer is a TEXT note, NOT an image)
    # sit BELOW its best score for an image-target query? If image vectors fire as strongly off-topic
    # as on-topic, a cross-space gate is NEEDED; clean separation means RRF alone could suffice.
    on_topic = [sp_clip_image.best_score(clip_image_q[i]) for i in image_q]
    off_topic = [sp_clip_image.best_score(clip_text_q[i]) for i in text_q]
    print("\n--- cross-space activation diagnostic (CLIP image space best-cosine) ---")
    print(
        f"  image-target queries (on-topic):  mean={np.mean(on_topic):.3f}  std={np.std(on_topic):.3f}"
    )
    print(
        f"  text-target  queries (off-topic): mean={np.mean(off_topic):.3f}  std={np.std(off_topic):.3f}"
    )
    sep = float(np.mean(on_topic) - np.mean(off_topic))
    print(f"  separation (on - off): {sep:+.3f}  (gate threshold mean+1·std = {image_gate:.3f})")

    print("\nDone.")


def _first_existing(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        d = base / name
        if (d / "model.onnx").is_file():
            return d
    return None


if __name__ == "__main__":
    main()
