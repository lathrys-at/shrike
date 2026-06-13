#!/usr/bin/env python3
"""Cross-space retrieval eval — the GO/NO-GO gate for the multi-space epic (#231 / #229).

NOT production code. This is the *cross-space* question, distinct from `scripts/eval_multimodal.py`
(the #193 single-shared-space eval). It answers: when there is no single omni embedder, does running
a **dedicated text embedder** and a **CLIP** as TWO SEPARATE vector spaces and **RRF-fusing the query
across the two spaces** actually work — and does it beat the *one* omni/CLIP space that already ships
(#532/#533)?

Three conditions, one shared corpus (all notes — text-bearing and image-bearing — in ONE collection,
so every query competes against the full distractor set, the realistic shape):

  (a) text-only baseline        : a dedicated text embedder (ONNX MiniLM), ONE index of note text.
  (b) single-CLIP-shared-space  : CLIP embeds note text AND images into ONE space, one index keyed by
                                  note_id (text vec + one vec per image, the shipped #532/#533 shape).
  (c) cross-space FUSED (NEW)   : a SEPARATE MiniLM text index + a SEPARATE CLIP index, the query
                                  embedded into BOTH, the two spaces' per-space rankings RRF-FUSED,
                                  results aggregated items -> notes by max-over-items.

For (b) and (c) the per-space ranker is itself per-vector max-over-items (a note's rank in a space is
its best vector for the query), exactly as the kernel's `search_by_modality` aggregates today.

Reported per condition × {text-target, image-target} queries: R@1 / R@5 / MRR.
  - text-target  : the answer is in a text note's body (the 12 text_notes) — does cross-space fusion
                   PRESERVE text<->text quality, i.e. not drag it toward CLIP's weaker text tower?
  - image-target : the answer is in a note's image (the 16 image_notes) — does the CLIP space surface
                   the image-bearing note at useful recall?

Models (the project's OWN native backends — NO torch / sentence-transformers; this is exactly what
Shrike would ship): ONNX MiniLM (`OnnxBackend`, 384-dim text-only) + the small CLIP fixture
(`ClipBackend`, int8 dual-encoder, 512-dim shared space). Both load from the shared test-model cache
(`$SHRIKE_TEST_MODEL_DIR` or `~/.cache/shrike-test-models`); fetch them with
`scripts/fetch-multimodal-model.sh` is NOT needed — the CLIP/MiniLM ONNX fixtures come from the test
suite (`tests/integration/model_cache.py`). Build the native extension first
(`scripts/build-native.sh`); `shrike_native.ClipEmbedder` is required.

Image selection is pinned in `eval/multimodal/resolved_urls.json` (committed, reused verbatim from
the #193 eval) and bytes cached locally, so the numbers replay; `--refresh` re-resolves.

Run:
    python scripts/eval_cross_space.py [-k 5] [--rrf-k 60] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval" / "multimodal"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"  # committed: pins image selection (shared with #193 eval)
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

    def rank(self, query: np.ndarray) -> list[int]:
        """note_ids best-first for one query vector, max-over-items per note."""
        ids, mat = self._matrix()
        q = _normalize(query.reshape(1, -1))[0]
        sims = mat @ q  # cosine per item
        best: dict[int, float] = {}
        for nid, s in zip(ids.tolist(), sims.tolist(), strict=True):
            if nid not in best or s > best[nid]:
                best[nid] = s
        return [nid for nid, _ in sorted(best.items(), key=lambda kv: -kv[1])]

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
    ap.add_argument("--refresh", action="store_true", help="re-resolve images (ignore pins/cache)")
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

    txt_dir = _first_existing(
        model_cache, ("all-MiniLM-L6-v2-onnx-fp32", "all-MiniLM-L6-v2-onnx")
    )
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
        all_text_bodies[n["id"]] = n["text"]  # the leaky-label body (e.g. "Cardiovascular figure")

    corpus_ids = list(all_text_bodies.keys())
    bodies = [all_text_bodies[i] for i in corpus_ids]

    # Pre-embed everything once.
    print("\nEmbedding corpus (text bodies, CLIP text, CLIP images) ...")
    t0 = time.time()
    mini_body = {i: v for i, v in zip(corpus_ids, text_embed(bodies), strict=True)}
    clip_body = {i: v for i, v in zip(corpus_ids, clip_text(bodies), strict=True)}
    img_vecs = {n["id"]: clip_image([n["image"]])[0] for n in img_notes}
    print(f"  embedded {len(corpus_ids)} notes ({len(img_vecs)} with images) in {time.time() - t0:.1f}s")

    # Query sets.
    text_q = {n["id"]: n["query"] for n in text_notes}
    image_q = {n["id"]: n["query"] for n in img_notes}

    mini_text_q = {i: v for i, v in zip(text_q, text_embed(list(text_q.values())), strict=True)}
    mini_image_q = {i: v for i, v in zip(image_q, text_embed(list(image_q.values())), strict=True)}
    clip_text_q = {i: v for i, v in zip(text_q, clip_text(list(text_q.values())), strict=True)}
    clip_image_q = {i: v for i, v in zip(image_q, clip_text(list(image_q.values())), strict=True)}

    # ---- (a) text-only baseline: ONE MiniLM index of note bodies. ----
    sp_text = Space("text")
    for i in corpus_ids:
        sp_text.add(i, mini_body[i])

    # ---- (b) single CLIP shared space: text bodies + image vectors in ONE CLIP index. ----
    sp_clip = Space("clip")
    for i in corpus_ids:
        sp_clip.add(i, clip_body[i])
    for i, v in img_vecs.items():
        sp_clip.add(i, v)

    # ---- (c) cross-space: the text space (a) AND a CLIP space holding text+image, RRF-fused. ----
    #    Two SEPARATE spaces. Per query we rank each independently, then fuse the two rankings.
    #    The CLIP space here carries BOTH its text bodies and its image vectors (so it can answer an
    #    image-target query); the text space is MiniLM-only. This is the "dedicated text embedder +
    #    CLIP" shape from #229/#232-#234.
    sp_clip_full = sp_clip  # identical contents — reuse

    rrf_weights = {"text": 1.0, "clip": 1.0}

    def fuse_two(text_rank: list[int], clip_rank: list[int]) -> list[int]:
        hits = rrf_fuse(
            {"text": text_rank, "clip": clip_rank}, weights=rrf_weights, k=args.rrf_k
        )
        return [h.note_id for h in hits]

    # Run all three conditions over both query sets.
    def run(query_vecs_mini: dict, query_vecs_clip: dict) -> dict[str, list[list[int]]]:
        a_rank, b_rank, c_rank = [], [], []
        for qid in query_vecs_mini:
            tr = sp_text.rank(query_vecs_mini[qid])
            cr = sp_clip.rank(query_vecs_clip[qid])
            cr_full = sp_clip_full.rank(query_vecs_clip[qid])
            a_rank.append(tr)
            b_rank.append(cr)
            c_rank.append(fuse_two(tr, cr_full))
        return {"a": a_rank, "b": b_rank, "c": c_rank}

    text_target = run(mini_text_q, clip_text_q)
    image_target = run(mini_image_q, clip_image_q)
    text_expect = list(text_q.keys())
    image_expect = list(image_q.keys())

    k = args.k
    print(f"\n=== CROSS-SPACE EVAL (corpus={len(corpus_ids)} notes, rrf_k={args.rrf_k}) ===")
    print(f"\n--- text-target queries ({len(text_expect)}: answer in a TEXT note) ---")
    print(fmt("(a) text-only [MiniLM]", metrics(text_target["a"], text_expect, k), k))
    print(fmt("(b) single CLIP shared space", metrics(text_target["b"], text_expect, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(text_target["c"], text_expect, k), k))

    print(f"\n--- image-target queries ({len(image_expect)}: answer in the IMAGE) ---")
    print(fmt("(a) text-only [MiniLM]", metrics(image_target["a"], image_expect, k), k))
    print(fmt("(b) single CLIP shared space", metrics(image_target["b"], image_expect, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(image_target["c"], image_expect, k), k))

    # ---- activation diagnostic: does CLIP fire spuriously on text-target queries? ----
    # The #201b intra-modal gate floors a non-text modality. Across SPACES the analogue is: does the
    # CLIP image sub-space's best score for a text-target query (where the right answer is a TEXT
    # note, NOT an image) sit ABOVE or BELOW its best score for an image-target query? If text-target
    # queries routinely light up image vectors as strongly as image-target ones, a cross-space gate
    # is NEEDED; if image vectors only fire for genuinely image-relevant queries, RRF alone suffices.
    img_space = Space("img")
    for i, v in img_vecs.items():
        img_space.add(i, v)
    on_topic = [img_space.best_score(clip_image_q[i]) for i in image_q]
    off_topic = [img_space.best_score(clip_text_q[i]) for i in text_q]
    print("\n--- cross-space activation diagnostic (CLIP image sub-space best-cosine) ---")
    print(f"  image-target queries (on-topic): mean={np.mean(on_topic):.3f}  std={np.std(on_topic):.3f}")
    print(f"  text-target  queries (off-topic): mean={np.mean(off_topic):.3f}  std={np.std(off_topic):.3f}")
    sep = np.mean(on_topic) - np.mean(off_topic)
    print(f"  separation (on - off): {sep:+.3f}  ({'separable' if sep > 0.02 else 'NOT separable'})")

    print("\nDone.")


def _first_existing(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        d = base / name
        if (d / "model.onnx").is_file():
            return d
    return None


if __name__ == "__main__":
    main()
