#!/usr/bin/env python3
"""Evaluation spike for image<->text retrieval (Phase 3a of #162).

NOT production code. Measures, on a small labelled set, whether a CLIP model lets a *text*
query retrieve a card by the content of its *image*, and whether the shared space regresses
plain text retrieval versus the current MiniLM text model. The numbers gate the build (the
CLIP backend + multi-vector index redesign).

What it does:
  1. Resolve each image note's `search_term` to a real current Wikimedia Commons image
     (MediaWiki API) and download it (skips failures).
  2. Embed note text + images with jina-clip-v2 (sentence-transformers) into one space.
  3. Build USearch indexes keyed by note_id — image-only, text-only, and `multi=True`
     (text + image under one key, the real index shape) — and report image-by-text
     recall@1/@5 + MRR. The text-only index should do *poorly* (the note text hides the
     answer), proving retrieval comes from the image.
  4. Baseline: embed the text-only notes + queries with jina's text encoder AND the project's
     MiniLM (OnnxBackend), and compare text-by-text recall — the single-shared-space tradeoff.

Run (heavy deps; install into the venv first):
    pip install sentence-transformers torch transformers pillow einops timm
    python scripts/eval_multimodal.py [--collection PATH]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "eval" / "multimodal" / "manifest.json"
CLIP_MODEL = "jinaai/jina-clip-v2"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "shrike-multimodal-eval/0.1 (https://github.com/lathrys-at/shrike)"}


# -- Wikimedia Commons image resolution --------------------------------------


def commons_image(term: str, width: int = 640) -> Any | None:
    """Resolve a search term to a current Commons raster image (PIL), or None."""
    from PIL import Image

    try:
        r = httpx.get(
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
                "iiurlwidth": width,
            },
            headers=UA,
            timeout=30,
            follow_redirects=True,
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
        if not url:
            return None
        img = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
        img.raise_for_status()
        return Image.open(io.BytesIO(img.content)).convert("RGB")
    except Exception as e:  # noqa: BLE001 — best-effort; a failed image is skipped
        print(f"    ! {term!r}: {e}")
        return None


# -- metrics + tiny USearch helper -------------------------------------------


def _index(dim: int, multi: bool = False) -> Any:
    from usearch.index import Index

    return Index(ndim=dim, metric="cos", dtype="f32", multi=multi)


def recall_mrr(
    index: Any, queries: np.ndarray, expected_ids: list[int], k: int = 5
) -> tuple[float, float, float]:
    """Return (recall@1, recall@k, MRR) of expected note_ids in the search results."""
    hits1 = hitsk = mrr = 0.0
    raw = index.search(queries, k)
    per_query = [raw] if len(queries) == 1 else [raw[i] for i in range(len(queries))]
    for matches, want in zip(per_query, expected_ids, strict=True):
        # Dedup to best rank per note_id (multi=True can return a key more than once).
        seen: list[int] = []
        for key in [int(m.key) for m in matches]:
            if key not in seen:
                seen.append(key)
        rank = seen.index(want) + 1 if want in seen else 0
        hits1 += rank == 1
        hitsk += 1 <= rank <= k
        mrr += (1.0 / rank) if rank else 0.0
    n = len(expected_ids)
    return hits1 / n, hitsk / n, mrr / n


def line(label: str, r1: float, rk: float, mrr: float, k: int = 5) -> str:
    return f"  {label:<28} R@1={r1:5.2f}  R@{k}={rk:5.2f}  MRR={mrr:5.2f}"


# -- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", help="(optional) real .anki2 for a qualitative pass")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    image_notes = data["image_notes"]
    text_notes = data["text_notes"]

    print(f"Loading {CLIP_MODEL} (sentence-transformers, trust_remote_code) ...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    clip = SentenceTransformer(CLIP_MODEL, trust_remote_code=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    def clip_text(texts: list[str]) -> np.ndarray:
        return np.asarray(
            clip.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def clip_image(images: list[Any]) -> np.ndarray:
        return np.asarray(
            clip.encode(images, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    # 1. Fetch images.
    print(f"\nFetching {len(image_notes)} Commons images ...")
    notes: list[dict[str, Any]] = []
    for n in image_notes:
        img = commons_image(n["search_term"])
        if img is not None:
            notes.append({**n, "image": img})
            print(f"    + #{n['id']} {n['search_term']!r}")
    print(f"  resolved {len(notes)}/{len(image_notes)} images")
    if len(notes) < 5:
        print("  too few images resolved to evaluate; aborting.")
        return

    # 2. Embed image-note text, images, queries.
    t0 = time.time()
    txt_vecs = clip_text([n["text"] for n in notes])
    img_vecs = clip_image([n["image"] for n in notes])
    q_vecs = clip_text([n["query"] for n in notes])
    dim = img_vecs.shape[1]
    embed_dt = time.time() - t0
    ids = [n["id"] for n in notes]
    keys = np.array(ids, dtype=np.int64)

    # 3. Three indexes: text-only, image-only, multi (text+image under one key).
    txt_idx = _index(dim)
    txt_idx.add(keys, txt_vecs)
    img_idx = _index(dim)
    img_idx.add(keys, img_vecs)
    multi_idx = _index(dim, multi=True)
    multi_idx.add(keys, txt_vecs)
    multi_idx.add(keys, img_vecs)  # same keys → two vectors per note

    print(f"\n=== Image-by-text retrieval ({len(notes)} notes, dim={dim}) ===")
    print("  (note text hides the answer, so text-only should be poor; image/multi prove it)")
    print(line("text-only index", *recall_mrr(txt_idx, q_vecs, ids, args.k), args.k))
    print(line("image-only index", *recall_mrr(img_idx, q_vecs, ids, args.k), args.k))
    print(line("multi (text+image)", *recall_mrr(multi_idx, q_vecs, ids, args.k), args.k))

    # 4. Text baseline: jina text encoder vs MiniLM (OnnxBackend) on text-only notes.
    print(f"\n=== Text-by-text retrieval, {len(text_notes)} notes (shared-space tradeoff) ===")
    t_ids = [n["id"] for n in text_notes]
    t_keys = np.array(t_ids, dtype=np.int64)
    jina_doc = clip_text([n["text"] for n in text_notes])
    jina_q = clip_text([n["query"] for n in text_notes])
    j_idx = _index(jina_doc.shape[1])
    j_idx.add(t_keys, jina_doc)
    print(line("jina-clip-v2 text", *recall_mrr(j_idx, jina_q, t_ids, args.k), args.k))

    mini = _minilm_embedder()
    if mini is not None:
        embed, mdim = mini
        m_idx = _index(mdim)
        m_idx.add(t_keys, embed([n["text"] for n in text_notes]))
        m_r = recall_mrr(m_idx, embed([n["query"] for n in text_notes]), t_ids, args.k)
        print(line("MiniLM baseline", *m_r, args.k))
    else:
        print("  (MiniLM baseline skipped — no cached ONNX MiniLM; set SHRIKE_TEST_MODEL_DIR)")

    print("\n=== Serving / cost ===")
    print(f"  embed {len(notes)} text+image+query via jina-clip-v2: {embed_dt:.1f}s  (dim={dim})")
    print("  loader: sentence-transformers (transformers); build targets the ONNX seam (3b).")

    if args.collection:
        print(
            f"\n(--collection {args.collection}: qualitative pass not wired; numbers are the gate.)"
        )


def _minilm_embedder() -> tuple[Any, int] | None:
    """The project's cached ONNX MiniLM as the text baseline, if available."""
    import os

    cache = os.environ.get("SHRIKE_TEST_MODEL_DIR") or os.path.expanduser(
        "~/.cache/shrike-test-models"
    )
    for name in ("all-MiniLM-L6-v2-onnx-fp32", "all-MiniLM-L6-v2-onnx-int8"):
        d = Path(cache) / name
        if (d / "model.onnx").is_file():
            from shrike.embedding_onnx import OnnxBackend

            be = OnnxBackend(model=str(d))
            be.start()
            dim = be.embedding_dim() or 384

            def embed(ts: list[str], _be: Any = be) -> np.ndarray:
                return np.asarray(_be.embed_texts(list(ts)), dtype=np.float32)

            return embed, dim
    return None


if __name__ == "__main__":
    main()
