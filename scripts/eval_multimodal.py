#!/usr/bin/env python3
"""Evaluation spike for image<->text retrieval (Phase 3a of #162).

NOT production code. Measures, on a small labelled set, whether a CLIP model lets a *text*
query retrieve a card by the content of its *image*, and whether the shared space regresses
plain text retrieval versus the current MiniLM text model. The numbers gate the build (the
CLIP backend + multi-vector index redesign).

What it does:
  1. Resolve each image note's `search_term` to a real Wikimedia Commons image (MediaWiki API)
     and download it. Resolved URLs are pinned in `eval/multimodal/resolved_urls.json` (committed)
     and bytes cached locally, so the committed numbers are replayable; `--refresh` re-resolves.
  2. Embed note text + images with jina-clip-v2 (sentence-transformers) into one space.
  3. Build USearch indexes keyed by note_id — image-only, text-only, and `multi=True` (text +
     image under one key, the real index shape) — and report image-by-text recall@1/@5 + MRR.
     The image-only index holds *only* image vectors, so its score is self-sufficient proof that
     a text query retrieves by image content. A "blind" text row (answer-independent filler) is
     the no-signal floor; the real note text leaks topic *semantically* (the eval is semantic,
     not lexical), so plain text-only is an observation, not a blind control.
  4. Baseline: embed the text-only notes + queries with jina's text encoder AND the project's
     MiniLM (OnnxBackend), and compare text-by-text recall — the single-shared-space tradeoff.

Run (heavy deps; jina-clip-v2's trust_remote_code needs transformers 4.x — 5.x removed
`clip_loss`; on a Python without `_lzma`, torchvision fails to import):
    pip install sentence-transformers torch 'transformers==4.49.0' pillow einops timm
    python scripts/eval_multimodal.py [--refresh] [--collection PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval" / "multimodal"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"  # committed: pins image selection
CACHE = EVAL_DIR / "cache"  # gitignored: image bytes
CLIP_MODEL = "jinaai/jina-clip-v2"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "shrike-multimodal-eval/0.1 (https://github.com/lathrys-at/shrike)"}


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


# -- metrics + tiny USearch helper -------------------------------------------


def _index(dim: int, multi: bool = False) -> Any:
    from usearch.index import Index

    return Index(ndim=dim, metric="cos", dtype="f32", multi=multi)


def recall_mrr(
    index: Any,
    queries: np.ndarray,
    expected_ids: list[int],
    k: int = 5,
    vectors_per_note: int = 1,
) -> tuple[float, float, float]:
    """(recall@1, recall@k, MRR) of expected note_ids, scored over k *distinct* notes.

    A `multi=True` index holds several vectors per note, so the top-k raw vectors can dedup to
    fewer than k distinct notes. Over-fetch ``k * vectors_per_note`` raw hits, dedup to distinct
    note_ids, then truncate to k distinct — so multi is scored over the same k candidate slots as
    a single-vector index (apples-to-apples). ``queries`` is always 2-D, so search returns
    BatchMatches indexable per query.
    """
    hits1 = hitsk = mrr = 0.0
    raw = index.search(queries, k * max(1, vectors_per_note))
    for i, want in enumerate(expected_ids):
        distinct: list[int] = []
        for key in [int(m.key) for m in raw[i]]:
            if key not in distinct:
                distinct.append(key)
        distinct = distinct[:k]
        rank = distinct.index(want) + 1 if want in distinct else 0
        hits1 += rank == 1
        hitsk += 1 <= rank <= k
        mrr += (1.0 / rank) if rank else 0.0
    n = len(expected_ids)
    return hits1 / n, hitsk / n, mrr / n


def line(label: str, r1: float, rk: float, mrr: float, k: int = 5) -> str:
    return f"  {label:<30} R@1={r1:5.2f}  R@{k}={rk:5.2f}  MRR={mrr:5.2f}"


# -- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-resolve images (ignore pins/cache)")
    ap.add_argument("--collection", help="(optional) real .anki2 for a qualitative pass")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    image_notes = data["image_notes"]
    text_notes = data["text_notes"]
    resolved: dict[str, str] = json.loads(RESOLVED.read_text()) if RESOLVED.exists() else {}

    print(f"Loading {CLIP_MODEL} (sentence-transformers, trust_remote_code) ...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    clip = SentenceTransformer(CLIP_MODEL, trust_remote_code=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    def clip_text(texts: list[str]) -> np.ndarray:
        return np.asarray(
            clip.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
        )

    def clip_image(images: list[Any]) -> np.ndarray:
        return np.asarray(
            clip.encode(images, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    # 1. Fetch images (pinned URLs).
    print(f"\nFetching {len(image_notes)} Commons images ...")
    notes: list[dict[str, Any]] = []
    for n in image_notes:
        img = fetch_image(n["search_term"], resolved, args.refresh)
        if img is not None:
            notes.append({**n, "image": img})
            print(f"    + #{n['id']} {n['search_term']!r}")
    RESOLVED.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"  resolved {len(notes)}/{len(image_notes)} images (URLs pinned in {RESOLVED.name})")
    if len(notes) < 5:
        print("  too few images resolved to evaluate; aborting.")
        return

    # 2. Embed image-note text, images, queries, and an answer-independent "blind" filler.
    t0 = time.time()
    txt_vecs = clip_text([n["text"] for n in notes])
    blind_vecs = clip_text([f"study flashcard {n['id']}" for n in notes])
    img_vecs = clip_image([n["image"] for n in notes])
    q_vecs = clip_text([n["query"] for n in notes])
    dim = img_vecs.shape[1]
    embed_dt = time.time() - t0
    ids = [n["id"] for n in notes]
    keys = np.array(ids, dtype=np.int64)

    # 3. Indexes: text-only (real text), blind (no-signal floor), image-only, multi.
    def built(vecs: np.ndarray, multi: bool = False, *extra: np.ndarray) -> Any:
        idx = _index(dim, multi=multi)
        idx.add(keys, vecs)
        for v in extra:
            idx.add(keys, v)
        return idx

    print(f"\n=== Image-by-text retrieval ({len(notes)} notes, dim={dim}) ===")
    print("  (image-only holds only image vectors → its score is self-sufficient proof)")
    print(
        line(
            "text-only (note text, leaky)",
            *recall_mrr(built(txt_vecs), q_vecs, ids, args.k),
            args.k,
        )
    )
    print(
        line(
            "text-only (blind filler)", *recall_mrr(built(blind_vecs), q_vecs, ids, args.k), args.k
        )
    )
    print(line("image-only", *recall_mrr(built(img_vecs), q_vecs, ids, args.k), args.k))
    multi_idx = built(txt_vecs, True, img_vecs)
    print(line("multi (text+image)", *recall_mrr(multi_idx, q_vecs, ids, args.k, 2), args.k))

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
    """The project's cached ONNX MiniLM as the text baseline, if available.

    The model-dir names come from ``tests.integration.model_cache`` (the single source of truth)
    so a model bump there doesn't silently drop this baseline. Imported lazily with a path insert
    because ``tests`` isn't pip-installed and ``scripts/`` is ``sys.path[0]`` when run directly.
    """
    import os
    import sys

    sys.path.insert(0, str(ROOT))
    from tests.integration.model_cache import ONNX_FP32_MODEL_DIR_NAME, ONNX_MODEL_DIR_NAME

    cache = os.environ.get("SHRIKE_TEST_MODEL_DIR") or os.path.expanduser(
        "~/.cache/shrike-test-models"
    )
    for name in (ONNX_FP32_MODEL_DIR_NAME, ONNX_MODEL_DIR_NAME):
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
