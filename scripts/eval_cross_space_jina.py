#!/usr/bin/env python3
"""Cross-space fusion eval with a STRONG vision space (jina-clip-v2) — robustness check for #231.

The companion `scripts/eval_cross_space.py` validates the cross-space RRF-fusion MECHANISM with the
project's small CI backends (ONNX MiniLM + the small clip-vit-base-patch32 fixture). That fixture's
image tower is perfect *in isolation* but has a narrow on/off-topic cosine separation (~0.05), which
makes the cross-space activation gate hard to calibrate on the pure-image case. This script re-runs
the same three-condition comparison with a PRODUCTION-quality model (jina-clip-v2, the #193 model) so
the mechanism verdict and the gate-margin recommendation don't hinge on a weak fixture.

Model-agnostic by design (the coordinator's framing): we are validating the FUSION across two
separate spaces, not the specific models. jina-clip-v2 is a dual encoder — its text tower stands in
for "a platform/dedicated TEXT embedder space" and its vision tower for "a separate platform VISION
embedder space"; they are indexed as TWO SEPARATE spaces and RRF-fused, exactly the no-omni
deployment shape (a platform text embedder + a separate platform CLIP, two distinct vector spaces).

Conditions (same as the companion):
  (a) text-only baseline        : jina TEXT tower over note bodies, one index.
  (b) single-shared-space ref   : jina text bodies + image vectors POOLED in one cosine index
                                  (the reference ceiling a unified space reaches where one exists).
  (c) cross-space FUSED         : the text space + the vision (image) space as SEPARATE RRF signals.
  (c+gate)                      : (c) with the vision signal gated by the #201b cross-space analogue.

Heavy deps (jina-clip-v2's trust_remote_code needs transformers 4.x; a Python built without _lzma
fails to import torchvision):
    pip install 'transformers==4.49.0' sentence-transformers torch einops timm pillow numpy httpx
    python scripts/eval_cross_space_jina.py [-k 5] [--rrf-k 60] [--rank-cap 10] [--gate-margin 2.0]
"""
# ruff: noqa: E501  (throwaway eval: prose docstrings + aligned print tables read better un-wrapped)

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval" / "multimodal"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"
CACHE = EVAL_DIR / "cache"
CLIP_MODEL = "jinaai/jina-clip-v2"
UA = {"User-Agent": "shrike-cross-space-eval/0.1 (https://github.com/lathrys-at/shrike)"}

sys.path.insert(0, str(ROOT / "src"))

# Reuse the companion script's Space / metrics / fmt and the project's frozen RRF reference.
from eval_cross_space import Space, fmt, metrics  # noqa: E402

from shrike.search_fusion import rrf_fuse  # noqa: E402


def fetch_image(term: str, resolved: dict[str, str]) -> Any | None:
    from PIL import Image

    url = resolved.get(term)
    if not url:
        return None
    cache_file = CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.img"
    try:
        if cache_file.exists():
            data = cache_file.read_bytes()
        else:
            data = httpx.get(url, headers=UA, timeout=30, follow_redirects=True).content
            CACHE.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(data)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"    ! {term!r}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rank-cap", type=int, default=10)
    ap.add_argument("--gate-margin", type=float, default=2.0)
    ap.add_argument("--blind-image-bodies", action="store_true")
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    image_notes = data["image_notes"]
    text_notes = data["text_notes"]
    resolved: dict[str, str] = json.loads(RESOLVED.read_text())

    print(f"Loading {CLIP_MODEL} (sentence-transformers, trust_remote_code) ...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    clip = SentenceTransformer(CLIP_MODEL, trust_remote_code=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    def enc_text(texts: list[str]) -> np.ndarray:
        return np.asarray(
            clip.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
        )

    def enc_image(images: list[Any]) -> np.ndarray:
        return np.asarray(
            clip.encode(images, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    print(f"\nFetching {len(image_notes)} Commons images (pinned) ...")
    img_notes = []
    for n in image_notes:
        img = fetch_image(n["search_term"], resolved)
        if img is not None:
            img_notes.append({**n, "image": img})
    print(f"  resolved {len(img_notes)}/{len(image_notes)}")
    if len(img_notes) < 8:
        print("  too few images; aborting.")
        return

    # Shared corpus: all notes; image notes carry an image vector too.
    bodies_map = {n["id"]: n["text"] for n in text_notes}
    for n in img_notes:
        bodies_map[n["id"]] = f"study flashcard {n['id']}" if args.blind_image_bodies else n["text"]
    ids = list(bodies_map.keys())
    bodies = [bodies_map[i] for i in ids]

    print("\nEmbedding corpus ...")
    t0 = time.time()
    body_vec = dict(zip(ids, enc_text(bodies), strict=True))
    img_vec = {n["id"]: enc_image([n["image"]])[0] for n in img_notes}
    print(f"  {len(ids)} bodies + {len(img_vec)} images in {time.time() - t0:.1f}s")

    text_q = {n["id"]: n["query"] for n in text_notes}
    image_q = {n["id"]: n["query"] for n in img_notes}
    tq_vec = dict(zip(text_q, enc_text(list(text_q.values())), strict=True))
    iq_vec = dict(zip(image_q, enc_text(list(image_q.values())), strict=True))

    # Spaces: a TEXT space (bodies via the text tower) and a VISION space (image vectors only),
    # plus a POOLED space for the (b) reference ceiling.
    sp_text = Space("text")
    for i in ids:
        sp_text.add(i, body_vec[i])
    sp_vision = Space("vision")
    for i, v in img_vec.items():
        sp_vision.add(i, v)
    sp_pooled = Space("pooled")
    for i in ids:
        sp_pooled.add(i, body_vec[i])
    for i, v in img_vec.items():
        sp_pooled.add(i, v)

    cap, rk = args.rank_cap, args.rrf_k
    off = np.array([sp_vision.best_score(tq_vec[i]) for i in text_q])
    gate = float(off.mean() + args.gate_margin * off.std())

    def fuse(sigs: dict[str, list[int]]) -> list[int]:
        return [h.note_id for h in rrf_fuse(sigs, weights={s: 1.0 for s in sigs}, k=rk)]

    def run(qvec: dict) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {"a": [], "b": [], "c": [], "c_gate": []}
        for qid in qvec:
            q = qvec[qid]
            tr, vr, pr = sp_text.rank(q, cap), sp_vision.rank(q, cap), sp_pooled.rank(q)
            out["a"].append(tr)
            out["b"].append(pr)
            out["c"].append(fuse({"text": tr, "vision": vr}))
            sigs = {"text": tr}
            if sp_vision.best_score(q) >= gate:
                sigs["vision"] = vr
            out["c_gate"].append(fuse(sigs))
        return out

    tt = run(tq_vec)
    it = run(iq_vec)
    te, ie = list(text_q), list(image_q)
    k = args.k
    label = " [BLIND image bodies]" if args.blind_image_bodies else " [leaky labels]"
    print(
        f"\n=== JINA-CLIP-V2 CROSS-SPACE (corpus={len(ids)}, rrf_k={rk}, rank_cap={cap}, "
        f"gate={gate:.3f}){label} ==="
    )
    print(f"\n--- text-target queries ({len(te)}: answer in a TEXT note) ---")
    print(fmt("(a) text-only [jina text tower]", metrics(tt["a"], te, k), k))
    print(fmt("(b) single pooled space (ref)", metrics(tt["b"], te, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(tt["c"], te, k), k))
    print(fmt("(c+gate) + vision gate", metrics(tt["c_gate"], te, k), k))
    print(f"\n--- image-target queries ({len(ie)}: answer in the IMAGE) ---")
    print(fmt("(a) text-only [jina text tower]", metrics(it["a"], ie, k), k))
    print(fmt("(b) single pooled space (ref)", metrics(it["b"], ie, k), k))
    print(fmt("(c) cross-space RRF fused", metrics(it["c"], ie, k), k))
    print(fmt("(c+gate) + vision gate", metrics(it["c_gate"], ie, k), k))

    iso = [sp_vision.rank(iq_vec[i], cap) for i in image_q]
    print("\n--- isolating control: VISION space only (image vectors) ---")
    print(fmt("image-only retrieval", metrics(iso, ie, k), k))
    on = np.array([sp_vision.best_score(iq_vec[i]) for i in image_q])
    print("\n--- cross-space activation diagnostic (vision space best-cosine) ---")
    print(f"  image-target (on-topic):  mean={on.mean():.3f} std={on.std():.3f} min={on.min():.3f}")
    print(
        f"  text-target  (off-topic): mean={off.mean():.3f} std={off.std():.3f} max={off.max():.3f}"
    )
    print(
        f"  separation (on-off): {on.mean() - off.mean():+.3f}  gate(mean+{args.gate_margin}std)={gate:.3f}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
