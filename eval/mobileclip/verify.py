"""Spike #568 — verify a MobileCLIP2 ONNX export against the ``ClipBackend`` contract.

Part of #565 (profile (c) image leg). This script settles one question: can a real
**MobileCLIP2 ONNX export** load through Shrike's :class:`shrike.embedding_clip.ClipBackend`
(``_resolve_files`` + the native ``shrike_native.ClipEmbedder`` dual encoder) and produce
text + image vectors in one shared space?

What it does (all bytes fetched on demand into a **gitignored** cache — nothing committed):

1. Fetch the pinned MobileCLIP2 ONNX export (default S0 — the cheapest size) into the
   contract layout (``<dir>/text_model.onnx`` + ``vision_model.onnx`` + ``tokenizer.json`` +
   ``preprocessor_config.json``), reusing the suite's retry/backoff downloader.
2. Assert ``ClipBackend._resolve_files`` resolves all four files.
3. Instantiate ``ClipBackend``, ``start()`` it (loads both ONNX graphs natively), and
   embed a few texts + two real Commons images.
4. Sanity-check the shared space: text and image vectors share a dimension, and a text
   query is more similar to the matching image than to the mismatched one (cross-modal
   retrieval works through the gap).

Run (from the repo root, with the venv + native extension built — ``scripts/dev-setup.sh``)::

    python eval/mobileclip/verify.py            # default: S0
    python eval/mobileclip/verify.py --size s2  # S2 (better quality, larger)

Models: Apple MobileCLIP2 (Apple Sample Code License), ONNX export by ``plhery``; image
bytes from Wikimedia Commons. Source URLs + attribution: ``eval/mobileclip/ASSETS.md``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import httpx

# Reuse the suite's resilient downloader (retry/backoff on HF 429/5xx).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from model_cache import download_with_retry  # noqa: E402  # type: ignore[import-not-found]

# Wikimedia Commons rejects UA-less requests with 403; identify the eval (same string the
# multimodal eval uses), mirroring scripts/eval_multimodal.py.
_COMMONS_UA = {"User-Agent": "shrike-mobileclip-spike/0.1 (https://github.com/lathrys-at/shrike)"}

# --- pinned export ----------------------------------------------------------------
# plhery/mobileclip2-onnx — transformers.js-style ONNX export of Apple MobileCLIP2.
# Per-size subdirs each hold the four files the ClipBackend contract wants, flat.
# Pinned to an exact commit so the bytes (and thus the verdict) replay.
_REPO = "plhery/mobileclip2-onnx"
_REV = "ba95759a5bdbaca53e9111e2550a76ec09c8fd9e"
_BASE = f"https://huggingface.co/{_REPO}/resolve/{_REV}"

# tokenizer.json lives at the repo root (shared across sizes); the graphs +
# preprocessor_config.json live under onnx/<size>/.
_TOKENIZER_URL = f"{_BASE}/tokenizer.json"


def _size_files(size: str) -> dict[str, str]:
    """The four contract files for a given MobileCLIP2 size, arranged flat in one dir."""
    sub = f"{_BASE}/onnx/{size}"
    return {
        "text_model.onnx": f"{sub}/text_model.onnx",
        "vision_model.onnx": f"{sub}/vision_model.onnx",
        "preprocessor_config.json": f"{sub}/preprocessor_config.json",
        "tokenizer.json": _TOKENIZER_URL,
    }


# --- cross-modal sanity images (Wikimedia Commons; attributed in ASSETS.md) -------
# Reused from the existing eval corpora (same Commons files), well-licensed:
#   cat    — CC BY-SA 3.0 (Von.grzanka)
#   guitar — CC0 (Wilfredor)
# Pinned to the exact 960px thumbnails the multimodal eval already uses (Commons restricts
# thumbnail widths to an allow-list; these are proven-valid). Same Commons files as
# eval/multimodal/resolved_urls.json.
_IMAGES = {
    "cat": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/"
        "Felis_catus-cat_on_snow.jpg/960px-Felis_catus-cat_on_snow.jpg"
    ),
    "guitar": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/"
        "Man_playing_an_acoustic_brazilian_guitar_%28Viol%C3%A3o%29_on_Marco_Zero_"
        "Square%2C_Refice%2C_Pernambuco%2C_Brazil.jpg/960px-Man_playing_an_acoustic_"
        "brazilian_guitar_%28Viol%C3%A3o%29_on_Marco_Zero_Square%2C_Refice%2C_"
        "Pernambuco%2C_Brazil.jpg"
    ),
}


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent / "cache"


def _fetch_model(size: str) -> Path:
    """Download the four contract files for *size* into the gitignored cache; return the dir."""
    model_dir = _cache_dir() / f"mobileclip2-{size}-onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _size_files(size).items():
        dest = model_dir / name
        if not (dest.exists() and dest.stat().st_size > 0):
            print(f"  fetching {name} ...", flush=True)
            download_with_retry(url, dest, timeout=300.0)
    return model_dir


def _fetch_images() -> dict[str, bytes]:
    img_dir = _cache_dir() / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for name, url in _IMAGES.items():
        dest = img_dir / f"{name}.jpg"
        if not (dest.exists() and dest.stat().st_size > 0):
            print(f"  fetching image {name} ...", flush=True)
            resp = httpx.get(url, headers=_COMMONS_UA, follow_redirects=True, timeout=300.0)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        out[name] = dest.read_bytes()
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        default="s0",
        choices=("s0", "s2", "b", "l14"),
        help="MobileCLIP2 size to verify (default s0 — the cheapest).",
    )
    args = parser.parse_args()

    print(f"== MobileCLIP2 ONNX spike (#568) — size {args.size} ==")
    print(f"export: {_REPO}@{_REV[:8]}")

    print("[1/4] fetching model files into the gitignored cache ...")
    model_dir = _fetch_model(args.size)
    print(f"      model dir: {model_dir}")
    for p in sorted(model_dir.iterdir()):
        print(f"        {p.name}  ({p.stat().st_size:,} bytes)")

    # The contract gate: ClipBackend._resolve_files must find all four files.
    print("[2/4] asserting ClipBackend._resolve_files ...")
    from shrike.embedding_clip import ClipBackend

    backend = ClipBackend(model=str(model_dir))
    text_path, vis_path, tok_path, pp_path = backend._resolve_files()
    print(f"      text:   {text_path.name}")
    print(f"      vision: {vis_path.name}")
    print(f"      tok:    {tok_path.name}")
    print(f"      prep:   {pp_path.name}")

    print("[3/4] starting the native dual encoder (loads both ONNX graphs) ...")
    backend.start()
    assert backend.running, "backend failed to start"
    print(f"      fingerprint: {backend.model_fingerprint()}")
    print(f"      health: {backend.health()}")

    print("[4/4] embedding text + images, checking the shared space ...")
    texts = ["a photograph of a cat", "an acoustic guitar"]
    text_vecs = backend.embed_texts(texts)
    images = _fetch_images()
    img_vecs = backend.embed_images([images["cat"], images["guitar"]])

    text_dim = len(text_vecs[0])
    img_dim = len(img_vecs[0])
    print(f"      text dim={text_dim}  image dim={img_dim}")
    assert text_dim == img_dim, f"dim mismatch: text {text_dim} vs image {img_dim}"
    assert backend.embedding_dim() == text_dim

    # Cross-modal cosine grid: matching pairs should outscore mismatched ones.
    cat_txt, guitar_txt = text_vecs
    cat_img, guitar_img = img_vecs
    grid = {
        "cat_txt~cat_img": _cosine(cat_txt, cat_img),
        "cat_txt~guitar_img": _cosine(cat_txt, guitar_img),
        "guitar_txt~guitar_img": _cosine(guitar_txt, guitar_img),
        "guitar_txt~cat_img": _cosine(guitar_txt, cat_img),
    }
    for k, v in grid.items():
        print(f"      cos {k:24s} = {v:+.4f}")

    ok = (
        grid["cat_txt~cat_img"] > grid["cat_txt~guitar_img"]
        and grid["guitar_txt~guitar_img"] > grid["guitar_txt~cat_img"]
    )
    print()
    if ok:
        print(
            f"VERDICT: PASS — MobileCLIP2-{args.size.upper()} loads through ClipBackend as-is; "
            f"dual-encoder produces a shared {text_dim}-dim space with correct cross-modal ranking."
        )
        return 0
    print(
        "VERDICT: LOADED but cross-modal ranking did NOT separate — inspect grid above "
        "(graphs load, but the shared-space sanity failed)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
