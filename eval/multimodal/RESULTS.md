# Multimodal eval results — image↔text via CLIP (Phase 3a of #162)

Reproduce: `pip install sentence-transformers torch transformers==4.49.0 pillow einops timm`
then `python scripts/eval_multimodal.py`. (jina-clip-v2's `trust_remote_code` needs
transformers **4.x** — 5.x removed `clip_loss`. On a Python built without `_lzma`, torchvision
fails to import; the ONNX build path avoids torch entirely.)

## Setup

- Model: **jina-clip-v2** (dual-encoder, 1024-dim, Matryoshka-truncatable) via sentence-transformers.
- 16 image notes (real Wikimedia images resolved by `search_term`; the note `text` is generic so
  the *image* carries the query's meaning) + 12 text notes. USearch `multi=True` keyed by `note_id`.
- Baseline: the project's ONNX MiniLM (`OnnxBackend`) on the text notes.

## Numbers

| index (16 image notes, dim=1024) | R@1 | R@5 | MRR |
|----------------------------------|-----|-----|-----|
| text-only (note text)            | 0.81 | 1.00 | 0.90 |
| **image-only**                   | **1.00** | **1.00** | **1.00** |
| multi (text + image)             | 0.81 | 1.00 | 0.90 |

Text-by-text (12 notes): jina-clip-v2 text **R@1=1.00** == MiniLM **R@1=1.00**.
Serving: ~0.7 s/item embed, 163 s first load (compiles, then cached), 0.9B params, dim 1024.

## Findings / decisions

1. **Image-by-text works → GO.** The image-only index is perfect (R@1=1.0): a text query
   reliably retrieves the right image. The core capability the epic targets is real.
2. **No text-retrieval regression (epic problem #1).** jina-clip-v2's text encoder matches the
   MiniLM baseline on the text set (both saturate — a larger eval would differentiate, but no
   regression is visible). The shared space does not cost text quality here.
3. **Embedding unit = multi-vector (`multi=True`), not fusion-into-one (epic problem #3).** A note
   maps to its text vector + one vector per image, all under the `note_id` key. `remove(note_id)`
   drops all of a note's vectors; search returns `note_id`; results dedup to the best vector per
   note. Recall is excellent (R@5=1.0).
4. **The CLIP modality gap is the real design consideration.** `multi` R@1 (0.81) == text-only,
   **below** image-only (1.0): a text query is closer to *text* vectors than image vectors, so in
   one cosine index the text vectors win rank-1 and image hits are **additive** (they catch what
   text misses; R@5 stays 1.0) rather than dominant. So the build ships `multi=True` for recall,
   and **rank quality across the gap — fusing/weighting text vs image hits, with per-result
   provenance — is exactly the Search epic (#180 rank fusion, #182 provenance).** Multimodal and
   Search converge; the fusion backbone serves both.
5. **Serving = the ONNX seam (epic problem #2), to confirm in the build.** The eval used the heavy
   transformers path (0.9B, torch). The build targets the in-process `OnnxBackend` seam —
   jina-clip-v2 ships ONNX exports (separate text + vision graphs + 512×512 image preprocessing);
   3b confirms input names/preprocessing and wires a `ClipBackend`. Plus a **small CLIP for CI**
   (jina-clip-v2 is too heavy to pin) — to pick in 3b.

## Gate: **GO.**

Build proceeds (each its own PR, gated here): `ClipBackend` (ONNX, `modalities={text,image}`) →
multi-vector index (`multi=True`) + media-aware reconcile (hash text + image content; vision model
in `model_id`) → search provenance (which modality surfaced each result), with rank-fusion shared
with / deferred to the Search epic.
