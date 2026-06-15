# MobileCLIP2 ONNX vs the `ClipBackend` contract (spike #568)

## Verdict: **MATCHES AS-IS** — MobileCLIP2-S0/S2 ONNX loads through `ClipBackend` unchanged.

Part of #565 (profile (c) image leg). This spike settles whether a real **MobileCLIP2
ONNX export** can satisfy `ClipBackend._resolve_files` (`src/shrike/embedding_clip.py`)
and produce text + image vectors in one shared space — or whether profile (c)'s image
leg must fall back to the proven **jina-clip-v2 ONNX** (`eval/multimodal/RESULTS.md`).

It can. The `plhery/mobileclip2-onnx` export ships the exact transformers.js layout the
contract wants (per-size subdirs with a flat `text_model.onnx` + `vision_model.onnx` +
`preprocessor_config.json` + a repo-root `tokenizer.json`). No rename, no re-export.

## The export tried (pinned)

- **Repo:** [`plhery/mobileclip2-onnx`](https://huggingface.co/plhery/mobileclip2-onnx)
- **Revision:** `ba95759a5bdbaca53e9111e2550a76ec09c8fd9e`
- **Base models:** Apple MobileCLIP2 (S0, S2, B, L-14) — ONNX-converted with OpenCLIP.
- **License:** Apple Sample Code License (`apple-amlr`). See `ASSETS.md`.
- **Sizes verified:** **S0** (vision 45 MB) and **S2** (vision 143 MB). Both PASS.

Verified file URLs (S0; swap `s0`→`s2`/`b`/`l14` for the others):

| File | URL |
| --- | --- |
| `text_model.onnx` | `…/resolve/ba95759a/onnx/s0/text_model.onnx` (254 MB) |
| `vision_model.onnx` | `…/resolve/ba95759a/onnx/s0/vision_model.onnx` (45 MB) |
| `preprocessor_config.json` | `…/resolve/ba95759a/onnx/s0/preprocessor_config.json` |
| `tokenizer.json` | `…/resolve/ba95759a/tokenizer.json` (repo-root, shared) |

(`…` = `https://huggingface.co/plhery/mobileclip2-onnx`.)

## File layout observed (matches the contract exactly)

```
plhery/mobileclip2-onnx/
  tokenizer.json                  # CLIP BPE (<|startoftext|>=49406, <|endoftext|>=49407)
  preprocessor_config.json        # = onnx/<size>/ copy (top-level mirror)
  onnx/
    s0/  { text_model.onnx, vision_model.onnx, config.json, preprocessor_config.json }
    s2/  { text_model.onnx, vision_model.onnx, config.json, preprocessor_config.json }
    b/   { ... }
    l14/ { ... }
    text_model.onnx, vision_model.onnx   # default (== s2) top-level pair
```

`ClipBackend(model="…/onnx/s0")` resolves all four flat files there. (Pointing at the
repo root works too — root `tokenizer.json` + root `preprocessor_config.json` +
`onnx/text_model.onnx` + `onnx/vision_model.onnx`, the default == S2 — but for a clean
per-size identity, point at the size subdir.)

**Per-size pairs are independent vector spaces.** S0 and S2 ship different text encoders
(distinct SHA-256, both 254 MB, same 512-dim) and different vision encoders, so each size
is its own model — the index `model_id` (which folds both graph sizes) distinguishes them.

### `preprocessor_config.json` (S0; S2 identical)

```json
{
  "image_processor_type": "CLIPImageProcessor",
  "crop_size": {"height": 256, "width": 256},
  "do_center_crop": true, "do_normalize": true, "do_resize": true,
  "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
  "resample": 3, "size": {"shortest_edge": 256}
}
```

This is the HF `CLIPImageProcessor` JSON the backend reads. `_read_edge` handles the
`{"shortest_edge": 256}` / `{"height": 256, "width": 256}` dict forms. Note
`image_mean=(0,0,0)`, `image_std=(1,1,1)` — MobileCLIP2 expects 0–1 rescaled pixels
with **no** per-channel standardization (a no-op normalize), unlike the standard
OpenAI-CLIP mean/std. The backend reads these scalars verbatim and passes them to the
native engine, so the export's own preprocessing is honored. Image size is **256** (not
the usual 224) — also read from the config, no code change needed.

## Why it loads without a rename or re-export

The native `shrike_native.ClipEmbedder` (`native/shrike-embed/src/clip.rs`) is tolerant
by construction:

- **Text inputs are graph-discovered**, not hard-coded: `graph_inputs(session, ["input_ids",
  "attention_mask"])` feeds whichever the graph declares (at its declared int width). The
  MobileCLIP2 text graph declares `input_ids` → it's fed. No `position_ids` requirement.
- **Vision input is taken positionally** (`vision_session.inputs().first()`), so the
  conventional `pixel_values` name is accepted without a name check.
- **Outputs are read positionally** (`outputs[0]` as a rank-2 f32 `[batch, dim]`), so the
  export's `text_embeds`/`image_embeds` output names don't matter.
- **The engine L2-normalizes itself** — and the export ships *unnormalized* embeddings
  (per its README), which is exactly right (no double-normalization).
- **Context is fixed at 77 tokens** (CLIP positional limit); the export's CLIP tokenizer
  pads/truncates to 77 cleanly.

## Run results (verified locally, `eval/mobileclip/verify.py`)

Both sizes: `_resolve_files` resolved all four files, the native dual encoder loaded both
graphs, and the shared space is well-formed (text dim == image dim == **512**, and the
batch-safety probe cleared the **fp32** export for **batched** embedding).

Cross-modal cosine grid (text query × image), `cat` photo + `guitar` photo (Commons):

| pair | **S0** | **S2** |
| --- | --- | --- |
| cat text ↔ **cat** image | **+0.2657** | **+0.2932** |
| cat text ↔ guitar image | +0.1206 | +0.1060 |
| guitar text ↔ **guitar** image | **+0.2239** | **+0.2617** |
| guitar text ↔ cat image | +0.0774 | +0.0725 |

The matching text↔image pair outscores the mismatched one in every row (correct
cross-modal retrieval), with the textbook CLIP modality gap (~0.25–0.29 matching,
~0.07–0.12 mismatched) — exactly the constant-offset regime the project's per-modality
RRF + activation-gate design already handles. S2 separates a touch more cleanly than S0.

Backend fingerprint (S0): `clip-rs:text_model.onnx:254053669:vision_model.onnx:45555784:imgprep=rs1:textprep=1`.

## Reproduce

From the repo root, with the venv + native extension built (`scripts/dev-setup.sh`):

```bash
python eval/mobileclip/verify.py --size s0   # 45 MB vision + 254 MB text; cheapest
python eval/mobileclip/verify.py --size s2   # 143 MB vision; better separation
```

The script fetches the pinned export + two Commons images on demand into the gitignored
`eval/mobileclip/cache/` (no model bytes committed) and prints the contract + embed checks.

## Recommendation for the profile-(c) image leg

- **MobileCLIP2 is viable as the named target.** Wire the **`plhery/mobileclip2-onnx`**
  export (size **S2** recommended for the quality/size balance; **S0** if image-leg footprint
  matters most) into `model_cache.py` / `MODULE.bazel` as an `@model_*` external the same way
  jina-clip-v2 / the CI CLIP fixture are — a `ClipBackend` `[text, image]` entry, no contract
  change. **(Out of scope for this spike — that integration is the profile-(c) image-leg wave.)**
- **One caveat to weigh**, not a blocker: the text encoder is **254 MB** (the same size across
  S0/S2), so the "mobile-fit" framing applies to the *vision* tower (45–143 MB), not the
  fp32 text tower. If a smaller image-leg footprint is wanted, an int8/fp16 re-export (the
  contract's variant-suffix resolver would pick it up) or the proven jina-clip-v2 fallback
  remain options. As-is, fp32 is the quality-best and it **batches** (no serial penalty).
- **License:** `apple-amlr` (Apple Sample Code License) is a research/sample-code license —
  confirm it's acceptable for a checked-in capability profile (model bytes are never committed;
  fetched at runtime, as with every other model). The OpenCLIP-derived
  `RuteNL/MobileCLIP2-*-OpenCLIP-ONNX` exports carry the same Apple base-model license.

## Alternatives surveyed (not used)

| Repo | Layout | Fits as-is? |
| --- | --- | --- |
| **`plhery/mobileclip2-onnx`** | `text_model.onnx`+`vision_model.onnx`+`tokenizer.json`+`preprocessor_config.json` (per size) | **YES** ✓ (chosen) |
| `RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX` | `text.onnx`/`visual.onnx` (+ `.onnx.data`), no `preprocessor_config.json` | No — needs rename + a synthesized preprocessor config |
| `ipsilondev/mobileclip2-s2-onnx` | `text_model.onnx`/`vision_model.onnx`+`preprocessor_config.json` but **no `tokenizer.json`** (only `vocab.json`+`merges.txt`) | No — needs a `tokenizer.json` built |
| `apple/MobileCLIP2-*`, `timm/MobileCLIP2-*-OpenCLIP` | PyTorch/`safetensors` (+ tflite/coreml on others) | No — not ONNX dual-graph |

So a fitting MobileCLIP2 ONNX export **exists today and loads unchanged**; the other ONNX
exports would need trivial-to-moderate fixups (rename / synthesize a preprocessor config /
build a tokenizer.json), and the upstream Apple/timm repos aren't ONNX at all.
