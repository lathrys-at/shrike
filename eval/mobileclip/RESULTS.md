# MobileCLIP2 ONNX vs the `ClipBackend` contract (spike #568)

Status: **IN PROGRESS** — verdict pending.

Part of #565 (profile (c) image leg). This spike answers one question: can a real
**MobileCLIP2 ONNX export** satisfy `ClipBackend._resolve_files`
(`src/shrike/embedding_clip.py`) as-is, after a rename/re-export, or not at all
(in which case profile (c)'s image leg uses the proven **jina-clip-v2 ONNX**
fallback, `eval/multimodal/RESULTS.md`)?

## The contract being verified

`ClipBackend._resolve_files` requires, in ONE model dir (it tries `root/onnx/<name>`
then `root/<name>` for each):

- `text_model<suffix>.onnx` **and** `vision_model<suffix>.onnx` — a *matching*
  variant-suffix pair (`_VARIANT_SUFFIXES = ("", "_fp16", "_quantized", "_int8",
  "_uint8", "_q4", "_bnb4")`), first precision for which **both** exist wins.
- `tokenizer.json` (HF fast-tokenizer JSON).
- `preprocessor_config.json` (HF image processor JSON; the backend reads
  `image_mean`, `image_std`, `size`, `crop_size`).

The native engine (`shrike_native.ClipEmbedder`) then loads both graphs with the
text graph taking `input_ids` and the vision graph taking `pixel_values`, both
projecting into one cosine-comparable space.

(Body filled in by the spike — see commits on `spike/568-mobileclip2-onnx`.)
