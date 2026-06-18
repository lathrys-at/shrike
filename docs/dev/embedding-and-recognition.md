# Embedding and recognition

These are the kernel's two injected capabilities. The embedder turns notes into
vectors; the recognizer turns note media into searchable text (OCR today, ASR
and image-describe seamed for later). Both are off until the harness attaches a
backend, and both can be cycled independently of the server.

## The embedding service lifecycle

`EmbeddingRuntime` (`harness/engines/embedding/runtime.py`) owns the current
backend (or `None`), the parameters needed to (re)start it, and the binding to
the index. It serializes start/stop under a lock.

The vector index is **always** created at boot, even with no embedder: it loads
any on-disk vectors and reports `unavailable` until a backend is attached.

- `shrike server start` starts embedding at boot if a model is configured, unless
  `--no-embedding` is passed.
- `shrike server embedding start` / `stop` cycle the service on a running server
  (for llama-server upgrades, model swaps, or freeing GPU/RAM). Stopping marks
  the index `unavailable` but keeps the on-disk vectors; starting re-attaches and
  rebuilds only if the model changed or the index drifted.

## Pluggable backends behind one protocol

The operational surface depends only on the minimal `EmbedderBackend` protocol
(`harness/engines/embedding/base.py`): `embed_texts`, `embedding_dim`,
`model_fingerprint`, `health`, lifecycle, and `modalities`. Every production
backend also hands the kernel a native composition (`native_embedder()`), so
`embed_texts` serves direct callers (the probe, tests) while kernel embeds run
crate-side end to end.

| Backend | What it is |
|---------|------------|
| `LlamaServerBackend` | A managed llama-server subprocess (GGUF/MLX). `EmbeddingService` is a back-compat alias. |
| `RemoteBackend` | An endpoint Shrike does not run — a config `endpoint` with `api_key_env`, or `managed.llama_server.manage: attach`. Start proves connectivity with one embed; it never spawns or stops the server. |
| `OnnxBackend` | In-process onnxruntime + tokenizers, text-only. |
| `ClipBackend` | In-process CLIP dual-encoder for image↔text; `modalities={text, image}`. |

The kernel's drift, hashing, and persistence machinery is backend-agnostic — only
the embed contract is called.

**Choosing a backend:** in-process ONNX for text-only collections (no subprocess,
port, or orphan-reaping; ~1 ms single-note upserts); llama-server for GGUF/MLX and
GPU offload; the CLIP shape (a `[text, image]` ONNX entry) for the multimodal path.

`modalities` is the graceful-degradation seam. Every backend advertises a
`frozenset[str]` of what it can embed. **Text-only is a permanent, first-class
capability** (the suites rely on small text-only models); a multimodal backend
advertises more. Search over media-by-content lights up where vectors
exist and quietly returns nothing where they don't — never an error.

### The CLIP backend

Multimodal search — a text query retrieving a card by the content of its image —
needs image and text in *one* vector space. `ClipBackend` loads two ONNX graphs
(text: `input_ids → text_embeds`; vision: `pixel_values → image_embeds`), both
projecting into the same space (L2-normalize, no pooling). Image preprocessing
(resize → center-crop → rescale → normalize) is read from the model's
`preprocessor_config.json` and done in PIL + numpy — no torch. It reuses
`OnnxBackend`'s provider resolution and the batch probe (the two graphs share one
export and one quantization, so a single text-path probe governs both), and adds
`embed_images()`.

The **modality gap** (text-text cosine ~0.7 vs text-image ~0.3) is why image
vectors are not simply mixed into the text ranking; see
[`indexing-and-search.md`](indexing-and-search.md) for how per-modality
sub-indexes and rank fusion handle it.

### ONNX specifics

The model is an ONNX directory (`model.onnx` + `tokenizer.json`, or
`onnx/model.onnx`) or a `.onnx` file with `tokenizer.json` beside it. Pooling
(`mean|cls|last`) and optional L2 normalization happen in numpy.

- **Pooling is vector-affecting, so it is folded into the fingerprint.**
  Normalization is scale-only (USearch's `cos` metric is scale-invariant), so it
  is deliberately *not* folded in — the same reasoning that makes llama's
  `--embd-normalize` moot. Fingerprints are namespaced by family (`onnx:…` vs
  llama's `meta:`/`file:…`), so the same model under two runtimes never shares a
  vector space.
- **Execution providers** (`--embedding-onnx-provider`, repeatable; default CPU)
  are resolved gracefully: intersected with `get_available_providers()`, an
  unavailable one dropped *with a warning*, CPU always appended as the final
  fallback. The **actually-loaded** provider is surfaced in `health()` and
  `server status`, so a silent CPU fallback is visible.
- **Packaging** mirrors onnxruntime's wheels: base onnxruntime (CPU + CoreML on
  macOS) is a hard dependency; the `gpu` extra installs `onnxruntime-gpu`
  *instead of* the base (they conflict); DirectML is a manual
  `onnxruntime-directml`.

### Batch safety is probed, not assumed

int8 ONNX exports use dynamic quantization, whose activation scales are computed
over the whole batch tensor — so a *batched* embed makes a note's vector depend on
its batch-mates' content. That would break the core index invariant that a
`reconcile` produces the same vectors as a full rebuild. fp32/fp16 ONNX (and
llama-server, fp) are bit-exact when batched, so the variance is int8-only.

Rather than guess from a model's quantization scheme, **every backend's `start()`
runs `probe_max_safe_batch`** (`harness/engines/embedding/batching.py`): embed a
magnitude-spiked probe set serially, then all in one batch, and compare within a
tolerance chosen to sit above float noise and below quant drift. `embed_texts`
then batches up to the proven probe-set size (further capped by
`--embedding-batch-size`), or serially (batch size 1) for a batch-variant model —
so "proven safe" and "what we batch" are the same size. The probe set is spiked
for activation magnitude (long, numeric, symbol, repeated, mixed-script inputs),
because int8 drift is magnitude-driven, not length-driven. This is locked by
exact-equality tests against real int8 (serial) and fp32 (batched == serial)
models.

## Model fingerprint and the embedding text

`model_id` comes from llama-server's `GET /v1/models` `meta` block (`n_params`,
`n_embd`, `n_vocab`, `n_ctx_train`, `size`) — fast, and it describes the *loaded*
model. It falls back to filename + on-disk size if absent. The model *name* is
deliberately excluded: it would force needless rebuilds on rename and miss
same-name re-quantizations the numeric fields catch.

Two suffixes are appended:

- An explicitly-set pooling type (`…:pool=last`), since it changes every vector
  but isn't in the metadata. Omitted when unset, so older indexes still match.
- The note-text normalization version (`…:textprep=N`, `EMBED_TEXT_VERSION` in
  `harness/engines/embedding/text.py`), appended **unconditionally** — the cleaned
  text is as much a part of the vector space as the model.

`embed_text.normalize_for_embedding()` turns each raw Anki field value into stable
plain text. It works on field *values*, not rendered cards: a note (not a card) is
the embedding unit, and templates add presentational scaffolding that is noise. It
delegates the HTML→text step to Anki's own `strip_html` (Rust-backed, robust on
malformed markup), and around that it reveals cloze (`{{c1::France}}` → `France`),
drops MathJax/LaTeX wrappers keeping the inner source, drops `[sound:…]`, and
converts block tags to spaces before stripping. The result is a function of the
field value and the pinned Anki version's stripper — identical whether freshly
upserted or re-read during a rebuild. **Bump `EMBED_TEXT_VERSION` whenever the
normalized output changes**, including an Anki upgrade whose stripping differs.

## Recognition (OCR)

Recognition is the kernel's second injected capability — the same slot pattern as
the embedder. An OCR engine the harness attaches turns note media into searchable
text. Off by default.

The **server build does not compile the Apple Vision engine** (the platform
engines are mobile-only on every OS; the server's replacement is the remote
recognizer). Selecting `--ocr-backend apple` on a server degrades the recognition
state to `error` without disturbing boot, exactly like the off-macOS case always
did. The engine impl (`shrike-engine::apple`, in mobile builds) parses the raw
JSON the glue crate `shrike-platform` returns; that glue is Swift behind a C ABI
driving Apple's `RecognizeTextRequest` (macOS 15+). Building it on macOS needs
full Xcode, which the server build no longer pays for.

The Python contract is `RecognizerBackend`
(`harness/engines/recognition/recognition.py`): a blocking
`recognize(items: list[bytes]) -> list[tuple[str, float, str]]` (text,
confidence, segments-JSON) plus `model_fingerprint()`. `PyRecognizer.capture`
bridges it to the kernel for the custom/test seam.

### One pass, many consumers

The kernel's `recognize_pending(max_items)` sweeps bounded batches of pending
(note, image) pairs. *Pending* means a resolvable image with no OCR row and no
below-gate marker — a gate-dropped item is recorded in the derived store's
`gated` table so it is judged once, not re-OCR'd every sweep — or everything
after the recognizer *fingerprint* changes (an OS upgrade re-derives rows and
markers, like a model change rebuilds vectors).

Each pass persists both:

- the flattened text as derived rows (`source='ocr'`), feeding substring/fuzzy
  search and provenance;
- the per-segment structure (the `segments` table; boxes today).

A `RecognitionGate` applies a confidence-and-substance bar to store text at all,
and a higher substance bar to mint a vector. Gated text embeds via the *text*
encoder as extra vectors under the note key in the `text` space (so there is no
modality gap, and max-over-items ranking falls out). The per-note fingerprint
folds the OCR text — byte-identical when there is none, so upgrades never
spuriously rebuild. The harness drives sweeps in the background
(`recognition_sweep`); `/status` carries `recognition: {state, backend}`.
