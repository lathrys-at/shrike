# Capability profiles (`scripts/profiles/*.yml`)

Checked-in, **path-free** capability declarations the offline-integration
dogfooding launcher (`//scripts:serve`, #565) boots a real Shrike server from.
Each profile names a capability *shape* — what embedding spaces and recognizers
to run — with no machine-absolute paths and no `collection:` key (run paths ride
as flags; bundled models are bare dir-names the launcher materializes).

| Profile | Shape | CI |
|---------|-------|----|
| `text-onnx` | one text-only ONNX space (pinned MiniLM int8) | runs in CI |
| `onnx-multispace` | two ONNX spaces: embeddinggemma text + MobileCLIP2 image | runs in CI |
| `jina-text-clip` | a managed-llama text space (jina-v5-text-nano) + an ONNX CLIP image space (MobileCLIP2) | **manual / local-only** |
| `jina-omni` | one text+image space on a managed llama-server (jina-v5-omni) | **manual / local-only** |

The CI-running profiles boot directly under the launcher:

```bash
./bazel run //scripts:serve -- --profile text-onnx [--seed qa] [--foreground|--daemon]
./bazel run //scripts:serve -- --profile onnx-multispace [--seed qa]
```

`jina-text-clip` and `jina-omni` are **not** served via `serve --profile` (see
their sections below) — their binaries and models are operator-provided, so each
is instantiated and run via `shrike server start --config`.

## `jina-text-clip` — manual / local-only (never in CI)

`jina-text-clip` runs a **HYBRID over two embedding spaces**, RRF-fused at search
time:

- a **dedicated TEXT space** — [jina-embeddings-v5-text-nano][model] on a managed
  llama-server (a high-quality, text-only model); and
- a **separate IMAGE space** — MobileCLIP2-S2 via the in-process ONNX
  `ClipBackend` (text + image into one shared space, so a text query retrieves a
  card by the content of its image).

### How it differs from `jina-omni`

Both let a text query find a card by its image, but the mechanism differs:

- **`jina-omni`** uses ONE shared text+image space (a multimodal *omni* model):
  text and images embed into the same vectors, **no modality gap**, no second
  sub-index — but it needs a multimodal model (and the patched fork + F16 GGUF).
- **`jina-text-clip`** pairs a **dedicated text model** (stronger text recall
  than an omni's text head) with a **separate ONNX CLIP image leg**: two spaces,
  fused as distinct per-modality RRF signals. The image leg is **in-process
  ONNX** — no GPU subprocess for images, no vision projector on the server. The
  CLIP modality gap is invisible to the rank-based fusion (the image signal joins
  on its calibrated activation floor). Exactly **one image-embedding space**
  (#580): MobileCLIP2 is the single image leg; jina-text is text-only.

Pick `jina-text-clip` when you want the best dedicated-text recall plus
search-by-image-content and prefer the image encoder off-server; pick `jina-omni`
when you want a single gapless text+image space.

### Why manual / local-only

The jina-text leg needs a **hand-built llama-server**: `jina-embeddings-v5-text`
is a **custom architecture** (`JinaEmbeddingsV5Model`) the official / pinned
llama-server does not recognize (the GGUF arch tag isn't in upstream llama.cpp).
Both the binary and the model are **operator-provided** — neither is a Bazel
external and neither is fetched on the CI lane (the patched binary can't be
pinned or cached), so this profile is gated out of CI exactly like `jina-omni`
and the `#501` image-embedding harness.

### The two hard constraints (jina-text leg)

1. **Patched binary.** Build the [`jina-ai/llama.cpp`][fork] fork — upstream
   llama.cpp does not yet support the `jina-embeddings-v5-text` architecture.
   Point `managed.llama_server.binary` at it.
2. **`--pooling last`.** jina-v5-text-nano is a last-token model whose pooling
   type isn't in the GGUF metadata; without `pooling: last` llama-server defaults
   to mean and produces **wrong** embeddings. The profile sets it on the entry.

Unlike `jina-omni` there is **no** image-embed through the managed server (no
`mmproj`, no F16-vs-quant `ggml_get_f32_1d` abort) — images go through the ONNX
CLIP leg. So the jina-text GGUF **may be a quant**: **Q5_K_M is recommended for
nano** (Q4_K_M drops very-short-input parity below ~0.9 cosine per jina's GGUF
card; Q5_K_M or higher keeps it).

### 1. Build the patched llama-server

```bash
git clone https://github.com/jina-ai/llama.cpp.git jina-llama.cpp
cd jina-llama.cpp && cmake -B build && cmake --build build --config Release -j
# the binary lands at build/bin/llama-server
# (macOS: it needs DYLD_LIBRARY_PATH=<build>/bin for the sibling dylibs)
```

> Check the jina-ai/llama.cpp branches for the one that carries the
> `jina-embeddings-v5-text` architecture support, and read the
> [model card][model] / the linked blog post for the current build recipe — the
> exact branch name moves as patches land upstream.

### 2. Get the jina-text GGUF + the MobileCLIP2-S2 ONNX dir

- **jina-text GGUF.** From [`jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF`][model]
  download a `*-Q5_K_M.gguf` (retrieval is the right task variant for a search
  corpus; Q5_K_M is the recommended nano quant). Place it anywhere local; nothing
  commits these bytes.
- **MobileCLIP2-S2 ONNX.** The flat `ClipBackend` layout — `text_model.onnx`,
  `vision_model.onnx`, `preprocessor_config.json`, `tokenizer.json` in one dir.
  The shared test-model cache already knows how to fetch the exact pinned export
  (`tests/integration/model_cache.cached_mobileclip2_model_dir`) if you want one
  download home:

  ```bash
  python -c "from pathlib import Path; \
    from tests.integration.model_cache import cached_mobileclip2_model_dir; \
    print(cached_mobileclip2_model_dir(Path.home() / '.cache' / 'shrike-test-models'))"
  ```

### 3. Point the profile at them and run

The committed `jina-text-clip.yml` is **path-free**: the three operator-provided
paths are `${ENV}` placeholders, so the template stays portable. Export the vars
and expand them into a local (gitignored) config the daemon reads:

```bash
export SHRIKE_JINA_TEXT_LLAMA_SERVER=/path/to/jina-llama.cpp/build/bin/llama-server
export SHRIKE_JINA_TEXT_MODEL=/path/to/jina-embeddings-v5-text-nano-retrieval-Q5_K_M.gguf
export SHRIKE_JINA_TEXT_CLIP_MOBILECLIP2=/path/to/mobileclip2-s2-onnx

# Instantiate the path-free template into a local config (envsubst expands the
# three vars); the committed YAML stays placeholder-only and path-free.
envsubst < scripts/profiles/jina-text-clip.yml > /tmp/jina-text-clip.local.yml

# Boot a server against a fresh empty collection with that config.
shrike --config /tmp/jina-text-clip.local.yml \
  server start --collection /tmp/jina-text-clip-run/collection.anki2 --foreground
```

> The `//scripts:serve` launcher only materializes **onnx** model dir-names from
> Bazel externals and leaves managed/remote entries untouched; it also can't
> inject the operator-provided patched binary (no env-substitution for v2 config
> values). So serve this profile via `shrike server start --config` with the
> instantiated config above, **not** `serve --profile jina-text-clip`.

When the `jina-embeddings-v5-text` patches land in upstream llama.cpp (so the
pinned `llama-server` recognizes the arch) **and** a dedicated text model becomes
a server default, this profile graduates to a pinned-fixture CI test (the GGUF
rides the shared model cache like every other fixture; only the binary blocks
CI today).

[model]: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF
[fork]: https://github.com/jina-ai/llama.cpp

## `jina-omni` — manual / local-only (never in CI)

`jina-omni` runs **one multimodal embedding space** (text + image) over
[jina-embeddings-v5-omni-nano][omni-model] on a managed llama-server: a text query
can retrieve a card by the content of its image, because text and images embed
into the **same** vector space (no modality gap, no second sub-index — unlike
the CLIP-shaped profiles).

It is **manual and local-only**: it needs a hand-built llama-server and an
unquantized model, both **operator-provided**. Neither is a Bazel external and
neither is fetched on the CI lane — the patched binary can't be pinned or
cached, so this profile is gated out of CI exactly like the `#501`
image-embedding harness (`tests/integration/test_multimodal.py`).

### The three hard constraints

Verified end to end (macOS/Metal); the full forensics live in the
`jina-v5-omni-image-embed` memory note.

1. **Patched binary.** The official / pinned llama-server loads the model and
   serves *text* embeds, then **segfaults** during image-embed extraction
   (verified on `b9415` and `b9616`). You must build the
   [`jina-ai/llama.cpp` `feat-v5-omni`][omni-fork] fork.
2. **F16 GGUF, not a quant.** The fork reads the token-embedding tensor
   element-by-element with `ggml_get_f32_1d`, which **aborts** on a
   block-quantized type. `Q4_K_M` *and* `Q5_K_M` (the quant jina's own GGUF card
   recommends) both crash on the first image embed; the `*-F16.gguf` text
   variant keeps the table readable.
3. **`--pooling last`.** jina-v5-omni is a last-token model whose pooling type
   isn't in the GGUF metadata; without `pooling: last` llama-server defaults to
   mean and produces **wrong** embeddings. The profile sets it.

### 1. Build the patched llama-server

```bash
git clone --branch feat-v5-omni https://github.com/jina-ai/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build --config Release -j
# the binary lands at build/bin/llama-server
# (macOS: it needs DYLD_LIBRARY_PATH=<build>/bin for the sibling dylibs;
#  Hopper GPUs need GGML_CUDA_DISABLE_GRAPHS=1; CPU/Metal/Vulkan are fine)
```

### 2. Get the F16 text GGUF + vision projector

From [`jinaai/jina-embeddings-v5-omni-nano-classification-GGUF`][omni-model] download
the `*-F16.gguf` (text — **must be unquantized**, constraint 2) and the
`*-vision-mmproj-F16.gguf` (the vision projector). Place them anywhere local;
nothing commits these bytes. The shared test-model cache already knows how to
fetch them (`tests/integration/model_cache.cached_multimodal_model_dir`) if you
prefer one download home:

```bash
python -c "from pathlib import Path; \
  from tests.integration.model_cache import cached_multimodal_model_dir; \
  print(cached_multimodal_model_dir(Path.home() / '.cache' / 'shrike-test-models'))"
```

### 3. Point the profile at them and run

The committed `jina-omni.yml` is **path-free**: the three operator-provided
paths are `${ENV}` placeholders, so the template stays portable. Export the vars
and expand them into a local (gitignored) config the daemon reads:

```bash
export SHRIKE_JINA_OMNI_LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
export SHRIKE_JINA_OMNI_MODEL=/path/to/jina-...-classification-F16.gguf
export SHRIKE_JINA_OMNI_VISION_MMPROJ=/path/to/jina-...-vision-mmproj-F16.gguf

# Instantiate the path-free template into a local config (envsubst expands the
# three vars); the committed YAML stays placeholder-only and path-free.
envsubst < scripts/profiles/jina-omni.yml > /tmp/jina-omni.local.yml

# Boot a server against a fresh empty collection with that config.
shrike --config /tmp/jina-omni.local.yml \
  server start --collection /tmp/jina-omni-run/collection.anki2 --foreground
```

> The `//scripts:serve` launcher only materializes **onnx** model dir-names from
> Bazel externals; it leaves managed/remote entries untouched. Because
> `jina-omni`'s binary and model are operator-provided (not externals), serve it
> via `shrike server start --config` with the instantiated config above, not
> `serve --profile jina-omni`.

When a multimodal embedding model becomes a server default (the #237 eval) **and**
the fork's patches land upstream, this profile graduates to a pinned-fixture CI
test (re-test the quant question against the then-current fork — the
`ggml_get_f32_1d` abort may get fixed, re-enabling a smaller quant).

[omni-model]: https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-classification-GGUF
[omni-fork]: https://github.com/jina-ai/llama.cpp/tree/feat-v5-omni
