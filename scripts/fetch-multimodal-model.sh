#!/usr/bin/env bash
# Fetch the multimodal embedding model for the manual #501 image-embed harness
# (tests/integration/test_multimodal.py) into the shared test-model cache — the
# same `~/.cache/shrike-test-models` (or $SHRIKE_TEST_MODEL_DIR) the other
# fixtures use. ~625 MB (jina-embeddings-v5-omni nano: F16 text GGUF + vision
# mmproj). Idempotent: already-present files are skipped.
#
# This only fetches the MODEL. The harness also needs a PATCHED llama-server
# (the official/pinned one segfaults on image embeddings) built from
# jina-ai/llama.cpp `feat-v5-omni` — see the recipe printed at the end and in
# test_multimodal.py.
#
#   scripts/fetch-multimodal-model.sh        # download/use cached
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${SHRIKE_TEST_MODEL_DIR:-$HOME/.cache/shrike-test-models}"
mkdir -p "$cache"

# Prefer the repo venv (it has httpx, which model_cache imports); fall back to
# $PYTHON or python3 for an already-activated env.
if [ -x "$repo_root/.venv/bin/python" ]; then
    py="$repo_root/.venv/bin/python"
else
    py="${PYTHON:-python3}"
fi

# Reuse model_cache's retry/backoff download + the canonical filenames, so the
# layout and URLs have one source of truth (no re-spelling here).
SHRIKE_TEST_MODEL_DIR="$cache" PYTHONPATH="$repo_root:${PYTHONPATH:-}" "$py" - <<'PY'
from pathlib import Path
from tests.integration.model_cache import (
    MULTIMODAL_TEXT_NAME,
    MULTIMODAL_VISION_MMPROJ_NAME,
    cached_multimodal_model_dir,
)

# fallback_dir is unused when SHRIKE_TEST_MODEL_DIR is set (it is, above).
model_dir = cached_multimodal_model_dir(Path("/tmp/unused-multimodal-fallback"))
text = model_dir / MULTIMODAL_TEXT_NAME
vision = model_dir / MULTIMODAL_VISION_MMPROJ_NAME
print("\nMultimodal model ready in the cache:")
print(f"  text:   {text}")
print(f"  vision: {vision}")
print("\nRun the harness (after building the patched llama-server):")
print(f"  export SHRIKE_MULTIMODAL_MODEL={text}")
print(f"  export SHRIKE_MULTIMODAL_VISION_MMPROJ={vision}")
print("  export SHRIKE_MULTIMODAL_LLAMA_SERVER=/path/to/jina-ai-llama.cpp/build/bin/llama-server")
print("  pytest tests/integration/test_multimodal.py -v -m multimodal")
PY

cat <<'EOF'

Build the patched llama-server (model needs jina-v5-omni patches not yet upstream):
  git clone --branch feat-v5-omni https://github.com/jina-ai/llama.cpp.git
  cd llama.cpp && cmake -B build && cmake --build build --config Release -j
  # binary: build/bin/llama-server   (Hopper GPUs: GGML_CUDA_DISABLE_GRAPHS=1)
EOF
