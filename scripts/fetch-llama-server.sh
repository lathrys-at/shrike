#!/usr/bin/env bash
#
# Download the latest llama-server binary and a small embedding model.
#
# Everything is cached in .cache/ at the project root. Prints the paths
# and a PATH export you can eval or paste into your shell.
#
# Usage:
#   ./scripts/fetch-llama-server.sh           # download/use cached
#   ./scripts/fetch-llama-server.sh --fresh   # re-download everything
#   eval "$(./scripts/fetch-llama-server.sh)" # export PATH directly
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache"
LLAMA_DIR="$CACHE/llama-server"
MODEL_DIR="$CACHE/models"
MODEL_NAME="all-MiniLM-L6-v2-Q4_0.gguf"
MODEL_URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/$MODEL_NAME"

for arg in "$@"; do
    if [[ "$arg" == "--fresh" ]]; then
        echo "Cleaning cache..." >&2
        rm -rf "$LLAMA_DIR" "$MODEL_DIR"
    fi
done

# -- llama-server --

if [[ -x "$LLAMA_DIR/llama-server" ]]; then
    echo "Using cached llama-server" >&2
else
    echo "Downloading llama-server..." >&2
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   PLATFORM="ubuntu-x64" ;;
        Linux-aarch64)  PLATFORM="ubuntu-arm64" ;;
        Darwin-arm64)   PLATFORM="macos-arm64" ;;
        Darwin-x86_64)  PLATFORM="macos-x64" ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2 && exit 1 ;;
    esac

    if command -v gh &>/dev/null; then
        TAG=$(gh api repos/ggml-org/llama.cpp/releases/latest --jq .tag_name)
    else
        TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
    fi
    echo "  Release: $TAG, platform: $PLATFORM" >&2

    mkdir -p "$CACHE"
    curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-${PLATFORM}.tar.gz" \
        | tar xz -C "$CACHE"
    mv "$CACHE/llama-${TAG}" "$LLAMA_DIR"
fi

echo "  $(${LLAMA_DIR}/llama-server --version 2>&1 || true)" >&2

# -- Embedding model --

if [[ -f "$MODEL_DIR/$MODEL_NAME" ]]; then
    echo "Using cached model" >&2
else
    echo "Downloading $MODEL_NAME..." >&2
    mkdir -p "$MODEL_DIR"
    curl -sL "$MODEL_URL" -o "$MODEL_DIR/$MODEL_NAME"
fi

echo "  Model: $(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1) $MODEL_NAME" >&2

# -- Output for eval or copy-paste --

echo "export PATH=\"$LLAMA_DIR:\$PATH\""
echo "export SHRIKE_EMBEDDING_MODEL=\"$MODEL_DIR/$MODEL_NAME\""
