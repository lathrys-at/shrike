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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE="$ROOT/.cache"
LLAMA_DIR="$CACHE/llama-server"
MODEL_DIR="$CACHE/models"
MODEL_NAME="all-MiniLM-L6-v2-Q4_0.gguf"
MODEL_URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/$MODEL_NAME"

# Pinned llama.cpp tag + per-platform checksums (shared with CI).
# shellcheck source=scripts/llama-server.lock
source "$SCRIPT_DIR/llama-server.lock"

# Fail unless $1 hashes to $2. Portable across Linux (sha256sum) and macOS (shasum).
verify_sha256() {
    local file="$1" expected="$2" actual
    if command -v sha256sum &>/dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    else
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    fi
    if [[ "$actual" != "$expected" ]]; then
        echo "Checksum mismatch for $(basename "$file")" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        exit 1
    fi
}

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
    echo "Downloading llama-server $LLAMA_TAG..." >&2
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   PLATFORM="ubuntu-x64" ;;
        Linux-aarch64)  PLATFORM="ubuntu-arm64" ;;
        Darwin-arm64)   PLATFORM="macos-arm64" ;;
        Darwin-x86_64)  PLATFORM="macos-x64" ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2 && exit 1 ;;
    esac

    # Look up the pinned checksum for this platform (lock keys use underscores).
    sha_var="SHA256_${PLATFORM//-/_}"
    expected="${!sha_var:-}"
    if [[ -z "$expected" ]]; then
        echo "No pinned SHA256 for platform '$PLATFORM' in llama-server.lock" >&2 && exit 1
    fi
    echo "  Release: $LLAMA_TAG, platform: $PLATFORM" >&2

    mkdir -p "$CACHE"
    TARBALL="$CACHE/llama-${LLAMA_TAG}-bin-${PLATFORM}.tar.gz"
    curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-${PLATFORM}.tar.gz" \
        -o "$TARBALL"
    verify_sha256 "$TARBALL" "$expected"
    tar xz -C "$CACHE" -f "$TARBALL"
    rm -f "$TARBALL"
    mv "$CACHE/llama-${LLAMA_TAG}" "$LLAMA_DIR"
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
