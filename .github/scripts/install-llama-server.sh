#!/usr/bin/env bash
#
# Install the pinned llama-server into $RUNNER_TEMP for CI and expose it on PATH.
# Shared by the `embedding` and `cross-platform` jobs in test.yml. Verifies the
# per-platform SHA256 from scripts/llama-server.lock before extracting.
#
# Run from the repo root (GitHub Actions' default working directory).

set -euo pipefail

# shellcheck source=scripts/llama-server.lock
source scripts/llama-server.lock

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)   PLATFORM="ubuntu-x64" ;;
    Linux-aarch64)  PLATFORM="ubuntu-arm64" ;;
    Darwin-arm64)   PLATFORM="macos-arm64" ;;
    Darwin-x86_64)  PLATFORM="macos-x64" ;;
    *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2 && exit 1 ;;
esac

sha_var="SHA256_${PLATFORM//-/_}"
expected="${!sha_var}"

INSTALL_DIR="${RUNNER_TEMP}/llama-server"

# Skip the download when a restored cache already holds the pinned binary. The
# cache is keyed on scripts/llama-server.lock (tag + SHA256), so a hit is the
# same verified build a download would produce; a tag/SHA bump changes the key
# and forces a fresh, re-verified download.
if [ -x "$INSTALL_DIR/llama-server" ]; then
    echo "llama-server already present at $INSTALL_DIR (cache hit) — skipping download"
else
    TARBALL="${RUNNER_TEMP}/llama-${LLAMA_TAG}-bin-${PLATFORM}.tar.gz"
    curl -fsSL \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-${PLATFORM}.tar.gz" \
        -o "$TARBALL"

    if command -v sha256sum >/dev/null; then
        actual=$(sha256sum "$TARBALL" | awk '{print $1}')
    else
        actual=$(shasum -a 256 "$TARBALL" | awk '{print $1}')
    fi
    if [ "$actual" != "$expected" ]; then
        echo "Checksum mismatch for $PLATFORM: expected $expected, got $actual" >&2 && exit 1
    fi

    tar xz -C "${RUNNER_TEMP}" -f "$TARBALL"
    mv "${RUNNER_TEMP}/llama-${LLAMA_TAG}" "$INSTALL_DIR"
fi

echo "$INSTALL_DIR" >> "$GITHUB_PATH"
echo "LLAMA_DIR=$INSTALL_DIR" >> "$GITHUB_ENV"
"$INSTALL_DIR/llama-server" --version || true
