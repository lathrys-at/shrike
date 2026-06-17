#!/usr/bin/env bash
#
# Regenerate tools/llama-server.lock: pin a llama.cpp release tag and record
# the SHA256 of each platform tarball we download.
#
# Usage:
#   ./tools/update-llama-lock.sh          # pin the latest release
#   ./tools/update-llama-lock.sh b9415    # pin a specific tag
#
# Downloads the four platform tarballs to a temp dir to hash them, then writes
# the lock. Commit the result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK_FILE="$SCRIPT_DIR/llama-server.lock"
PLATFORMS=(ubuntu-x64 ubuntu-arm64 macos-arm64 macos-x64)

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    if command -v gh &>/dev/null; then
        TAG=$(gh api repos/ggml-org/llama.cpp/releases/latest --jq .tag_name)
    else
        TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
    fi
fi
echo "Pinning llama.cpp $TAG" >&2

sha256() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Build the file in a buffer first (avoids bash-3.2 associative arrays, so this
# also runs on stock macOS bash).
{
    echo "# Pinned llama.cpp release, consumed by MODULE.bazel's llama-server http_archive"
    echo "# shas and the CI model-cache keys (.github/actions/bazel-setup)."
    echo "#"
    echo "# We pin a known tag and verify a per-platform SHA256 instead of pulling"
    echo "# \`releases/latest\` unverified — a release could be re-tagged or a mirror"
    echo "# tampered with. To bump: run \`tools/update-llama-lock.sh [TAG]\` (defaults to"
    echo "# the latest release) and commit the result."
    echo "#"
    echo "# Shell-sourceable: KEY=VALUE, no spaces. Platform keys use underscores because"
    echo "# \`ubuntu-x64\` isn't a valid shell variable name."
    echo "LLAMA_TAG=$TAG"
    for P in "${PLATFORMS[@]}"; do
        url="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-${P}.tar.gz"
        echo "  fetching $P..." >&2
        curl -fsSL "$url" -o "$TMP/$P.tar.gz"
        echo "SHA256_${P//-/_}=$(sha256 "$TMP/$P.tar.gz")"
    done
} > "$LOCK_FILE"

echo "Wrote $LOCK_FILE" >&2
