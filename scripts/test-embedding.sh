#!/usr/bin/env bash
#
# Run embedding integration tests locally.
#
# Uses fetch-llama-server.sh to get llama-server and a small GGUF model
# (cached in .cache/), then runs the embedding test suite.
#
# Usage:
#   ./scripts/test-embedding.sh           # run embedding tests
#   ./scripts/test-embedding.sh --fresh   # re-download everything
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

FETCH_ARGS=()
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--fresh" ]]; then
        FETCH_ARGS+=("--fresh")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

eval "$("$ROOT/scripts/fetch-llama-server.sh" "${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}")"
export LD_LIBRARY_PATH="${ROOT}/.cache/llama-server:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${ROOT}/.cache/llama-server:${DYLD_LIBRARY_PATH:-}"

echo ""
echo "Running embedding tests..."
cd "$ROOT"
exec python -m pytest tests/integration -v -m embedding ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
