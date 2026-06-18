#!/usr/bin/env bash
# Build the Shrike native extension via Bazel and install it into the active
# venv — the fast inner loop for the pip lane:
#
#   source .venv/bin/activate && scripts/build-native.sh
#   pytest shrike-py/tests/unit -q     # facades now see the real extension
#
# Bazel builds //shrike-core/bindings/shrike-pyo3:native_so — the SAME _native.so the
# release wheel ships (//shrike-py:wheel names it) — so the inner loop and the
# canonical artifact share ONE build graph instead of a parallel cargo path.
# `bazel cquery --output=files` locates the built .so (the house idiom from
# tools/build-wheel.sh / tools/build-sdist.sh); we copy it into the source-tree
# package dir and pip install that. Bazel builds anki + the engines hermetically,
# so this lane needs no protoc on PATH.
#
# Flags:
#   --release   optimized build (bazel `-c opt`; default: fastbuild)
#
# There is no system-SQLite linkage check here: the hermetic Bazel
# build always bundles SQLite (FTS5 + trigram guaranteed, no system-linkage
# config), and platform linkage is by definition a non-hermetic cargo concern.
# For that rare local check run cargo directly:
#   (cd shrike-core && cargo build -p shrike-pyo3 \
#       --no-default-features --features "anki-core,engine-ort,engine-remote,manage-llama")
#
# Lives in shrike-core/scripts/ (with the workspace it builds) and is symlinked
# back into top-level scripts/; resolve the repo root via git so it works from
# either invocation path.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BAZEL_FLAGS=()
MODE="fastbuild"
for arg in "$@"; do
  case "$arg" in
    --release) BAZEL_FLAGS+=(-c opt); MODE="opt" ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

TARGET="//shrike-core/bindings/shrike-pyo3:native_so"
# Python imports `shrike_native._native` from a file literally named `_native.so`;
# this is the in-source-tree copy pip installs from.
DEST="shrike-core/bindings/shrike-pyo3/python/shrike_native/_native.so"

# The first-class way to get a build artifact OUT of Bazel: build the target,
# then `cquery --output=files` for its declared output path (config-correct —
# pass the same flags so an `-c opt` build resolves the opt output dir).
./bazel build ${BAZEL_FLAGS[@]+"${BAZEL_FLAGS[@]}"} "$TARGET" >&2
BUILT="$(./bazel cquery --output=files ${BAZEL_FLAGS[@]+"${BAZEL_FLAGS[@]}"} "$TARGET" 2>/dev/null)"

# Copy out of the build dir (Bazel outputs are read-only; cp -f + chmod keeps the
# source-tree copy writable for the next rebuild and for tooling).
cp -f "$BUILT" "$DEST"
chmod u+w "$DEST"
echo "built $DEST (bazel ${MODE})"

# Plain (non-editable) install: hatchling editables use an import hook that
# mypy/stubtest cannot resolve, and the .so changes each rebuild anyway.
python -m pip install -q --force-reinstall shrike-core/bindings/shrike-pyo3/python
python - <<'PY'
import shrike_native
print(f"shrike_native {shrike_native.version()} — {shrike_native.build_info()}")
PY

# Record the staleness stamp keyed to this venv, so scripts/native-stale.sh
# (and the .envrc / pytest backstop) can tell a fresh extension from a stale one.
# Reuse scripts/native-stamp.sh — the single source of truth, never inlined here.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  "$(dirname "$0")/native-stamp.sh" >"$VIRTUAL_ENV/.shrike-native-stamp"
  echo "stamped $VIRTUAL_ENV/.shrike-native-stamp"
fi
