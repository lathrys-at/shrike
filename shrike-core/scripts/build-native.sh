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
#   --release    optimized build (bazel `-c opt`; default: fastbuild)
#   --synthetic  add the deterministic synthetic embedder (#865), for the perf
#                lane and fast deterministic tests. OFF by default: the release
#                wheel and the per-PR `//...` lane build the lean extension, so a
#                config naming `runtime: synthetic` is refused there. With this
#                flag the staged extension is NOT byte-identical to the wheel's.
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

# Worktree isolation guard. The build reads this tree (cd above) but the install +
# stamp below land in $VIRTUAL_ENV; if that venv belongs to a different checkout,
# the build lands in the wrong venv and stamps it — the silent cross-wire that
# makes pytest import another worktree's .so. The test is checkout *identity*, not
# path containment: agent worktrees nest under the main checkout
# (.claude/worktrees/*), so a containment test would wave a worktree's venv
# through while standing in main. Resolve the venv's own git worktree root and
# require it to be this checkout. A venv outside any git tree (a deliberate
# external venv) belongs to no checkout and is left alone, as is an unset
# VIRTUAL_ENV (the CI/system-python path: uv pip install --system).
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  here="$(pwd -P)"
  venv_real="$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd -P || echo "$VIRTUAL_ENV")"
  vtop="$(git -C "$venv_real" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$vtop" ]]; then vtop="$(cd "$vtop" && pwd -P)"; fi
  if [[ -n "$vtop" && "$vtop" != "$here" ]]; then
    echo "build-native: refusing to cross-wire checkouts." >&2
    echo "  active venv : $VIRTUAL_ENV" >&2
    echo "  venv's tree : $vtop" >&2
    echo "  this tree   : $here" >&2
    echo "  That venv belongs to a different checkout (worktree mix-up)." >&2
    echo "  Fix: deactivate, then in THIS tree run" >&2
    echo "       scripts/dev-setup.sh && source .venv/bin/activate" >&2
    exit 1
  fi
fi

BAZEL_FLAGS=()
MODE="fastbuild"
SYNTH=""
for arg in "$@"; do
  case "$arg" in
    --release) BAZEL_FLAGS+=(-c opt); MODE="opt" ;;
    --synthetic) BAZEL_FLAGS+=(--define shrike_synthetic=on); SYNTH=" +synthetic" ;;
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
echo "built $DEST (bazel ${MODE}${SYNTH})"

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
