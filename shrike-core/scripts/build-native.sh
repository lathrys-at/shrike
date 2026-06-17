#!/usr/bin/env bash
# Build the Shrike native extension with cargo and install it (editable) into the
# active venv (#269). The canonical release artifact is the Bazel wheel (//:wheel —
# the platform-tagged shrike-mcp wheel ships shrike_native inside it since #497);
# this is the fast inner loop for the pip lane:
#
#   source .venv/bin/activate && scripts/build-native.sh
#   pytest tests/unit -q          # facades now see the real extension
#
# Flags:
#   --release         optimized build (default: debug, fastest compile)
#   --system-sqlite   link the platform SQLite instead of bundling (#300);
#                     FTS5/trigram availability is then probed at runtime
#
# Since the cutover the anki collection core is a DEFAULT cargo feature —
# every build pulls the anki tree and needs protoc on PATH (brew/apt).
#
# Lives in shrike-core/scripts/ (with the workspace it builds) and is symlinked
# back into top-level scripts/; resolve the repo root via git so it works from
# either invocation path.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PROFILE="debug"
CARGO_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --release) PROFILE="release"; CARGO_FLAGS+=(--release) ;;
    # Drop ONLY the bundling: re-enable the rest of the default (server) set
    # (#499 — a bare --no-default-features would also drop anki-core and every
    # engine, leaving an extension the server can't run on).
    --system-sqlite) CARGO_FLAGS+=(--no-default-features --features "anki-core,engine-ort,engine-remote,manage-llama") ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

(cd shrike-core && cargo build -p shrike-pyo3 ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"})

case "$(uname -s)" in
  Darwin) LIB="shrike-core/target/${PROFILE}/libshrike_pyo3.dylib" ;;
  *)      LIB="shrike-core/target/${PROFILE}/libshrike_pyo3.so" ;;
esac
DEST="shrike-core/shrike-pyo3/python/shrike_native/_native.so"
cp "$LIB" "$DEST"
echo "built $DEST (${PROFILE})"

# Plain (non-editable) install: hatchling editables use an import hook that
# mypy/stubtest cannot resolve, and the .so changes each rebuild anyway.
python -m pip install -q --force-reinstall shrike-core/shrike-pyo3/python
python - <<'PY'
import shrike_native
print(f"shrike_native {shrike_native.version()} — {shrike_native.build_info()}")
PY

# Record the staleness stamp keyed to this venv (#573), so scripts/native-stale.sh
# (and the .envrc / pytest backstop) can tell a fresh extension from a stale one.
# Reuse scripts/native-stamp.sh — the single source of truth, never inlined here.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  "$(dirname "$0")/native-stamp.sh" >"$VIRTUAL_ENV/.shrike-native-stamp"
  echo "stamped $VIRTUAL_ENV/.shrike-native-stamp"
fi
