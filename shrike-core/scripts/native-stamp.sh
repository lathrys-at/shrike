#!/usr/bin/env bash
# Compute the native-build stamp: a hash of every input that decides whether the
# compiled shrike_native extension in the active venv is current (#573). The
# single source of truth for staleness — both scripts/native-stale.sh (the check)
# and scripts/build-native.sh (the writer) call this, so the two can't drift.
#
# Inputs, in order:
#   1. the committed shrike-core/ tree hash (git rev-parse HEAD:shrike-core)
#   2. the working-tree shrike-core diff (git diff HEAD -- shrike-core/)
#   3. untracked shrike-core files (git ls-files --others --exclude-standard shrike-core/)
#   4. the active interpreter (sys.executable + version — the abi3 .so is
#      venv-bound, so a different interpreter is a different build)
#
# git plumbing throughout, not mtimes: mtimes lie across checkout/stash/pull,
# the content hashes don't. Prints the hex digest and nothing else.
#
# Lives in shrike-core/scripts/ (with the workspace it stamps) and is symlinked
# back into top-level scripts/; resolve the repo root via git so it works from
# either invocation path.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Pick a SHA-256 command that exists on both macOS (shasum) and Linux (sha256sum).
if command -v shasum >/dev/null 2>&1; then
  _sha256() { shasum -a 256 | cut -d' ' -f1; }
elif command -v sha256sum >/dev/null 2>&1; then
  _sha256() { sha256sum | cut -d' ' -f1; }
else
  echo "native-stamp: neither shasum nor sha256sum found on PATH" >&2
  exit 1
fi

# 1. Committed shrike-core tree hash. A repo with no shrike-core/ tree (or not a
#    git repo) falls back to a stable sentinel rather than failing the whole stamp.
tree_hash="$(git rev-parse HEAD:shrike-core 2>/dev/null || echo 'no-native-tree')"

# 2. Working-tree shrike-core edits, content-hashed.
diff_hash="$(git diff HEAD -- shrike-core/ 2>/dev/null | _sha256)"

# 3. Untracked (but not ignored) shrike-core files: their names AND contents.
others="$(git ls-files --others --exclude-standard shrike-core/ 2>/dev/null || true)"
if [[ -n "$others" ]]; then
  # Hash each file's bytes too, not just its path, so editing an untracked file
  # changes the stamp.
  others_hash="$(printf '%s\n' "$others" | while IFS= read -r f; do
    printf '%s\0' "$f"
    [[ -f "$f" ]] && cat -- "$f"
  done | _sha256)"
else
  others_hash="no-untracked"
fi

# 4. The active interpreter (the abi3 .so is venv-bound). SHRIKE_NATIVE_PYTHON
#    lets a caller pin the exact interpreter to probe — the venv's own python —
#    so the stamp is identical whether pytest runs activated (bare `python` is
#    the venv) or as .venv/bin/pytest with no VIRTUAL_ENV (where bare `python`
#    would otherwise resolve to a system interpreter and spuriously shift it).
PYTHON="${SHRIKE_NATIVE_PYTHON:-python}"
interp="$("$PYTHON" -c 'import sys; print(sys.executable, sys.version)' 2>/dev/null || echo 'no-python')"

printf '%s\n%s\n%s\n%s\n' "$tree_hash" "$diff_hash" "$others_hash" "$interp" | _sha256
