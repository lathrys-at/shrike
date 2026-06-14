#!/usr/bin/env bash
# Compute the native-build stamp: a hash of every input that decides whether the
# compiled shrike_native extension in the active venv is current (#573). The
# single source of truth for staleness — both scripts/native-stale.sh (the check)
# and scripts/build-native.sh (the writer) call this, so the two can't drift.
#
# Inputs, in order:
#   1. the committed native/ tree hash (git rev-parse HEAD:native)
#   2. the working-tree native diff (git diff HEAD -- native/)
#   3. untracked native files (git ls-files --others --exclude-standard native/)
#   4. the active interpreter (sys.executable + version — the abi3 .so is
#      venv-bound, so a different interpreter is a different build)
#
# git plumbing throughout, not mtimes: mtimes lie across checkout/stash/pull,
# the content hashes don't. Prints the hex digest and nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick a SHA-256 command that exists on both macOS (shasum) and Linux (sha256sum).
if command -v shasum >/dev/null 2>&1; then
  _sha256() { shasum -a 256 | cut -d' ' -f1; }
elif command -v sha256sum >/dev/null 2>&1; then
  _sha256() { sha256sum | cut -d' ' -f1; }
else
  echo "native-stamp: neither shasum nor sha256sum found on PATH" >&2
  exit 1
fi

# 1. Committed native tree hash. A repo with no native/ tree (or not a git repo)
#    falls back to a stable sentinel rather than failing the whole stamp.
tree_hash="$(git rev-parse HEAD:native 2>/dev/null || echo 'no-native-tree')"

# 2. Working-tree native edits, content-hashed.
diff_hash="$(git diff HEAD -- native/ 2>/dev/null | _sha256)"

# 3. Untracked (but not ignored) native files: their names AND contents.
others="$(git ls-files --others --exclude-standard native/ 2>/dev/null || true)"
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

# 4. The active interpreter.
interp="$(python -c 'import sys; print(sys.executable, sys.version)' 2>/dev/null || echo 'no-python')"

printf '%s\n%s\n%s\n%s\n' "$tree_hash" "$diff_hash" "$others_hash" "$interp" | _sha256
