#!/usr/bin/env bash
# Is the compiled shrike_native in the active venv stale or unbuilt? (#573)
#
#   exit 0  fresh — the extension matches the current native/ inputs and imports
#   exit 1  stale, unbuilt, or unimportable — needs scripts/build-native.sh
#
# Fast (git plumbing only, no compile — well under 100ms in the fresh case), so
# it's cheap enough to run on every shell cd (.envrc) and every pytest session
# (the tests/conftest.py backstop). Run it from anywhere — the repo root is
# resolved from the script's own location.
#
# Two conditions must BOTH hold for "fresh":
#   - the recorded stamp ($VIRTUAL_ENV/.shrike-native-stamp) equals the current
#     stamp from scripts/native-stamp.sh, and
#   - `import shrike_native._native` actually succeeds (a wiped site-packages
#     with a leftover stamp file must read as stale, not fresh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "native-stale: no \$VIRTUAL_ENV active (activate the venv first)" >&2
  exit 1
fi

stamp_file="$VIRTUAL_ENV/.shrike-native-stamp"
if [[ ! -f "$stamp_file" ]]; then
  exit 1  # never built into this venv
fi

recorded="$(cat "$stamp_file")"
current="$("$HERE/native-stamp.sh")"
if [[ "$recorded" != "$current" ]]; then
  exit 1  # inputs moved since the last build
fi

# The stamp matches, but the .so must still actually be importable — a stamp can
# outlive its extension (a pip wipe, a half-removed site-packages).
if ! python -c 'import shrike_native._native' >/dev/null 2>&1; then
  exit 1
fi

exit 0
