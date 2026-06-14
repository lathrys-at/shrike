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
#   - the recorded stamp ($venv/.shrike-native-stamp) equals the current stamp
#     from scripts/native-stamp.sh, and
#   - `import shrike_native._native` actually succeeds (a wiped site-packages
#     with a leftover stamp file must read as stale, not fresh).
#
# The venv is resolved from $VIRTUAL_ENV (an activated shell / .envrc) OR
# $SHRIKE_NATIVE_VENV — the latter is how the tests/conftest.py backstop passes
# the running interpreter's sys.prefix, so the check works under .venv/bin/pytest
# / an IDE runner / `uv run`, where VIRTUAL_ENV is unset (#574). Both empty is a
# bare CLI run with no venv at all — fair to reject.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

venv="${VIRTUAL_ENV:-${SHRIKE_NATIVE_VENV:-}}"
if [[ -z "$venv" ]]; then
  echo "native-stale: no venv (activate it, or set SHRIKE_NATIVE_VENV)" >&2
  exit 1
fi

# Probe the venv's own interpreter, not whatever `python` PATH resolves to — when
# VIRTUAL_ENV is unset, bare `python` could be a system interpreter. This is the
# same interpreter build-native.sh recorded (its sys.executable is venv-absolute,
# identical activated or not), so a fresh stamp matches either invocation.
PY="$venv/bin/python"
[[ -x "$PY" ]] || PY="python"

stamp_file="$venv/.shrike-native-stamp"
if [[ ! -f "$stamp_file" ]]; then
  exit 1  # never built into this venv
fi

recorded="$(cat "$stamp_file")"
current="$(SHRIKE_NATIVE_PYTHON="$PY" "$HERE/native-stamp.sh")"
if [[ "$recorded" != "$current" ]]; then
  exit 1  # inputs moved since the last build
fi

# The stamp matches, but the .so must still actually be importable — a stamp can
# outlive its extension (a pip wipe, a half-removed site-packages).
if ! "$PY" -c 'import shrike_native._native' >/dev/null 2>&1; then
  exit 1
fi

exit 0
