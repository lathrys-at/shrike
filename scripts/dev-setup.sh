#!/usr/bin/env bash
# One-step local dev setup for the pip lane (#573). Idempotent and safe to re-run
# — every step no-ops when already satisfied, so this doubles as a repair button
# when the venv or the native extension drifts.
#
#   scripts/dev-setup.sh
#
# What it does, in order:
#   1. create .venv with the pinned Python (.python-version) if absent
#   2. pip install -e ".[dev]"  (the harness + dev tooling)
#   3. build the native extension — only if it's stale (scripts/native-stale.sh)
#   4. verify onnxruntime and shrike_native both import
#
# Python selection: pyenv if present (it honors .python-version), else the
# matching python3.X off PATH. (Adopting `uv` for the whole venv is a separable
# future option — not built here.)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PYVER="$(tr -d '[:space:]' <.python-version)"   # e.g. 3.12.13
PYVER_MM="${PYVER%.*}"                            # major.minor, e.g. 3.12

# ---------------------------------------------------------------- 1. venv ------
if [[ ! -d .venv ]]; then
  echo "==> creating .venv (Python $PYVER)"
  if command -v pyenv >/dev/null 2>&1; then
    # pyenv reads .python-version from cwd and resolves the interpreter for us.
    PY="$(pyenv which python 2>/dev/null || true)"
    if [[ -z "$PY" ]]; then
      echo "dev-setup: pyenv is present but Python $PYVER isn't installed." >&2
      echo "           run:  pyenv install $PYVER" >&2
      exit 1
    fi
    "$PY" -m venv .venv
  elif command -v "python$PYVER_MM" >/dev/null 2>&1; then
    "python$PYVER_MM" -m venv .venv
  else
    echo "dev-setup: no pyenv and no python$PYVER_MM on PATH." >&2
    echo "           install Python $PYVER_MM (or pyenv + 'pyenv install $PYVER')." >&2
    exit 1
  fi
else
  echo "==> .venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# --------------------------------------------------------------- 2. deps -------
# pyproject.toml lives in the shrike-py/ unit (#731); install editable from there.
# The .venv stays at the repo root so native-stale.sh / .envrc / the pytest
# backstop (all keyed off $VIRTUAL_ENV / sys.prefix) keep working unchanged.
echo "==> installing the harness and dev tooling (pip install -e \"shrike-py/[dev]\")"
pip install -q --upgrade pip
pip install -q -e "shrike-py/[dev]"

# ------------------------------------------------------------- 3. native -------
if "$HERE/native-stale.sh"; then
  echo "==> native extension is current — skipping build"
else
  echo "==> building the native extension (scripts/build-native.sh)"
  "$HERE/build-native.sh"
fi

# ------------------------------------------------------------- 4. verify -------
echo "==> verifying imports"
python - <<'PY'
import onnxruntime
import shrike_native
print(f"    onnxruntime  {onnxruntime.__version__}")
print(f"    shrike_native {shrike_native.version()} — {shrike_native.build_info()}")
PY

echo
echo "ready. run:  pytest shrike-py/tests/unit -q"
