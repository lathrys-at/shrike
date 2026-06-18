#!/usr/bin/env bash
# One-step local dev setup for the pip lane. Idempotent and safe to re-run
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
# matching python3.X off PATH.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Root at the checkout we're standing in, not at this script's location. Run from
# a git worktree — even via the main checkout's copy of the script — and we set up
# THAT worktree: its own .venv and its own native extension. This cwd-rooting is
# what keeps each worktree's dev env isolated instead of cross-wiring into another
# checkout's venv. Fall back to the script dir only outside a git tree.
ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/..")"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"

# A venv from ANOTHER checkout being active is the classic worktree cross-wire.
# We always target this checkout's own .venv below (the activate on the next step
# re-points VIRTUAL_ENV), but say so plainly rather than letting it look honored.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  active="$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd -P || echo "$VIRTUAL_ENV")"
  case "$active/" in
    "$ROOT/"*) : ;;
    *) echo "==> ignoring active VIRTUAL_ENV from another checkout ($VIRTUAL_ENV)" ;;
  esac
fi

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
# pyproject.toml lives in the shrike-py/ unit; install editable from there.
# The .venv stays at the checkout root so native-stale.sh / .envrc / the pytest
# backstop (all keyed off $VIRTUAL_ENV / sys.prefix) keep working.
echo "==> installing the harness and dev tooling (pip install -e \"shrike-py/[dev]\")"
pip install -q --upgrade pip
pip install -q -e "shrike-py/[dev]"

# ------------------------------------------------------------- 3. native -------
# Use this checkout's own copies (not $HERE's): ROOT is cwd-rooted, so when the
# script is invoked through another checkout these still target the right tree.
if "$ROOT/scripts/native-stale.sh"; then
  echo "==> native extension is current — skipping build"
else
  echo "==> building the native extension (scripts/build-native.sh)"
  "$ROOT/scripts/build-native.sh"
fi

# ------------------------------------------------------------- 4. verify -------
echo "==> verifying imports"
python - <<'PY'
import onnxruntime
import shrike_native
print(f"    onnxruntime  {onnxruntime.__version__}")
print(f"    shrike_native {shrike_native.version()} — {shrike_native.build_info()}")
PY

BRANCH="$(git -C "$ROOT" symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)"
echo
echo "ready — this checkout is isolated:"
echo "    checkout : $ROOT  ($BRANCH)"
echo "    venv     : $ROOT/.venv"
echo
echo "activate THIS checkout's venv (not another worktree's), then test:"
echo "    source $ROOT/.venv/bin/activate"
echo "    pytest shrike-py/tests/unit -q"
