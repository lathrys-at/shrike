#!/usr/bin/env bash
# Local coverage run — the same measurement CI does on rc PRs and 3x/week on main
# (.github/workflows/coverage.yml), so the number matches. CI runs *plain* tests on
# every PR for speed (no tracer) and only enforces the fail_under gate on rc, so run
# this locally to keep coverage healthy as you work rather than discovering a drop at
# release time.
#
# Usage:
#   scripts/coverage.sh                 # full suite, prints report, exits non-zero below fail_under
#   scripts/coverage.sh -k upsert       # subset (the % will be lower — partial run)
#   scripts/coverage.sh --html          # also write a browsable htmlcov/ report
#
# Run from the repo root inside your venv (the one with `pip install -e ".[dev]"`,
# plus the native extension — see the guard below).
set -euo pipefail

# Repo root from the script's own location, so the committed coverage hook
# (tools/coverage_subprocess.pth) and the harness pyproject (shrike-py/pyproject.toml,
# which carries [tool.coverage.*]) are found regardless of cwd.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Since the native cutover the suite imports shrike_native, which pip alone
# doesn't build — fail early with the fix instead of an ImportError wall (#400).
if ! python -c 'import shrike_native' 2>/dev/null; then
  echo "shrike_native is not importable — build it first: scripts/build-native.sh" >&2
  exit 1
fi

want_html=0
pytest_args=()
for arg in "$@"; do
  if [ "$arg" = "--html" ]; then
    want_html=1
  else
    pytest_args+=("$arg")
  fi
done

# The integration suite drives a `python -m shrike.server` subprocess; without this
# the server's lines read as uncovered. The .pth only imports coverage when
# COVERAGE_PROCESS_START is set, so it costs nothing on other interpreter starts —
# the same guard coverage's own auto-.pth uses. Copy the single committed hook
# (tools/coverage_subprocess.pth — one source of truth, shared with CLAUDE.md's
# by-hand recipe) into site-packages. Idempotent.
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
HOOK="$SITE/coverage_subprocess.pth"
if [ ! -f "$HOOK" ]; then
  cp "$ROOT/tools/coverage_subprocess.pth" "$HOOK"
fi

# The [tool.coverage.*] config lives in the harness pyproject, now under shrike-py/ (#731).
export COVERAGE_PROCESS_START="$ROOT/shrike-py/pyproject.toml"

# Combined `-n auto` run over both suites, matching CI. `-m "not embedding and not
# search_quality"` keeps the embedding-gated AND the manual search-quality tests out
# (they need local models / a downloaded Commons corpus and aren't part of the
# measured number). The .pth fires for each xdist worker and each spawned server, so
# `coverage combine` merges everything to one total.
coverage erase
# ${arr[@]+...} guard: bash 3.2 (macOS /bin/bash) treats an EMPTY array as
# unbound under `set -u`, so a bare "${pytest_args[@]}" aborts a no-arg run.
coverage run --parallel-mode -m pytest \
  shrike-py/tests/unit shrike-py/tests/integration shrike-py/tests/native \
  -q -m "not embedding and not search_quality" -n auto ${pytest_args[@]+"${pytest_args[@]}"}
coverage combine

if [ "$want_html" -eq 1 ]; then
  coverage html
  echo "HTML report: htmlcov/index.html"
fi

# Prints the per-file table (show_missing) and exits non-zero below fail_under,
# exactly like the rc gate.
coverage report
