#!/usr/bin/env bash
# Performance harness entry point (#865): run gold workloads against a
# deterministic corpus under a profile and report latency distributions.
#
#   scripts/perf.sh --profile stub --size 500 --variant text --workloads search,rebuild
#   scripts/perf.sh --profile real --size 5000 --variant text+image --workloads search
#
# Prerequisites:
#   - Build OPTIMIZED. The runner times the staged extension, and the default
#     build is fastbuild (unoptimized — meaningless for perf). Pass --release:
#         scripts/build-native.sh --release [--synthetic]
#     The runner records the build mode and warns/refuses on a debug build.
#   - The STUB profile (kernel isolation) also needs the synthetic embedder
#     (--synthetic); a lean build refuses `runtime: synthetic` at resolution.
#   - The REAL profile (end-to-end) needs the onnx/CLIP models; they are fetched
#     into the shared model cache ($SHRIKE_TEST_MODEL_DIR / ~/.cache/shrike-dev).
#
# Off the per-PR critical path; never run in CI. Results land under
# .cache/perf/runs/ (gitignored).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Activate the worktree's venv if the caller hasn't (the same venv dev-setup builds).
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python shrike-py/tests/manual/perf/run.py "$@"
