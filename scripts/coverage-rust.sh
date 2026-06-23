#!/usr/bin/env bash
# Rust coverage via cargo-llvm-cov — the native counterpart of scripts/coverage.sh.
# Reported, never gated in CI (the same off-gate posture as Python coverage); the
# optional --fail-under-lines below is a LOCAL ratchet, exactly like pyproject's
# fail_under for the Python side.
#
# Usage:
#   scripts/coverage-rust.sh                      # run the workspace test suites, print the per-crate table
#   scripts/coverage-rust.sh --html               # also write a browsable HTML report
#   scripts/coverage-rust.sh --fail-under-lines 80 # exit non-zero below 80% line coverage (local ratchet)
#
# Needs cargo-llvm-cov + the llvm-tools-preview component:
#   rustup component add llvm-tools-preview && cargo install cargo-llvm-cov --locked
# protoc must be on PATH for the anki-coupled crates (collection/kernel/derived) —
# the same prerequisite a bare `cargo test --workspace` has.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  echo "cargo-llvm-cov is not installed — run:" >&2
  echo "  rustup component add llvm-tools-preview && cargo install cargo-llvm-cov --locked" >&2
  exit 1
fi

want_html=0
passthrough=()
for arg in "$@"; do
  if [ "$arg" = "--html" ]; then
    want_html=1
  else
    passthrough+=("$arg")
  fi
done

cd "$ROOT/shrike-core"

# Line + region coverage (cargo-llvm-cov's default on stable; region is llvm's
# finer-than-line granularity — branch coverage proper is a nightly-only `-Z` flag,
# not worth pinning a nightly toolchain for here). The binding crates (shrike-pyo3,
# shrike-cabi) are EXCLUDED: their Rust unit tests are thin and their real contract is
# the FFI boundary, exercised by the Python `native` suite and (per the testing ADRs) a
# scheduled miri lane — not by cargo coverage. Measuring them here would dilute the
# kernel/store/engine number this report tracks.
common=(--workspace
  --exclude shrike-pyo3
  --exclude shrike-cabi)

if [ "$want_html" -eq 1 ]; then
  cargo llvm-cov "${common[@]}" --html ${passthrough[@]+"${passthrough[@]}"}
  echo "HTML report: shrike-core/target/llvm-cov/html/index.html"
else
  cargo llvm-cov "${common[@]}" ${passthrough[@]+"${passthrough[@]}"}
fi
