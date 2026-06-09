#!/usr/bin/env bash
# Regenerate the pinned Bazel dependency lock (requirements_lock.txt) from
# pyproject.toml using uv. Run this after changing dependencies in pyproject.toml,
# then commit the updated lock. This is the source of truth pip.parse resolves
# against in MODULE.bazel.
#
# The lock is universal (cross-platform, marker-guarded) and hashed, and includes
# the extras the Bazel test/build graph needs (dev tooling + the onnx/clip
# embedding backends). onnx-gpu is excluded on purpose: it conflicts with onnx.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

exec uv pip compile pyproject.toml \
  --python-version 3.12 \
  --universal \
  --generate-hashes \
  --extra dev \
  --extra onnx \
  --extra clip \
  --extra socks \
  -o requirements_lock.txt
