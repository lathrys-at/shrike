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

# Pin the resolver so the lock is reproducible regardless of the contributor's
# installed uv: uvx fetches this exact version. Bump deliberately (and commit the
# regenerated lock). Mirrors the pin-everything bootstrap (bazelisk, llama-server).
UV_VERSION=0.11.19

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uvx "uv@${UV_VERSION}" pip compile pyproject.toml \
  --python-version 3.12 \
  --universal \
  --generate-hashes \
  --extra dev \
  --extra onnx \
  --extra clip \
  --extra socks \
  -o requirements_lock.txt

# Build-time tools for //:sdist, in their own lock so they stay out of the runtime
# dependency set (consumed by the @shrike_sdist_pip hub in MODULE.bazel).
exec uvx "uv@${UV_VERSION}" pip compile tools/sdist-requirements.in \
  --python-version 3.12 \
  --universal \
  --generate-hashes \
  -o requirements_sdist_lock.txt
