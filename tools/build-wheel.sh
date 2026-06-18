#!/usr/bin/env bash
# Build the release wheel via Bazel and give it its real filename.
#
# py_wheel stamps the tag-derived version (STABLE_VERSION) into the wheel's
# METADATA + .dist-info, but Bazel computes the output *filename* at analysis time
# — before stamping — so it ships as `shrike_py-{STABLE_VERSION}-<tags>.whl`.
# This copies it to the metadata's actual version, preserving the python/abi/platform
# tags (cp312-abi3-<platform> — the wheel carries the native extension).
# Used by release.yml and locally.
#
# Usage: tools/build-wheel.sh [OUT_DIR]   (default: dist/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-$repo_root/dist}"
cd "$repo_root"

./bazel build //shrike-py:wheel --stamp >&2
built="$(./bazel cquery --output=files //shrike-py:wheel 2>/dev/null)"
version="$(unzip -p "$built" '*.dist-info/METADATA' | sed -n 's/^Version: //p' | head -1)"
# The built name is shrike_py-{STABLE_VERSION}-<python>-<abi>-<platform>.whl;
# keep everything after the placeholder.
tags="$(basename "$built" | sed 's/^shrike_py-{STABLE_VERSION}-//')"

mkdir -p "$out_dir"
dest="$out_dir/shrike_py-${version}-${tags}"
cp -f "$built" "$dest"
echo "$dest"
