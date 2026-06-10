#!/usr/bin/env bash
# Build the release wheel via Bazel and give it its real filename (#245).
#
# py_wheel stamps the tag-derived version (STABLE_VERSION) into the wheel's
# METADATA + .dist-info, but Bazel computes the output *filename* at analysis time
# — before stamping — so it ships as `shrike_mcp-{STABLE_VERSION}-py3-none-any.whl`.
# This copies it to the metadata's actual version. Used by release.yml and locally.
#
# Usage: tools/build-wheel.sh [OUT_DIR]   (default: dist/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-$repo_root/dist}"
cd "$repo_root"

./bazel build //:wheel --stamp >&2
built="$(./bazel cquery --output=files //:wheel 2>/dev/null)"
version="$(unzip -p "$built" '*.dist-info/METADATA' | sed -n 's/^Version: //p' | head -1)"

mkdir -p "$out_dir"
dest="$out_dir/shrike_mcp-${version}-py3-none-any.whl"
cp -f "$built" "$dest"
echo "$dest"
