#!/usr/bin/env bash
# Build the release sdist via Bazel and give it its real filename (#245).
#
# The //shrike-py:sdist rule stamps the tag-derived version (STABLE_VERSION) into the sdist,
# but Bazel computes the output *filename* at analysis time — before stamping — so it
# ships as `sdist.tar.gz`. This copies it to the real shrike_mcp-<version>.tar.gz
# (the version read from the tarball's top-level directory). Used by release.yml and
# locally.
#
# Usage: tools/build-sdist.sh [OUT_DIR]   (default: dist/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-$repo_root/dist}"
cd "$repo_root"

./bazel build //shrike-py:sdist --stamp >&2
built="$(./bazel cquery --output=files //shrike-py:sdist 2>/dev/null)"
# The sdist's top-level dir is shrike_mcp-<version>/ — read the version from it.
version="$(tar tzf "$built" | head -1 | sed -E 's#^shrike_mcp-(.*)/.*#\1#')"

mkdir -p "$out_dir"
dest="$out_dir/shrike_mcp-${version}.tar.gz"
cp -f "$built" "$dest"
echo "$dest"
