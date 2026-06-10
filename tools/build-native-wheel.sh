#!/usr/bin/env bash
# Build the shrike-native platform wheel via Bazel and give it its real filename
# (#269), mirroring tools/build-wheel.sh: py_wheel stamps STABLE_VERSION into the
# METADATA, but the output *filename* keeps the placeholder — copy it to the
# metadata's actual version, preserving the platform/abi tags.
#
# Usage: tools/build-native-wheel.sh [OUT_DIR]   (default: dist/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-$repo_root/dist}"
cd "$repo_root"

./bazel build //native/shrike-py:wheel --stamp >&2
built="$(./bazel cquery --output=files //native/shrike-py:wheel 2>/dev/null)"
version="$(unzip -p "$built" '*.dist-info/METADATA' | sed -n 's/^Version: //p' | head -1)"
# The built name is shrike_native-{STABLE_VERSION}-<python>-<abi>-<platform>.whl;
# keep everything after the placeholder.
tags="$(basename "$built" | sed 's/^shrike_native-{STABLE_VERSION}-//')"

mkdir -p "$out_dir"
dest="$out_dir/shrike_native-${version}-${tags}"
cp -f "$built" "$dest"
echo "$dest"
