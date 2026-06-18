#!/usr/bin/env bash
# Refresh the checked-in anki protobuf descriptor set after an anki tag bump.
# The Bazel build of the `anki` crate reads this file via the
# DESCRIPTORS_BIN override (see shrike-core/third_party/anki/BUILD.bazel); cargo
# regenerates the live copy under target/ on every build of anki_proto.
#
#   scripts/update-anki-descriptors.sh        # after bumping the anki tag
set -euo pipefail
cd "$(dirname "$0")/.."

(cd shrike-core && cargo build -p anki_proto)
SRC=shrike-core/target/debug/build/anki_descriptors.bin
DEST=shrike-core/third_party/anki/anki_descriptors.bin
[ -f "$SRC" ] || { echo "descriptor set not found at $SRC" >&2; exit 1; }
cp "$SRC" "$DEST"
echo "updated $DEST ($(wc -c < "$DEST") bytes)"
