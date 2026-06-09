#!/usr/bin/env bash
# Refresh the pinned build-system bootstrap: tools/bazel.lock + .bazelversion.
# Run after bumping bazelisk or Bazel, then commit both files. Mirrors
# scripts/update-llama-lock.sh.
#
# Usage:
#   tools/update-bazel-lock.sh                                 # keep versions, refresh shas
#   tools/update-bazel-lock.sh --bazel 8.8.0 --bazelisk v1.30.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/bazel.lock
. "$repo_root/tools/bazel.lock"

bazelisk_ver="$BAZELISK_VERSION"
bazel_ver="$BAZEL_VERSION"
while [ $# -gt 0 ]; do
  case "$1" in
    --bazelisk) bazelisk_ver="$2"; shift 2 ;;
    --bazel) bazel_ver="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

fetch_sha() { curl -fsSL --retry 5 --retry-connrefused --connect-timeout 30 "$1" | awk '{print $1}'; }

echo "Fetching bazelisk $bazelisk_ver shas…" >&2
bzlk="https://github.com/bazelbuild/bazelisk/releases/download/${bazelisk_ver}"
bzlk_darwin_arm64="$(fetch_sha "${bzlk}/bazelisk-darwin-arm64.sha256")"
bzlk_darwin_amd64="$(fetch_sha "${bzlk}/bazelisk-darwin-amd64.sha256")"
bzlk_linux_amd64="$(fetch_sha "${bzlk}/bazelisk-linux-amd64.sha256")"
bzlk_linux_arm64="$(fetch_sha "${bzlk}/bazelisk-linux-arm64.sha256")"

echo "Fetching Bazel $bazel_ver shas…" >&2
# Note: Bazel release artifacts use x86_64 (not amd64).
bz="https://releases.bazel.build/${bazel_ver}/release"
bz_darwin_arm64="$(fetch_sha "${bz}/bazel-${bazel_ver}-darwin-arm64.sha256")"
bz_darwin_amd64="$(fetch_sha "${bz}/bazel-${bazel_ver}-darwin-x86_64.sha256")"
bz_linux_amd64="$(fetch_sha "${bz}/bazel-${bazel_ver}-linux-x86_64.sha256")"
bz_linux_arm64="$(fetch_sha "${bz}/bazel-${bazel_ver}-linux-arm64.sha256")"

printf '%s\n' "$bazel_ver" > "$repo_root/.bazelversion"

cat > "$repo_root/tools/bazel.lock" <<LOCK
# Pinned build-system bootstrap — the only thing a clean checkout needs.
#
# The committed ./bazel wrapper downloads + checksum-verifies bazelisk (below),
# then bazelisk fetches Bazel (BAZEL_VERSION, mirrored in .bazelversion) and the
# wrapper hands it the matching BAZEL_SHA256 via BAZELISK_VERIFY_SHA256 — so the
# whole chain (launcher -> bazelisk -> Bazel) is hash-pinned end to end. Mirrors
# scripts/llama-server.lock.
#
# Bump with tools/update-bazel-lock.sh (refreshes .bazelversion + every sha here).
BAZELISK_VERSION=${bazelisk_ver}
SHA256_darwin_arm64=${bzlk_darwin_arm64}
SHA256_darwin_amd64=${bzlk_darwin_amd64}
SHA256_linux_amd64=${bzlk_linux_amd64}
SHA256_linux_arm64=${bzlk_linux_arm64}

# Bazel itself. BAZEL_VERSION must match .bazelversion (the wrapper checks, and
# only pins the sha when they agree). bazelisk downloads the binary; the wrapper
# verifies it via BAZELISK_VERIFY_SHA256 (shas from releases.bazel.build).
BAZEL_VERSION=${bazel_ver}
BAZEL_SHA256_darwin_arm64=${bz_darwin_arm64}
BAZEL_SHA256_darwin_amd64=${bz_darwin_amd64}
BAZEL_SHA256_linux_amd64=${bz_linux_amd64}
BAZEL_SHA256_linux_arm64=${bz_linux_arm64}
LOCK

echo "Updated tools/bazel.lock + .bazelversion (bazelisk ${bazelisk_ver}, Bazel ${bazel_ver})."
