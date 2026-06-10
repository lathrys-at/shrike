#!/usr/bin/env bash
# Bazel workspace status — emit STABLE_VERSION from the git tag (#245), so the
# Bazel py_wheel stamps the same tag-derived version hatch-vcs produces on the pip
# path. On a clean release tag `vX.Y.Z` this is exactly `X.Y.Z` (so the artifact ↔
# tag guarantee holds and the parity gate matches the hatchling build); off-tag it
# is a PEP 440 dev version `X.Y.(Z+1).devN+g<hash>[.dirty]`, mirroring
# setuptools-scm's default guess-next-dev scheme.
#
# Wired via .bazelrc `build --workspace_status_command`. Keys printed as
# `KEY value`; STABLE_* keys go in stable-status.txt (a change re-stamps).
set -euo pipefail

# `vX.Y.Z-<commits since tag>-g<hash>` (+ `-dirty`); empty if no tag yet.
describe="$(git describe --tags --long --dirty --match 'v[0-9]*' 2>/dev/null || true)"

if [ -z "$describe" ]; then
  version="0.0.0+unknown"
else
  dirty=""
  case "$describe" in *-dirty) dirty=".dirty"; describe="${describe%-dirty}" ;; esac
  hash="${describe##*-g}"          # trailing g<hash>
  rest="${describe%-g*}"           # vX.Y.Z-<n>
  count="${rest##*-}"              # <n>
  tag="${rest%-*}"                 # vX.Y.Z
  base="${tag#v}"                  # X.Y.Z

  if [ "$count" = "0" ] && [ -z "$dirty" ]; then
    version="$base"               # exact release tag
  else
    # Bump patch and mark a dev version with the commit count + hash (PEP 440).
    major="${base%%.*}"; rest2="${base#*.}"; minor="${rest2%%.*}"; patch="${rest2##*.}"
    version="${major}.${minor}.$((patch + 1)).dev${count}+g${hash}${dirty}"
  fi
fi

echo "STABLE_VERSION ${version}"
