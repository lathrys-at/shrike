#!/usr/bin/env bash
# Bazel workspace status — emit STABLE_VERSION from the git tag (#245), so the
# Bazel py_wheel stamps the same tag-derived version hatch-vcs produces on the pip
# path. On a clean tag this is exactly what setuptools-scm emits, so the artifact ↔
# tag guarantee holds and the parity gate matches the hatchling build:
#   - a release tag  `vX.Y.Z`      -> `X.Y.Z`
#   - a pre-release  `vX.Y.Z-rc.N` -> `X.Y.ZrcN`  (PEP 440-normalized)
# Off-tag it is a PEP 440 dev version: `X.Y.ZrcN.devM+g<hash>` continuing a
# pre-release, else `X.Y.(Z+1).devM+g<hash>` (setuptools-scm guess-next-dev) — both
# `[.dirty]`-suffixed for a dirty tree. (The project only tags vX.Y.Z / vX.Y.Z-rc.N,
# per CLAUDE.md; the bash stays self-contained so a clean checkout builds without a
# Python/hatch dependency on every build.)
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
  tag="${rest%-*}"                 # vX.Y.Z or vX.Y.Z-rc.N
  base="${tag#v}"                  # X.Y.Z   or X.Y.Z-rc.N

  # PEP 440-normalize a SemVer pre-release: `X.Y.Z-rc.N` -> `X.Y.ZrcN` (what
  # setuptools-scm/hatch-vcs emit), so the Bazel and hatchling wheels match.
  norm="${base/-rc./rc}"

  if [ "$count" = "0" ] && [ -z "$dirty" ]; then
    version="$norm"               # exact tag (release or rc) — matches the pip path
  elif [ "$base" != "${base%%-*}" ]; then
    # Off a pre-release tag: continue it with a dev segment (don't bump the patch).
    version="${norm}.dev${count}+g${hash}${dirty}"
  else
    # Off a final release tag: bump the patch (setuptools-scm guess-next-dev).
    major="${base%%.*}"; rest2="${base#*.}"; minor="${rest2%%.*}"; patch="${rest2##*.}"
    version="${major}.${minor}.$((patch + 1)).dev${count}+g${hash}${dirty}"
  fi
fi

echo "STABLE_VERSION ${version}"
