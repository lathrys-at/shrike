#!/usr/bin/env bash
# Bazel-lane coverage (#262) — runs `bazel coverage` over the py suites and prints
# a per-file + total report from the merged lcov. Spawned server subprocesses ARE
# captured (each py_binary's rules_python bootstrap self-instruments off the
# inherited COVERAGE_DIR; the integration conftest gives each server its own
# COVERAGE_DIR subdirectory so the fixed-name pylcov.dat reports don't collide).
# The coverage-specific flags (instrumentation filter, the CC-collector bypass,
# the combined report) live in .bazelrc under `coverage --…`.
#
# Report-only, like CI's coverage posture: the fail_under ratchet stays on the
# pip lane (scripts/coverage.sh), which also remains the basis of the published
# number until the bazel number replaces it (tracked on #262). Note the metric
# difference when comparing: lcov totals are line coverage; coverage.py's total
# folds in branch coverage (pyproject sets branch=true), so the two can differ
# by a point or so.
#
# Usage:
#   scripts/coverage-bazel.sh             # py suites, prints report
#   scripts/coverage-bazel.sh --html      # also render htmlcov-bazel/ (needs genhtml)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

want_html=0
for arg in "$@"; do
  [ "$arg" = "--html" ] && want_html=1
done

# The non-embedding py suites — the same scope as scripts/coverage.sh's
# `-m "not embedding"` run. //shrike-py/tests/unit/... covers both the pre- and
# post-#259 target layout. A failing test still prints the (partial) report —
# bazel collects no coverage for a failed target, so the number is an
# undercount then — and the script exits with bazel's status at the end.
bazel_status=0
./bazel coverage //shrike-py/tests/unit/... //shrike-py/tests/integration:integration //shrike-py/tests/native:native || bazel_status=$?

report="$(./bazel info output_path)/_coverage/_coverage_report.dat"
if [ ! -f "$report" ]; then
  echo "no combined report at $report — did the coverage run fail?" >&2
  exit "${bazel_status:-1}"
fi

python3 - "$report" <<'PY'
import sys

per_file: dict[str, tuple[int, int]] = {}
cur = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("SF:"):
        cur = line[3:]
    elif line.startswith("LF:") and cur:
        per_file[cur] = (int(line[3:]), per_file.get(cur, (0, 0))[1])
    elif line.startswith("LH:") and cur:
        per_file[cur] = (per_file.get(cur, (0, 0))[0], int(line[3:]))

width = max(len(f) for f in per_file)
print(f"{'file':<{width}}  {'lines':>11}  {'cover':>6}")
total_lf = total_lh = 0
for f in sorted(per_file):
    lf, lh = per_file[f]
    total_lf += lf
    total_lh += lh
    print(f"{f:<{width}}  {lh:>5}/{lf:<5}  {100 * lh / lf if lf else 0.0:>5.1f}%")
print("-" * (width + 21))
print(f"{'TOTAL':<{width}}  {total_lh:>5}/{total_lf:<5}  {100 * total_lh / total_lf:>5.1f}%")
PY

if [ "$want_html" = 1 ]; then
  if command -v genhtml >/dev/null; then
    genhtml --quiet --output-directory htmlcov-bazel "$report"
    echo "HTML report: htmlcov-bazel/index.html"
  else
    echo "--html needs genhtml (brew install lcov / apt-get install lcov)" >&2
    exit 1
  fi
fi

# Surface a failed test run after the report (the number above is an
# undercount in that case — failed targets contribute no coverage).
if [ "$bazel_status" -ne 0 ]; then
  echo "warning: bazel coverage exited $bazel_status (test failures above); the report is partial" >&2
fi
exit "$bazel_status"
