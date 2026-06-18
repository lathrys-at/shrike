#!/usr/bin/env bash
# Parity gate: the Bazel py_wheel is the release artifact;
# the hatchling wheel is the dev-lane build (pip install -e). The Python payload and
# metadata must stay equivalent so the dev lane exercises what ships. Builds both and
# compares the substantive metadata (name, deps, extras, project URLs,
# entry-point-bearing fields, long description) and the packaged `shrike/` source
# files. The Bazel wheel ADDITIONALLY ships the `shrike_native/` package (the
# platform-tagged wheel carries the extension); the gate asserts that payload is
# present (incl. the compiled _native.so) and excludes it from the file comparison
# (the hatchling wheel is pure Python by design; the dev venv gets shrike_native from
# scripts/build-native.sh instead).
#
# It checks SEMANTIC equivalence, not byte-identity, because the two builders emit a
# different wheel Metadata-Version (rules_python py_wheel = 2.1, hatchling = 2.4): the
# license and homepage live in differently-named fields (License vs License-Expression;
# Home-page vs Project-URL). Those format fields are tolerated — the license *value*
# and the URLs are still verified. The version is reported, not gated (identical on a
# clean release/rc tag, differing only in the dev/dirty local segment off-tag). Run
# before tagging a release.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "Building the Bazel wheel…" >&2
bazel_whl="$("$repo_root/tools/build-wheel.sh" "$work/bazel")"

echo "Building the hatchling wheel…" >&2
# pyproject.toml lives in the shrike-py/ unit — point uv at it.
uv build --wheel "$repo_root/shrike-py" --out-dir "$work/hatch" >&2
hatch_whl="$(ls "$work"/hatch/*.whl | head -1)"

python3 - "$bazel_whl" "$hatch_whl" <<'PY'
import sys, zipfile
from email.parser import Parser

# Compared as sets (field order differs between builders).
SET_FIELDS = {"Requires-Dist", "Provides-Extra", "Classifier", "Project-URL"}
# Reflect the wheel Metadata-Version difference (py_wheel 2.1 vs hatchling 2.4) — the
# license/homepage live in renamed fields. The license value + URLs are verified
# separately, so the raw fields themselves are tolerated.
TOLERATE = {"Metadata-Version", "License", "License-Expression", "License-File", "Home-page"}
REPORT_ONLY = {"Version"}
# hatchling ships a generated _version.py; the Bazel wheel resolves the version via
# importlib.metadata instead (an accepted divergence — see src/shrike/__init__.py).
IGNORE_FILES = {"shrike/_version.py"}


def load(path):
    with zipfile.ZipFile(path) as z:
        meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        msg = Parser().parsestr(z.read(meta_name).decode())
        names = {n for n in z.namelist() if ".dist-info/" not in n}
        native = {n for n in names if n.startswith("shrike_native/")}
        files = names - native - IGNORE_FILES
    return msg, files, native


def lic(meta):  # license value, across the 2.1 License / 2.4 License-Expression split
    return meta.get("License-Expression") or meta.get("License")


def compare(label, a, b):
    if a != b:
        print(f"  MISMATCH {label}:\n    bazel:    {sorted(a) if isinstance(a, set) else a!r}\n"
              f"    hatchling:{sorted(b) if isinstance(b, set) else b!r}")
        return False
    return True


bazel_meta, bazel_files, bazel_native = load(sys.argv[1])
hatch_meta, hatch_files, _ = load(sys.argv[2])

ok = True
# The release wheel must actually carry the extension — a pure-Python
# shrike-mcp cannot run (the kernel is shrike_native).
if "shrike_native/_native.so" not in bazel_native:
    print(f"  MISMATCH native payload: bazel wheel lacks shrike_native/_native.so "
          f"(found: {sorted(bazel_native)})")
    ok = False
for field in sorted((set(bazel_meta.keys()) | set(hatch_meta.keys())) - TOLERATE - REPORT_ONLY):
    a, b = bazel_meta.get_all(field, []), hatch_meta.get_all(field, [])
    ok &= compare(f"{field} (set)", set(a), set(b)) if field in SET_FIELDS else compare(field, a, b)

ok &= compare("license expression", lic(bazel_meta), lic(hatch_meta))
# rstrip: py_wheel writes a trailing newline after the body that hatchling doesn't —
# cosmetic, renders identically on PyPI.
ok &= compare("description body",
              bazel_meta.get_payload().rstrip("\n"), hatch_meta.get_payload().rstrip("\n"))
ok &= compare("packaged files", bazel_files, hatch_files)

bv, hv = bazel_meta.get("Version"), hatch_meta.get("Version")
if bv != hv:
    print(f"  note: version differs (bazel={bv} hatchling={hv}) — expected only off a clean tag")
print(f"  note: tolerated metadata-format fields (py_wheel 2.1 vs hatchling 2.4): "
      f"{', '.join(sorted(TOLERATE))}")

if ok:
    print("PARITY OK — Bazel and hatchling wheels carry equivalent metadata + files.")
    sys.exit(0)
print("PARITY FAILED — see mismatches above.", file=sys.stderr)
sys.exit(1)
PY
