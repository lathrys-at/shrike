# `tools/` — invoked by the build

`tools/` holds everything the **build** runs: Bazel macros and toolchain
helpers, the version-pin locks and their tripwires/writers, the sdist/wheel/
requirements builders, and the workspace-status stamp. `//tools` is the Bazel
idiom for exactly this.

## The boundary

The repo has three top-level directories for non-package code. The line between
them is **who invokes it**:

| Directory | Who invokes it | Holds |
|-----------|----------------|-------|
| [`bin/`](../bin/README.md) | the end user / a spawned server | Shipped/runnable product entry points (`py_binary` launchers over `//src/shrike:shrike`). **Load-bearing**, not cruft. |
| `tools/` | **the build** (Bazel, CI) | Build-system internals: Bazel macros, version-pin locks + their writers/checkers, sdist/wheel/requirements builders, workspace-status, hermetic-toolchain CI smoke tests. |
| [`scripts/`](../scripts/README.md) | **a human** at a dev shell | Dev/maintenance entry points: environment setup, the native build, coverage runners. |

A file's home follows the strongest coupling. A *version-pin lock* is consumed
by the build (Bazel reads it, CI cache keys hash it, a `py_test` validates it),
so the lock — and the script that **regenerates** it, and the tripwire that
**checks** it — all live here, together. A script a developer runs by hand to
fix up their environment lives in `scripts/`, even if its output feeds a build.

## Contents

### Bazel build system (`//tools/bazel`)
- `bazel/` — shared Bazel macros + launchers (`pytest_test`, the sdist builder, `defs.bzl`/`sdist.bzl`).
- `bazel.lock` — the pinned build-system bootstrap (bazelisk + Bazel + their shas), consumed by the committed `./bazel` wrapper.
- `update-bazel-lock.sh` — regenerates `bazel.lock` + `.bazelversion`. Mirrors `update-llama-lock.sh`.
- `workspace_status.sh` — the `--workspace_status_command` stamp (git version → `STABLE_VERSION`, consumed by `//:wheel`/`//:sdist`).

### Version-pin locks
- `llama-server.lock` — the pinned llama.cpp release tag + per-platform SHA256s. Consumed by `MODULE.bazel`'s `llama_server_*` http_archives (duplicated there because Bazel can't read the lock at module-resolution time) and by the CI model-cache keys.
- `update-llama-lock.sh` — regenerates `llama-server.lock` (pins a tag, hashes each platform tarball). Mirrors `update-bazel-lock.sh`.
- `check_llama_lock.py` — the de-dup tripwire (#566): asserts `llama-server.lock` and `MODULE.bazel` pin the same tag + shas. Runs both as a Bazel `py_test` (`//tools:llama_lock_in_sync_test`) and as a pip-lane pytest unit (`tests/unit/test_llama_lock_sync.py`).

> The llama-lock concern was previously split — writer + data in `scripts/`,
> checker here. #695 reunited it in `tools/`, beside its sibling build-pin
> `bazel.lock`, since the lock is consumed by the build and the writer is its
> regenerator (parallel to `update-bazel-lock.sh`).

### Packaging / requirements
- `build-wheel.sh` / `build-sdist.sh` — build the release `//:wheel` / `//:sdist` Bazel targets.
- `check-wheel-parity.sh` — asserts the Bazel wheel's metadata matches `pyproject.toml`.
- `update-requirements.sh` — regenerates the pinned `requirements*.txt` from the lock inputs.
- `sdist-requirements.in` — the sdist builder's build-tool deps (kept out of the runtime lock).

### Hermetic-toolchain CI smoke tests
- `import_spike.py` — `//tools:import_spike` (#242): the native-dependency wheels (`anki`, `usearch`, `onnxruntime`) import and *run* on Bazel's hermetic CPython, on every target platform.
- `library_smoke.py` — `//tools:library_smoke` (#243): the `shrike` package + its declared deps import cleanly and a pure function runs.

  These two began as one-off Phase 0/Phase 1 spikes, but they are **live,
  non-`manual` `py_test` targets** — every `./bazel test //...` runs them, and
  they still guard the hermetic-toolchain and library-wiring assumptions. They
  are build-invoked checks, so `tools/` is their correct home today. (A longer
  term move under `tests/` is deferred so it doesn't collide with the in-flight
  test-layout reshape, epic #694.)

## Not Bazel-ified (yet)

Several shell helpers here are still plain scripts rather than `sh_binary`/
`sh_test`/`genrule` targets; expressing the dev/maintenance scripts as Bazel
idioms is tracked separately (#700). Categorizing a file into `tools/` does not
require rewriting it as a Bazel rule.
