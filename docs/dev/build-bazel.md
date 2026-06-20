# Building Shrike with Bazel

Bazel is the authoritative build: CI's test gate, the release artifacts, and
the polyglot (Python + Rust + Swift) graph all run through it. You don't need
it for day-to-day iteration — the pip lane (below) is faster for that — but
anything you merge is ultimately proven by `bazel test`. The "why" lives in
the ADR in [`docs/dev/decisions.md`](decisions.md) ("Bazel as the polyglot build
system").

## Zero-install quick start

Bazel is not assumed on PATH and you should not install it. The committed
wrapper at the repo root bootstraps everything:

```bash
./bazel test //...
```

The wrapper downloads a pinned, sha256-verified bazelisk, which downloads the
pinned, sha256-verified Bazel named in `.bazelversion` (all hashes in
`tools/bazel.lock`). Everything lands under your XDG cache dir. CI invokes
the same wrapper, so local and CI builds are identical. If you have your own
Bazel and insist, `SHRIKE_BAZEL=/path/to/bazel ./bazel ...` overrides the
chain.

The first run is slow (toolchains, the crate graph, wheels); after that the
local disk cache makes unchanged targets free. Python itself is hermetic
(rules_python's 3.12) — no venv or system Python is involved.

## The two lanes

| | pip lane | Bazel lane |
|---|---|---|
| Purpose | fast iteration, debugging | the authoritative gate (CI, release) |
| Build | `scripts/build-native.sh` (bazel → venv) | `./bazel build/test ...` |
| Test | `pytest shrike-py/tests/unit shrike-py/tests/native -q` | `./bazel test //...` |
| Debug affordances | `-x`, `-s`, `pdb`, `--testmon`, warm xdist | `--test_output=errors`, per-target logs |
| Coverage | `scripts/coverage.sh` (the published number + the fail_under gate) | `scripts/coverage-bazel.sh` (#262; subprocess-capturing, report-only) |

Iterate on the pip lane; run `./bazel test //...` before a PR for anything
that touches BUILD files, the Rust workspace, or dependencies. The two lanes
share sources, so there is nothing to sync — but after a Rust change the pip
lane needs `scripts/build-native.sh` (pytest otherwise tests a stale
extension).

## Common invocations

```bash
./bazel test //...                       # everything non-manual: all crate tests,
                                         #   unit/native/integration py suites,
                                         #   stubtest, layering_check
./bazel test //shrike-py/tests/unit:unit           # one suite
./bazel test -- //... \
  //shrike-py/tests/integration:embedding_core \
  //shrike-py/tests/integration:embedding_semantic \
  //shrike-py/tests/integration:embedding_backends # what CI runs (adds the manual lanes)
./bazel build //shrike-py:wheel --stamp           # the platform wheel (or tools/build-wheel.sh)
./bazel build //shrike-py:sdist --stamp           # the sdist (or tools/build-sdist.sh)
./bazel build //shrike-skills:skill      # the create-cards.skill bundle (unversioned)
```

**Manual targets.** The embedding test lanes, the llama-server alias, and
the release artifacts are tagged `manual`, so `//...` never touches them —
that is what keeps the default lane from fetching hundreds of MB of models.
Name them explicitly when you want them (the embedding lanes need ~1 GB of
externals on first fetch, then they're cached).

**Running the embedding tests locally.** The Bazel embedding lane *is* the local
embedding-test path (it replaced the old `scripts/test-embedding.sh` +
`scripts/fetch-llama-server.sh` — #657). It's hermetic: the pinned llama-server
(`MODULE.bazel` http_archive) and the GGUF/ONNX fixtures (sha256-pinned
http_files) ride in as `data` deps, the conftest puts llama-server on `PATH` and
assembles the models into `$SHRIKE_TEST_MODEL_DIR`, so there is **no separate
fetch step**:

```bash
./bazel test //shrike-py/tests/integration:embedding_core      # out-of-process llama-server + GGUF lifecycle lane
./bazel test //shrike-py/tests/integration:embedding_semantic  # in-process ONNX semantic/search behaviour
./bazel test //shrike-py/tests/integration:embedding_backends  # the onnx/clip/llama backend zoo
```

For a non-Bazel manual run (the QA harness, a `serve --profile` with a llama
model) the single Python fetch source is `tests/integration/model_cache.py` (its
`cached_*_model_dir` / `download_with_retry` — no URL is spelled anywhere else);
bring your own `llama-server` on `PATH` or set `LLAMA_SERVER_PATH`.

**Read the full summary.** Bazel prints per-target results and an
`Executed N out of M` line — read that, not just the tail. A green tail with
a failed target above it has shipped breakage before.

**Versions are stamped, not stored.** `--stamp` runs
`tools/workspace_status.sh` (`git describe` → PEP 440) and feeds the wheel,
sdist, and `_version.py`. Off a tag you get a dev version; on a `v*` tag the
clean release version. The pip lane reads the same tag through hatch-vcs.

## The polyglot seam

- **Rust** comes in through rules_rust + crate_universe: `MODULE.bazel`'s
  `crate.from_cargo` consumes the workspace's `Cargo.toml`s and the single
  `Cargo.lock`, so cargo stays the Rust source of truth. A few crates need
  `crate.annotation` patches (usearch's cxxbridge symlinks; anki's
  proto-descriptor and FTL build scripts, fed from the checked-in
  descriptors and the `@anki_src` archive).
- **The PyO3 extension** (`//shrike-core/bindings/shrike-pyo3:shrike_pyo3_native`) builds as
  an abi3-py312 cdylib — no libpython at build time — and is wrapped into
  the importable `//shrike-core/bindings/shrike-pyo3:shrike_native` py_library that
  `//shrike-py/src/shrike` depends on. `:mobile_skeleton` is a build-only proof that
  the mobile feature set keeps compiling; `:stubtest` pins the `.pyi` stubs
  against the real module; `//shrike-core:layering_check` enforces the crate
  layering rules (pyo3 only in binding crates, no kernel→engine deps).
- **Swift** (the Apple Vision engine) is mobile-build-only since #496; on a
  mac it needs full Xcode when you build a target that includes it.
- **Python deps** are PyPI wheels via `pip.parse` — upstream native packages
  (anki, usearch, onnxruntime, tokenizers) are never built from source.
- **Embedding externals** (llama-server, fixture models) are
  sha256-pinned `http_archive`/`http_file` repos — hermetic, no
  HuggingFace flake.

## Regenerating the locks

Three lock layers, three triggers:

- **Python deps changed** (`pyproject.toml`): run
  `tools/update-requirements.sh` (uv pip compile, universal, hashed) and
  commit `requirements_lock.txt` + `requirements_sdist_lock.txt`.
- **Rust deps changed** (`shrike-core/Cargo.lock` moved): the next Bazel run
  re-splices the crate graph and refreshes `MODULE.bazel.lock` — commit it.
  If the splice hangs (~600s then "Timed out") run with
  `CARGO_BAZEL_ISOLATED=0 ./bazel test //...` to reuse your warm `~/.cargo`.
  A new cargo dep also needs a matching `deps` entry in the consuming
  `BUILD.bazel` (cargo building ≠ bazel building), and a brand-new workspace
  crate needs its manifest listed in `MODULE.bazel` *and* in
  `layering_check`'s `data`.
- **Bazel itself**: `.bazelversion` + `tools/bazel.lock` pin bazelisk and
  Bazel by hash. Bump them together.

## Caching

Locally, Bazel's default on-disk caches are enough; nothing to configure. A
`.bazelrc.user` (gitignored, `try-import`ed) is the hook for personal flags
— CI writes `--disk_cache`/`--repository_cache` there via the
`.github/actions/bazel-setup` composite.

In CI the disk cache (action + **test** results) and repository cache
(downloads + Bazel 9's extracted-repo contents cache) persist through
`actions/cache`. PRs can only restore from `main`'s cache scope, and the
test workflow never runs on `main` — so `warm-cache.yml` runs the same
invocation on `main` daily (and on lock-file merges) to seed it. An
unchanged target on a PR replays as `(cached) PASSED`. The documented
upgrade path when `actions/cache`'s 10 GB eviction budget bites: a real
remote cache (BuildBuddy / self-hosted bazel-remote), swapped in by flag in
the composite action.

## Coverage

`scripts/coverage-bazel.sh` runs `bazel coverage` over the py suites and
prints a per-file report from the merged lcov
(`bazel-out/_coverage/_coverage_report.dat`; `--html` renders it with
genhtml). The pieces that make it correct, all wired in `.bazelrc` and the
test plumbing (#262):

- **The spawned server subprocess is captured.** A py_binary's rules_python
  bootstrap self-instruments when it inherits `COVERAGE_DIR`, but every
  process writes the same fixed `pylcov.dat` name — so the integration
  conftest gives each spawned server its own `COVERAGE_DIR` subdirectory and
  Bazel's lcov merger picks them all up.
- **Serial under coverage.** xdist workers are execnet subprocesses the
  in-process tracer can't see, so the pytest runner drops `-n` when
  `COVERAGE_DIR` is set (a parallel coverage run silently measures only the
  controller).
- **The CC collector is bypassed.** rules_rust instruments the cdylib
  whenever the global coverage flag is on, and the resulting `.profraw`
  trips Bazel's CC collector (no `LLVM_PROFDATA` on the autodetected
  toolchain); `IGNORE_COVERAGE_COLLECTION_FAILURES=1` downgrades that to a
  no-op. Rust coverage is a separate lane (cargo), not measured here.

The published number and the `fail_under` ratchet stay on the pip path
(`scripts/coverage.sh`, `coverage.yml`) until the Bazel number proves out
side-by-side — note lcov totals are line coverage while coverage.py's total
folds in branches, so expect a point or so of difference.

## Optional: direnv

`.envrc` activates the coexistence `.venv` (for the pip lane) when you have
direnv installed and have run `direnv allow`. Without direnv it does
nothing, and nothing depends on it — `./bazel` needs no environment at all.
