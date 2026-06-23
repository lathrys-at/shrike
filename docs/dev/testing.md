# Development setup, tests, and checks

## Setting up

```bash
scripts/dev-setup.sh        # creates .venv, installs shrike-py[dev], builds the native extension
source .venv/bin/activate
```

This is one idempotent step — re-run it any time as a repair button. It picks
the pinned interpreter (`.python-version`, via pyenv if present, else
`python3.12`), installs the editable package plus dev tooling, and builds the
Rust `shrike_native` extension. **`pip install` alone does not build the
extension** — that is a separate Bazel build, run by `scripts/build-native.sh`.

`dev-setup.sh` roots at **the checkout you run it from** (the current git
worktree), not at the script's location, so each worktree gets its **own**
isolated `.venv` and native extension. This is load-bearing: the native
extension is pip-installed into `$VIRTUAL_ENV` and its staleness stamp lives
there, so two worktrees sharing one venv clobber each other's `.so` and stamp —
`pytest` then imports the wrong build. In a worktree, run `scripts/dev-setup.sh`
first and activate the venv it prints (`source <worktree>/.venv/bin/activate`);
never activate another checkout's. `build-native.sh` refuses a `$VIRTUAL_ENV`
that lives outside the current checkout rather than cross-wiring it.

Python 3.12 is the supported interpreter (the `anki` package requires Python
≥ 3.11). After setup, refreshes of the native extension happen for you: with
direnv, `.envrc` rebuilds a stale extension on `cd`; without it, `pytest` fails
loudly (before importing the extension) rather than silently loading a stale
`.so`.

Cacheable dev artifacts live under a Shrike-owned cache (never `/tmp`), in two
tiers by lifetime:

- **Shared dev cache `~/.cache/shrike-dev/`** (`$XDG_CACHE_HOME` honored) — the
  checkout-*independent* artifacts shared across worktrees: the `./bazel`/bazelisk
  toolchains (under `build/`) and the downloaded test-fixture models (under
  `models/`, overridable via `SHRIKE_TEST_MODEL_DIR`), each pinned under a
  version-encoded subdir so two checkouts can't collide or serve a stale artifact.
- **Per-checkout cache `<repo>/.cache/`** (gitignored) — cheap, checkout-*specific*
  scratch and derived data, and the home for any throwaway file.

The `shrike-dev` namespace is deliberately distinct from the *application's* own
runtime cache (on Linux `~/.cache/shrike/` via XDG — see
[`server-runtime.md`](server-runtime.md)), which it would otherwise collide with.

## Running the tests

```bash
pytest shrike-py/tests/unit -v                        # fast, no server
pytest shrike-py/tests/integration -v -m integration  # starts a real server subprocess
```

The integration suite shares one server with a per-test reset. All non-embedding
integration tests share a single session-scoped server (one boot per xdist
worker); an autouse fixture resets the collection to a pristine baseline after
each test. The clients track what a test mutated so the reset is cheap. So a test
always starts clean, and even collection-wide assertions hold regardless of run
order — **you don't need to clean up after yourself**. Just don't assume the
collection is empty mid-suite; prefer asserting on your own deck or tag.

For the rare test that needs an exclusive, un-reset collection (e.g.
collection-wide tag counts, which the reset can't restore because Anki keeps the
tag registry), use the `isolated_server` / `isolated_mcp` / `isolated_runner`
fixtures, which spawn a dedicated collection. Embedding tests use their own
`collection_server` and are untouched by the reset.

## Performance lane

The perf harness (#865) is a **manual** lane — off the per-PR critical path — that
times gold workflows against deterministic corpora and reports latency
distributions. Run it by name:

```bash
# kernel-isolation run (the #445 hotspot class, no model inference):
scripts/build-native.sh --release --synthetic   # OPTIMIZED (-c opt) + synthetic
scripts/perf.sh --profile stub --size 5000 --variant text --workloads search-batch,rebuild,upsert-batch

# end-to-end run (real onnx + CLIP; models fetched to the model cache):
scripts/build-native.sh --release                # optimized extension
scripts/perf.sh --profile real --size 5000 --variant text+image --workloads search-batch
```

**Build optimized.** The runner times whatever extension is staged in the venv, and
the default `build-native.sh` is `fastbuild` (unoptimized — meaningless for perf).
Pass `--release` (`-c opt`); the runner records whether the build was optimized in
the result conditions (and warns + refuses to diff a debug run against a release
one), so a debug-build number can't be mistaken for a real one.

Both modes boot the **same** harness from a config profile
(`tests/manual/perf/profiles/perf-{stub,real}.yml`); the only difference is the
embedder, and the two are comparable because they share the modality shape.
`--profile stub` selects `runtime: synthetic`, which a lean build refuses — hence
the `--synthetic` extension build. To benchmark a different engine set, pass
`--profile-path PATH` (a path-free profile YAML of the same shape) in place of a
built-in `--profile`.

- **Workloads** drive the transport-neutral **actions API** directly (the
  maintained serving path, off the FastMCP transport — we measure the system, not
  the wire adapter). The read/write ops come in two shapes on the batching axis:
  `search-{batch,seq}`, `upsert-{batch,seq}`, `delete-{batch,seq}` (one call with N
  items vs N calls of one), plus `rebuild`, `reconcile`, and `ingest`.
- **Two-phase timing.** A write returns once committed with index/derived
  maintenance *enqueued*, so each write/reconcile workload is timed in two phases —
  `response` (the action returns) and `settle` (drain to quiescence) — and reports
  both plus their per-iteration `total`. Read workloads have no tail and report
  `response` only. Under `--profile stub` the synthetic embedder makes
  `settle`/`total` reflect orchestration cost only, not real embed/index work; the
  runner prints a note when a settling workload runs under it.
- **`--ops N`** scales the operations each data-plane workload performs per timed
  iteration (search queries, upsert/delete notes, reconcile drift) — the N in
  `N ops × M repeats`, where `--repeats` is M. Default 100; `rebuild` ignores it
  (one O(collection) pass). It's a comparison invariant: runs at different N don't
  diff.
- **Sizes** 500 / 5k / 50k notes; **variants** `text` and `text+image`. Corpora
  are deterministic, built through the real write path, and cached + gitignored
  under `.cache/perf/corpora/`.
- **Output format.** The terminal table is fixed-width by default;
  `--output-format table` renders it (and the baseline diff) as a GitHub-flavored
  markdown table to paste into a PR/issue comment. The result JSON is written
  either way.
- **Results** are distributions (p50/p90/p99/max) plus the conditions they were
  taken under (machine, build, corpus, mode), written to `.cache/perf/runs/`.
  `--baseline <result.json>` diffs a prior run (per phase) and refuses to compare
  across mismatched conditions. Logging is captured to an in-memory buffer and
  flushed to `run.log` beside the result — never to the terminal, so it can't
  contaminate the timings.
- The pure pieces (distribution math, the artifact, the diff) are unit-tested on
  the per-PR lane (`//shrike-py/tests/manual/perf:pure_test`); the corpus and the
  boot+drive are `manual` Bazel targets (the latter needs `--define
  shrike_synthetic=on`).

### Profiling a run

To turn "this workflow is slow" into "this line is slow", add `--instrument` to
profile a **single** workload under a sampling profiler. The default is
[py-spy](https://github.com/benfred/py-spy) `--native` — the one profiler that
merges the Python harness and the Rust kernel into a single flamegraph (a hot
frame reads `run.py → search_notes → kernel.search → usearch…` across the
boundary):

```bash
pip install py-spy                                  # a manual-lane dev tool
scripts/build-native.sh --release --synthetic --frame-pointers
sudo scripts/perf.sh --profile stub --size 5000 --variant text --workloads search-batch --instrument
```

`--instrument` takes an optional **tool**, with `--instrument-arg` to pass options
through to it (repeatable; one token each):

| `--instrument[=tool]` | Tool | What you get | Platform |
|-----------------------|------|--------------|----------|
| `--instrument` (or `=py-spy`) | py-spy | `flame-<workload>.svg` — **merged** Python + Rust flamegraph | Linux/Windows (native); macOS Python-only |
| `--instrument=samply` | [samply](https://github.com/mstange/samply) | `profile-<workload>.json` — deep Rust detail, opaque Python frames | Linux, macOS-arm64 |
| `--instrument=xctrace` | Apple Instruments | `profile-<workload>.trace` — deep Rust detail, opaque Python frames | macOS only |

```bash
# macOS: native Rust detail (Python frames are opaque), no sudo needed
scripts/perf.sh --profile stub --size 5000 --workloads search-batch --instrument=samply
scripts/perf.sh --profile stub --size 5000 --workloads search-batch --instrument=xctrace
samply load .cache/perf/runs/<run>/profile-search-batch.json   # opens the Firefox-profiler UI
open   .cache/perf/runs/<run>/profile-search-batch.trace        # opens Instruments.app

# tune the tool: e.g. samply at 4 kHz
scripts/perf.sh ... --instrument=samply --instrument-arg=--rate --instrument-arg=4000
```

- **One workload per instrumented run.** `--instrument` profiles a single
  `--workloads` entry (it errors on a list) and writes its artifact next to that
  run's `result.json` under `.cache/perf/runs/`.
- **Build with `--frame-pointers`.** An optimized build drops frame pointers, which
  degrades native unwinding; the flag forces them across the Rust crates. Keep it
  OUT of a clean-timing build — a reserved register would skew the distribution, so
  it's a separate profiling build.
- **The merged view is Linux's.** py-spy's `--native` unwinding is Linux/Windows
  only; on macOS the flag is dropped and the flamegraph is Python-only (the kernel
  shows as opaque native leaves). For the cross-boundary Python+Rust view on a Mac,
  run py-spy `--native` inside a Linux container; for Rust hotspots locally, reach
  for `--instrument=samply` / `=xctrace`.
- **`sudo` is py-spy's.** py-spy attaches via the OS process-inspection API, which
  usually needs root (especially on macOS); run the whole thing under `sudo`
  (preserve `PATH`/`VIRTUAL_ENV` so it finds the venv interpreter). samply and
  xctrace launch the target themselves and need no elevation.

Numeric per-span stage timings (parse → write → derive → embed → index, from the
kernel's `tracing` spans) are the observability work (#800); the flamegraph already
gives the visual breakdown. Perf is checked by hand against a stored baseline — there is
no automated regression gate (a deliberate choice; see `decisions.md`).

## The native (Rust) workspace

The Rust workspace lives in `shrike-core/` (run `cargo` from there for
`fmt`/`clippy`/`test`). The `shrike_native` extension is rebuilt into the venv by
`scripts/build-native.sh`, which builds it **via Bazel** — the same `_native.so`
the release wheel ships, so the inner loop and the canonical artifact share one
build graph. You rarely run it by hand: direnv rebuilds a stale extension on
`cd`, and `pytest` aborts loudly if the `.so` is stale.

The full local gate for a native change:

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd shrike-core && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
./bazel test //...      # the authoritative CI lane: all crate tests + layering check + py suites
```

**Bazel is not on `PATH`** — use the committed `./bazel` launcher at the repo
root. It bootstraps bazelisk and the pinned Bazel from `.bazelversion`, the same
entry point CI uses. See [`build-bazel.md`](build-bazel.md) for the operational
guide.

### Rust test taxonomy & quality bar

The bar is **exhaustive, fast, API-level tests against each crate's public
surface** — a unit suite that stands on its own, not one that leans on the
Python integration suite as a backstop. Aim adversarial: a test earns its place
by trying to *break* an invariant (boundaries, negative cases, malformed input,
ordering, idempotence), never by re-asserting the happy path. The categories,
and where each belongs:

| Category | What it is | Where it lives |
|----------|-----------|----------------|
| **Unit** | One function/type against its contract; table-driven edge cases. | `#[cfg(test)] mod tests` inline in `src/`. |
| **Property / generative** | An invariant over many generated inputs (round-trip, idempotence, an oracle cross-check against a simpler reference). | Inline, using `proptest` (below). |
| **Oracle / differential** | Output checked against an independent reference — a golden fixture, a brute-force re-implementation, or the frozen Python spec (fusion vs `search_fusion.py`). | Inline or `tests/`. |
| **Robustness / fuzz-style** | "Never panics, only `Err`s" over mutated/garbage bytes at a trust boundary (schemas deserialize, image decode, SSRF URL parsing, MIME sniff). | Inline, driven by `proptest`. |
| **Integration (in-crate)** | Cross-module behaviour reachable purely in Rust — concurrency, persistence, FFI lifecycle. | `tests/*.rs` (its own binary). |

**The generator: `proptest`.** Property/generative/fuzz tests use
[`proptest`](https://docs.rs/proptest): `Strategy`-generated inputs, the
`proptest!` macro, `prop_assert*!`, and model-based oracle traces (a generated
`Vec` of ops replayed against a `BTreeMap`/`BTreeSet` reference). The canonical
exemplar is `shrike-core/contracts/shrike-store/src/lib.rs` — the FxHasher
injectivity property and the `FxI64Map`/`FxI64Set`-vs-std oracle traces. The win
over a hand-rolled PRNG is **shrinking**: a failure is reduced to a minimal
witness, not a raw seed. The two trust-boundary properties to preserve when
extending: the reference must be **independently structured** (not a clone of the
impl), and every sweep must be **hermetic** — pure code only, never live I/O
(no `getaddrinfo`, no sockets). `cargo-fuzz`/`miri` remain a separate scheduled,
non-`//...` lane (see the testing ADRs in [`decisions.md`](decisions.md)).

**Wiring.** Inline tests run under each crate's `rust_test(crate = ":<lib>")`
target (`srcs` globs `src/**/*.rs`). To use `proptest` in a crate: add `proptest =
{ workspace = true }` to its `[dev-dependencies]` and `@crates//:proptest` to its
`rust_test` `deps`. A new `tests/*.rs` binary needs its own `rust_test` target with
explicit `deps`.

## Linting and type checking

All three must pass cleanly:

```bash
ruff check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/bindings/shrike-pyo3/python/
ruff format --check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/bindings/shrike-pyo3/python/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
```

`shrike-core/bindings/shrike-pyo3/python/` is the extension's Python shim. It sits outside
`src/`, so it must be named explicitly or it falls into no lint scope.

CI runs on every PR (`.github/workflows/test.yml`): a `lint` job and a `tests`
job (one `bazel test` over the full graph plus the embedding lanes). The
expensive cross-platform ARM legs are opt-in by label — `rc` selects all legs,
`macos` and `linux-arm` select one each — and never run on a plain PR or on merge
to `main`.

## Coverage

Coverage is two numbers from two tools — Python (`coverage.py`) and Rust
(`cargo-llvm-cov`) — both living in the `Coverage` workflow, both **reported,
never enforced as a CI gate** (the rationale, and why off-gate, is in the coverage
ADR in [`decisions.md`](decisions.md)). The floors are local ratchets: Python's
`fail_under` in `[tool.coverage.report]`, Rust's `--fail-under-lines`. Run either
locally to keep the number healthy:

```bash
scripts/coverage.sh                        # Python: full suite; report, exits non-zero below fail_under
scripts/coverage.sh --html                 # also writes htmlcov/index.html
scripts/coverage-rust.sh                    # Rust: shrike-core workspace; prints the per-crate table
scripts/coverage-rust.sh --html            # also writes shrike-core/target/llvm-cov/html/index.html
scripts/coverage-rust.sh --fail-under-lines 88   # local ratchet
```

`scripts/coverage-rust.sh` needs `cargo-llvm-cov` + the `llvm-tools-preview`
component (`rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
--locked`) and excludes the binding crates (their contract is the FFI boundary, not
cargo coverage — see the ADR). Neither coverage number rides Bazel: `bazel
coverage` can't see the spawned server subprocess, so each language uses its own
native tool.

A plain `pytest --cov=shrike` reads well below the real number because it can't
see the spawned server subprocess. The capture happens through a committed `.pth`
hook (`tools/coverage_subprocess.pth`) that imports coverage only when
`COVERAGE_PROCESS_START` is set. `scripts/coverage.sh` wires this up; the
coverage workflow runs the identical command, so the numbers are comparable.

`pytest-xdist` (`-n auto`) parallelizes across cores and roughly halves the
server-spawn-bound integration suite. The coverage hook fires for each xdist
worker and each spawned server, so `coverage combine` merges them to one total.
CI runs `-n auto`; locally the default stays serial so `-x`, `-s`, and `pdb`
keep working.

## Running the server manually

```bash
# Directly (foreground):
python -m shrike.server --collection /path/to/collection.anki2

# Via the CLI (daemon):
shrike server start --collection /path/to/collection.anki2
shrike server status
shrike server stop
```
