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
job (one `bazel test` over the full graph plus the embedding halves). The
expensive cross-platform ARM legs are opt-in by label — `rc` selects all legs,
`macos` and `linux-arm` select one each — and never run on a plain PR or on merge
to `main`.

## Coverage

Coverage lives in its own workflow and is **reported, never enforced as a CI
gate**. The `fail_under` target in `[tool.coverage.report]` is enforced only
locally. Run it locally to keep the number healthy:

```bash
scripts/coverage.sh            # full suite; prints report, exits non-zero below fail_under
scripts/coverage.sh --html     # also writes htmlcov/index.html
```

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
