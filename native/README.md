# Shrike native workspace

The Rust compute plane behind Shrike's Python facades (epic #265). A single
Cargo workspace; Bazel (`crate_universe`) resolves third-party crates from the
one `Cargo.lock` here, and `rules_rust` builds everything against the hermetic
CPython toolchain.

## Crates

| Crate | Role | pyo3? |
|---|---|---|
| `shrike-ffi` | FFI conventions: error taxonomy (`NativeError`: `invalid_input` / `unavailable` / `internal`) + the marshaling rules (crate docs) | no |
| `shrike-py` | The one PyO3 binding crate — builds the `shrike_native._native` extension module | **yes** |
| `_demo` | The #247 polyglot proof (kept as the smoke target) | yes |

**Layering rule (epic #265 convention 5):** `pyo3` may appear only in
`shrike-py` and `_demo`. Compute/kernel crates stay pure Rust so the eventual
no-CPython kernel (#279) is structural. Enforced by `//native:layering_check`.

## Conventions every native module follows

- **Protocol → Python facade → native module.** The Python protocol
  (`embedding_base.py`) is the interface truth; the facade (e.g. `OnnxBackend`)
  stays a plain Python class — patchable, `spec=`-able, strictly typed — and
  imports `shrike_native` lazily so a missing native install degrades to a
  clean `ImportError`. The native module is internal: **no test file imports
  it.**
- **Marshaling:** only strings, bytes, f32 vectors, i64 key arrays, and small
  JSON-able dicts cross the boundary; coarse, batched calls; zero-copy numpy
  interchange where arrays must cross. No live Python objects into worker
  threads.
- **GIL:** all compute under `py.detach(..)` (pyo3 ≥ 0.26's name for
  `allow_threads`).
- **Errors:** native code returns `shrike_ffi::NativeError`; `shrike-py` maps
  the kinds to `NativeInputError` (ValueError — the expected-bad-input tier,
  logged without traceback by the facades), `NativeUnavailableError`, and
  `NativeInternalError` (RuntimeError). `parallel_sum`/`checked_div` in
  `shrike-py` are the permanent executable exemplar.
- **Typing:** the `shrike_native` package ships `.pyi` stubs + `py.typed`;
  `//native/shrike-py:stubtest` (mypy.stubtest) fails when a Rust signature
  drifts from its stub.

## Building and testing

```bash
# Bazel (what CI runs — the gated native lane in test.yml):
./bazel test //native/...                  # crates, layering check, smoke, stubtest
./bazel build //native/shrike-py:wheel     # the shrike-native platform wheel (manual tag)

# Plain cargo (fast inner loop; .cargo/config.toml carries the macOS link flags):
cd native && cargo build && cargo test

# Build + install the extension into the active venv so the pip-lane tests
# exercise the facades against the real native module:
scripts/build-native.sh            # debug; --release for optimized
```

After changing any `Cargo.toml`: `cd native && cargo generate-lockfile` (or
`cargo update -p <crate>`), then commit `Cargo.lock` — MODULE.bazel pins the
Bazel crate graph to it.

## Packaging

The canonical release artifact is `//native/shrike-py:wheel` — a
`shrike-native` abi3 (cp312+) platform wheel carrying `_native.so`, the stubs,
and `py.typed`. The main `shrike-mcp` wheel stays pure Python; the native
package is an optional accelerator the facades probe for. Linux wheels use
plain `linux_*` tags until PyPI publishing is wired (#43) — manylinux auditing
happens then. ort linkage decision for #270 is recorded there: **load-dynamic
against the pinned onnxruntime the Python backend already installs** (no
`download-binaries`, no duplicated runtime, guaranteed version match for the
parity bake).

## Licensing inventory

All AGPL-3.0-compatible: pyo3 (MIT/Apache-2.0), usearch (Apache-2.0),
ort/onnxruntime (MIT), tokenizers (Apache-2.0), rusqlite (MIT),
image (MIT/Apache-2.0).
