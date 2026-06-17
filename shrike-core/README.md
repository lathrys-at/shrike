# Shrike native workspace

The Rust compute core (epics #265 → #279/#332). The kernel owns the collection,
the vector index, the derived-text store, and search fusion; the Python side is
the assembling harness (server, CLI, client). A single Cargo workspace; Bazel
(`crate_universe`) resolves third-party crates from the one `Cargo.lock` here,
and `rules_rust` builds everything against the hermetic CPython toolchain.

## Crates

| Crate | Role | pyo3? |
|---|---|---|
| `shrike-kernel` | **The kernel**: collection ops, index orchestration, derived-text ingest, recognition sweeps, RRF fusion. Owns the process's tokio runtime; pure plugin host (engines are injected, never named) | no |
| `shrike-collection` | anki via its protobuf service layer — the **only** anki coupling | no |
| `shrike-index` | Per-modality USearch HNSW vector index | no |
| `shrike-derived` | FTS5 trigram engine (substring/fuzzy search, OCR rows) | no |
| `shrike-engine-api` | The engine contract (#342): kernel-facing async traits, sync compute traits, the `Blocking` adapter, the batch-safety probe | no |
| `shrike-embed` | ort/tokenizers text + CLIP embedding engines | no |
| `shrike-embed-remote` | Text embedding over any OpenAI-compatible endpoint | no |
| `shrike-describe-remote` | VLM image→text description over OpenAI-compatible chat completions (#433) | no |
| `shrike-llama-server` | llama-server subprocess lifecycle (spawn/health/reap/stop) — a manager, not an engine | no |
| `shrike-recognize-apple` | Apple Vision OCR engine (Swift glue behind Rust, macOS 15+; stub elsewhere) | no |
| `shrike-schemas` | serde+schemars wire types — canonical; `shrike/schemas.py` binds them | no |
| `shrike-ffi` | FFI conventions: error taxonomy (`NativeError`: `invalid_input` / `unavailable` / `internal`) + the marshaling rules (crate docs) | no |
| `shrike-pyo3` | The one PyO3 binding crate — builds the `shrike_native._native` extension module | **yes** |

**Layering rule:** `pyo3` may appear only in `shrike-pyo3`. Every other crate
stays pure Rust, which is what makes the kernel deployable in non-Python hosts.
Enforced by `//shrike-core:layering_check`.

## Conventions every native module follows

- **Protocol → Python facade → native module.** The Python protocol
  (`embedding_base.py`, `recognition.py`) is the harness-side interface truth;
  the facade (e.g. `OnnxBackend`) stays a plain Python class — patchable,
  `spec=`-able, strictly typed — and hands the kernel a native engine
  composition at attach. The binding surface itself is tested in
  `tests/native/`; the rest of the suite goes through the facades.
- **Marshaling:** only strings, bytes, f32 vectors, i64 key arrays, and small
  JSON-able dicts cross the boundary; coarse, batched calls; zero-copy numpy
  interchange where arrays must cross. No live Python objects into worker
  threads.
- **GIL:** all compute under `py.detach(..)` (pyo3 ≥ 0.26's name for
  `allow_threads`).
- **Errors:** native code returns `shrike_ffi::NativeError`; `shrike-pyo3` maps
  the kinds to `NativeInputError` (ValueError — the expected-bad-input tier,
  logged without traceback by the facades), `NativeUnavailableError`, and
  `NativeInternalError` (RuntimeError). `parallel_sum`/`checked_div` in
  `shrike-pyo3` are the permanent executable exemplar.
- **Typing:** the `shrike_native` package ships `.pyi` stubs + `py.typed`;
  `//shrike-core/shrike-pyo3:stubtest` (mypy.stubtest) fails when a Rust signature
  drifts from its stub.

## Building and testing

```bash
# Bazel (CI runs one `./bazel test //...` over the whole graph; this is its native slice):
./bazel test //shrike-core/...                  # crates, layering check, smoke, stubtest
./bazel build //:wheel --stamp             # the platform-tagged shrike-mcp wheel (manual tag)

# Plain cargo (fast inner loop; .cargo/config.toml carries the macOS link flags):
cd native && cargo build && cargo test

# Build + install the extension into the active venv so the pip-lane tests
# exercise the facades against the real native module:
scripts/build-native.sh            # debug; --release for optimized
```

**The build matrix (#499; docs/distribution.md is canonical):** the
engine/manager crates are cargo features on the binding crate — `engine-ort`
(text + CLIP via ort), `engine-remote` (OpenAI-compatible embeddings;
shrike-describe-remote joins at #485), `engine-apple` (Vision OCR — MOBILE
builds only, never the server set, on any OS; binding coverage is #514),
`manage-llama` (llama-server lifecycle). The cargo default is the SERVER
set, pinned verbatim on
`//shrike-core/shrike-pyo3:shrike_pyo3_native`; the MOBILE set (anki-core +
engine-remote + engine-apple — no ort, no managers) is proven by
`//shrike-core/shrike-pyo3:mobile_skeleton`, which the per-PR `//...` lane builds so
an ungated engine reference fails CI (the Bazel target is the authoritative
proof — it really links; the cargo line below is the convenience check):

```bash
cd native && cargo check -p shrike-pyo3 --no-default-features \
  --features bundled-sqlite,anki-core,engine-remote,engine-apple
```

**SQLite linkage (#300):** `shrike-derived` bundles its own SQLite by default
(FTS5 + trigram guaranteed — the #281 property; release wheels keep this).
`scripts/build-native.sh --system-sqlite` (or `cargo build -p shrike-pyo3
--no-default-features --features <the server set minus bundled-sqlite>`)
links the platform libsqlite3 instead — any sqlite3-ABI-compatible library works via the standard
libsqlite3-sys overrides (pkg-config / `SQLITE3_LIB_DIR`), including libsql
builds exposing the sqlite3 C API. Under platform linkage FTS5/trigram are
probed at runtime (`shrike_native.derived_fts5_probe`), and the store degrades
to the `find_notes` fallback exactly like the stdlib engine when they're absent.

After changing any `Cargo.toml`: `cd native && cargo generate-lockfile` (or
`cargo update -p <crate>`), then commit `Cargo.lock` — MODULE.bazel pins the
Bazel crate graph to it.

## Packaging

The release artifact is `//:wheel`: an abi3 (cp312+) **platform-tagged
`shrike-mcp` wheel** that ships the Python package *and* the `shrike_native`
package (`_native.so`, the stubs, `py.typed`) in one artifact — since the
kernel inversion the server *requires* `shrike_native` (it is the kernel, not
an accelerator), so there is no separate `shrike-native` distribution and no
version-skew surface (#497; the plugin-extension-wheel idea stays recorded on
#340). release.yml builds it per platform (macOS arm64/x86_64, linux
x86_64/aarch64); the linux wheels are auditwheel-repaired to manylinux tags
before publishing. ort linkage decision for #270: **load-dynamic against the
pinned onnxruntime wheel shrike-mcp hard-depends on** (no `download-binaries`,
no duplicated runtime, guaranteed version match for the parity bake).

## Licensing inventory

All AGPL-3.0-compatible: pyo3 (MIT/Apache-2.0), usearch (Apache-2.0),
ort/onnxruntime (MIT), tokenizers (Apache-2.0), rusqlite (MIT),
image (MIT/Apache-2.0).
