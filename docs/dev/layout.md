# Repository layout

The repository is two self-contained units that mirror each other, bridged by
the `shrike_native` extension:

- **`shrike-core/`** — the Rust workspace (the compute core).
- **`shrike-py/`** — the Python harness (the host, CLI, and server).

Each unit owns its own build. Shared dev and build tooling — `scripts/`,
`tools/`, `MODULE.bazel`, and the requirements lockfiles — stays at the repo
root. The units are designed to be subtree-extractable: `shrike-py/` depends on
the Rust side only through the published extension, never the workspace
internals.

## `shrike-py/` — the Python harness

```
shrike-py/
├── pyproject.toml                # the unit's build config
├── BUILD.bazel                   # //shrike-py:wheel + :sdist
├── bin/                          # py_binary launchers (shrike, server, server_embedding)
├── tests/                        # unit / native / integration
└── src/shrike/                   # the package (src layout)
```

The package under `src/shrike/` is organised into **layered subpackages**. Each
layer imports only from the layers below it, in this order:

```
platform/  ←  schemas + errors + client  ←  harness/  ←  api/  ←  server/  ←  cli/
```

| Layer | What lives there |
|-------|------------------|
| `platform/` | Near-leaf process infra: `paths`, `log`, `daemon`, `pathsafety`. |
| `schemas.py`, `errors.py`, `client.py` | The wire models (Pydantic), the client-side error taxonomy, and the standalone HTTP client. |
| `harness/` | Kernel-wrapping assembly (`harness.py`), the collection facade (`collection.py`), index/derived host policy, and the engine bindings under `engines/{embedding,recognition}/`. |
| `api/` | The operation surface: the action registry (`actions.py`), the MCP binding (`tools.py`), and the call adapter (`mcp_adapter.py`). |
| `server/` | The MCP host: FastMCP routes, transport security, the export store. |
| `cli/` | The Click CLI, one module per command group. |

`daemon` sits in `platform/` (not with the server) because the standalone
`client` consumes it for liveness probing; placing it under `server/` would
invert the layering. The same reasoning puts `pathsafety` in `platform/` — it is
shared by both `api/actions` and `server/`.

The engine bindings mirror the Rust engine families:

```
harness/engines/
├── embedding/   # base protocol, runtime (backend lifecycle), onnx, clip, batching, text
└── recognition/ # recognizer protocol + make_recognizer (OCR today)
```

## `shrike-core/` — the Rust workspace

Crates are grouped by role (a pure path grouping; no crate is renamed by it):

```
shrike-core/
├── contracts/                # the floor: wire types, error taxonomy, the engine firewall
│   ├── shrike-error/            # the shared error taxonomy
│   ├── shrike-schemas/          # serde + schemars wire types (canonical; schemas.py binds to these)
│   ├── shrike-engine-api/       # the engine contract: kernel-facing async traits, sync compute
│   │                            #   traits, the Blocking adapter, WithPolicy, the batch probe
│   └── shrike-store/            # the derived-cache store primitive
├── runtime/                  # the kernel and the stores it orchestrates
│   ├── shrike-kernel/           # the kernel: collection + index orchestration + derived + fusion
│   ├── shrike-collection/       # anki via its protobuf service layer (the only anki coupling)
│   ├── shrike-index/            # per-modality USearch engine
│   ├── shrike-derived/          # FTS5 trigram engine
│   └── shrike-cache/            # per-collection cache layout (index/derived subdirs + namespacing)
├── engines/                  # engine-contract impls + the platform glue they parse
│   ├── shrike-engine/           # engine impls, feature-gated by family (onnx / remote / apple)
│   └── shrike-platform/         # raw Swift/C-ABI recognizer glue (Vision OCR + Speech ASR)
├── managed/                  # subprocess lifecycle (manage-class, not engines)
│   ├── shrike-process/          # generic managed-subprocess lifecycle (ManagedProcess + reaper)
│   └── shrike-llama-server/     # llama-server lifecycle (spawn/health/reap/stop)
├── utility/                  # low-level crates below the kernel and engines
│   ├── shrike-network/          # the SSRF-pinned network primitives
│   ├── shrike-image/            # the CLIP pixel pipeline
│   └── shrike-media/            # inbound media: SSRF fetch + decode + MIME
└── bindings/                 # the host-language bindings
    ├── shrike-pyo3/             # the pyo3 binding (the only pyo3 crate) + the shrike_native package
    └── shrike-cabi/             # the C-ABI mobile binding (cdylib, zero CPython)
```

The layering check (`shrike-core/layering_check.py`) enforces the role order —
the kernel may depend on contracts and utility crates, never on an engine crate.

## `scripts/` vs `tools/` vs `bin/`

The dividing line is **who invokes it**. Each directory has a `README.md` with
its inventory.

- **`shrike-py/bin/`** — shipped product entry points. The `py_binary` launchers
  over the package (`//shrike-py/bin:shrike`, `:server`, `:server_embedding`).
  They live outside the `shrike` package so a binary's output path never collides
  with a package subdir.
- **`tools/`** — invoked by the build. Bazel macros, the version-pin locks and
  their writers/checkers, the wheel/sdist/requirements builders,
  `workspace_status.sh`. A file follows its strongest coupling: the
  `llama-server.lock` plus its regenerator and tripwire all live here, beside the
  sibling `bazel.lock`.
- **`scripts/`** — human-facing dev and maintenance entry points. `dev-setup.sh`,
  `build-native.sh`, the coverage runners, the dogfooding launchers. A script a
  developer runs by hand stays here even when its output feeds a build.

## See also

- [`architecture.md`](architecture.md) — how these pieces interact at runtime.
- [`testing.md`](testing.md) — setting up the dev environment and running the
  suites.
- [`build-bazel.md`](build-bazel.md) — the Bazel build graph and the two lanes.
