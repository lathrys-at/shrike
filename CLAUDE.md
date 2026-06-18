# CLAUDE.md

Orientation for working on Shrike. This file is deliberately short; the deep
material lives in [`docs/dev/`](docs/dev/), one concern per file. When you need
detail, follow the links rather than expecting it here.

## What is Shrike?

Shrike manages Anki flashcard collections without running Anki's GUI. It exposes
Anki's collection operations through an MCP server and a `shrike` CLI, with
semantic search over notes computed locally.

License: AGPL-3.0.

## Architecture at a glance

Shrike is a **Rust compute core** (`shrike-core/`) wrapped by a **Python harness**
(`shrike-py/`), bridged by one compiled extension, `shrike_native`.

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (server/ host; api/ verb surface)
                                      └──▶ Harness (assembly + verbs)
                                              └──▶ AsyncKernel (Rust, via shrike-pyo3)
                                                      ├── collection.anki2 (anki protobuf services)
                                                      ├── vector index (per-modality USearch)
                                                      ├── derived-text store (FTS5 sidecar)
                                                      ├── Embed slot  ◀── shrike-engine
                                                      └── Recognize slot ◀── shrike-engine
```

The kernel owns the collection, the indexes, and search fusion; it is a **plugin
host** that runs whatever embedder/recognizer the harness attaches. The full
picture — the plugin contracts, the tokio runtime, the action exchange, and the
load-bearing invariants — is in [`docs/dev/architecture.md`](docs/dev/architecture.md).

## Development

One-step, idempotent setup (re-run any time as a repair button):

```bash
scripts/dev-setup.sh        # creates .venv, installs shrike-py[dev], builds shrike_native
source .venv/bin/activate
```

The full local gate before a native change:

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd shrike-core && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
./bazel test //...      # the authoritative CI lane (Bazel is not on PATH; use ./bazel)
```

Lint and type-check (all three must be clean):

```bash
ruff check  shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
ruff format --check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
```

`pip install` does **not** build the native extension — that is a separate cargo
step, run for you by direnv or `scripts/build-native.sh`; `pytest` aborts loudly
if the `.so` is stale. Python 3.12 (managed via pyenv; `.python-version` at the
repo root). Coverage, the test-sharing model, and the Bazel lanes are in
[`docs/dev/testing.md`](docs/dev/testing.md).

## Conventions you must follow

- **Type annotations on every function** (mypy `disallow_untyped_defs`); **ruff**
  for lint + format, line length 100; **`raise ... from err`**;
  **`contextlib.suppress`** over bare `try/except/pass`; **`datetime.UTC`**.
- **Make illegal states unrepresentable** in `schemas.py` — correlated fields are a
  discriminated union, never a bag of optionals.
- **Ground performance at a 100k-note collection** — no collection reads inside
  per-item loops; one transaction per batch; never hold a lock across file writes.
- **One INFO log line per served call**; per-module loggers; `%s` formatting, not
  f-strings.

The detail and the reasoning are in [`docs/dev/conventions.md`](docs/dev/conventions.md).

## Documentation map

Developer docs — how the code works — live in [`docs/dev/`](docs/dev/):

| Doc | Covers |
|-----|--------|
| [`architecture.md`](docs/dev/architecture.md) | The Rust/Python split, the plugin kernel, the runtime, the action exchange. |
| [`layout.md`](docs/dev/layout.md) | Where every crate and package lives; the `scripts`/`tools`/`bin` boundary. |
| [`testing.md`](docs/dev/testing.md) | Dev setup, the suites, the native build, coverage, linting. |
| [`server-runtime.md`](docs/dev/server-runtime.md) | Collection lifecycle and locking, the transport trust boundary, the daemon, config. |
| [`embedding-and-recognition.md`](docs/dev/embedding-and-recognition.md) | The embedding service, its backends, and OCR/recognition. |
| [`indexing-and-search.md`](docs/dev/indexing-and-search.md) | Vector-index consistency, the derived-text sidecar, search fusion (RRF). |
| [`tools.md`](docs/dev/tools.md) | The 26 MCP tools: where they live and the behaviours to preserve. |
| [`conventions.md`](docs/dev/conventions.md) | Code style, the schema house style, performance rules, logging. |
| [`agent-workflow.md`](docs/dev/agent-workflow.md) | Review gates, the PR loop, multi-agent team development, the defect workflow. |
| [`decisions.md`](docs/dev/decisions.md) | The "why" behind non-obvious choices, and the alternatives rejected. |
| [`build-bazel.md`](docs/dev/build-bazel.md) | The Bazel build graph, the two lanes, caching, the locks. |

Reference docs for *users and integrators* live at [`docs/`](docs/): the
[CLI reference](docs/cli-reference.md), the [MCP tool reference](docs/mcp-tools.md),
and the [distribution profiles](docs/distribution.md).

## Working in this repo

- **Trunk-based.** Every change goes through a `‹type›/‹issue#›-‹slug›` branch → PR
  → squash merge. Open every PR as a **draft**; mark it ready only once complete and
  reviewed. CI always runs on every PR.
- **Roadmap and tracked work live in GitHub issues and milestones**, not in this
  file or the README.
- **Review and audit gates are mandatory:** a code review on every behavioural
  change, a security review whenever the server API surface changes, and a
  security + performance audit before a release.
- **Found a defect out of scope?** Don't fix it inline — open an issue, push a
  branch with a failing/xfail test as the spec, and link it.

The human conventions (branching, versioning, releasing, repo settings) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md); the agent operating procedure (the PR loop,
team development, the gates, the defect workflow) is in
[`docs/dev/agent-workflow.md`](docs/dev/agent-workflow.md).
