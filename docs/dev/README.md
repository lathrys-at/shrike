# Developer docs

Orientation for working on Shrike. Start with `CLAUDE.md` at the repo root for the
short version; these docs go deeper, one concern each.

| Doc | What it covers |
|-----|----------------|
| [`architecture.md`](architecture.md) | The Rust core / Python harness split, the plugin kernel, the runtime, the action exchange. |
| [`layout.md`](layout.md) | Where every crate and package lives; the `scripts`/`tools`/`bin` boundary. |
| [`testing.md`](testing.md) | Dev setup, running the suites, the native build, coverage, linting. |
| [`server-runtime.md`](server-runtime.md) | Collection lifecycle and locking, the transport trust boundary, the daemon, platform dirs, config. |
| [`embedding-and-recognition.md`](embedding-and-recognition.md) | The embedding service and its backends; OCR/recognition. |
| [`indexing-and-search.md`](indexing-and-search.md) | Vector-index consistency, the derived-text sidecar, search fusion (RRF). |
| [`tools.md`](tools.md) | The 26 MCP tools: where they live and the behaviours to preserve. |
| [`decisions.md`](decisions.md) | The "why" behind non-obvious design choices, and the alternatives rejected. |
| [`build-bazel.md`](build-bazel.md) | The Bazel build graph, the two lanes, caching, the locks. |

Reference docs for *users and integrators* live one level up in [`../`](..):
the [CLI reference](../cli-reference.md), the [MCP tool reference](../mcp-tools.md),
and the [distribution profiles](../distribution.md).
