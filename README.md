# Shrike

A desktop application for managing Anki flashcard collections without running Anki. Shrike opens the collection directly via Anki's Python library, exposes a set of MCP tools for LLM-driven card management, and maintains a local vector index for semantic search over card content.

Designed for a workflow where cards are created on desktop (through conversation with an LLM) and reviewed on mobile.

## Why

Anki's plugin ecosystem — particularly [anki-connect](https://git.sr.ht/~foosoft/anki-connect) — is the standard way to programmatically interact with a collection. It works, but:

- **Requires Anki desktop to be running.** anki-connect is an Anki plugin, so the full GUI application must be open for the HTTP API to be available. If you only review on your phone, this means running a desktop app you never use just to serve an API.
- **Fragile.** Plugin breakage across Anki updates is common. The plugin ecosystem in general is poorly maintained.
- **Bloated interface.** anki-connect exposes 80+ endpoints, most of which exist to remote-control Anki's GUI (browsing, reviewing, answering cards). An LLM creating flashcards needs maybe 10 of them.

Shrike replaces this with a standalone application that owns the collection directly, doesn't depend on Anki desktop or any plugins, and exposes a minimal MCP interface purpose-built for LLM interaction.

## Architecture

```
  MCP Clients
  (Claude Desktop, Claude Code, etc.)
            │
            │ HTTP/SSE
            │
┌───────────┼──────────────────────────────────────────┐
│  Tauri Shell                                         │
│  - System tray / menu bar presence                   │
│  - Lifecycle management for sidecars                 │
│  - Settings UI (collection path, sync, embedding)    │
│  - Duplicate detection alerts                        │
│                                                      │
│  ┌─────────┼───────────┐  ┌────────────────────────┐ │
│  │  Python Sidecar     │  │  llama-server          │ │
│  │                     │  │                        │ │
│  │  - MCP server       │  │  - Local embedding     │ │
│  │    (HTTP/SSE)       │  │    model (GGUF)        │ │
│  │  - anki.Collection  │◄─┤  - HTTP /embedding     │ │
│  │  - USearch index    │  │    endpoint            │ │
│  │                     │  │                        │ │
│  └────────┬────────────┘  └────────────────────────┘ │
│           │                                          │
└───────────┼──────────────────────────────────────────┘
            │
  ┌─────────▼─────────┐
  │  collection.anki2  │
  │  (SQLite)          │
  │                    │
  │  + shrike.usearch  │
  │  (vector index)    │
  └────────────────────┘
```

### Components

**Tauri shell** — the desktop application frame. Provides a system tray icon, manages the lifecycle of both sidecars, hosts any configuration UI, and surfaces duplicate detection alerts. Minimal surface area; most logic lives in the Python sidecar.

**Python sidecar** — the core process. Opens the Anki collection via `anki.Collection` (the official headless Python API, installable via `pip install anki` — no Qt or GUI dependencies). Runs the MCP server over HTTP/SSE on localhost and manages the USearch index. This is where all six MCP tools are implemented. Tauri owns the process lifecycle; MCP clients connect over the network.

**llama-server** — serves a local embedding model (e.g., Jina Embeddings v5 Nano as a GGUF) over HTTP. The Python sidecar calls its `/embedding` endpoint to vectorize card content. Keeps the Python process free of ML framework dependencies (no PyTorch, no ONNX runtime). Small footprint: the llama.cpp binary is ~50MB, the embedding model is ~50-100MB.

**collection.anki2** — the standard Anki SQLite database. Shrike reads and writes it directly through `anki.Collection`, which handles all schema invariants (USN tracking, timestamps, model validation). This is the same file that Anki desktop and AnkiWeb sync operate on.

**shrike.usearch** — a [USearch](https://github.com/unum-cloud/usearch) index file storing vector embeddings for all notes. Kept separate from the Anki collection to avoid interfering with Anki's sync. Rebuilt from scratch on first run; incrementally updated as notes are created or modified.

### Collection access and sync

Shrike assumes exclusive ownership of the collection while running. The `anki.Collection` API acquires a write lock on the SQLite database, so Anki desktop cannot open the same collection simultaneously.

`anki.Collection` exposes sync methods directly, so Shrike can sync to AnkiWeb (or a self-hosted sync server) without involving Anki desktop at all. The Tauri app can expose a "sync now" action in the tray menu, or sync automatically on a schedule. Anki desktop is never needed.

### Vector index

The embedding index enables semantic search over card content (the `search_notes` MCP tool) and powers the application-level duplicate detection system.

**Embedding:** All text fields of each note are concatenated and embedded as a single vector. The Python sidecar calls llama-server's `/embedding` endpoint over localhost.

**Storage:** Vectors are stored in a USearch HNSW index, with a separate mapping of note IDs to modification timestamps for incremental updates. On startup, the sidecar diffs the index against the collection (by note ID and mod time), embeds anything new or changed, and removes entries for deleted notes. USearch indexes are memory-mapped, so the full index doesn't need to be loaded into RAM.

**Search:** USearch provides fast approximate nearest neighbor search via HNSW. For the `search_notes` tool, query strings are embedded via llama-server and searched against the index. When searching by note ID, the existing vector is looked up directly.

**Duplicate detection:** When new notes are created via `upsert_notes`, the application (not the MCP tool) checks the new embeddings against the index and surfaces warnings in the Tauri UI for notes above a configurable similarity threshold. The user decides whether to keep or remove flagged notes.

## MCP interface

Six tools, designed around what an LLM actually needs when helping a user manage flashcards.

| Tool | Purpose |
|---|---|
| `collection_info` | Schema discovery — note types, decks, tags, stats. |
| `list_notes` | Structured search — filter by deck, tags, note type, IDs, modification date. |
| `search_notes` | Semantic search — find conceptually similar notes by query string or note ID. |
| `upsert_notes` | Create or update notes in bulk (1–100). |
| `upsert_note_types` | Create or update note type definitions — fields, card templates (HTML), CSS. |
| `delete_notes` | Delete notes by ID. |

Design principles:

- **No pagination.** Results are capped by `limit` with a `total` count in the response. If the result set is too large, the LLM should narrow its filters. LLMs don't paginate well, and they have no use for thousands of note IDs.
- **No duplicate detection in tools.** Dedup is an application concern surfaced in the Tauri UI, not a parameter the LLM tunes. The LLM can use `search_notes` to check for existing coverage before creating cards, reasoning about overlap from content rather than numeric thresholds.
- **No permission scaffolding.** MCP clients already provide per-tool permission controls. The tools execute what they're asked to; authorization is the client's responsibility.
- **Upsert over separate create/update.** The LLM doesn't need to decide whether a note exists yet. If an ID is present, it's an update; if absent, it's a create.
- **Note types are first-class.** LLMs are good at authoring HTML/CSS card templates from natural-language descriptions. Users shouldn't need to write CSS to get a note type suited to their material.

Full tool schemas and documentation are in `docs/mcp-tools.md` and `docs/mcp-schema.json`.

## Project structure

```
shrike/
├── src-tauri/              # Tauri application (Rust)
│   ├── src/
│   │   └── main.rs         # Tray icon, sidecar lifecycle, settings
│   ├── tauri.conf.json
│   └── Cargo.toml
├── sidecar/                # Python MCP server
│   ├── shrike/
│   │   ├── __init__.py
│   │   ├── server.py       # MCP server entry point (HTTP/SSE transport)
│   │   ├── collection.py   # anki.Collection wrapper, note CRUD
│   │   ├── note_types.py   # Note type CRUD
│   │   ├── index.py        # USearch vector index management
│   │   └── tools.py        # MCP tool definitions and handlers
│   ├── pyproject.toml
│   └── requirements.txt    # anki, usearch
├── docs/
│   ├── mcp-tools.md        # Human-readable tool documentation
│   └── mcp-schema.json     # MCP tool definitions (JSON schema)
├── README.md
└── LICENSE
```

## Dependencies

**Runtime:**

- [Tauri v2](https://v2.tauri.app/) — desktop shell
- [Python 3.11+](https://www.python.org/) — sidecar runtime
- [`anki`](https://pypi.org/project/anki/) — headless collection access (pip package, no Qt)
- [llama.cpp / llama-server](https://github.com/ggml-org/llama.cpp) — local embedding inference
- A GGUF embedding model (e.g., Jina Embeddings v5 Nano)
- [USearch](https://github.com/unum-cloud/usearch) — vector index (HNSW, Python bindings)

**Development:**

- Rust toolchain (for Tauri)
- Node.js (for Tauri's build tooling)

## Status

Design phase. The MCP tool interface is specified; implementation has not started.

## License

Shrike is licensed under the GNU Affero General Public License, version 3 or later.
