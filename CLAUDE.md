# CLAUDE.md

## What is Shrike?

Shrike is a desktop application for managing Anki flashcard collections without running Anki's GUI. It exposes Anki's collection operations through an MCP (Model Context Protocol) server that any MCP client, CLI tool, or local automation can talk to.

**License:** AGPL-3.0

### Architecture

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (FastMCP)
                                      │
                                      ├──▶ CollectionWrapper (anki.Collection)
                                      │         └──▶ collection.anki2 (SQLite)
                                      │
                                      └──▶ VectorIndex (stub)
                                               └──▶ shrike.usearch (future)
```

The Python sidecar lives in `sidecar/`. A Tauri desktop shell will wrap it later but is not yet implemented.

## Project layout

```
sidecar/                          # Python package root (cd here to work)
├── pyproject.toml                # Package config, deps, tool settings
├── requirements.txt              # Flat deps list (mirrors pyproject.toml)
├── shrike/
│   ├── __init__.py               # Just __version__
│   ├── server.py                 # MCP server entry point (argparse, FastMCP)
│   ├── collection.py             # CollectionWrapper — all Anki DB operations
│   ├── note_types.py             # upsert_note_types() — create/update note types
│   ├── tools.py                  # Registers 6 MCP tools, Pydantic input models
│   ├── index.py                  # VectorIndex stub (search_notes not yet implemented)
│   └── cli/
│       ├── __init__.py           # Root Click group, global options (--config, --url, --json)
│       ├── client.py             # ShrikeClient — HTTP client for MCP JSON-RPC calls
│       ├── config.py             # YAML config loading/saving (~/.config/shrike/config.yml)
│       ├── server_cmd.py         # shrike server start/stop/status (daemon management)
│       ├── info_cmd.py           # shrike info
│       ├── note_cmd.py           # shrike note list/show/create/update/delete/search
│       ├── type_cmd.py           # shrike type list/show/create/update
│       └── output.py             # Rich-based formatting (tables, panels, styled output)
├── tests/
│   ├── unit/                     # 53 tests — direct CollectionWrapper calls, no server
│   │   ├── conftest.py           # wrapper fixture (temp collection), basic_note fixture
│   │   ├── test_collection_info.py
│   │   ├── test_list_notes.py
│   │   ├── test_upsert_notes.py
│   │   ├── test_delete_notes.py
│   │   └── test_note_types.py
│   └── integration/              # 27 tests — real server subprocess + HTTP transport
│       ├── conftest.py           # server fixture (session-scoped), mcp fixture
│       └── test_tools.py
docs/
├── mcp-tools.md                  # Tool documentation (human-readable)
└── mcp-schema.json               # Full JSON schema for all 6 tools
```

## Development setup

```bash
cd sidecar
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.12 is used (managed via pyenv; `.python-version` is at repo root). The `anki` package requires Python >= 3.11.

## Running commands

All commands below assume you're in `sidecar/`.

### Tests

```bash
pytest tests/unit -v                           # Unit tests (fast, no server)
pytest tests/integration -v -m integration     # Integration tests (starts a server)
```

### Linting

```bash
ruff check shrike/                  # Lint
ruff format --check shrike/         # Format check
mypy shrike/                        # Type check
```

All three must pass cleanly. CI runs them in a `lint` job alongside `unit` and `integration` jobs.

### Running the server manually

```bash
# Directly (foreground):
python -m shrike.server --collection /path/to/collection.anki2

# Via CLI (daemon):
shrike server start --collection /path/to/collection.anki2
shrike server status
shrike server stop
```

## Key technical details

### anki.Collection

The `anki` pip package provides a headless Python API to Anki's SQLite database — no Qt or GUI dependencies. It acquires an exclusive write lock on the database, so only one process can have a collection open at a time. The `CollectionWrapper` class handles lifecycle (open/close via atexit).

### MCP transport

The server uses FastMCP with streamable HTTP transport (`stateless_http=True`, `json_response=True`). It listens on `http://127.0.0.1:8372/mcp` by default. All communication is JSON-RPC 2.0: clients POST to the endpoint with `method: "tools/call"` and receive structured JSON responses.

### MCP tools (6 total)

| Tool | Status | Purpose |
|------|--------|---------|
| `collection_info` | Working | Collection structure, note types, decks, tags, stats |
| `list_notes` | Working | Filter/retrieve notes by deck, tags, type, IDs, date |
| `search_notes` | Stub | Semantic similarity search (returns "not available" message) |
| `upsert_notes` | Working | Create or update notes in bulk (1-100) |
| `upsert_note_types` | Working | Create or update note type definitions (1-10) |
| `delete_notes` | Working | Permanently delete notes by ID |

Tool input schemas are defined as Pydantic models (`NoteInput`, `NoteTypeInput`, `TemplateInput`) in `tools.py`. The authoritative schema is in `docs/mcp-schema.json`.

### CLI structure

The CLI uses Click with rich for output formatting. Command hierarchy:

```
shrike [--config PATH] [--url URL] [--json]
├── server start|stop|status
├── info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
├── note list|show|create|update|delete|search
└── type list|show|create|update
```

The CLI talks to the MCP server over HTTP — it can target a remote server via `--url` or `SHRIKE_URL`.

### Daemon management

`shrike server start` spawns the server as a background process. State files live in `~/.local/state/shrike/`:
- `server.pid` — PID file
- `server.json` — metadata (URL, port, collection path, start time)
- `server.log` — stdout/stderr

### Config file

YAML at `~/.config/shrike/config.yml`. Auto-created on first `shrike server start`. Resolution order: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`) → CLI flags.

## Code style and conventions

- **Type annotations** on all functions (enforced by mypy with `disallow_untyped_defs`)
- **Ruff** for linting (rules: E, F, W, I, UP, B, SIM) and formatting, line length 100
- **Error handling:** batch operations (upsert_notes, upsert_note_types) use per-item try/except so one failure doesn't block the batch. Results include `status: "created"|"updated"|"error"` per item.
- **`raise ... from err`** in except blocks (enforced by ruff B904)
- **`contextlib.suppress`** instead of bare `try/except/pass`
- **`datetime.UTC`** not `timezone.utc` (ruff UP017)

## What's not yet implemented

- **Semantic search** (`search_notes`): `index.py` is a stub. Needs llama-server for embeddings and usearch for the vector index.
- **Tauri desktop shell**: Not started. The sidecar is designed to be wrapped by Tauri.
- **AnkiWeb sync**: Config has a placeholder for sync settings but no implementation.
