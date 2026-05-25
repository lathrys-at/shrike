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
│   ├── log.py                    # Logging config, log parsing and styling
│   ├── index.py                  # VectorIndex stub (search_notes not yet implemented)
│   └── cli/
│       ├── __init__.py           # Root Click group, global options (--config, --url, --json, --pretty)
│       ├── client.py             # ShrikeClient — HTTP client for MCP JSON-RPC calls
│       ├── config.py             # YAML config loading/saving (~/.config/shrike/config.yml)
│       ├── server_cmd.py         # shrike server start/stop/status/logs (daemon management)
│       ├── info_cmd.py           # shrike info
│       ├── note_cmd.py           # shrike note list/show/create/update/delete/search
│       ├── type_cmd.py           # shrike type list/show/create/update
│       └── output.py             # Rich formatting, output_options decorator
├── tests/
│   ├── unit/                     # 53 tests — direct CollectionWrapper calls, no server
│   │   ├── conftest.py           # wrapper fixture (temp collection), basic_note fixture
│   │   ├── test_collection_info.py
│   │   ├── test_list_notes.py
│   │   ├── test_upsert_notes.py
│   │   ├── test_delete_notes.py
│   │   ├── test_note_types.py
│   │   └── test_logging.py
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
shrike [--config PATH] [--url URL] [--json] [--pretty/--no-pretty]
├── server start|stop|status|logs
├── info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
├── note list|show|create|update|delete|search
└── type list|show|create|update
```

The CLI talks to the MCP server over HTTP — it can target a remote server via `--url` or `SHRIKE_URL`.

`--json` and `--pretty/--no-pretty` are global options but also accepted on every leaf command (via the `@output_options` decorator in `output.py`), so both `shrike --json info` and `shrike info --json` work. `--json` implies `--no-pretty`; combining `--json --pretty` is an error.

### Daemon management

`shrike server start` spawns the server as a background process. State files live in `~/.local/state/shrike/`:
- `server.pid` — PID file
- `server.json` — metadata (URL, port, collection path, start time, log dir)
- `logs/shrike.log` — rotating log file (10 MB, 5 backups)

### Config file

YAML at `~/.config/shrike/config.yml`. Auto-created on first `shrike server start`. Resolution order: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`) → CLI flags.

## Code style and conventions

- **Type annotations** on all functions (enforced by mypy with `disallow_untyped_defs`)
- **Ruff** for linting (rules: E, F, W, I, UP, B, SIM) and formatting, line length 100
- **Error handling:** batch operations (upsert_notes, upsert_note_types) use per-item try/except so one failure doesn't block the batch. Results include `status: "created"|"updated"|"error"` per item.
- **`raise ... from err`** in except blocks (enforced by ruff B904)
- **`contextlib.suppress`** instead of bare `try/except/pass`
- **`datetime.UTC`** not `timezone.utc` (ruff UP017)

### Logging

Logging is configured in `shrike/log.py`. Log format, parsing, and styling all live in that module — formatting knowledge should not be spread across CLI commands.

**Logger names** — Use per-module loggers: `shrike.server`, `shrike.tools`, `shrike.collection`, `shrike.note_types`. This makes the config's per-logger level overrides (`logging.levels.shrike.collection: debug`) actually work. Never log everything under a bare `shrike` logger.

**Principles for log messages:**

1. **Say what happened and include the key context.** "Collection ready: 847 notes, 5 decks, 12 note types" not "Collection opened". Include counts, IDs, paths, durations — the things that make a log line useful without having to correlate it with other lines.
2. **Log operational boundaries at INFO.** Startup, shutdown, configuration loaded, server listening. These are the anchors you scan for when reading a log.
3. **Log every tool call at INFO.** This is a server; knowing what it did is the point. Log the tool name with its parameters on entry, and a result summary on completion: `list_notes deck=Test limit=50` → `list_notes returned 3/3 notes`.
4. **Use DEBUG for internals.** Individual note creates/updates, query construction, index lookups. Things you'd turn on when debugging a specific module, not things you want in production logs.
5. **Use WARNING for recoverable failures that deserve attention.** A single note failing in a batch upsert, a note type update that was rejected. Not normal empty-result responses.
6. **Use `%s` formatting, not f-strings.** Lazy evaluation — the format string is only interpolated if the log level is enabled.
7. **Don't repeat what the logger name already says.** The log line already shows `shrike.tools` — don't prefix the message with "tools:".
8. **Log the signal name on shutdown**, not just "shutting down" — you want to know whether it was SIGTERM (normal stop) or SIGINT (Ctrl+C) or something else.

**Log file format** (defined in `log.py`):
```
2025-05-24T10:30:00 INFO  shrike.tools  list_notes deck=Test limit=50
```
Timestamp is `%Y-%m-%dT%H:%M:%S` (19 chars), level is left-padded to 5 chars, logger and message are separated by double-space. `parse_log_line()` and `style_log_line()` in `log.py` know this format — keep them in sync if you change it.

## Upcoming tasks

### 1. CLI integration tests and bug fixes

Systematically exercise every `shrike` CLI command against a running server and verify it works end-to-end. At least one command currently fails with an exception traceback. Add integration tests in `tests/integration/` that cover the full CLI surface: `server start/stop/status/logs`, `info`, `note list/show/create/update/delete/search`, `type list/show/create/update`. Fix any broken commands found during the audit.

Note: CLI commands that talk to the server (everything except `server start/stop/status/logs`) should auto-launch the daemon if it isn't already running. This makes the explicit `shrike server start` a convenience, not a prerequisite. Provide a `--no-autostart` flag (or config option) to disable this for users who want manual control. The auto-start needs a collection path, so it must be resolvable from config/env without `--collection` being passed to every command.

### 2. CLI output UI/UX review

Once all commands work, systematically review the pretty output of each command for readability and visual consistency. Check: table alignment, color usage, error presentation, empty-state messages, `--no-pretty` degradation, `--json` output structure. The goal is compact, scannable output — no decorative noise.

### 3. Tab completion for the CLI

Add shell completion support (bash, zsh, fish) via Click's built-in completion system. Consider dynamic completions where useful — e.g., deck names for `--deck`, note type names for `--type`, tag names for `--tags`.

### 4. CLI transparent batching for large requests

The MCP tools enforce per-call limits (100 notes, 10 note types) which are appropriate for LLM tool calls. The CLI client should transparently batch larger requests — e.g., a 500-note upsert becomes 5 server calls with results stitched together. This keeps the server simple (one set of limits, no caller-aware logic) and makes the CLI a capable scripting tool. Implement in `ShrikeClient` so individual commands don't need to think about it.

### 5. Sync auth and operations

Support syncing the collection to both AnkiWeb (ankiweb.net) and self-hosted anki-sync-server instances, configurable in `config.yml`. Add `shrike sync` commands for authentication (store credentials securely), triggering sync, and checking sync status. Both sync targets should work interchangeably.

Sync operations are delegated through the MCP daemon — the CLI sends a sync request to the server, which manages the operation. The daemon already holds the collection lock, so it's the right place to own sync. This means sync commands require the daemon to be running (or auto-start it, per task #1).

### 6. Sync server lifecycle management

The `anki` Python package provides an entrypoint for the sync server. Add `shrike sync-server start/stop/status` as a separate command group (parallel to `shrike server`) to launch and manage a local sync server instance. This lets users run a self-hosted sync target without needing Anki desktop. Separate PID file and metadata from the MCP server.

The sync server is a child process managed by the MCP daemon, not launched independently by the CLI. The daemon handles forking, PID tracking, and log routing (same pattern we'll use for llama-server). `shrike sync-server start/stop/status` sends commands to the daemon, which manages the actual process. Logs go through the same rotating log infrastructure (`sync-server.log` via `--process sync-server` in `shrike server logs`).

## What's not yet implemented

- **Semantic search** (`search_notes`): `index.py` is a stub. Needs llama-server for embeddings and usearch for the vector index.
- **Tauri desktop shell**: Not started. The sidecar is designed to be wrapped by Tauri.
