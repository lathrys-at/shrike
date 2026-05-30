# CLAUDE.md

## What is Shrike?

Shrike manages Anki flashcard collections without running Anki's GUI. It exposes Anki's collection operations through an MCP server and CLI.

**License:** AGPL-3.0

### Architecture

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (FastMCP)
                                      │
                                      ├──▶ CollectionWrapper (anki.Collection)
                                      │         └──▶ collection.anki2 (SQLite)
                                      │
                                      └──▶ VectorIndex (USearch HNSW)
                                               ├──▶ EmbeddingService (llama-server)
                                               └──▶ index.usearch + index.meta.json
```

## Project layout

```
src/shrike/                       # Python package (src layout)
├── __init__.py                   # Just __version__
├── server.py                     # MCP server entry point (argparse, FastMCP)
├── collection.py                 # CollectionWrapper — all Anki DB operations
├── note_types.py                 # upsert_note_types() — create/update note types
├── daemon.py                     # Daemon lifecycle — file locks, spawn, shutdown
├── tools.py                      # Registers 7 MCP tools; returns response models (emits outputSchema)
├── schemas.py                    # Pydantic models — single source of truth for every tool request/response + status shape
├── client.py                     # ShrikeClient — standalone HTTP client; typed per-tool methods, daemon lifecycle
├── paths.py                      # Platform-canonical directories (via platformdirs)
├── log.py                        # Logging config, log parsing and styling
├── embedding.py                  # EmbeddingService (llama-server subprocess) + EmbeddingRuntime (start/stop lifecycle)
├── index.py                      # VectorIndex — USearch HNSW index for note embeddings
└── cli/
    ├── __init__.py               # Root Click group, global options (--config, --url, --json, --pretty)
    ├── client.py                 # Re-export shim → shrike.client (keeps imports working)
    ├── config.py                 # YAML config loading/saving
    ├── completion_cmd.py         # shrike completion {bash,zsh,fish}
    ├── embedding_cmd.py          # shrike embedding status/start/stop
    ├── index_cmd.py              # shrike index rebuild/status
    ├── server_cmd.py             # shrike server start/stop/status/logs (daemon management)
    ├── info_cmd.py               # shrike info
    ├── note_cmd.py               # shrike note list/show/create/update/delete/search
    ├── type_cmd.py               # shrike type list/show/create/update/delete
    └── output.py                 # Rich formatting, output_options decorator
tests/
├── unit/                         # 326 tests — direct calls, no server
│   ├── conftest.py               # wrapper fixture (temp collection), basic_note fixture
│   ├── test_collection_info.py
│   ├── test_list_notes.py
│   ├── test_upsert_notes.py
│   ├── test_delete_notes.py
│   ├── test_note_types.py
│   ├── test_client_batching.py
│   ├── test_logging.py
│   ├── test_embedding.py         # EmbeddingService unit tests (mocked subprocess)
│   ├── test_config.py            # Config loading and embedding args
│   ├── test_index.py             # VectorIndex unit tests (mocked embeddings)
│   ├── test_note_embedding_text.py  # CollectionWrapper.note_texts_for_embedding
│   ├── test_tools_search.py     # search_notes, upsert neighbors, delete index updates
│   ├── test_server_security.py  # loopback guard + transport-security helpers
│   ├── test_daemon.py           # stop_server HTTP→SIGTERM→SIGKILL escalation
│   └── test_collection_concurrency.py  # single-worker-thread serialization
└── integration/                  # 159 tests — real server subprocess + HTTP transport
    ├── conftest.py               # server fixture (session-scoped), mcp fixture
    ├── test_tools.py
    ├── test_cli.py
    ├── test_security.py          # custom-route Host/Origin guard + non-loopback refusal
    ├── test_embedding.py         # Embedding tests + orphan reaping (requires llama-server + GGUF model)
    └── test_semantic.py          # Semantic search, neighbors, index CLI (requires llama-server)
docs/
└── mcp-tools.md                  # Tool documentation (human-readable; machine schema is served
                                  # live by the server and defined in shrike/schemas.py)
```

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.12 is used (managed via pyenv; `.python-version` is at repo root). The `anki` package requires Python >= 3.11.

## Running commands

### Tests

```bash
pytest tests/unit -v                           # Unit tests (fast, no server)
pytest tests/integration -v -m integration     # Integration tests (starts a server)
```

#### Coverage

CI enforces a coverage gate (`fail_under` in `[tool.coverage.report]`). The
integration suite runs the server as a `python -m shrike.server` subprocess, so
reproducing the real number locally needs subprocess coverage enabled — install
the hook once, then run both suites under `coverage` and combine:

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo 'import coverage; coverage.process_startup()' > "$SITE/coverage_subprocess.pth"

export COVERAGE_PROCESS_START="$PWD/pyproject.toml"
coverage run --parallel-mode -m pytest tests/unit -q
coverage run --parallel-mode -m pytest tests/integration -q -m "integration and not embedding"
coverage combine && coverage report      # exits non-zero below fail_under
```

A plain `pytest tests/unit --cov=shrike` reads ~18 points lower because it can't
see the server subprocess — use the combined flow above when checking the gate.

### Linting

```bash
ruff check src/shrike/             # Lint
ruff format --check src/shrike/    # Format check
mypy src/shrike/                   # Type check
```

All three must pass cleanly. CI (`.github/workflows/test.yml`) runs, on every PR (Linux x64 only): a `lint` job, a `test` job (unit + non-embedding integration under the coverage gate), and an `embedding` job. macOS and ARM run the full integration suite on **push to `main`** (the `cross-platform` job, gated on `github.event_name == 'push'`) — Actions minutes are limited, and macOS bills at 10×, so the expensive lanes only run at merge time.

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

**Trust boundary.** Every endpoint is unauthenticated, so the server binds loopback only; binding a non-loopback host requires `--allow-remote` (it refuses to start otherwise, with a loud warning) and llama-server stays pinned to `127.0.0.1` regardless. For loopback binds, DNS-rebinding/CSRF protection is on: `_build_transport_security()` allow-lists only loopback `Host`/`Origin` values, applied to the MCP endpoint *and* — via the `_guard` wrapper in `_register_custom_routes` — to the custom routes (`/status`, `/shutdown`, `/index/rebuild`, `/embedding/*`), which bypass MCP middleware. Bearer-token auth for remote use is deferred to the v0.4/v0.6 roadmap.

### MCP tools (7 total)

| Tool | Status | Purpose |
|------|--------|---------|
| `collection_info` | Working | Collection structure, note types, decks, tags, stats |
| `list_notes` | Working | Filter/retrieve notes by deck, tags, type, IDs, date |
| `search_notes` | Working | Semantic similarity search over note embeddings |
| `upsert_notes` | Working | Create or update notes in bulk (1-100), returns similar neighbors |
| `upsert_note_types` | Working | Create or update note type definitions (1-10) |
| `delete_notes` | Working | Permanently delete notes by ID |
| `delete_note_types` | Working | Delete note types by ID (only if unused) |

Every tool request and response shape — plus the server-status shapes — is a Pydantic model in `shrike/schemas.py` (the single source of truth). Tool functions in `tools.py` return the response models, so FastMCP emits an `outputSchema` for each tool, and `_safe_tool` runs each docstring through `inspect.cleandoc` so the advertised descriptions carry no source indentation. The standalone `ShrikeClient` exposes a typed per-tool method for each (e.g. `list_notes(...) -> ListNotesResponse`) that validates the wire response into the model; `ShrikeClient._call()` is the untyped escape hatch. There is no checked-in schema file: the authoritative machine schema is whatever the running server advertises via `tools/list`, derived from these models. `docs/mcp-tools.md` is the human-readable companion.

**Make illegal states unrepresentable — the schemas.py house style.** When a field's presence is *correlated* with another (a hidden state), model it as a **discriminated union**, never a bag of optionals. The pattern is an `Annotated` type alias: each variant is a `BaseModel` with a `Literal` discriminator field, and `Thing = Annotated[A | B, Field(discriminator="status")]`. Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field* typed as `Thing` validates automatically). Examples: per-item results (`UpsertNoteResult = UpsertNoteOk | UpsertNoteError` — success has `id`+`neighbors`, error has `index`+`error`), `IndexStatus` (`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored` — `progress` only on building, `error` only on errored), and the `/index/rebuild` + `/embedding/*` endpoint responses (unions on `status`). Two fields that always appear or vanish *as a pair* are the same smell at smaller scale — group them into a nested sub-model (`NoteTypeInfo.detail: NoteTypeDetail | None`), not two optionals. A bare `X | None` is reserved for *genuinely independent* optionality (a datum absent on its own — `col_mod` before the index is built, a field omitted from a partial update, a caller-selected `collection_info` section); annotate why so it reads as deliberate. Response models carry **no `error` field**: a whole-call failure (bad input, unhandled exception) is raised in the tool and surfaces as an MCP `isError` result, which `ShrikeClient._call` turns into a `ServerError`. Expected bad input raises `ToolInputError` (logged without a traceback); genuine bugs log with one. The only optional advisory on a success response is `message` (e.g. index-building notice, neighbor-retry hint).

Input bounds (e.g. `limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as `Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so FastMCP **rejects** out-of-range input rather than silently clamping. Optional list filters use `Field(default_factory=list)` (keyword-only params) so they render as a plain array in the schema, not a noisy `anyOf:[array, null]`.

### CLI structure

The CLI uses Click with rich for output formatting. Command hierarchy:

```
shrike [--config PATH] [--url URL] [--json] [--pretty/--no-pretty]
├── server start|stop|status|logs
├── info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
├── note list|show|create|update|delete|search
├── type list|show|create|update|delete
├── index rebuild|status|save
└── embedding status|start|stop
```

The CLI talks to the MCP server over HTTP — it can target a remote server via `--url` or `SHRIKE_URL`.

`--json` and `--pretty/--no-pretty` are global options but also accepted on every leaf command (via the `@output_options` decorator in `output.py`), so both `shrike --json info` and `shrike info --json` work. `--json` implies `--no-pretty`; combining `--json --pretty` is an error.

**Identifier resolution** — `type show`, `type update`, and `type delete` accept either a name or numeric ID. Note commands accept IDs with an optional `#` prefix (e.g., `note show #123`). The `NoteIDType` custom Click type in `output.py` handles `#` stripping. `note show` is sugar for `note list --ids ID`; `type show` is sugar for `type list IDENTIFIER`.

**Output conventions:**
- Colors: cyan for names/paths/URLs, green for `#ID` identifiers, yellow for tags, dim for labels/headers, no color on plain counts or dates
- Headers: `"Showing X of Y note(s) in DeckName from /path/to/collection"` pattern
- Tables: flush-left, no borders, dim underlined column headers (matches `gh` CLI style)
- Detail views: Rich `Panel` with dim border, bold title
- Spinners: `output.spinner()` context manager, dots style, no-op when `--no-pretty`
- Results: `+` (green) for created, `~` (yellow) for updated, `!` (red) for errors
- Validation errors use `click.UsageError` (shows usage line); runtime errors use `click.ClickException`

### Daemon management

`shrike server start` spawns the server as a background process. Lifecycle is managed by `shrike/daemon.py`.

**Liveness detection** uses file locks via `filelock` (fcntl on Unix, msvcrt on Windows). The server holds an exclusive lock on `server.lock` for its entire lifetime. When the server exits — cleanly or via crash — the OS releases the lock. Clients probe liveness by attempting a non-blocking lock acquisition. This avoids PID recycling issues entirely.

**Shutdown** is cross-platform via an HTTP endpoint (`POST /shutdown` on the running server, registered via FastMCP's `custom_route`). The CLI's `stop_server()` uses a three-tier strategy:
1. HTTP POST `/shutdown` — clean, works on all platforms
2. SIGTERM (Unix only) — fallback if HTTP is unresponsive
3. SIGKILL / TerminateProcess — last resort for hung processes

Signal handlers (SIGTERM, SIGINT) remain as a secondary path for Unix `kill` commands and Ctrl+C in foreground mode.

**HTTP endpoints** beyond MCP:
- `GET /status` — returns JSON with pid, url, collection, log_level, log_dir, uptime, embedding, index. Used by `shrike server status` and auto-start health checks.
- `POST /shutdown` — triggers graceful server shutdown.
- `POST /index/rebuild` — triggers a full index rebuild (returns immediately with status/progress). Requires the embedding service to be running.
- `POST /index/save` — forces an immediate flush of the in-memory index to disk (off the event loop). Returns `saved` (with `size` and the `pending` count it flushed), `empty` (no index built yet), or `building` (refused mid-rebuild). Backs `shrike index save`; the index also saves automatically (debounced flush + shutdown).
- `POST /embedding/start` — starts the embedding service (optional JSON body overrides model/port/etc.; falls back to the params the server booted with). Attaches it to the index and triggers a rebuild if the model changed or the index drifted. Returns `started` / `already_running` / a 400 if no model is configured.
- `POST /embedding/stop` — saves the index, then stops the embedding service and marks the index `unavailable`. The server and collection stay up.

State files live in the platform state directory (see `shrike/paths.py`):
- `server.lock` — exclusive file lock held by the running server
- `server.pid` — PID file (convenience for diagnostics, not used for liveness)
- `server.json` — metadata (URL, port, collection path, start time, log dir)

### Platform directories

All file paths are resolved via `platformdirs` in `shrike/paths.py`:

| Purpose | macOS | Linux (XDG) | Windows |
|---------|-------|-------------|---------|
| Config | `~/Library/Application Support/shrike/` | `~/.config/shrike/` | `%APPDATA%\shrike\` |
| State | `~/Library/Application Support/shrike/` | `~/.local/state/shrike/` | `%LOCALAPPDATA%\shrike\` |
| Logs | `~/Library/Logs/shrike/` | `~/.local/state/shrike/log/` | `%LOCALAPPDATA%\shrike\Logs\` |
| Cache | `~/Library/Caches/shrike/` | `~/.cache/shrike/` | `%LOCALAPPDATA%\shrike\Cache\` |

On Linux, XDG env vars (`XDG_CONFIG_HOME`, `XDG_STATE_HOME`, etc.) are respected.

### Config file

YAML at the platform config directory (`config.yml`). Auto-created on first `shrike server start` (the `embedding` section, including the model path, is persisted there too). Resolution order: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`, `SHRIKE_EMBEDDING_MODEL`, `SHRIKE_EMBEDDING_PORT`, `LLAMA_SERVER_PATH`, `SHRIKE_CACHE_DIR`, `SHRIKE_INDEX_SAVE_DELAY`, `SHRIKE_INDEX_SAVE_THRESHOLD`) → CLI flags. Embedding params follow the same cascade via `config.resolve_embedding()`, shared by `shrike server start` and `shrike embedding start`; the index cache dir and flush tuning follow it via `config.resolve_cache_dir()` / `config.resolve_index_save()`. `save_config` persists `collection`, `cache_dir`, non-default `server.*`, `embedding.*`, and any set `index.save_delay` / `index.save_threshold`; **logging overrides are read from config but not written by auto-save** — set `logging.level` / `logging.dir` in `config.yml` directly. The `index.*` flush knobs and `cache_dir` resolve to `None` in config, meaning "use the server's built-in defaults" (`IndexSaver`'s 60s/100 and the platform cache dir) — the numeric defaults live in `shrike.index`, not duplicated in config.

### Embedding service lifecycle

The embedding service (llama-server) can be cycled independently of the Shrike server. `EmbeddingRuntime` (`embedding.py`) owns the current `EmbeddingService` (or `None`), the params needed to (re)start it, and the binding to the index; it serializes start/stop under a lock. The `VectorIndex` is **always** created at boot, even with no embedder — it loads on-disk vectors and reports `unavailable` until a service is attached.

- `shrike server start` starts embedding at boot if a model is configured, unless `--no-embedding` is passed (lets a server run with embedding deliberately off).
- `shrike embedding start` / `shrike embedding stop` cycle the service on a running server (for llama-server upgrades, model swaps, freeing GPU/RAM). Stopping marks the index `unavailable` but keeps the on-disk vectors; starting re-attaches and rebuilds only if needed.
- Starting llama-server blocks (model load + health wait), so the HTTP handler runs it via `asyncio.to_thread` to keep the event loop responsive.
- **Orphan reaping:** `EmbeddingService` records the llama-server PID in `<state-dir>/embedding.pid` (written after spawn, removed on clean stop). If Shrike is hard-killed (SIGKILL, incl. the daemon's own force-kill path), llama-server is orphaned and keeps holding its port. On the next `start()`, `_reap_orphan` detects a recorded PID that is still alive *and* holding the port and terminates it (SIGTERM→SIGKILL) before binding. `PR_SET_PDEATHSIG` is intentionally avoided: the parent-death signal keys on the spawning *thread*, and start runs under `asyncio.to_thread`, so a reclaimed pool thread could kill a live server.

### Vector index and consistency

The vector index is a **derived cache**, not a co-equal store. The Anki collection (SQLite) is always the source of truth. SQLite handles its own crash recovery via WAL/journal, so the collection is self-consistent after any crash. If the index is stale, corrupt, or missing, it can be rebuilt from the collection by re-embedding all notes.

**Consistency model:** the index may lag behind the collection (notes added/modified/deleted without the index being updated), but the collection never lags behind the index. This means search results may be stale, but data operations (upsert, delete, list) are always correct.

**Drift detection:** the index metadata (`index.meta.json`) stores `col_mod` — the value of `col.mod` (collection-level modification timestamp, milliseconds) at the time the index was last built — and `model_id`, a fingerprint of the embedding model that produced the vectors. On startup (and whenever the embedding service is attached), compare both against the stored values. If they match, nothing changed — skip reindexing. If `col.mod` differs (something changed outside our control) **or** `model_id` differs (the embedding model changed, so every vector lives in a different space and is invalid), trigger a full rebuild. No watermarks, no per-note diffing, no fragile heuristics. Shrike owns the collection most of the time; anything that changed externally (Anki GUI, sync, imports) should force a clean rebuild for correctness. When sync is implemented (v0.3.0), its implications for index maintenance will be revisited.

**Model fingerprint:** `model_id` comes from llama-server's `GET /v1/models` `meta` block (`n_params`, `n_embd`, `n_vocab`, `n_ctx_train`, `size`) — fast and describes the *loaded* model. It falls back to model filename + on-disk size if that metadata is unavailable. The model *name* is deliberately excluded (it would force needless rebuilds on rename and miss same-name re-quantizations, which the numeric fields catch). `EmbeddingService.embed()` also pins `"model": <id>` in the request body as insurance against a future external multi-model endpoint.

**Implementation:**

1. **Startup check** — on server start, compare `col.mod` against the stored value in index metadata. Match → load existing index. Mismatch, missing, or corrupt → full rebuild in a background thread. Server starts accepting requests immediately; `search_notes` returns actionable status messages ("building 2847/5000 notes, try again shortly") until ready.

2. **Incremental updates** — after `upsert_notes` and `delete_notes` succeed on the collection, the index is updated in the same call (`index.add()` / `index.remove()`). Stored `col_mod` is updated after each successful index update. Index update failures log a warning but don't fail the tool call — the next startup detects the `col.mod` mismatch and rebuilds.

3. **Persistence** — the index is saved to disk on graceful shutdown (signal handler and `POST /shutdown`), at the end of a rebuild, and via a **debounced flush** during normal operation. `IndexSaver` (in `index.py`) owns the debounce: the upsert/delete tools call `saver.request_save()` after each incremental update (once `col_mod` is set), and the index is written either **`save_delay` seconds after the last change** (idle debounce, default 60s) **or immediately once `save_threshold` unsaved changes accumulate** (burst cap, default 100), whichever comes first. The save runs off the event loop (`asyncio.to_thread`); the debounce timer is `loop.call_later`, so there is no background timer *thread* and no fixed-interval polling — the flush is driven by edit activity. This bounds how much incremental work a hard kill / crash discards: once a flush lands and the server goes idle, the on-disk index (and its `col_mod`) are current, so it reloads without a rebuild. For edits since the last flush, the `col.mod` mismatch on next startup still triggers a full rebuild — correctness is preserved either way, at the cost of a re-embed. `save_delay`/`save_threshold` are configurable (config `index.*`, env, `--index-save-*` flags); the cache location is `cache_dir`/`SHRIKE_CACHE_DIR`/`--cache-dir`. (Tombstone compaction is unnecessary on the pinned USearch — see the index code comments.)

4. **Full rebuild** — `shrike index rebuild` CLI and `POST /index/rebuild` endpoint. Drops existing index and re-embeds all notes. Progress reporting via CLI and `/status`.

5. **State machine** — states: `ready`, `building` (with progress), `unavailable` (embedding service not running — never configured or stopped), `error` (build failed). Exposed via `/status` endpoint, `search_notes` responses, and `shrike server status` CLI.

**Cost considerations:** full rebuilds are the only reindexing strategy — no incremental reconciliation. A typical collection (1K notes) rebuilds in seconds; a large one (10K+) takes minutes. Rebuilds run in a background thread, so the server is never blocked. During normal operation, incremental updates from `upsert_notes`/`delete_notes` keep the index current without rebuilds.

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

## Roadmap

### v0.1.0 — CLI + MCP Server ✓

- CLI integration tests and bug fixes (full command coverage) ✓
- Daemon auto-start from any CLI command ✓
- CLI output UI/UX review ✓
- Tab completion (bash, zsh, fish) ✓
- Transparent batching in ShrikeClient for large requests ✓
- `delete_note_types` MCP tool and `type delete` CLI command ✓
- `/status` HTTP endpoint for health checks ✓

### v0.2.0 — Semantic Search ✓

- llama-server integration for local embeddings ✓
- USearch vector index (HNSW) for note content ✓
- Wire `search_notes` tool to the vector index ✓
- Incremental index updates on note create/modify/delete ✓
- Startup drift detection (`col.mod` comparison) and background rebuild ✓
- Index persistence on graceful shutdown, after rebuild, and via a count-based flush every N incremental edits ✓ (no time-based timer; a crash mid-burst still self-heals via `col.mod` drift)
- `shrike index rebuild` CLI command for full re-indexing ✓
- Index build state machine and progress tracking (ready/building/unavailable/error) ✓
- Index status in `/status` endpoint and `search_notes` responses (actionable messages) ✓
- `shrike embedding status` and `shrike index status` CLI commands ✓
- Embedding service lifecycle ✓: `shrike embedding start` / `shrike embedding stop` cycle llama-server independently of the Shrike server; `shrike server start --no-embedding` boots with it off; the index records the embedding model's fingerprint (`model_id`) and forces a rebuild when the model changes.
- Contextual upsert responses ✓: `upsert_notes` returns `neighbors` for each created/updated note — the k most similar existing notes as `{id, score, tags}` objects ranked by cosine similarity (defaults: `top_k_neighbors=5`, `neighbor_threshold=0.5`). Same search operation as `search_notes` (which returns `{id, score, tags, content}`), triggered by the upserted note's own content as the query. Neighbors below the threshold are excluded; batch notes are excluded from each other's results. Raw neighbor data — the server makes no tag suggestions; callers decide what to do with it. Grounds LLM-driven card creation in the collection's existing taxonomy and surfaces near-duplicates for investigation.
- Duplicate detection ✓: threshold-based, using the same similarity infrastructure. High similarity scores in neighbors or search results indicate potential duplicates — callers apply their own threshold. No separate duplicate detection endpoint; `search_notes` and contextual upsert neighbors are the same operation.

### v0.3.0 — Skill Plugin

- Extract `ShrikeClient` from CLI into a standalone Python client (`shrike.client`) usable outside the CLI — daemon lifecycle, MCP tool calls, server status. CLI becomes a thin layer over this client.
- Reference skill plugin (Claude custom skill format): encodes pedagogical best practices for LLM-driven card creation — minimum information principle, cloze discipline, prefer existing decks over new ones, tag consistency via contextual upsert data, broad decks with tags over fine-grained deck hierarchies. Keeps opinions in the skill, not the server. Designed for Project-style setups with course materials as context. Initial goal is real-use iteration, not packaging.

### v0.4.0 — Sync

- AnkiWeb sync (auth, trigger, status)
- Self-hosted anki-sync-server support
- `shrike sync` commands
- Sync server lifecycle management (`shrike sync-server start/stop/status`)
- Credential storage

### v0.5.0 — Desktop Application

- Tauri shell wrapping the Python process
- System tray / menu bar
- Settings UI
- Duplicate detection alerts in UI

### v0.6.0 — Relay Prototype

- Lightweight relay server: authenticates and forwards MCP JSON-RPC to a user's local Shrike instance
- Removes the need for Tailscale or similar tunneling tools
- Motivating use case: "study companion" workflow — student in Claude.ai with course materials in a Project and skill plugin shaping card creation, talking to their local collection via the relay
- Scope is minimal: auth, forwarding, rate limiting, nothing else
- Explicitly a prototype to test demand before investing in a hosted solution (which would need sync, multi-tenancy, storage)

## What's not yet implemented

- **Skill plugin**: Not started. Depends on contextual upsert responses from v0.2.0 (now complete).
- **Sync**: No sync support yet.
- **Desktop application**: Not started.
- **Relay**: Not started.
