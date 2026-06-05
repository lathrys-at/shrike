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
├── __init__.py                   # Re-exports __version__ from generated _version.py (hatch-vcs)
├── server.py                     # MCP server entry point (argparse, FastMCP)
├── collection.py                 # CollectionWrapper — all Anki DB operations
├── note_types.py                 # upsert_note_types() — create/update note types
├── daemon.py                     # Daemon lifecycle — file locks, spawn, shutdown
├── tools.py                      # Registers 17 MCP tools; returns response models (emits outputSchema)
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
    ├── note_cmd.py               # shrike note list/show/create/update/tag/delete/search
    ├── tag_cmd.py                # shrike tag rename (collection-level tag ops)
    ├── deck_cmd.py               # shrike deck create/rename/delete
    ├── type_cmd.py               # shrike type list/show/create/update/delete
    └── output.py                 # Rich formatting, output_options decorator
tests/
├── unit/                         # 570 tests — direct calls, no server
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
└── integration/                  # 206 tests — real server subprocess + HTTP transport
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
coverage run --parallel-mode -m pytest tests/unit tests/integration -q -m "not embedding" -n auto
coverage combine && coverage report      # exits non-zero below fail_under
```

Both suites run in one combined `-n auto` invocation (xdist balances them, so
workers don't idle between phases — faster than two separate runs); `-m "not
embedding"` drops the embedding-gated tests. This is exactly what CI runs.

A plain `pytest tests/unit --cov=shrike` reads ~18 points lower because it can't
see the server subprocess — use the combined flow above when checking the gate.

**`-n auto`** (pytest-xdist) parallelizes the suite across cores — the integration
suite is server-spawn-bound and roughly halves (each server gets its own free
port + temp state/log/cache dirs, so workers don't collide). It composes with the
coverage hook above (the `.pth` fires for each xdist worker *and* each spawned
server, so `coverage combine` merges everything to the same total). CI runs both
suites with `-n auto`. Locally it's opt-in — the default (no `-n`) stays serial so
`-x`, `-s`, and `pdb` keep working for debugging.

### Linting

```bash
ruff check src/shrike/             # Lint
ruff format --check src/shrike/    # Format check
mypy src/shrike/                   # Type check
```

All three must pass cleanly. CI (`.github/workflows/test.yml`) runs, on every PR (Linux x64 only): a `lint` job, a `test` job (unit + non-embedding integration under the coverage gate), and an `embedding` job. macOS and ARM run the full integration suite via the `cross-platform` job, gated on `contains(github.event.pull_request.labels.*.name, 'rc')` — i.e. **only** on a PR labelled `rc` (release candidate), never on plain PRs and never on merge to `main`. Actions minutes are limited and macOS bills at 10×, so these lanes stay off the normal iterate-and-merge loop entirely; apply the `rc` label before tagging a release to get cross-platform coverage first. The PR trigger lists `labeled` in its `types` so adding the label re-triggers CI.

The embedding/cross-platform jobs **cache** the pinned llama-server and the GGUF test model (`actions/cache`) so they aren't re-downloaded every run. But an `actions/cache` entry is only restorable from the run's own branch or the **default branch** — and `test.yml` runs on PRs only, so nothing ever seeds `main`'s cache scope and every PR would cold-download the model from HuggingFace (which intermittently `429`s). A separate **cache-warmer** (`.github/workflows/warm-cache.yml`) closes that gap: it runs on `main` twice weekly (and via `workflow_dispatch`), downloads the pinned llama-server + model, and lets `actions/cache` save them into `main`'s scope, which every PR then restores from. It uses the *same* cache paths/keys as the embedding job; llama-server stays pinned via `scripts/llama-server.lock` and the model via the `EMBEDDING_MODEL_*` constants in `tests/integration/model_cache.py` (both bumped manually). The fixture's `download_with_retry` (backoff on `429`/5xx) remains the backstop for a cold/evicted run (#83, #93).

A separate **release workflow** (`.github/workflows/release.yml`) fires on `push` of a `v*` tag (not on PRs): it runs the full cross-platform integration suite on all three platforms unconditionally, builds the release artifacts — Python sdist + wheel (hatch-vcs derives the version from the tag, so the build/test jobs check out with `fetch-depth: 0`), the `anki-cards.skill` bundle (`scripts/package-skill.py`), and a `SHA256SUMS` — and cuts a GitHub Release with them all attached. Release notes come from the matching `## [X.Y.Z]` section of `CHANGELOG.md` (falling back to auto-generated commit notes); a pre-release tag (`vX.Y.Z-rc.N`, detected by the SemVer hyphen) is published as a GitHub pre-release. PyPI publishing is intentionally not wired up (#43).

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

**Trust boundary.** Every endpoint is unauthenticated, so the server binds loopback by default; binding a non-loopback host requires `--allow-remote` (it refuses to start otherwise, with a loud warning) and llama-server stays pinned to `127.0.0.1` regardless. DNS-rebinding/CSRF protection (`_build_transport_security()`) validates `Host`/`Origin`, applied to the MCP endpoint *and* — via the `_guard` wrapper in `_register_custom_routes` — to the custom routes (`/status`, `/shutdown`, `/index/rebuild`, `/embedding/*`), which bypass MCP middleware. The guard is **independent of the bind address**: a loopback bind allow-lists loopback `Host`/`Origin`; `--allowed-host`/`--allowed-origin` (config `server.allowed_hosts`/`allowed_origins`, env `SHRIKE_ALLOWED_HOSTS`/`SHRIKE_ALLOWED_ORIGINS`, comma-separated) *add* trusted values for a reverse-proxy or VPN hostname — note a proxy forwards `Host: name:port`, so use the SDK's `name:*` port-wildcard form; and `--no-dns-rebinding-protection` (config `server.no_dns_rebinding_protection`) turns the guard off entirely for deployments where the network is the trust boundary (behind Caddy, on a tailnet, firewalled). A non-loopback bind with no explicit allow-list also leaves the guard off (preserves the original `--allow-remote` behaviour). In every mode the endpoints stay unauthenticated — the guard is anti-CSRF/DNS-rebinding, not authentication.

**OAuth is required for native connectors (deferred; tracked here).** Claude Desktop / claude.ai *URL connectors* require OAuth 2.1 + Dynamic Client Registration: they try to register against the MCP server's sign-in service and fail against an unauthenticated endpoint, regardless of TLS or network exposure. So the "run Shrike behind a reverse proxy / on a VPN and add it as a connector" story depends on implementing MCP server auth (`mcp.server.auth`, OAuth 2.0 + PKCE — see audit §1.1); it's a v0.4/v0.6-class project, intentionally not started. Until then the unauthenticated + network-boundary model serves CLI / programmatic / `mcp-remote` clients: a native client reaches Shrike through the **`mcp-remote` stdio bridge** (`npx mcp-remote http://127.0.0.1:8372/mcp --allow-http --transport http-only`), which connects without auth because Shrike demands none. This is how the QA harness drives Claude Desktop.

### MCP tools (17 total)

| Tool | Status | Purpose |
|------|--------|---------|
| `collection_info` | Working | Collection structure, note types, decks, tags, stats |
| `list_notes` | Working | Filter/retrieve notes by deck, tags, type, IDs, date |
| `search_notes` | Working | Per-query semantic similarity **and** exact-substring search, folded into annotated results (`score`?/`substring`?) |
| `collection_query` | Working | Raw Anki search expression (`is:due`, `prop:`, …) → notes; same shape as `list_notes` |
| `upsert_notes` | Working | Create or update notes in bulk (1-100); `on_duplicate` policy + `dry_run`; returns similar neighbors |
| `upsert_note_types` | Working | Create or update note type definitions (1-10) |
| `update_note_type_fields` | Working | Edit a note type's fields by name: add/remove/rename/reposition (data-safe) |
| `update_note_type_templates` | Working | Edit a note type's card templates by name: add/remove/rename/reposition (data-safe) |
| `find_replace_note_types` | Working | Find/replace text in one note type's template HTML + CSS (front/back/css selectors); returns a count |
| `update_note_tags` | Working | Edit tags on a note set (1-1000): `set` (replace) XOR `add`/`remove` |
| `rename_tag` | Working | Rename a tag collection-wide or on a note set (exact match) |
| `find_replace_notes` | Working | Bulk find/replace across fields in a scoped set; literal or regex; `dry_run` preview |
| `upsert_decks` | Working | Create or rename/reparent decks in bulk (1-100); id = rename |
| `delete_decks` | Working | Delete decks by name, only if empty (else reported `not_empty`) |
| `delete_notes` | Working | Permanently delete notes by ID |
| `delete_note_types` | Working | Delete note types by ID (only if unused) |
| `collection_prune` | Working | Cleanup: unused tags + empty notes + empty cards; `dry_run` default **true** (preview) |

**Duplicate handling lives *inside* `upsert_notes`, not in a separate pre-check (#77).** Before each new note is written, Anki's own add-note validation (`note.fields_check()`, via `CollectionWrapper._check_new_note`) runs. A first-field duplicate (same first field as an existing note of that type — Anki's rule, collection-wide and **deck-independent**) is governed by the `on_duplicate` param: `error` (**default** — reported, not written), `skip` (`status: skipped`), or `allow` (written anyway). Structurally invalid notes — empty first field, broken cloze — are *always* reported as errors with a `reason` regardless of policy, and never written. `dry_run: true` runs the exact same validation but writes nothing: every result is `ok` (with `action: create|update`), `skipped`, or `error`, and the response echoes `dry_run: true` — so `dry_run` + the default policy is a full `fields_check`-based sanity pass over a batch. The per-item result union (`UpsertNoteResult` in `schemas.py`) carries the four variants (`UpsertNoteOk` / `UpsertNoteValidated` / `UpsertNoteSkipped` / `UpsertNoteError`, discriminated on `status`); `UpsertNoteError.reason` is the machine-readable `NoteValidationReason`. This is Anki's *exact* first-field rule, distinct from the *semantic* near-duplicate signal the returned `neighbors` provide. A standalone `canAddNotes`-style tool was rejected: it would be racy (check-then-write) and only actionable by a follow-up call (see `docs/decisions.md`). Note the validation applies to **creates**; updates are validated for existence/fields but not duplicate/empty. `dry_run` does not catch intra-batch duplicates (it writes nothing, so two identical new notes in one call both validate clean — a real run catches the second).

**Note-type field/template edits come in two flavours, both data-safe (#76).** `upsert_note_types` replaces the whole `fields`/`templates` list **by position** (the field/template at each position keeps its note data/cards even when renamed; only shortening drops the tail) — fixed in #99 after the original rebuild-from-scratch blanked note data and deleted cards. `update_note_type_fields` and `update_note_type_templates` are the **by-identity** counterparts: a sequence of `add`/`remove`/`rename`/`reposition` ops addressed by field/template *name*, so they can express a true move, an insert, or a non-trailing remove that position-replace can't. They delegate to Anki's own data-safe primitives (`rename_field`/`reposition_field`/`add_field`/`remove_field` and the `*_template` equivalents), which migrate note data / cards by identity — so a reposition is a true move and a non-trailing remove drops only that field's data / that template's cards. (A template `rename` is a pure label change: cards key by ordinal, not name.) Each call is **atomic** (the op sequence is validated against a simulated name list before any primitive runs — an invalid op changes nothing) and persists with a single `update_dict`. Like `upsert_note_types`, they do no inline index maintenance — the `col.mod` bump from `update_dict` triggers a drift-rebuild on next startup (a removed field changes embedding text, so a rebuild is correct; for the rest it's conservative). All live in `note_types.py`, sharing `_simulate_struct_op` and the soundness check; bad ops raise `NoteTypeOpError` → `ToolInputError` (logged without a traceback). The position vs identity tools are **reconciled** so they can't overlap dangerously: the positional `fields`/`templates` replace in `upsert_note_types` *rejects* any update where an existing name moves to a different position (a reorder/insert/non-trailing-remove — which would silently re-label note data / cards), erroring with a pointer to the matching identity tool (`_reject_unsound_positional_replace`). It keeps only the unambiguous positional edits: rename/edit-in-place, append, trailing-remove. A third tool, `find_replace_note_types` (anki-connect's `findAndReplaceInModels`), does a literal-or-regex text rewrite over one model's template `qfmt`/`afmt` and shared `css` (selected by `front`/`back`/`css`), returning a replacement count — it touches *template text*, not fields or note data, so it migrates nothing and (unlike the field/template structure ops) bumps `col_mod` without a re-embed. Literal mode escapes the pattern and inserts the replacement verbatim (no `\1` interpretation); `match_case` defaults **true** (template/CSS is code). It lives in `note_types.py` (`find_and_replace_note_types` + `_subn_text`) and raises `NoteTypeOpError` → `ToolInputError` for an unknown model or a bad regex.

The tag tools (#73), deck tools (#74), **and** `find_replace_note_types` (#76) are a derived-index-aware special case: tags, deck names, and a note type's template/CSS text are **not** part of a note's embedding text, so these ops leave every vector valid but bump `col.mod`. Each advances the stored `index.col_mod` (and requests a debounced save) **without** re-embedding — via the shared `_bump_col_mod_after_metadata_change` helper in `tools.py` — so such a metadata-only change doesn't force a spurious full rebuild on next startup. Full-replace of tags lives in both `update_note_tags` (`set`) and incidentally in `upsert_notes` (`{id, tags}`); the additive/subtractive logic lives only in `update_note_tags`. `upsert_decks` mirrors `upsert_notes` (id present = rename the existing deck; absent = create); **decks never merge** — renaming onto an existing deck name is an error, and `delete_decks` is **empty-only** (move notes out first), so deck deletion can never delete a note.

**Collection maintenance is one tool, `collection_prune` (#89), not scattered cleanups.** It runs any of three cleanups — clear **unused tags** (`tags.clear_unused_tags`), remove **empty notes**, remove **empty cards** (`get_empty_cards` → `remove_cards_and_orphaned_notes`); selecting none runs all. It is **preview-by-default**: `dry_run` defaults **true** (unlike `find_replace_notes`, which applies by default — prune is collection-wide and deletes notes/cards, so it errs the safe way; the CLI `shrike collection prune` previews unless `--apply`). The earlier standalone `clear_unused_tags`/`shrike tag clean` (#73) was removed (#90) to land here, and this supersedes a standalone remove-empty-notes (#78). An **empty note** is one whose every field is blank by `embed_text.field_is_blank` — no text *and* no media (`<img|audio|video|object|embed|source>`/`[sound:]`), so an image- or audio-only note is never deleted. Index handling is **mixed**, which is why this isn't a pure metadata-bump op like the tag/deck ones: empty-note and empty-card removal **delete notes**, so their vectors leave the index via `index.remove` exactly like `delete_notes`; clearing unused tags is vectors-unchanged. The tool does both in one pass (`index.remove(removed_note_ids)` then advance `col_mod` + save) when anything changed. On apply the order is empty notes → empty cards → unused tags, so tags freed by the deletions get cleared in the same call; the dry-run previews each cleanup independently (so an apply may clear a few more tags than the preview showed). Logic lives in `CollectionWrapper._prune`/`_find_empty_notes`; the CLI `collection` group (`cli/collection_cmd.py`) also houses `collection query` (below).

**Three retrieval surfaces, by intent.** `list_notes` is structured filters (deck/tags/type/ids/date, ANDed); `search_notes` is meaning + exact text (semantic + substring, annotated); `collection_query` (#97) is the **raw Anki escape hatch** — the string goes straight to `col.find_notes`, so the full expression language works (`is:due`, `prop:ivl>=30`, `added:`, `rated:`, `flag:`, `nid:`/`cid:`, `OR`/`-`/brackets). It exists because #86 removed `note list --query` (the leaky raw param) in favour of an explicit tool. It is **read-only** — filtering by `is:due` returns notes and performs no review, so the full grammar is allowed despite Shrike's non-review stance (no whitelist to police). It reuses `list_notes`' serialization (`_note_to_dict`) and `ListNotesResponse`; a malformed expression raises `anki.errors.SearchError`, which the tool maps to `ToolInputError` (stripping Anki's U+2068/U+2069 isolation marks). Lives in `CollectionWrapper._query`.

`find_replace_notes` (#85) is the opposite case: it edits **field bodies**, which *are* embedding text, so it re-embeds the changed notes via the `upsert_notes` index path (not the col_mod-only helper). The actual edit runs Anki's `col.find_and_replace` (Rust regex, undo-able); the changed set is detected by diffing `notes.flds` before/after (note `mod` is only second-resolution, so a same-second edit shows no bump) and exactly those are re-embedded. The dry-run preview is computed in Python (`apply_replacement`): exact for literal, illustrative for regex. A scope (`deck`/`tags`/`note_type`/`ids`) is required.

Every tool request and response shape — plus the server-status shapes — is a Pydantic model in `shrike/schemas.py` (the single source of truth). Tool functions in `tools.py` return the response models, so FastMCP emits an `outputSchema` for each tool, and `_safe_tool` runs each docstring through `inspect.cleandoc` so the advertised descriptions carry no source indentation. The standalone `ShrikeClient` exposes a typed per-tool method for each (e.g. `list_notes(...) -> ListNotesResponse`) that validates the wire response into the model; `ShrikeClient._call()` is the untyped escape hatch. There is no checked-in schema file: the authoritative machine schema is whatever the running server advertises via `tools/list`, derived from these models. `docs/mcp-tools.md` is the human-readable companion.

**Make illegal states unrepresentable — the schemas.py house style.** When a field's presence is *correlated* with another (a hidden state), model it as a **discriminated union**, never a bag of optionals. The pattern is an `Annotated` type alias: each variant is a `BaseModel` with a `Literal` discriminator field, and `Thing = Annotated[A | B, Field(discriminator="status")]`. Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field* typed as `Thing` validates automatically). Examples: per-item results (`UpsertNoteResult = UpsertNoteOk | UpsertNoteError` — success has `id`+`neighbors`, error has `index`+`error`), `IndexStatus` (`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored` — `progress` only on building, `error` only on errored), and the `/index/rebuild` + `/embedding/*` endpoint responses (unions on `status`). Two fields that always appear or vanish *as a pair* are the same smell at smaller scale — group them into a nested sub-model (`NoteTypeInfo.detail: NoteTypeDetail | None`), not two optionals. A bare `X | None` is reserved for *genuinely independent* optionality (a datum absent on its own — `col_mod` before the index is built, a field omitted from a partial update, a caller-selected `collection_info` section); annotate why so it reads as deliberate. Response models carry **no `error` field**: a whole-call failure (bad input, unhandled exception) is raised in the tool and surfaces as an MCP `isError` result, which `ShrikeClient._call` turns into a `ServerError`. Expected bad input raises `ToolInputError` (logged without a traceback); genuine bugs log with one. The only optional advisory on a success response is `message` (e.g. index-building notice, neighbor-retry hint).

Input bounds (e.g. `limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as `Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so FastMCP **rejects** out-of-range input rather than silently clamping. Optional list filters use `Field(default_factory=list)` (keyword-only params) so they render as a plain array in the schema, not a noisy `anyOf:[array, null]`.

### CLI structure

The CLI uses Click with rich for output formatting. Command hierarchy:

```
shrike [--config PATH] [--url URL] [--json] [--pretty/--no-pretty]
├── server start|stop|status|logs
├── info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
├── note list|show|create|update|tag|delete|search
├── deck create|rename|delete
├── tag rename
├── type list|show|create|update|delete
├── collection query|prune
├── index rebuild|status|save
└── embedding status|start|stop
```

The CLI talks to the MCP server over HTTP — it can target a remote server via `--url` or `SHRIKE_URL`.

`--json` and `--pretty/--no-pretty` are global options but also accepted on every leaf command (via the `@output_options` decorator in `output.py`), so both `shrike --json info` and `shrike info --json` work. `--json` implies `--no-pretty`; combining `--json --pretty` is an error.

**Identifier resolution** — `type show`, `type update`, and `type delete` accept either a name or numeric ID. Note commands accept IDs with an optional `#` prefix (e.g., `note show #123`). The `NoteIDType` custom Click type in `output.py` handles `#` stripping. `note show` is sugar for `note list --ids ID`; `type show` is sugar for `type list IDENTIFIER`. **Decks** are referenceable by name, numeric ID, or `#id` wherever a deck is *taken* (`--deck` on note list/create/update/search, `deck rename`/`delete`) — `#id` is always an ID, a bare number is tried as an ID then falls back to a literal name. Resolution is server-side in `CollectionWrapper._resolve_deck_ref` (used by `list_notes`/`search_notes`/upsert/`delete_decks`); the CLI passes refs through untouched except `deck rename`, which resolves to an ID client-side via `_match_deck` in `deck_cmd.py` for the `upsert_decks` call (#88).

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

YAML at the platform config directory (`config.yml`). **User-managed: `shrike server start` never writes it unless `--save-config` is passed** (#56). With `--save-config`, start persists the resolved flags (the `embedding` section, including the model path, is written there too) so later runs and `shrike embedding start` pick them up; without it, start is a no-write operation and always reflects exactly the flags it was given. (The old behaviour wrote the file once on first run and then silently ignored later flags — the divergence that was the #56 bug.) Resolution order: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`, `SHRIKE_EMBEDDING_MODEL`, `SHRIKE_EMBEDDING_PORT`, `SHRIKE_EMBEDDING_POOLING`, `SHRIKE_EMBEDDING_ARGS`, `LLAMA_SERVER_PATH`, `SHRIKE_CACHE_DIR`, `SHRIKE_INDEX_SAVE_DELAY`, `SHRIKE_INDEX_SAVE_THRESHOLD`) → CLI flags. Embedding params follow the same cascade via `config.resolve_embedding()`, shared by `shrike server start` and `shrike embedding start`; the index cache dir and flush tuning follow it via `config.resolve_cache_dir()` / `config.resolve_index_save()`. `save_config` persists `collection`, `cache_dir`, non-default `server.*`, `embedding.*`, and any set `index.save_delay` / `index.save_threshold`; **logging overrides are read from config but never written by `--save-config`** — set `logging.level` / `logging.dir` in `config.yml` directly. The `index.*` flush knobs and `cache_dir` resolve to `None` in config, meaning "use the server's built-in defaults" (`IndexSaver`'s 60s/100 and the platform cache dir) — the numeric defaults live in `shrike.index`, not duplicated in config.

### Embedding service lifecycle

The embedding service (llama-server) can be cycled independently of the Shrike server. `EmbeddingRuntime` (`embedding.py`) owns the current `EmbeddingService` (or `None`), the params needed to (re)start it, and the binding to the index; it serializes start/stop under a lock. The `VectorIndex` is **always** created at boot, even with no embedder — it loads on-disk vectors and reports `unavailable` until a service is attached.

- `shrike server start` starts embedding at boot if a model is configured, unless `--no-embedding` is passed (lets a server run with embedding deliberately off).
- `shrike embedding start` / `shrike embedding stop` cycle the service on a running server (for llama-server upgrades, model swaps, freeing GPU/RAM). Stopping marks the index `unavailable` but keeps the on-disk vectors; starting re-attaches and rebuilds only if needed.
- **Pooling type:** `--embedding-pooling {mean|last|cls|none}` (config `embedding.pooling`, `SHRIKE_EMBEDDING_POOLING`) is passed to llama-server as `--pooling`. Unset means "use the model's GGUF default" — fine for BERT-family models (`all-MiniLM-L6-v2`, `bge-m3`) that carry mean pooling in metadata. **Last-token models (Jina v5, Qwen3-Embedding) need `--embedding-pooling last`**: their pooling type isn't in the GGUF metadata, so without it llama-server defaults to mean and produces wrong embeddings (and some of these architectures may need a newer llama.cpp than the pinned `LLAMA_TAG` — see `scripts/llama-server.lock`). Pooling is folded into `model_id` (below) so changing it forces an index rebuild.
- **Generic arg passthrough:** `--embedding-arg` (repeatable; config `embedding.extra_args` as a list; `SHRIKE_EMBEDDING_ARGS` as one shlex string) appends raw tokens to the llama-server command for the long tail of **runtime-only** flags (`--flash-attn`, `--ubatch-size`, gpu split, …). Each entry is `shlex`-split at command-build time and appended last. Two guardrails: (1) Shrike-owned flags (`--model`/`-m`/`--host`/`--port`/`--embeddings`/`--embedding`, plus their value token) are stripped with a warning — `--host` especially, since llama-server is pinned to loopback (audit §1.1); (2) the effective passthrough is folded into `model_id`, so **any** change forces a rebuild (conservative — Shrike can't tell a vector-affecting flag from a perf-only one in an opaque bag). **Vector-affecting flags must be typed settings** (like `--embedding-pooling`), not buried here. Normalization is *not* such a setting: USearch's `cos` metric is scale-invariant (verified in `index.py`), so `--embd-normalize` is moot.
- Starting llama-server blocks (model load + health wait), so the HTTP handler runs it via `asyncio.to_thread` to keep the event loop responsive.
- **Orphan reaping:** `EmbeddingService` records the llama-server PID in `<state-dir>/embedding.pid` (written after spawn, removed on clean stop). If Shrike is hard-killed (SIGKILL, incl. the daemon's own force-kill path), llama-server is orphaned and keeps holding its port. On the next `start()`, `_reap_orphan` detects a recorded PID that is still alive *and* holding the port and terminates it (SIGTERM→SIGKILL) before binding. `PR_SET_PDEATHSIG` is intentionally avoided: the parent-death signal keys on the spawning *thread*, and start runs under `asyncio.to_thread`, so a reclaimed pool thread could kill a live server.

### Vector index and consistency

The vector index is a **derived cache**, not a co-equal store. The Anki collection (SQLite) is always the source of truth. SQLite handles its own crash recovery via WAL/journal, so the collection is self-consistent after any crash. If the index is stale, corrupt, or missing, it can be rebuilt from the collection by re-embedding all notes.

**Consistency model:** the index may lag behind the collection (notes added/modified/deleted without the index being updated), but the collection never lags behind the index. This means search results may be stale, but data operations (upsert, delete, list) are always correct.

**Drift detection:** the index metadata (`index.meta.json`) stores `col_mod` — the value of `col.mod` (collection-level modification timestamp, milliseconds) at the time the index was last built — and `model_id`, a fingerprint of the embedding model that produced the vectors. On startup (and whenever the embedding service is attached), compare both against the stored values. If they match, nothing changed — skip reindexing. If `col.mod` differs (something changed outside our control) **or** `model_id` differs (the embedding model changed, so every vector lives in a different space and is invalid), trigger a full rebuild. No watermarks, no per-note diffing, no fragile heuristics. Shrike owns the collection most of the time; anything that changed externally (Anki GUI, sync, imports) should force a clean rebuild for correctness. When sync is implemented (v0.4.0), its implications for index maintenance will be revisited (#38).

**Model fingerprint:** `model_id` comes from llama-server's `GET /v1/models` `meta` block (`n_params`, `n_embd`, `n_vocab`, `n_ctx_train`, `size`) — fast and describes the *loaded* model. It falls back to model filename + on-disk size if that metadata is unavailable. The model *name* is deliberately excluded (it would force needless rebuilds on rename and miss same-name re-quantizations, which the numeric fields catch). An explicitly-set pooling type is appended (`…:pool=last`) because it changes every vector but isn't reflected in the model metadata; it's omitted when unset so indexes built before this setting existed still match. The note-text normalization version (`…:textprep=N`, `EMBED_TEXT_VERSION` in `embed_text.py`) is appended **unconditionally**: the cleaned text we feed the model is as much a part of the vector space as the model itself, so changing how notes are rendered for embedding must invalidate the index (unlike pooling, it's never omitted — an index built under the prior raw-text scheme *should* rebuild). `EmbeddingService.embed()` also pins `"model": <id>` in the request body as insurance against a future external multi-model endpoint.

**Note text for embedding:** `embed_text.normalize_for_embedding()` turns each raw Anki field value into stable plain text before embedding. It operates on field *values*, not rendered cards — a note (not a card) is the embedding unit, a cloze note generates N cards, and templates add presentational scaffolding (`{{FrontSide}}`, `<hr id=answer>`, the hidden `[...]` on a cloze question side) that is noise for search. The HTML→text + entity step **delegates to Anki's own `strip_html`** (Rust-backed, robust on malformed markup, and it leaves an encoded `&lt;tag&gt;` as the literal `<tag>`); around it we do what that stripper doesn't: reveal cloze (`{{c1::France}}` → `France`, hint dropped), drop MathJax/LaTeX wrappers keeping the inner source (`\(…\)`, `$$…$$`, `[latex]…[/latex]`), drop `[sound:…]`, and convert block tags to spaces *before* calling Anki's stripper (which otherwise glues `a<br>b` → `ab`). The module lazily `set_lang("en")`s once so the stripper works headless (locale doesn't affect stripping output). The result is a function of the field value plus the pinned Anki version's stripper — identical whether a note is freshly upserted or re-read during a rebuild, and independent of which card a cloze generates. `CollectionWrapper.note_texts()` applies it per field; both embed call sites (`upsert_notes`, rebuild) route through it, so consistency is structural. Bump `EMBED_TEXT_VERSION` whenever the normalized output changes — including an Anki upgrade whose stripping differs.

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

## Branching, releases & issue tracking

Full conventions live in [`CONTRIBUTING.md`](CONTRIBUTING.md) — this is the
working summary.

- **Trunk-based.** `main` is always releasable and protected; every change goes
  through a `‹type›/‹issue#›-‹slug›` branch → PR → **squash merge**. No direct
  pushes to `main`.
- **SemVer**, `vX.Y.Z` annotated tags. `0.x` may break the public surface (MCP
  schemas, CLI, config) between minor versions. The version is **derived from the
  git tag** by hatch-vcs (no `__version__` constant to bump): the build writes
  `src/shrike/_version.py`, re-exported by `__init__.py`. Just tag to release.
- **Roadmap and tracked work live in GitHub issues + milestones** (one milestone
  per minor version, each with an `epic` tracking issue) — *not* in this file or
  the README, which is how the old prose roadmaps drifted. `gh issue list` /
  `gh issue list --milestone "..."` is the current state of the project.
- **Shipped-design rationale** (the "why" behind decisions like contextual-upsert
  neighbours, duplicate detection, full-replace tags) lives in
  [`docs/decisions.md`](docs/decisions.md).

### Defect workflow — follow this when you find a defect or limitation

When you hit a bug, a limitation, or a missing API surface that is **out of scope
for the task in hand**, do not silently fix it inline and do not leave it as a
prose note. Capture it as resumable state:

1. Open a GitHub issue with a clear problem statement (repro / expected vs actual /
   scope, or the intended API surface that's missing).
2. Create a branch `fix/‹issue#›-‹slug›` (or `feat/…` for a missing capability).
3. Add failing test(s) that exercise the defect / pin the intended API —
   asserting the *desired* behaviour and marked
   `@pytest.mark.xfail(strict=True, reason="#‹n›: …")` so the branch's CI stays
   green while the test is red-by-design, and a future fix that makes it pass
   forces the marker's removal.
4. Push the branch to origin and link it from the issue.

The failing test is the spec; the pushed branch is the handoff. See the full
rationale in `CONTRIBUTING.md`.
