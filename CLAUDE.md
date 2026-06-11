# CLAUDE.md

## What is Shrike?

Shrike manages Anki flashcard collections without running Anki's GUI. It exposes Anki's collection operations through an MCP server and CLI.

**License:** AGPL-3.0

### Architecture

**Since the kernel inversion (#279/#332, June 2026) the compute core is Rust.**
The server is an *assembling harness*: it opens one `AsyncKernel` (the Rust
kernel bound for asyncio) on the event loop, registers services on it (the
embedding backend via `attach_embedder` — the #342 service slot), and serves.
The kernel owns the collection (anki via its protobuf service layer ONLY), the
vector index orchestration (drift, per-note fingerprints, debounced saves),
and the derived-text ingest; write actions route through maintained kernel ops
(`upsert_notes_json`, `delete_notes`, `reindex_notes`, `forget_notes`,
`metadata_changed`). Scheduling is *injected*: the harness thread runs the
kernel's `WorkerExecutor`; the kernel owns no threads and assumes no runtime
(no tokio — the asyncio loop polls kernel futures via the runtime-less bridge).

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (FastMCP, server.py = the host)
                                      │
                                      └──▶ Harness (harness.py: assembly + operational verbs)
                                              │
                                              └──▶ AsyncKernel (Rust shrike-kernel via shrike-py)
                                                      ├──▶ collection.anki2 (anki protobuf services)
                                                      ├──▶ IndexOrchestrator (per-modality USearch HNSW)
                                                      │       └──▶ index.usearch (+ index.image.usearch) + index.meta.json
                                                      ├──▶ DerivedEngine (FTS5 trigram sidecar, shrike.db)
                                                      └──▶ EmbedService slot ◀── EmbeddingRuntime backend (llama/onnx/clip)
```

Embedded hosts skip Python entirely: `native/shrike-cabi` is the manual
C-ABI surface (#333) over the minimal kernel profile (#338, no ort).

## Project layout

```
src/shrike/                       # Python package (src layout) — the harness half
├── __init__.py                   # Re-exports __version__ from generated _version.py (hatch-vcs)
├── server.py                     # MCP host: argparse, FastMCP, routes; main() tail = asyncio.run(_serve())
├── harness.py                    # Harness (kernel-mode core: assembly + verbs) + KernelIndexView
├── collection.py                 # CollectionWrapper — async facade over the shared core
│                                 #   (kernel mode via over_kernel; standalone mode for tests)
├── note_types.py                 # Note-type op shims over the native core
├── daemon.py                     # Daemon lifecycle — file locks, spawn, shutdown
├── actions.py                    # The action registry (24 actions) — kernel-mode write paths
├── tools.py                      # Binds the registry to MCP; returns response models (outputSchema)
├── mcp_adapter.py                # _safe_tool policy + per-call INFO logging
├── schemas.py                    # Pydantic wire models (the BINDING; shrike-schemas in Rust is canonical)
├── client.py                     # ShrikeClient — standalone HTTP client; typed per-tool methods
├── paths.py                      # Platform-canonical directories (via platformdirs)
├── log.py                        # Logging config, log parsing and styling
├── embedding.py                  # EmbedderBackend facades + EmbeddingRuntime (backend lifecycle)
├── recognition.py                # RecognizerBackend protocol + AppleVisionBackend (#228 OCR; `vision` extra)
├── index.py                      # VectorIndex/IndexSaver — STANDALONE/TEST contexts only since the
│                                 #   harness rebase (#353); the server's index lives in the kernel (#355)
├── derived.py                    # DerivedTextStore — FTS5 facade (read paths; kernel ingests in server mode)
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
    ├── media_cmd.py              # shrike media store/fetch/list/delete
    ├── type_cmd.py               # shrike type list/show/create/update/delete
    └── output.py                 # Rich formatting, output_options decorator
tests/
├── unit/                         # direct calls, no server
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
│   ├── test_index.py             # VectorIndex unit tests (standalone facade; see #355)
│   ├── test_derived.py           # DerivedTextStore unit tests (real FTS5: build/substring/fuzzy/seam/drift)
│   ├── test_note_embedding_text.py  # CollectionWrapper.note_texts_for_embedding
│   ├── test_tools_search.py     # search_notes, upsert neighbors, delete index updates
│   ├── test_server_security.py  # loopback guard + transport-security helpers
│   ├── test_daemon.py           # stop_server HTTP→SIGTERM→SIGKILL escalation
│   ├── test_collection_concurrency.py  # single-worker-thread serialization
│   └── test_media.py            # media store/fetch/list/delete, SSRF guard, prune unused_media (mocked URL fetch)
├── native/                       # the native extension + kernel bindings (asyncio bridge,
│                                 #   AsyncKernel, KernelIndex, Harness assembly, C-ABI parity)
└── integration/                  # real server subprocess + HTTP transport
    ├── conftest.py               # shared session server + per-test collection reset; mcp/runner; isolated_server
    ├── test_tools.py
    ├── test_cli.py
    ├── test_media.py             # media tools + `shrike media`/`collection check` (isolated; media isn't reset-tracked)
    ├── test_security.py          # custom-route Host/Origin guard + non-loopback refusal
    ├── test_embedding.py         # Embedding tests + orphan reaping (requires llama-server + GGUF model)
    └── test_semantic.py          # Semantic search, neighbors, index CLI (requires llama-server)
native/                           # the Rust workspace (the compute core)
├── shrike-kernel/                # THE kernel: collection + index orchestration + derived + fusion
├── shrike-collection/            # anki via its protobuf service layer (the ONLY anki coupling)
├── shrike-index/                 # per-modality USearch engine
├── shrike-derived/               # FTS5 trigram engine
├── shrike-embed/                 # ort/tokenizers text + CLIP encoders (feature-gated, #338)
├── shrike-compute/               # rrf_fuse + fused embed→index paths
├── shrike-schemas/               # serde+schemars wire types (CANONICAL; schemas.py binds)
├── shrike-ffi/                   # the shared error taxonomy
├── shrike-py/                    # the pyo3 binding (the ONLY pyo3 crate) + shrike_native package
└── shrike-cabi/                  # the manual C-ABI surface (#333) + embedded C smoke host
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

#### Native (Rust) workspace

The Rust workspace lives in `native/` (run `cargo` from there); the Python
extension is rebuilt into the venv with `scripts/build-native.sh` (the fast
pip-lane inner loop — run it after any Rust change before pytest, which
otherwise tests the stale extension). **Bazel is NOT on PATH** — use the
committed `./bazel` launcher at the repo root (it bootstraps bazelisk + the
pinned Bazel from `.bazelversion`; same entry point CI uses). The full local
gate for a native change:

```bash
(cd native && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd native && cargo clippy -p shrike-compute --no-default-features --all-targets -- -D warnings)
(cd native && cargo test --workspace)
scripts/build-native.sh && pytest tests/unit tests/native -q
./bazel test //...      # the authoritative CI lane: all crate tests + layering check + py suites
```

#### Coverage

Coverage lives in its own workflow (`.github/workflows/coverage.yml`), **off the
per-PR path and never a CI gate** — it's reported, never enforced. It runs on
**`rc`-labelled PRs** (so a release candidate carries a fresh number — skipped on every
other PR), **3x/week on `main`**, and on demand (`workflow_dispatch`); every run uploads
an `htmlcov` artifact + a job-summary report and never fails on a threshold (`coverage
report --fail-under=0`). The **main/scheduled runs also publish the README badge** (a
shields-endpoint JSON force-pushed to the orphan `badges` branch). The `fail_under` in
`[tool.coverage.report]` is the **target**, and **`scripts/coverage.sh` enforces it
locally** (exits non-zero below it) — that's the proactive self-check. Every PR still
runs the full suite as the correctness gate, just *plain* (no tracer, in `test.yml`).
So **run coverage locally** to keep the number healthy:

```bash
scripts/coverage.sh            # full suite; prints the report, exits non-zero below fail_under
scripts/coverage.sh --html     # also writes a browsable htmlcov/index.html
```

The script enables subprocess coverage and runs the exact CI command. The hook
matters because the integration suite runs the server as a `python -m shrike.server`
subprocess; without it ~all of `server.py`/`daemon.py` reads as uncovered. It's a
`.pth` that only imports coverage when `COVERAGE_PROCESS_START` is set, so it costs
nothing on other interpreter starts (same guard coverage's own auto-`.pth` uses).
By hand it's:

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo 'import os; os.getenv("COVERAGE_PROCESS_START") and __import__("coverage").process_startup()' > "$SITE/coverage_subprocess.pth"
export COVERAGE_PROCESS_START="$PWD/pyproject.toml"
coverage run --parallel-mode -m pytest tests/unit tests/integration -q -m "not embedding" -n auto
coverage combine && coverage report      # exits non-zero below fail_under
```

Both suites run in one combined `-n auto` invocation (xdist balances them, so
workers don't idle between phases — faster than two separate runs); `-m "not
embedding"` drops the embedding-gated tests. `coverage.yml` and `scripts/coverage.sh`
run this identical command, so the numbers are comparable.

A plain `pytest tests/unit --cov=shrike` reads ~18 points lower because it can't
see the server subprocess — use `scripts/coverage.sh` when checking the number.

**`-n auto`** (pytest-xdist) parallelizes the suite across cores — the integration
suite is server-spawn-bound and roughly halves (each server gets its own free
port + temp state/log/cache dirs, so workers don't collide). It composes with the
coverage hook above (the `.pth` fires for each xdist worker *and* each spawned
server, so `coverage combine` merges everything to the same total). CI runs both
suites with `-n auto`. Locally it's opt-in — the default (no `-n`) stays serial so
`-x`, `-s`, and `pdb` keep working for debugging.

**Integration tests share one server, with a per-test reset.** Spawning a server
subprocess per test class dominated the suite (each boots `anki` under coverage),
so all non-embedding integration tests share a single session-scoped `server`
(one boot per xdist worker), and an autouse fixture (`_reset_shared_collection`)
resets the collection to its pristine baseline after each test. The `mcp`/`runner`
test clients record what a test mutated (a `_ResetTracker`), so the reset is
cheap: a read-only test resets to nothing, created notes are deleted by tracked
id (never re-listed), and one `collection_info` catches auto-created decks plus
any untracked note (the safety net — a tracking gap costs an extra enumeration,
never leaked state). So a test always starts clean, and even collection-wide
assertions (`total_notes == 0`) hold regardless of run order.
**When writing an integration test you don't need to clean up after yourself** —
just don't assume the collection is empty mid-suite without the reset, and prefer
asserting on your own deck/tag. Two affordances: `scoped_collection(url)` (a
context manager that snapshots and unrolls a sub-section explicitly) and
`isolated_server` / `isolated_mcp` / `isolated_runner` (opt-in fixtures that spawn
a *dedicated* collection for the rare test that needs an exclusive, un-reset one —
e.g. asserting on collection-wide tag counts, which the reset can't restore since
Anki keeps the tag registry). Embedding tests use their own `embedding_server` /
`collection_server` and are untouched by the reset.

### Linting

```bash
ruff check src/shrike/             # Lint
ruff format --check src/shrike/    # Format check
mypy src/shrike/                   # Type check
```

All three must pass cleanly. CI (`.github/workflows/test.yml`) is **gated on the `ci` label** — to spare the free-tier Actions budget, none of the test lanes run on a PR until it carries `ci` (or `rc`, which implies full CI for a release candidate), so a PR that churns through code review doesn't re-run the whole suite on every push. Add the `ci` label once review is complete to run CI on the final state. The gate lives on the `changes` job's `if` (the root of the dependency graph — every test lane `needs: changes`, so skipping `changes` cascades and not a single test runner spins up); `ci-ok` (the single required status check) **fails** with an actionable message until the label is present, so CI is *deferred, not skipped* — it must still pass before merge. The trigger types include `labeled`/`unlabeled` so toggling the label re-evaluates. When the label is present (Linux x64 only): a `lint` job, a `test` job (unit + non-embedding integration, run **plain** — no coverage tracer, for speed), and an `embedding` job. **Coverage is reported but never gated**, and lives in its own workflow (`.github/workflows/coverage.yml`, off the per-PR path entirely): it runs on `rc`-labelled PRs so a release candidate carries a fresh number, *and* 3x/week on `main` + on demand (`workflow_dispatch`) — each uploads an `htmlcov` artifact + a job-summary report and never fails on a threshold, and the main/scheduled runs publish the README coverage badge to the orphan `badges` branch. It is deliberately separate from `ci-ok` (it must not block a merge). The `fail_under` target is enforced only locally by `scripts/coverage.sh` — see the Coverage section above. macOS and ARM run the full integration suite via the `cross-platform` job, gated on `contains(github.event.pull_request.labels.*.name, 'rc')` — i.e. **only** on a PR labelled `rc` (release candidate), never on plain PRs and never on merge to `main`. Actions minutes are limited and macOS bills at 10×, so these lanes stay off the normal iterate-and-merge loop entirely; apply the `rc` label before tagging a release to get cross-platform coverage first. The PR trigger lists `labeled` in its `types` so adding the label re-triggers CI.

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

**Cooperative locking (#64, opt-in).** By default the daemon holds that exclusive lock for its whole life, so you can't launch Anki desktop against the same collection while it runs. `--cooperative-lock` (config `server.cooperative_lock`, env `SHRIKE_COOPERATIVE_LOCK`) makes `CollectionWrapper` open on demand and **release the collection after a short idle window** (`--lock-hold-seconds`, config `server.lock_hold_seconds`, env `SHRIKE_LOCK_HOLD_SECONDS`, default `DEFAULT_LOCK_HOLD` = 5 s), so an *idle* daemon no longer blocks launching Anki. It's cooperative *time-slicing*, not concurrent sharing (Anki desktop never releases mid-session). Mechanics: `self._open_flag` tracks held-vs-released; every op routes through `_locked` on the worker thread (re-open if released, then run the op); `run` (re)arms an idle-release timer (`loop.call_later`, mirroring `IndexSaver`); `_release` closes on the worker thread, guarded so a close racing a re-acquire is a no-op. On each **re-acquire** an `on_acquire` hook (set by the server) runs a cheap `index.check_drift(col.mod)` and, only on real drift (an external edit during the idle gap), reads note texts under the lock and rebuilds off-lock — reusing the boot drift machinery. The boot path opens, builds the index, then `release_now()` so a never-touched idle daemon doesn't hold the lock. Default (permanent) mode leaves `_open_flag` permanently True, so all of this is inert and behaviour is unchanged. `server.lock` (daemon liveness) is a **separate** lock from the collection lock — `server status` / `/status` report both (`locking`, `collection_held`). The contention failure mode is a clean SQLite "database is locked" busy error, not corruption. #79 shipped the underlying reopen + read-`self.col`-at-execution-time primitive.

**Busy-acquire surface (#65).** When a re-acquire can't open the collection (another process — usually Anki desktop — holds it), `_locked` catches Anki's `DBError` and raises `CollectionBusyError` **immediately** (no retry — the caller decides). It's modelled as a typed error, *not* a per-tool response variant: busy is orthogonal to every tool's response (the op never ran), so adding a `CollectionBusy` variant to all 18 schemas would be wrong. Instead it rides the existing two-layer error split (server `ToolInputError` ↔ client `ServerError`): the server-side `CollectionBusyError` (in `collection.py`) carries a message prefixed with the `COLLECTION_BUSY_CODE` sentinel (`"collection_busy"`, the single source of truth in `schemas.py`); `_safe_tool` logs it at WARNING (no traceback, like `ToolInputError`) and re-raises so FastMCP emits an `isError` carrying that text; `ShrikeClient._call` detects the prefix and raises the **client-side** `CollectionBusyError(ShrikeError)` (a sibling of `ServerError`, so callers catch-and-retry instead of parsing a string); the CLI's root `ShrikeError`→`ClickException` handler renders the human message ("the collection is in use by another process…") with no stack trace. This surface is cooperative-only — permanent mode never re-opens.

### MCP transport

The server uses FastMCP with streamable HTTP transport (`stateless_http=True`, `json_response=True`). It listens on `http://127.0.0.1:8372/mcp` by default. All communication is JSON-RPC 2.0: clients POST to the endpoint with `method: "tools/call"` and receive structured JSON responses.

**Trust boundary.** Every endpoint is unauthenticated, so the server binds loopback by default; binding a non-loopback host requires `--allow-remote` (it refuses to start otherwise, with a loud warning) and llama-server stays pinned to `127.0.0.1` regardless. DNS-rebinding/CSRF protection (`_build_transport_security()`) validates `Host`/`Origin`, applied to the MCP endpoint *and* — via the `_guard` wrapper in `_register_custom_routes` — to the custom routes (`/status`, `/media/{name}`, `/shutdown`, `/index/rebuild`, `/index/save`, `/embedding/*`, `/reload`), which bypass MCP middleware (every route's guard is asserted in `tests/integration/test_security.py`, #166). The guard is **independent of the bind address**: a loopback bind allow-lists loopback `Host`/`Origin`; `--allowed-host`/`--allowed-origin` (config `server.allowed_hosts`/`allowed_origins`, env `SHRIKE_ALLOWED_HOSTS`/`SHRIKE_ALLOWED_ORIGINS`, comma-separated) *add* trusted values for a reverse-proxy or VPN hostname — note a proxy forwards `Host: name:port`, so use the SDK's `name:*` port-wildcard form; and `--no-dns-rebinding-protection` (config `server.no_dns_rebinding_protection`) turns the guard off entirely for deployments where the network is the trust boundary (behind Caddy, on a tailnet, firewalled). A non-loopback bind with no explicit allow-list also leaves the guard off (preserves the original `--allow-remote` behaviour). A non-loopback bind given *only* `--allowed-origin` (no `--allowed-host`) builds a guard whose Host allow-list is empty → **every** request is rejected 421 (fail-closed, a config footgun, not a hole); `_build_transport_security` logs a startup warning for that case. In every mode the endpoints stay unauthenticated — the guard is anti-CSRF/DNS-rebinding, not authentication.

**OAuth is required for native connectors (deferred; tracked here).** Claude Desktop / claude.ai *URL connectors* require OAuth 2.1 + Dynamic Client Registration: they try to register against the MCP server's sign-in service and fail against an unauthenticated endpoint, regardless of TLS or network exposure. So the "run Shrike behind a reverse proxy / on a VPN and add it as a connector" story depends on implementing MCP server auth (`mcp.server.auth`, OAuth 2.0 + PKCE — see audit §1.1); it's a v0.4/v0.6-class project, intentionally not started. Until then the unauthenticated + network-boundary model serves CLI / programmatic / `mcp-remote` clients: a native client reaches Shrike through the **`mcp-remote` stdio bridge** (`npx mcp-remote http://127.0.0.1:8372/mcp --allow-http --transport http-only`), which connects without auth because Shrike demands none. This is how the QA harness drives Claude Desktop.

### MCP tools (24 total)

| Tool | Status | Purpose |
|------|--------|---------|
| `collection_info` | Working | Collection structure, note types, decks, tags, stats |
| `list_notes` | Working | Filter/retrieve notes by deck, tags, type, IDs, date |
| `search_notes` | Working | Per-query **per-modality** semantic similarity (text + image) **and** exact-substring search, RRF-fused into annotated results (`score`?/`substring`?) |
| `collection_query` | Working | Raw Anki search expression (`is:due`, `prop:`, …) → notes; same shape as `list_notes` |
| `upsert_notes` | Working | Create or update notes in bulk (1-100); `on_duplicate` policy + `dry_run`; returns similar neighbors |
| `upsert_note_types` | Working | Create or update note type definitions (1-10) |
| `update_note_type_fields` | Working | Edit a note type's fields by name: add/remove/rename/reposition (data-safe) |
| `update_note_type_templates` | Working | Edit a note type's card templates by name: add/remove/rename/reposition (data-safe) |
| `find_replace_note_types` | Working | Find/replace text in one note type's template HTML + CSS (front/back/css selectors); returns a count |
| `update_note_type_field_metadata` | Working | Set a note type's per-field editor metadata (font/size/description); metadata-only, col_mod bump |
| `update_note_tags` | Working | Edit tags on a note set (1-1000): `set` (replace) XOR `add`/`remove` |
| `rename_tag` | Working | Rename a tag collection-wide or on a note set (exact match) |
| `find_replace_notes` | Working | Bulk find/replace across fields in a scoped set; literal or regex; `dry_run` preview |
| `migrate_note_type` | Working | Change notes' note type via field/template name map; reports drops; `dry_run` preview |
| `upsert_decks` | Working | Create or rename/reparent decks in bulk (1-100); id = rename |
| `delete_decks` | Working | Delete decks by name, only if empty (else reported `not_empty`) |
| `delete_notes` | Working | Permanently delete notes by ID |
| `delete_note_types` | Working | Delete note types by ID (only if unused) |
| `collection_prune` | Working | Cleanup: unused tags + empty notes + empty cards + unused media; `dry_run` default **true** (preview) |
| `collection_check` | Working | Read-only media diagnostics: unused/missing media, missing-media notes, trash state |
| `store_media` | Working | Store media (1-10) from base64 `data`, `url` (SSRF-guarded), or server-local `path` (off by default; opt-in repeatable `--media-path-root` on a purely-local daemon); per-item results; dedup/collision via Anki |
| `fetch_media` | Working | Locate media (1-10); per-item `found`(url+path+mime+size)/`missing` union — **never returns bytes** (GET the `url`) |
| `list_media` | Working | List media filenames (+ `media_dir`), optional glob `pattern`; each with `url`/`mime`/`size_bytes` |
| `delete_media` | Working | Delete media by name → Anki's recoverable trash (no ref-check); `deleted`/`not_found` |

**Duplicate handling lives *inside* `upsert_notes`, not in a separate pre-check (#77).** Before each new note is written, Anki's own add-note validation (`note.fields_check()`, via `CollectionWrapper._check_new_note`) runs. A first-field duplicate (same first field as an existing note of that type — Anki's rule, collection-wide and **deck-independent**) is governed by the `on_duplicate` param: `error` (**default** — reported, not written), `skip` (`status: skipped`), or `allow` (written anyway). Structurally invalid notes — empty first field, broken cloze — are *always* reported as errors with a `reason` regardless of policy, and never written. `dry_run: true` runs the exact same validation but writes nothing: every result is `ok` (with `action: create|update`), `skipped`, or `error`, and the response echoes `dry_run: true` — so `dry_run` + the default policy is a full `fields_check`-based sanity pass over a batch. The per-item result union (`UpsertNoteResult` in `schemas.py`) carries the four variants (`UpsertNoteOk` / `UpsertNoteValidated` / `UpsertNoteSkipped` / `UpsertNoteError`, discriminated on `status`); `UpsertNoteError.reason` is the machine-readable `NoteValidationReason`. This is Anki's *exact* first-field rule, distinct from the *semantic* near-duplicate signal the returned `neighbors` provide. A standalone `canAddNotes`-style tool was rejected: it would be racy (check-then-write) and only actionable by a follow-up call (see `docs/decisions.md`). Note the validation applies to **creates**; updates are validated for existence/fields but not duplicate/empty. `dry_run` does not catch intra-batch duplicates (it writes nothing, so two identical new notes in one call both validate clean — a real run catches the second).

**Note-type field/template edits come in two flavours, both data-safe (#76).** `upsert_note_types` replaces the whole `fields`/`templates` list **by position** (the field/template at each position keeps its note data/cards even when renamed; only shortening drops the tail) — fixed in #99 after the original rebuild-from-scratch blanked note data and deleted cards. `update_note_type_fields` and `update_note_type_templates` are the **by-identity** counterparts: a sequence of `add`/`remove`/`rename`/`reposition` ops addressed by field/template *name*, so they can express a true move, an insert, or a non-trailing remove that position-replace can't. They delegate to Anki's own data-safe primitives (`rename_field`/`reposition_field`/`add_field`/`remove_field` and the `*_template` equivalents), which migrate note data / cards by identity — so a reposition is a true move and a non-trailing remove drops only that field's data / that template's cards. (A template `rename` is a pure label change: cards key by ordinal, not name.) Each call is **atomic** (the op sequence is validated against a simulated name list before any primitive runs — an invalid op changes nothing) and persists with a single `update_dict`. Like `upsert_note_types`, they do no inline index maintenance — the `col.mod` bump from `update_dict` triggers a drift-rebuild on next startup (a removed field changes embedding text, so a rebuild is correct; for the rest it's conservative). All live in `note_types.py`, sharing `_simulate_struct_op` and the soundness check; bad ops raise `NoteTypeOpError` → `ToolInputError` (logged without a traceback). The position vs identity tools are **reconciled** so they can't overlap dangerously: the positional `fields`/`templates` replace in `upsert_note_types` *rejects* any update where an existing name moves to a different position (a reorder/insert/non-trailing-remove — which would silently re-label note data / cards), erroring with a pointer to the matching identity tool (`_reject_unsound_positional_replace`). It keeps only the unambiguous positional edits: rename/edit-in-place, append, trailing-remove. A third tool, `find_replace_note_types` (anki-connect's `findAndReplaceInModels`), does a literal-or-regex text rewrite over one model's template `qfmt`/`afmt` and shared `css` (selected by `front`/`back`/`css`), returning a replacement count — it touches *template text*, not fields or note data, so it migrates nothing and (unlike the field/template structure ops) bumps `col_mod` without a re-embed. Literal mode escapes the pattern and inserts the replacement verbatim (no `\1` interpretation); `match_case` defaults **true** (template/CSS is code). It lives in `note_types.py` (`find_and_replace_note_types` + `_subn_text`) and raises `NoteTypeOpError` → `ToolInputError` for an unknown model or a bad regex.

The tag tools (#73), deck tools (#74), `find_replace_note_types` (#76), **and** `update_note_type_field_metadata` (#119) are a derived-index-aware special case: tags, deck names, a note type's template/CSS text, and per-field editor metadata (font/size/description) are **not** part of a note's embedding text, so these ops leave every vector valid but bump `col.mod`. Each advances the stored `index.col_mod` (and requests a debounced save) **without** re-embedding — via the shared `_bump_col_mod_after_metadata_change` helper in `tools.py` — so such a metadata-only change doesn't force a spurious full rebuild on next startup. Full-replace of tags lives in both `update_note_tags` (`set`) and incidentally in `upsert_notes` (`{id, tags}`); the additive/subtractive logic lives only in `update_note_tags`. `upsert_decks` mirrors `upsert_notes` (id present = rename the existing deck; absent = create); **decks never merge** — renaming onto an existing deck name is an error, and `delete_decks` is **empty-only** (move notes out first), so deck deletion can never delete a note.

**Collection maintenance is one tool, `collection_prune` (#89), not scattered cleanups.** It runs any of four cleanups — clear **unused tags** (`tags.clear_unused_tags`), remove **empty notes**, remove **empty cards** (`get_empty_cards` → `remove_cards_and_orphaned_notes`), trash **unused media** (`col.media.check().unused` → `trash_files`, #70); selecting none runs all. It is **preview-by-default**: `dry_run` defaults **true** (unlike `find_replace_notes`, which applies by default — prune is collection-wide and deletes notes/cards, so it errs the safe way; the CLI `shrike collection prune` previews unless `--apply`). The earlier standalone `clear_unused_tags`/`shrike tag clean` (#73) was removed (#90) to land here, and this supersedes a standalone remove-empty-notes (#78). An **empty note** is one whose every field is blank by `embed_text.field_is_blank` — no text *and* no media (`<img|audio|video|object|embed|source>`/`[sound:]`), so an image- or audio-only note is never deleted. Index handling is **mixed**, which is why this isn't a pure metadata-bump op like the tag/deck ones: empty-note and empty-card removal **delete notes**, so their vectors leave the index via `index.remove` exactly like `delete_notes`; clearing unused tags and trashing unused media are vectors-unchanged (media isn't embedding text). The tool does both in one pass (`index.remove(removed_note_ids)` then advance `col_mod` + save) when anything changed. On apply the order is empty notes → empty cards → unused tags → unused media, so tags and media freed by the deletions get cleared in the same call; the dry-run previews each cleanup independently (so an apply may clear a few more than the preview showed). Logic lives in `CollectionWrapper._prune`/`_find_empty_notes`/`_find_unused_media`; the CLI `collection` group (`cli/collection_cmd.py`) also houses `collection query`, `collection check`, and `collection reload`. The read-only sibling **`collection_check`** (#70) wraps `col.media.check()` to report `unused`/`missing`/`missing_media_notes`/`have_trash` without mutating — the place to inspect media issues (and the home for future bookkeeping checks).

**Media files are first-class, via five tools (#70), and orthogonal to the vector index.** `store_media`/`fetch_media`/`list_media`/`delete_media` wrap `col.media` (`write_data`/`add_file`, the media-dir read, `trash_files`); media ops **never touch `col.mod` or embedding text** (verified), so unlike the tag/deck metadata ops they need *no* `index.col_mod` bump at all — the index is simply unconcerned. The authoring write path is `store_media`: a bulk (1-10) per-item batch where each item is base64 `data` (requires a `filename` with extension — the type carrier, since bytes are opaque) **or** a `url` the server fetches; one item failing (bad base64, blocked URL, oversize) doesn't sink the batch (per-item `StoreMediaOk`/`StoreMediaError` union). A server-local **`path`** source (#164/#170) stores a file on the server's own disk zero-copy via `col.media.add_file`, and is **off by default**. It's honored only when **all three** hold: the operator set one or more **`--media-path-root DIR`** (repeatable; config `server.media_path_roots` list, env `SHRIKE_MEDIA_PATH_ROOTS` `os.pathsep`-separated); the server is **purely-local** (`_server_is_purely_local`: loopback bind, no `--allow-remote`, guard on, no added `--allowed-host`/`--allowed-origin`); and the path is **contained in one of those roots** after `..`/symlink resolution. The two gates **compose** — purely-local stops a remote/proxied caller from *reaching* it, the roots bound *what* a permitted caller may read — so roots set on a non-purely-local server are refused (warn), not half-enabled (relaxing purely-local waits on server auth / the OAuth track). `main()` validates **each** root once at startup (`_validate_media_path_root`: realpath, reject the filesystem root via `dirname(p)==p`, require an existing dir — *per element*, since the disjunction means the weakest root governs and one `/` re-opens everything), dedups/canonicalizes, and passes the list into `register_tools` only when purely-local; the containment check is a disjunction of `os.path.commonpath([root, realpath(src)]) == root` over the roots (`_path_within_any_root` in `collection.py` — `commonpath` not `startswith`, on `realpath`'d sides, so `..`, symlink-escape — `add_file` follows symlinks — and the `/srv/media-evil` prefix bug are all closed; a check→`add_file` TOCTOU is an accepted local-trust residual). Anything else → a per-item error (off-by-default, not a request rejection). It's a **process-level capability gate, no per-request peer check** (redundant under the gate, and would couple to MCP-SDK internals). The CLI offers both `shrike media store PATH` (reads + uploads bytes, works against any daemon) and `--server-path` (server reads it, requires a configured root). Even so, within a root `path` is an **arbitrary read of those files at the server user's privileges** (store then `fetch_media`/`GET /media`) — a deliberate, documented part of the unauthenticated-loopback trust model, intended for single-user/local use; the explicit narrow roots keep the blast radius small on a shared host. Anki resolves collisions (identical content → same name, reported `deduped`; different content → hashed suffix), so callers must use the *returned* filename. **URL fetch is SSRF-guarded** (`_fetch_media_url` in `collection.py`): http/https only; the host is resolved and refused unless every address is **globally routable** (`ipaddress.is_global` — an *allowlist*, so it also rejects carrier-grade NAT 100.64/10, `192.0.0.0/24`, benchmarking ranges a denylist misses, plus loopback/RFC1918/link-local/metadata/reserved/multicast) — unless `--allow-private-media-fetch` / `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH` / config `server.media_allow_private_fetch` (threaded through `register_tools(..., allow_private_fetch=)` and `ServerSpec`). **Redirects are followed manually, re-running the guard on every hop's host** (capped at `MAX_MEDIA_REDIRECTS`), because httpx's `follow_redirects` would jump to an attacker-chosen private/metadata address unchecked. **The connection is pinned to the vetted IP (#165):** `_resolve_public_ip` returns one validated address, the request URL's host is that IP (so httpx connects there and never re-resolves the name at connect time — closing the DNS-rebinding TOCTOU), the `Host` header carries the original name for routing, and for HTTPS the `sni_hostname` request extension carries it too so TLS SNI + cert validation verify against the name, not the IP. Each item is prepared **concurrently and off the worker thread** (`asyncio.gather` over `to_thread`: URL downloads overlap, base64 decodes off the event loop with the size cap applied to the *encoded* length before decoding); honors httpx proxy env (SOCKS via the optional `httpx[socks]` extra); capped at `MEDIA_MAX_BYTES`. (`--allow-private-media-fetch` turns off both the guard and the pinning, connecting to the URL as given.) `fetch_media` **never returns bytes** — base64 in a tool response is useless to a model (it can't render it) and wrecks context. Each present file returns as `MediaFile` (`status: "found"`) carrying a **`url`** (the server's `GET /media/<name>`) + server-side `path`; the model GETs the url with its own download/fetch tool (no base64) or reads the path if co-located, and a missing file is `MediaMissing`. There is **no inline/base64 option** — the only way to bytes is the url (or path). The `url` is built in the tool layer (`_media_url`, from a `media_base_url` threaded into `register_tools`; `list_media` carries it too). `media_base_url` defaults to the bind host but is overridable for reverse-proxy deployments via `--public-url` / `SHRIKE_PUBLIC_URL` / config `server.public_url` (the bind host isn't externally reachable behind a proxy). The **`GET /media/{filename}`** custom route (in `_register_custom_routes`, behind the same `_guard` Host/Origin check, `FileResponse`) serves the bytes and resolves the media dir **lock-free and absolutized** via `CollectionWrapper.media_dir` (`media_paths_from_col_path`, no CollectionBusyError; abspath'd so a relative `--collection` doesn't make the route cwd-dependent). For programmatic byte access the standalone client has `ShrikeClient.read_media(name) -> bytes` (GETs that route). `list_media` defaults to 100 files. fetch/delete and the route sanitize filenames to a basename inside the media dir (`_safe_media_name`, traversal guard). The future multimodal-embedding epic (search media by content, #162) builds *on top of* this store path. Logic lives in `CollectionWrapper.store_media/fetch_media/list_media/delete_media/media_check`; CLI in `cli/media_cmd.py` (which downloads a found file's `url` or copies its `path`).

**Three retrieval surfaces, by intent.** `list_notes` is structured filters (deck/tags/type/ids/date, ANDed); `search_notes` is meaning + exact text (semantic + substring, annotated), **fused by Reciprocal Rank Fusion** (#180 — `search_fusion.rrf_fuse`: each signal ranks its own candidates, a note's fused score is `Σ w_s·1/(k+rank_s)`, a missing signal contributes nothing → graceful degradation, ordering stable across queries; an **exact-match override** tiers literal hits above the rest, the one place RRF's blindness to magnitude is wrong). The **per-modality semantic rankers are live (#201a):** `index.search_by_modality` ranks notes separately per modality, so `text` and `image` enter the fusion as **distinct RRF signals** alongside `exact` — a rank-based combiner makes the CLIP modality gap's constant cosine offset invisible, so a text query that matches a card's *image* now surfaces it (the payoff a single deduped cosine ranking couldn't deliver). The `image` ranking is **not** subject to the text-calibrated cosine `threshold` (meaningless across the gap); instead the offline-calibrated intra-modal **activation gate (#201b)** floors it — a non-text modality joins the fusion only when its best match for the query clears `mean + ACTIVATION_MARGIN·std` of that modality's *typical* best match, estimated by sampling stored text vectors as pseudo-queries against each modality (`index._calibrate_activation`, self-matches excluded; stored in `index.meta.json` under `activation`, surfaced in `server status`). So an off-topic query no longer injects weak image cards; text-only collections are unaffected (no image sub-index → nothing to calibrate or gate). The same signal-agnostic seam is what the **`fuzzy` signal (#98) plugs into** by just "producing a ranking" (tag-centroid #179 is the next to follow). The `fuzzy` signal is the **derived-text store's** trigram/typo ranking (`DerivedTextStore.search_fuzzy` — see the derived-store section below): it joins the fusion alongside `text`/`image`/`exact`, weighted **below** them (`SEARCH_WEIGHTS["fuzzy"]=0.5` — a near-miss is weaker evidence than a literal or semantic hit), and surfaces near-misses an exact substring search misses (`protien` → `protein`). The **same store also feeds the substring (`exact`) candidates** now — `search_notes` prefers `DerivedTextStore.search_substring` (a fast FTS5 trigram pre-filter) over the linear `find_notes` scan, falling back to `find_notes` when the store is unavailable/unbuilt or the query is sub-trigram (`<3` chars); either way `substring_info` stays the **authority** that confirms + annotates each candidate, so the swap is behaviour-preserving and the exact tier is unchanged. Both lexical hits carry **source-aware provenance** (`SubstringInfo.source`/`ref`, the new `FuzzyMatch.source`/`ref`/`snippet`), today always `source="field"` but seamed for #199's `ocr`/`asr`. `collection_query` (#97) is the **raw Anki escape hatch** — the string goes straight to `col.find_notes`, so the full expression language works (`is:due`, `prop:ivl>=30`, `added:`, `rated:`, `flag:`, `nid:`/`cid:`, `OR`/`-`/brackets). It exists because #86 removed `note list --query` (the leaky raw param) in favour of an explicit tool. It is **read-only** — it only *finds* notes; `is:due`/`rated:` filters return matching notes but perform no review or scheduling, so the full grammar is allowed without a whitelist to police. It reuses `list_notes`' serialization (`_note_to_dict`) and `ListNotesResponse`; a malformed expression raises `anki.errors.SearchError`, which the tool maps to `ToolInputError` (stripping Anki's U+2068/U+2069 isolation marks). Lives in `CollectionWrapper._query`.

`find_replace_notes` (#85) is the opposite case: it edits **field bodies**, which *are* embedding text, so it re-embeds the changed notes via the `upsert_notes` index path (not the col_mod-only helper). The actual edit runs Anki's `col.find_and_replace` (Rust regex, undo-able); the changed set is detected by diffing `notes.flds` before/after (note `mod` is only second-resolution, so a same-second edit shows no bump) and exactly those are re-embedded. The dry-run preview is computed in Python (`apply_replacement`): exact for literal, illustrative for regex. A scope (`deck`/`tags`/`note_type`/`ids`) is required.

**Changing a note's note type is `migrate_note_type` (#75), a dedicated tool — `upsert_notes` still hard-refuses a type change.** It wraps Anki's `col.models.change(source, nids, target, fmap, cmap)`: a history-safe migration (note IDs preserved; card scheduling carried across mapped templates) for Basic↔Cloze conversions, consolidating redundant note types, or adopting a richer template. The user-facing `field_map`/`template_map` are by **name**; `_migrate_note_type` (in `collection.py`) translates them to the ordinal `fmap`/`cmap` Anki wants (every source field ord → target ord or `None`=drop). It's **data-affecting and explicit**: `field_map` is required and non-empty, a source field not in it is *dropped* (reported in `dropped_fields`, content lost), target fields nothing maps into are reported (`new_empty_fields`), and unknown names / two-sources-to-one-target / mixed source types / target==source all raise `ValueError` → `ToolInputError` (no guessing). All `note_ids` must currently share one source type. Like `find_replace_notes` it edits embedding text (the fields move), so on apply it re-embeds the changed notes via the `upsert_notes` index path (IDs are unchanged). Applies by default with a `dry_run` preview; the CLI `note migrate-type` previews the drops, confirms, then applies.

Every tool request and response shape — plus the server-status shapes — is a Pydantic model in `shrike/schemas.py` (the single source of truth). Tool functions in `tools.py` return the response models, so FastMCP emits an `outputSchema` for each tool, and `_safe_tool` runs each docstring through `inspect.cleandoc` so the advertised descriptions carry no source indentation. The standalone `ShrikeClient` exposes a typed per-tool method for each (e.g. `list_notes(...) -> ListNotesResponse`) that validates the wire response into the model; `ShrikeClient._call()` is the untyped escape hatch. There is no checked-in schema file: the authoritative machine schema is whatever the running server advertises via `tools/list`, derived from these models. `docs/mcp-tools.md` is the human-readable companion.

**Make illegal states unrepresentable — the schemas.py house style.** When a field's presence is *correlated* with another (a hidden state), model it as a **discriminated union**, never a bag of optionals. The pattern is an `Annotated` type alias: each variant is a `BaseModel` with a `Literal` discriminator field, and `Thing = Annotated[A | B, Field(discriminator="status")]`. Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field* typed as `Thing` validates automatically). Examples: per-item results (`UpsertNoteResult = UpsertNoteOk | UpsertNoteError` — success has `id`+`neighbors`, error has `index`+`error`), `IndexStatus` (`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored` — `progress` only on building, `error` only on errored), and the `/index/rebuild` + `/embedding/*` endpoint responses (unions on `status`). Two fields that always appear or vanish *as a pair* are the same smell at smaller scale — group them into a nested sub-model (`NoteTypeInfo.detail: NoteTypeDetail | None`), not two optionals. A bare `X | None` is reserved for *genuinely independent* optionality (a datum absent on its own — `col_mod` before the index is built, a field omitted from a partial update, a caller-selected `collection_info` section); annotate why so it reads as deliberate. Response models carry **no `error` field**: a whole-call failure (bad input, unhandled exception) is raised in the tool and surfaces as an MCP `isError` result, which `ShrikeClient._call` turns into a `ServerError`. Expected bad input raises `ToolInputError` (logged without a traceback); genuine bugs log with one. The only optional advisory on a success response is `message` (e.g. index-building notice, neighbor-retry hint).

Input bounds (e.g. `limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as `Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so FastMCP **rejects** out-of-range input rather than silently clamping. Optional list filters use `Field(default_factory=list)` (keyword-only params) so they render as a plain array in the schema, not a noisy `anyOf:[array, null]`.

### CLI structure

The CLI uses Click with rich for output formatting. Command hierarchy:

```
shrike [--config PATH] [--url URL] [--json] [--pretty/--no-pretty]
├── server start|stop|status|logs
├── info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
├── note list|show|create|update|tag|delete|search|replace|migrate-type
├── deck create|rename|delete
├── tag rename
├── type list|show|create|update|delete
├── media store|fetch|list|delete
├── collection query|prune|check|reload
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
- `GET /status` — returns JSON with pid, url, collection, log_level, log_dir, uptime, embedding, index, and the collection-lock state (`locking`: permanent/cooperative, `collection_held`: bool). Used by `shrike server status` and auto-start health checks.
- `GET /media/{filename}` — streams a media file (`FileResponse`, content-type from extension) so `fetch_media`/`list_media` can hand back a `url` instead of base64 (#70). Read-only, behind the `_guard` Host/Origin check; filename basename-sanitized; the media dir is resolved lock-free (`wrapper.media_dir`). 404 for a missing/escaping name.
- `POST /shutdown` — triggers graceful server shutdown.
- `POST /index/rebuild` — triggers a full index rebuild (returns immediately with status/progress). Requires the embedding service to be running.
- `POST /index/save` — forces an immediate flush of the in-memory index to disk (off the event loop). Returns `saved` (with `size` and the `pending` count it flushed), `empty` (no index built yet), or `building` (refused mid-rebuild). Backs `shrike index save`; the index also saves automatically (debounced flush + shutdown).
- `POST /embedding/start` — starts the embedding service (optional JSON body overrides model/port/etc.; falls back to the params the server booted with). Attaches it to the index and triggers a rebuild if the model changed or the index drifted. Returns `started` / `already_running` / a 400 if no model is configured.
- `POST /embedding/stop` — saves the index, then stops the embedding service and marks the index `unavailable`. The server and collection stay up.
- `POST /reload` — closes and re-opens the collection (picks up on-disk changes — a restored backup, a file-level sync/swap) and re-checks index drift, rebuilding in the background if `col.mod` moved. Returns `{status: "reloaded", col_mod, rebuilding}`. Backs `shrike collection reload`. **First slice of cooperative locking (#64):** the reopen primitive (`CollectionWrapper.reopen`/`_do_reopen`) plus reading `self.col` at execution time in `run`/`run_sync` (so an op queued after a reopen sees the new handle) is exactly what #64's open-on-demand lifecycle will reuse. Under today's permanent-hold lock its utility is narrow (the lock blocks most external edits while the daemon runs) — it widens once #64 lands, where the per-acquire drift check makes reload mostly automatic. It's a control endpoint + CLI, not an MCP tool (operational, like the index/embedding routes).

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

YAML at the platform config directory (`config.yml`). **User-managed: `shrike server start` never writes it unless `--save-config` is passed** (#56). With `--save-config`, start persists the resolved flags (the `embedding` section, including the model path, is written there too) so later runs and `shrike embedding start` pick them up; without it, start is a no-write operation and always reflects exactly the flags it was given. (The old behaviour wrote the file once on first run and then silently ignored later flags — the divergence that was the #56 bug.) Resolution order: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`, `SHRIKE_EMBEDDING_BACKEND`, `SHRIKE_EMBEDDING_MODEL`, `SHRIKE_EMBEDDING_PORT`, `SHRIKE_EMBEDDING_POOLING`, `SHRIKE_EMBEDDING_ARGS`, `SHRIKE_EMBEDDING_ONNX_PROVIDERS`, `SHRIKE_EMBEDDING_BATCH_SIZE`, `LLAMA_SERVER_PATH`, `SHRIKE_CACHE_DIR`, `SHRIKE_INDEX_SAVE_DELAY`, `SHRIKE_INDEX_SAVE_THRESHOLD`, `SHRIKE_COOPERATIVE_LOCK`, `SHRIKE_LOCK_HOLD_SECONDS`, `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH`, `SHRIKE_PUBLIC_URL`, `SHRIKE_MEDIA_PATH_ROOTS`) → CLI flags. Embedding params follow the same cascade via `config.resolve_embedding()`, shared by `shrike server start` and `shrike embedding start`; the index cache dir and flush tuning follow it via `config.resolve_cache_dir()` / `config.resolve_index_save()`. `save_config` persists `collection`, `cache_dir`, non-default `server.*`, `embedding.*`, and any set `index.save_delay` / `index.save_threshold`; **logging overrides are read from config but never written by `--save-config`** — set `logging.level` / `logging.dir` in `config.yml` directly. The `index.*` flush knobs and `cache_dir` resolve to `None` in config, meaning "use the server's built-in defaults" (`IndexSaver`'s 60s/100 and the platform cache dir) — the numeric defaults live in `shrike.index`, not duplicated in config.

### Embedding service lifecycle

The embedding service can be cycled independently of the Shrike server. `EmbeddingRuntime` (`embedding.py`) owns the current backend (or `None`), the params needed to (re)start it, and the binding to the index; it serializes start/stop under a lock. The `VectorIndex` is **always** created at boot, even with no embedder — it loads on-disk vectors and reports `unavailable` until a backend is attached.

**Pluggable backends behind one protocol (#172).** The index and server depend only on the minimal `EmbedderBackend` protocol (`embedding_base.py`): `embed_texts`, `embedding_dim`, `model_fingerprint`, `health`, lifecycle, and `modalities`. Three implementations: `LlamaServerBackend` (the original llama-server subprocess; GGUF/MLX), `OnnxBackend` (`embedding_onnx.py`, in-process onnxruntime + tokenizers, text-only), and `ClipBackend` (`embedding_clip.py`, in-process CLIP dual-encoder for **image↔text**, `modalities={text,image}`). `EmbeddingService` is a back-compat alias of `LlamaServerBackend`. The runtime picks one by *kind* via `--embedding-backend {llama|onnx|clip}` (default `llama`; config `embedding.backend`, `SHRIKE_EMBEDDING_BACKEND`); `_make_backend` constructs it, importing the onnx/clip deps lazily so `shrike[onnx]`/`shrike[clip]` stay **optional extras** (a missing dep surfaces as `ImportError` only when that backend is selected — caught on both the `/embedding/start` path and the boot path so it degrades to no-embedding rather than crashing). The drift/hash/persistence machinery in `index.py` is backend-agnostic — only `embed_texts` is called. **Default stays `llama`** so existing setups are unchanged, but the user-facing guidance (README) leads with **ONNX for text-only collections** — it's in-process (no subprocess/port/orphan-reaping), lower-latency for single-note upserts (~1 ms vs llama-server's HTTP round-trip), and small/quantized-friendly; **llama-server** is for GGUF/MLX and GPU offload; the **CLIP backend** (`shrike[clip]`, #162 Phase 3b) is the multimodal path — it embeds text *and* images into one shared space (a dual text+vision ONNX encoder + image preprocessing in PIL/numpy, no torch), so a text query retrieves a card by the content of its image. It advertises `modalities={text,image}` (the seam below); reuses `OnnxBackend`'s provider resolution + the batch-safety probe (one text-path probe governs both graphs, which share the model's quantization); and adds `embed_images(...)`. Verified against `Xenova/clip-vit-base-patch32` (small CI fixture) and proven on `jina-clip-v2` in the Phase-3a eval (#193). The **multi-vector index** that holds a note's text + image vectors (`multi=True`) shipped in **#162 Phase 3c** (below): a note maps to its text vector + one vector per image under the `note_id` key, the collection extracts `<img src>` names (`embed_text.extract_image_refs` → `note_embed_inputs`), and the index reads image bytes lazily via an injected resolver. Note the CLIP **modality gap** (text-text cos ~0.7 vs text-image ~0.3): a *single* deduped cosine ranking buries image vectors under text ones, so 3c only made image vectors *indexed and retrievable* (image hits additive, not rank-1). **#201a** (the search epic) closes that — it split the index into **per-modality sub-indexes** (`index.usearch` for text, `index.image.usearch` for images) so a text query is ranked separately against each modality (`index.search_by_modality`) and each feeds RRF as its own signal; the gap's constant offset is invisible to a rank-based combiner, so a text query now surfaces a card by its *image*. The intra-modal **activation gate (#201b)** then stops a weak modality injecting its top-k on every query: a non-text modality contributes only when its best match clears `mean + margin·std` of that modality's typical best match, calibrated offline by sampling stored text vectors as pseudo-queries (self-excluded) and persisted in the index meta (`activation`).
  - **`modalities` is the graceful-degradation seam.** Every backend advertises a `frozenset[str]` of what it can embed; both Phase-1 backends are text-only (`{"text"}`), so there is no behavioural branch yet. The contract is what matters: **text-only is a permanent, first-class capability** (the unit/integration suites rely on small text-only models), and a future multimodal backend just advertises more modalities — search over media-by-content then lights up where vectors exist and quietly returns nothing where they don't, never erroring. Multimodal itself (the rest of #162: media embedding, the multi-vector-vs-fusion index question — *decided after evaluation*) builds behind this seam.
  - **`OnnxBackend` specifics:** model is an ONNX dir (`model.onnx` + `tokenizer.json`, or `onnx/model.onnx`) or a `.onnx` file with `tokenizer.json` beside it; pooling (`mean|cls|last`) and optional L2 normalization are done in numpy (llama-server does these internally). Pooling is **vector-affecting → folded into the fingerprint**; normalization is scale-only (USearch `cos` is scale-invariant) → **deliberately not**, mirroring llama's `--embd-normalize`. Fingerprints are namespaced by family (`onnx:…` vs llama's `meta:`/`file:…`), so the same model under two runtimes never shares a vector space and an existing llama index needs no change. onnxruntime providers via `--embedding-onnx-provider` (repeatable; `SHRIKE_EMBEDDING_ONNX_PROVIDERS`, comma-separated; default CPU) are **resolved gracefully**: intersected with `get_available_providers()`, an unavailable one dropped *with a warning* (not onnxruntime's silent CPU fallback), CPU always appended as the final fallback, and the **actually-loaded** provider (`session.get_providers()`) surfaced in `health()`/`server status` so a silent CPU fallback is visible. Packaging mirrors onnxruntime's wheels: `shrike[onnx]` = base (CPU + CoreML on macOS), `shrike[onnx-gpu]` = `onnxruntime-gpu` (CUDA/TensorRT), installed *instead of* `[onnx]` (they conflict); DirectML is a manual `onnxruntime-directml`. **Batch safety is probed at startup, not assumed (#174):** int8 exports use dynamic quantization whose activation scales are computed over the whole batch tensor, so a *batched* embed makes a note's vector depend on its batch-mates' content — breaking the `reconcile`==full-rebuild invariant. fp32/fp16 ONNX (and llama-server, fp) are bit-exact batched, so the variance is int8-only. Rather than guess, **every backend's `start()` runs `embed_batching.probe_max_safe_batch`** — embed a magnitude-spiked probe set serially vs **all in one batch**, within a tolerance above float noise and below quant drift — and `embed_texts` then batches **up to the proven probe-set size** (further capped by `--embedding-batch-size`, any backend) or **serially** (batch size 1) for a batch-variant model, so "proven safe" and "what we batch" are the same size (the probe retries a transient failure before falling back to serial). The set is spiked for activation magnitude (long/numeric/symbol/repeated/mixed-script), since int8 drift is magnitude- not length-driven; its sensitivity is pinned by a >10×-tol-drift test on the real int8 fixtures. Locked by exact-equality (`np.array_equal`) tests against *real* int8 (serial) and fp32 (batched==serial) models. Both backends — and a second ONNX lineage (DistilRoBERTa, 768-dim, BPE) plus the fp32 MiniLM — are CI-tested (`tests/integration/test_backends.py`; `tests/integration/test_onnx_models.py`).

- `shrike server start` starts embedding at boot if a model is configured, unless `--no-embedding` is passed (lets a server run with embedding deliberately off).
- `shrike embedding start` / `shrike embedding stop` cycle the service on a running server (for llama-server upgrades, model swaps, freeing GPU/RAM). Stopping marks the index `unavailable` but keeps the on-disk vectors; starting re-attaches and rebuilds only if needed.
- **Pooling type:** `--embedding-pooling {mean|last|cls|none}` (config `embedding.pooling`, `SHRIKE_EMBEDDING_POOLING`) is passed to llama-server as `--pooling`. Unset means "use the model's GGUF default" — fine for BERT-family models (`all-MiniLM-L6-v2`, `bge-m3`) that carry mean pooling in metadata. **Last-token models (Jina v5, Qwen3-Embedding) need `--embedding-pooling last`**: their pooling type isn't in the GGUF metadata, so without it llama-server defaults to mean and produces wrong embeddings (and some of these architectures may need a newer llama.cpp than the pinned `LLAMA_TAG` — see `scripts/llama-server.lock`). Pooling is folded into `model_id` (below) so changing it forces an index rebuild.
- **Generic arg passthrough:** `--embedding-arg` (repeatable; config `embedding.extra_args` as a list; `SHRIKE_EMBEDDING_ARGS` as one shlex string) appends raw tokens to the llama-server command for the long tail of **runtime-only** flags (`--flash-attn`, `--ubatch-size`, gpu split, …). Each entry is `shlex`-split at command-build time and appended last. Two guardrails: (1) Shrike-owned flags (`--model`/`-m`/`--host`/`--port`/`--embeddings`/`--embedding`, plus their value token) are stripped with a warning — `--host` especially, since llama-server is pinned to loopback (audit §1.1); (2) the effective passthrough is folded into `model_id`, so **any** change forces a rebuild (conservative — Shrike can't tell a vector-affecting flag from a perf-only one in an opaque bag). **Vector-affecting flags must be typed settings** (like `--embedding-pooling`), not buried here. Normalization is *not* such a setting: USearch's `cos` metric is scale-invariant (verified in `index.py`), so `--embd-normalize` is moot.
- Starting llama-server blocks (model load + health wait), so the HTTP handler runs it via `asyncio.to_thread` to keep the event loop responsive.
- **Orphan reaping:** `EmbeddingService` records the llama-server PID in `<state-dir>/embedding.pid` (written after spawn, removed on clean stop). If Shrike is hard-killed (SIGKILL, incl. the daemon's own force-kill path), llama-server is orphaned and keeps holding its port. On the next `start()`, `_reap_orphan` detects a recorded PID that is still alive *and* holding the port and terminates it (SIGTERM→SIGKILL) before binding. `PR_SET_PDEATHSIG` is intentionally avoided: the parent-death signal keys on the spawning *thread*, and start runs under `asyncio.to_thread`, so a reclaimed pool thread could kill a live server.

### Recognition (OCR) — #228

**Recognition is the kernel's second injected capability** (the #342 slot pattern,
sibling of the embed slot): an OCR engine the harness attaches at assembly turns
note media into searchable text. Off by default; `--ocr-backend apple` (config
`recognition.ocr`, env `SHRIKE_OCR_BACKEND`) selects macOS Vision via the
`shrike[vision]` extra (pyobjc; no model download — Vision ships with the OS). A
missing dependency degrades the recognition state to `error` without disturbing
boot. The Python contract is `RecognizerBackend` (`recognition.py`): a *blocking*
`recognize(items: list[bytes]) -> list[tuple[str, float, str]]` — (text,
confidence, segments-JSON) — plus `model_fingerprint()`; `PyRecognizer.capture`
bridges it to the kernel with the PyEmbedder dispatch shape (loop → thread pool →
oneshot, never the collection executor).

**One pass, many consumers** (the epic's load-bearing rule): the kernel's
`recognize_pending(max_items)` sweeps bounded batches of pending (note, image)
pairs — pending = a resolvable image with no OCR row, or everything after the
recognizer *fingerprint* changes (an OS upgrade re-derives, like a model change
rebuilds vectors) — and persists BOTH the flattened text (derived rows,
`source='ocr'` → substring/fuzzy search + provenance light up through the
existing seam) and the per-segment structure (the `segments` table; boxes today,
#230 occlusion's input). Gating (#199, `RecognitionGate` kernel-side): confidence
+ substance to store at all, a higher substance bar to mint a vector. Gated text
embeds via the TEXT encoder as extra vectors under the note key in the `text`
space (no modality gap; max-over-items ranking falls out), and the per-note
fingerprint folds the OCR text — byte-identical with none, so upgrades never
spuriously rebuild. The derived store's drift rebuild is **field-source-scoped**
(schema v2): a boot drift never discards recognition rows. The harness drives
sweeps in the background (`recognition_sweep`, one batch per executor occupancy);
`/status` carries `recognition: {state, backend}`.

### Vector index and consistency

The vector index is a **derived cache**, not a co-equal store. The Anki collection (SQLite) is always the source of truth. SQLite handles its own crash recovery via WAL/journal, so the collection is self-consistent after any crash. If the index is stale, corrupt, or missing, it can be rebuilt from the collection by re-embedding all notes.

**Consistency model:** the index may lag behind the collection (notes added/modified/deleted without the index being updated), but the collection never lags behind the index. This means search results may be stale, but data operations (upsert, delete, list) are always correct.

**Drift detection:** the index metadata (`index.meta.json`) stores `col_mod` — the value of `col.mod` (collection-level modification timestamp, milliseconds) at the time the index was last built — and `model_id`, a fingerprint of the embedding model that produced the vectors. On startup (and whenever the embedding service is attached), compare both against the stored values. If they match, nothing changed — skip reindexing. If `model_id` differs (the embedding model changed, so every vector lives in a different space) the whole index is invalid → full rebuild. If only `col.mod` differs (something changed outside our control — Anki GUI, sync, imports), the index **reconciles incrementally** rather than re-embedding everything (#38): a per-note embedding fingerprint sidecar (`index.hashes.json`, `{note_id: blake2b}`) lets `VectorIndex.reconcile` diff current notes against the indexed ones and re-embed only the changed ones, add new notes, and drop deleted ones — the end state is identical to a full rebuild over the same notes (verified), but a drift that touched a handful of notes re-embeds a handful, not the whole collection (measured 87× on a 1K-note index with 5 edits). The fingerprint is **media-aware** (#162 Phase 3c): for an image-capable backend it folds in the sorted filenames of a note's images **that actually resolve** (`_note_hash`), so adding/removing/swapping — or *later storing* — a note's image re-embeds it, while a referenced-but-never-stored image stays out of the hash (no re-embed loop); for a text-only backend it's the text hash unchanged — *byte-identical to the pre-3c scheme*, so text-only users pay no spurious rebuild on upgrade. Anki content-addresses media (a filename is a stable content identity), so the fingerprint hashes *names of present images* — a cheap presence `stat` (the resolver's `exists` half, no byte read), not the bytes — keeping the diff cheap (DB text + a regex + a stat); image bytes are read only for a note actually being embedded, lazily and lock-free via the index's image resolver. Folding in only *resolvable* names is what keeps reconcile == a full rebuild (which re-reads files at embed time) even for a note authored before its media landed. A `col.mod` bump from a non-embedding edit (tags/deck/template) finds no fingerprint changes and just advances the watermark. `reconcile` falls back to a full `rebuild` when the model changed, there's no prior hash state (an index built before this existed, or first build), or an image-capable backend meets a **pre-#201a (v1) single-index layout** that can't be split into per-modality sub-indexes (detected by the `schema` marker in `index.meta.json`; a text-only v1 index loads losslessly as the v2 text sub-index, so text-only users never rebuild on upgrade). The hash map is maintained incrementally by `add`/`remove` (so Shrike's own upserts don't trigger a re-embed on the next reconcile) and persisted alongside the index; a missing/corrupt sidecar safely degrades to a full rebuild. **Explicit** rebuilds (`shrike index rebuild`, `POST /index/rebuild`) stay full — reconcile is only the automatic drift path (`_maybe_rebuild`).

**Model fingerprint:** `model_id` comes from llama-server's `GET /v1/models` `meta` block (`n_params`, `n_embd`, `n_vocab`, `n_ctx_train`, `size`) — fast and describes the *loaded* model. It falls back to model filename + on-disk size if that metadata is unavailable. The model *name* is deliberately excluded (it would force needless rebuilds on rename and miss same-name re-quantizations, which the numeric fields catch). An explicitly-set pooling type is appended (`…:pool=last`) because it changes every vector but isn't reflected in the model metadata; it's omitted when unset so indexes built before this setting existed still match. The note-text normalization version (`…:textprep=N`, `EMBED_TEXT_VERSION` in `embed_text.py`) is appended **unconditionally**: the cleaned text we feed the model is as much a part of the vector space as the model itself, so changing how notes are rendered for embedding must invalidate the index (unlike pooling, it's never omitted — an index built under the prior raw-text scheme *should* rebuild). `EmbeddingService.embed()` also pins `"model": <id>` in the request body as insurance against a future external multi-model endpoint.

**Note text for embedding:** `embed_text.normalize_for_embedding()` turns each raw Anki field value into stable plain text before embedding. It operates on field *values*, not rendered cards — a note (not a card) is the embedding unit, a cloze note generates N cards, and templates add presentational scaffolding (`{{FrontSide}}`, `<hr id=answer>`, the hidden `[...]` on a cloze question side) that is noise for search. The HTML→text + entity step **delegates to Anki's own `strip_html`** (Rust-backed, robust on malformed markup, and it leaves an encoded `&lt;tag&gt;` as the literal `<tag>`); around it we do what that stripper doesn't: reveal cloze (`{{c1::France}}` → `France`, hint dropped), drop MathJax/LaTeX wrappers keeping the inner source (`\(…\)`, `$$…$$`, `[latex]…[/latex]`), drop `[sound:…]`, and convert block tags to spaces *before* calling Anki's stripper (which otherwise glues `a<br>b` → `ab`). The module lazily `set_lang("en")`s once so the stripper works headless (locale doesn't affect stripping output). The result is a function of the field value plus the pinned Anki version's stripper — identical whether a note is freshly upserted or re-read during a rebuild, and independent of which card a cloze generates. `CollectionWrapper.note_texts()` applies it per field; both embed call sites (`upsert_notes`, rebuild) route through it, so consistency is structural. Bump `EMBED_TEXT_VERSION` whenever the normalized output changes — including an Anki upgrade whose stripping differs.

**Implementation:**

1. **Startup check** — on server start, compare `col.mod` against the stored value in index metadata. Match → load existing index. `col.mod` mismatch → incremental reconcile in a background thread; missing/corrupt index or model change → full rebuild. Server starts accepting requests immediately; `search_notes` returns actionable status messages ("building 2847/5000 notes, try again shortly") until ready.

2. **Incremental updates** — after `upsert_notes` and `delete_notes` succeed on the collection, the index is updated in the same call (`index.add()` / `index.remove()`). Stored `col_mod` is updated after each successful index update. Index update failures log a warning but don't fail the tool call — the next startup detects the `col.mod` mismatch and rebuilds.

3. **Persistence** — the index is saved to disk on graceful shutdown (signal handler and `POST /shutdown`), at the end of a rebuild, and via a **debounced flush** during normal operation. `IndexSaver` (in `index.py`) owns the debounce: the upsert/delete tools call `saver.request_save()` after each incremental update (once `col_mod` is set), and the index is written either **`save_delay` seconds after the last change** (idle debounce, default 60s) **or immediately once `save_threshold` unsaved changes accumulate** (burst cap, default 100), whichever comes first. The save runs off the event loop (`asyncio.to_thread`); the debounce timer is `loop.call_later`, so there is no background timer *thread* and no fixed-interval polling — the flush is driven by edit activity. This bounds how much incremental work a hard kill / crash discards: once a flush lands and the server goes idle, the on-disk index (and its `col_mod`) are current, so it reloads without a rebuild. For edits since the last flush, the `col.mod` mismatch on next startup still triggers a full rebuild — correctness is preserved either way, at the cost of a re-embed. `save_delay`/`save_threshold` are configurable (config `index.*`, env, `--index-save-*` flags); the cache location is `cache_dir`/`SHRIKE_CACHE_DIR`/`--cache-dir`. (Tombstone compaction is unnecessary on the pinned USearch — see the index code comments.)

4. **Full rebuild** — `shrike index rebuild` CLI and `POST /index/rebuild` endpoint. Drops existing index and re-embeds all notes. Progress reporting via CLI and `/status`.

5. **State machine** — states: `ready`, `building` (with progress), `unavailable` (embedding service not running — never configured or stopped), `error` (build failed). Exposed via `/status` endpoint, `search_notes` responses, and `shrike server status` CLI.

**Cost considerations:** the automatic drift path reconciles incrementally (re-embed only changed/new notes, drop deleted), so an external edit to a few notes costs a few embeddings, not a whole-collection re-embed; a full rebuild (model change, explicit `index rebuild`, missing hash state) re-embeds everything — seconds for ~1K notes, minutes for 10K+. Both run in a background thread, so the server is never blocked. During normal operation, incremental updates from `upsert_notes`/`delete_notes` keep the index (and its per-note hashes) current without any rebuild.

### Derived-text store (#98)

**Shrike derives data locally from notes, and that derived/computed data wants one home — a sidecar SQLite `shrike.db` in `cache_dir()`, separate from Anki's synced collection.** `DerivedTextStore` (`derived.py`) is that store; its first artifact is an **FTS5 trigram index** over note text, backing the `search_notes` substring (`exact`) candidates and the new **`fuzzy`** RRF signal (typo/partial matches). With the relay in view (offload heavy compute to a user's desktop instance, sync the artifacts back), this is the natural sync target — so it's designed for more than the trigram index from day one.

**Source-seamed.** Every indexed row is keyed `(note_id, source, ref)` — `source` is *where* the text came from, `ref` the field name or media filename. Today the only source is `field` (raw field text, ingested via `CollectionWrapper.derived_field_rows`/`note_field_map`). The seam is for the rest of the search/OCR epic: #199 will feed `ocr`/`asr` recognized text (→ trigram **and** the text-embedding space, provenance-tagged so a result can say "matched the OCR text of diagram.png — here's the window," the win that finally gives an LLM/MCP client insight into image/audio cards it can't be shown). A future VLM image-describe goes to the **embedding space only, never the trigram index** — a literal-search hit on invisible VLM metadata can't be cleanly explained to a user. A row stores the text (so FTS5 `snippet()` gives the window, and #199's embedder can read it back); a two-table layout (`idx` FTS5 + a `rowmap(rowid → note_id, source, ref)` with an index on `note_id`) keeps incremental delete-by-note cheap (FTS5 has no secondary indexes).

**Why a sidecar, not tables in `collection.anki2` (settled, recorded in `docs/decisions.md`):** Anki's sync, "Check Database", media check, and version-upgrade migrations own that schema — foreign tables get dropped or error, and it would ship rebuildable derived data over sync. We already time-share the collection lock with Anki desktop (#64); the community norm is add-ons keep their own files. A sidecar in our cache dir is both safe and the correct home for derived/rebuildable data.

**Same derived-cache contract as `VectorIndex`, with two deliberate divergences.** It's rebuildable from the collection, detects `col.mod` drift, updates incrementally on upsert/delete (`ingest`/`remove` alongside `index.add`/`remove` at the five tool hook sites — guarded on `derived.available`, independent of the vector index so it works with embeddings off), and surfaces a `ready/building/unavailable/error` state (`/status` → `DerivedStatus` → `shrike server status`). The boot/`/reload`/cooperative-reacquire paths build it in the background on drift (a cheap text-only build — no models). Divergences: (1) **no `IndexSaver`** — persistence is inherent to the SQLite file (writes are transactional/durable), so there's no debounce; incremental writes just advance the stored `col_mod` watermark (the `col_mod` setter) so the next boot sees no drift. (2) **Graceful absence is first-class**: if the runtime's SQLite lacks FTS5/the trigram tokenizer (probed at construction on a throwaway conn), the store reports `unavailable` and every lookup signals the caller to fall back to the linear `find_notes` scan — current behaviour, no feature regression. Likewise a `<3`-char query (FTS5 trigram can't match it) returns `None` → fallback. All MATCH expressions are FTS5-quoted (`_fts_quote`) so query punctuation can't be parsed as FTS5 syntax (injection-safe). Metadata-only edits (tag/deck rename, template/CSS, field metadata) advance the store's `col_mod` *without* re-ingesting (field text unchanged) via the shared `_bump_col_mod_after_metadata_change`, exactly like the index.

## Code style and conventions

- **Type annotations** on all functions (enforced by mypy with `disallow_untyped_defs`)
- **Ruff** for linting (rules: E, F, W, I, UP, B, SIM) and formatting, line length 100
- **Error handling:** batch operations (upsert_notes, upsert_note_types) use per-item try/except so one failure doesn't block the batch. Results include `status: "created"|"updated"|"error"` per item.
- **`raise ... from err`** in except blocks (enforced by ruff B904)
- **`contextlib.suppress`** instead of bare `try/except/pass`
- **`datetime.UTC`** not `timezone.utc` (ruff UP017)

### Logging

Logging is configured in `shrike/log.py`. Log format, parsing, and styling all live in that module — formatting knowledge should not be spread across CLI commands.

**Logger names** — Use per-module loggers: `shrike.server`, `shrike.kernel`, `shrike.tools`, `shrike.collection`, `shrike.embedding`, `shrike.index`, `shrike.derived`, `shrike.daemon`. This makes the config's per-logger level overrides (`logging.levels.shrike.collection: debug`) actually work. Never log everything under a bare `shrike` logger. Native (Rust) tracing forwards through pyo3-log under the crate's module path (e.g. `shrike_collection.media_fetch`), so the same per-logger overrides govern it.

**Principles for log messages:**

1. **Say what happened and include the key context.** "Collection ready: 847 notes, 5 decks, 12 note types" not "Collection opened". Include counts, IDs, paths, durations — the things that make a log line useful without having to correlate it with other lines.
2. **Log operational boundaries at INFO.** Startup, shutdown, configuration loaded, server listening. These are the anchors you scan for when reading a log.
3. **One INFO line per served call.** This is a server; knowing what it did is the point — and one line per request says it all: tool name + given params + outcome + duration, e.g. `list_notes deck='Test' limit=50 -> 3/3 notes (12ms)`. The adapter (`_safe_tool`) emits it; actions contribute the outcome fragment via `note_outcome(...)`. Custom HTTP routes get the same treatment from `_guard`: `GET /media/x.png -> 200 (3ms)`. Anything else logged while serving a call is a WARNING/ERROR (exceptional) or DEBUG (internals, e.g. the entry line) — never a second INFO line.
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
- **Roadmap and tracked work live in GitHub issues + milestones** (each milestone
  is a themed body of work — e.g. *Sync*, *Terminal UI (TUI)* — with an `epic`
  tracking issue; milestones are **not** tied to specific version numbers, since
  what ships in a given release is decided at tag time) — *not* in this file or
  the README, which is how the old prose roadmaps drifted. `gh issue list` /
  `gh issue list --milestone "..."` is the current state of the project.
- **Shipped-design rationale** (the "why" behind decisions like contextual-upsert
  neighbours, duplicate detection, full-replace tags) lives in
  [`docs/decisions.md`](docs/decisions.md).

### Review & audit gates — mandatory

These are required, not optional, and run in addition to the CI lint/test gates:

- **Code review on every significant change and feature addition** before merge —
  not trivial typo/doc/dep-bump PRs, but anything that adds or changes behaviour.
  Use `/code-review` (escalate to `ultra` for larger changes).
- **Security review whenever the server API surface changes** — a new/changed MCP
  tool or custom HTTP route, auth/transport/SSRF/path handling, anything touching
  the trust boundary. Run it *in addition* to the code review, via
  `/security-review`.
- **Before cutting a release**, run a fresh pair of passes over the release
  candidate: a **security audit** and a **performance audit**. Apply the `rc`
  label first so the cross-platform CI lane also runs (see the CI notes above).

Reviews/audits are launched by the user (the `ultra`/cloud passes are billed and
user-triggered — the agent can't start them); the agent's job is to surface that
a change crosses one of these thresholds and to act on the findings.

### The agent's PR loop — delegated, self-driving

For work the user has delegated, the agent owns the whole PR cycle and keeps it
pipelined rather than serial:

- **PR at each natural checkpoint.** Don't contort in-progress work to make it
  mergeable when going a bit further lands a larger, coherent section — but
  don't hoard mergeable work either.
- **Self-review before the `ci` label, not after.** Run a self-check code
  review against the requirements on the ready PR — via a **subagent
  (prefer the latest Opus model)** — *before* labeling, so review findings
  never burn a CI re-run. Keep working while the review is in flight.
- **Then label, auto-merge, move on.** Apply `ci` once review findings are
  addressed, and immediately set the PR to merge when green
  (`gh pr merge --auto --squash`) — don't poll for green; while CI runs,
  proactively move on to the next step.
- **Subagents assist with research, orientation, and review — never
  authorship.** All code and tests are developed by the agent itself; use
  subagents wherever they speed up or improve the work (codebase orientation,
  API research, parallel fact-finding, the self-check review above).

This composes with the gates above: the user-triggered billed passes
(`ultra`/cloud review, security/perf audits) stay user-triggered; the agent's
self-review is the floor, not a replacement, for anything crossing those
thresholds.

### Defect workflow — follow this when you find a defect or limitation

When you hit a bug, a limitation, or a missing API surface that is **out of scope
for the task in hand**, do not silently fix it inline and do not leave it as a
prose note. Capture it as resumable state:

1. Open a GitHub issue with a clear problem statement (repro / expected vs actual /
   scope, or the intended API surface that's missing).
2. Create a branch `fix/‹issue#›-‹slug›` (or `feat/…` for a missing capability;
   `xfail/‹issue#›-‹slug›` when you're only capturing the red reproducing spec
   without a fix — e.g. the weekly performance audit).
3. Add failing test(s) that exercise the defect / pin the intended API —
   asserting the *desired* behaviour and marked
   `@pytest.mark.xfail(strict=True, reason="#‹n›: …")` so the branch's CI stays
   green while the test is red-by-design, and a future fix that makes it pass
   forces the marker's removal.
4. Push the branch to origin and link it from the issue.

The failing test is the spec; the pushed branch is the handoff. See the full
rationale in `CONTRIBUTING.md`.

### Approved-plan check-in — record the plan on its home issue

When the user **approves** a plan (plan mode / ExitPlanMode), post it as a comment
on the GitHub issue it's homed in, so the issue carries the agreed design as
resumable state. End the comment with the same attribution line used on PR bodies
(`🤖 Generated with [Claude Code](https://claude.com/claude-code)`). Only
*approved* plans, never drafts. **Strip notes-to-self before posting** — drop the
process/meta sections (e.g. "Workflow updates", "Branch"); the comment carries the
substantive plan only (context, design, surface, files, verification), not the
scaffolding.
