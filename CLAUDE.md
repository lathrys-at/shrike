# CLAUDE.md

## What is Shrike?

Shrike manages Anki flashcard collections without running Anki's GUI. It exposes Anki's collection operations through an MCP server and CLI.

**License:** AGPL-3.0

### Architecture

**The compute core is Rust; the kernel is a pure plugin host.** The server is an
*assembling harness*: it opens one `AsyncKernel` (the Rust kernel bound for
asyncio) on the event loop, constructs engines from config, attaches them to the
kernel's service slots (`attach_embedder`/`attach_recognizer`), and serves. The
kernel composes `Arc<dyn Embedder>`/`Arc<dyn Recognizer>` it is *given* — it names
no engine, no platform, no transport; the contracts live in `shrike-engine-api`
(async traits the kernel consumes; sync compute traits engines implement; the one
`Blocking` adapter onto the runtime's blocking pool; the batch-safety probe). Each
engine is its own crate: `shrike-embed` (ort text + CLIP), `shrike-recognize-apple`
(Vision OCR), `shrike-embed-remote` (any OpenAI-compatible embeddings endpoint) —
with `shrike-llama-server` as the *lifecycle manager* producing the local endpoint
the remote engine talks to (manage-class, not an engine). Every production backend
attaches native — kernel embeds/recognitions never enter Python;
`PyEmbedder`/`PyRecognizer` capture remains the custom/test-backend escape hatch.
The kernel owns the collection (anki via its protobuf service layer ONLY), the
vector index orchestration (drift, per-note fingerprints, debounced saves), and the
derived-text ingest; write actions route through maintained kernel ops
(`upsert_notes_json`, `delete_notes`, `reindex_notes`, `forget_notes`,
`metadata_changed`).

**The kernel owns its runtime**: idiomatic async Rust on a process-global tokio
runtime (`shrike_kernel::runtime` — only the Handle escapes; `init_runtime` is the
builder seam, proven down to a single-thread `current_thread` mode). The collection
serializes through a task-actor (one spawned task looping an mpsc inline — FIFO by
construction, no thread affinity, no `block_in_place`); timers (the debounced index
flush) ride `tokio::time`; engine compute runs on the blocking pool via the one
`Blocking<E>` adapter (eager `spawn_blocking`, which preserves the search/batch
overlap properties); independent batch futures are `try_join`ed by the kernel.
**The action exchange is the host boundary**: the binding spawns each op onto the
kernel runtime (`spawn_op`) and awaits a oneshot-backed completion future through
the one-wake asyncio bridge — dropping it detaches observation, never aborts the
work. **Invariant — sync ops never execute on a runtime worker thread**: anki's
sync paths (the only `block_on` callers; client sync #33/#362 will wake them) panic
from any runtime-worker thread, so kernel-side sync ops MUST dispatch via
`spawn_blocking` (a legal `block_on` site, the `py_embedder`/`py_recognizer`
pattern). The `sync_dispatch_pin` panic-repro test in `shrike_kernel::runtime` pins
this; see `docs/decisions.md`.

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
                                                      ├──▶ EmbedService slot ◀── engine crates via shrike-engine-api
                                                      │       (shrike-embed ort/CLIP; shrike-embed-remote ◀── shrike-llama-server)
                                                      └──▶ RecognizeService slot ◀── shrike-recognize-apple (Vision OCR)
```

## Project layout

```
src/shrike/                       # Python package (src layout) — the harness half
├── __init__.py                   # Re-exports __version__ from generated _version.py (hatch-vcs)
├── server.py                     # MCP host: argparse, FastMCP, routes; main() tail = asyncio.run(_serve())
├── harness.py                    # Harness (kernel-mode core: assembly + verbs) + KernelIndexView
├── collection.py                 # CollectionWrapper — async facade over the shared core
│                                 #   (kernel mode via over_kernel; standalone mode for tests)
├── daemon.py                     # Daemon lifecycle — file locks, spawn, shutdown
├── actions.py                    # The action registry (24 actions) — kernel-mode write paths
├── tools.py                      # Binds the registry to MCP; returns response models (outputSchema)
├── mcp_adapter.py                # _safe_tool policy + per-call INFO logging
├── schemas.py                    # Pydantic wire models (the BINDING; shrike-schemas in Rust is canonical)
├── client.py                     # ShrikeClient — standalone HTTP client; typed per-tool methods
├── paths.py                      # Platform-canonical directories (via platformdirs)
├── log.py                        # Logging config, log parsing and styling
├── embedding.py                  # EmbedderBackend facades + EmbeddingRuntime (backend lifecycle)
├── recognition.py                # RecognizerBackend protocol + make_recognizer (OCR; native engine)
├── index.py                      # IndexState + activation_floor (host-side index policy; index lives in kernel)
├── derived.py                    # DerivedTextStore — FTS5 facade (read paths; kernel ingests in server mode)
├── pathsafety.py                 # Shared server-local path-safety gate (store_media #164, export #71, import #72)
├── export_store.py               # Export download store: server-named temp packages + one-shot tokens (#71)
└── cli/
    ├── __init__.py               # Root Click group, global options (--config, --url, --json, --pretty)
    ├── groups.py                 # OrderedGroup (canonical §G order) + SearchGroup (default-command)
    ├── client.py                 # Re-export shim → shrike.client (keeps imports working)
    ├── config.py                 # YAML config loading/saving
    ├── completion_cmd.py         # shrike completion {bash,zsh,fish}
    ├── search_cmd.py             # shrike search {<query>,query,coverage} (the retrieval group)
    ├── embedding_cmd.py          # shrike server embedding status/start/stop
    ├── export_cmd.py             # shrike collection export (.apkg/.colpkg; download or --server-path)
    ├── import_cmd.py             # shrike collection import (.apkg/.colpkg)
    ├── index_cmd.py              # shrike server index rebuild/status/save
    ├── server_cmd.py             # shrike server start/stop/status/logs (+ embedding/index subgroups)
    ├── info_cmd.py               # shrike collection info
    ├── note_cmd.py               # shrike note create/update/delete/list/show/tag/replace/migrate-type
    ├── tag_cmd.py                # shrike collection tag rename (collection-level tag ops)
    ├── collection_cmd.py         # shrike collection info/check/export/import/prune/reload/tag/media
    ├── deck_cmd.py               # shrike deck create/rename/delete
    ├── media_cmd.py              # shrike collection media store/fetch/list/delete
    ├── type_cmd.py               # shrike type create/update/delete/list/show
    └── output.py                 # Rich formatting, output_options decorator
tests/
├── unit/                         # direct calls, no server (conftest: wrapper fixture, temp collection)
├── native/                       # native extension + kernel bindings (asyncio bridge, AsyncKernel, Harness)
└── integration/                  # real server subprocess + HTTP transport
                                  #   conftest: shared session server + per-test reset; mcp/runner; isolated_server
native/                           # the Rust workspace (the compute core)
├── shrike-kernel/                # THE kernel: collection + index orchestration + derived + fusion
├── shrike-collection/            # anki via its protobuf service layer (the ONLY anki coupling)
├── shrike-index/                 # per-modality USearch engine
├── shrike-derived/               # FTS5 trigram engine
├── shrike-engine-api/            # THE engine contract: kernel-facing traits, sync compute
│                                 #   traits, the Blocking adapter, WithPolicy, the batch probe
├── shrike-embed/                 # ort/tokenizers text + CLIP engines (implement the contract in-crate)
├── shrike-embed-remote/          # EmbedText over any OpenAI-compatible endpoint (ureq; llama/cloud/tailnet)
├── shrike-describe-remote/       # VLM image→text describe over OpenAI-compatible chat completions
│                                 #   (embedding-space-only destination — attach waits on the
│                                 #   kernel's per-engine destination policy)
├── shrike-llama-server/          # llama-server lifecycle ONLY (spawn/health/reap/stop) — not an engine
├── shrike-recognize-apple/       # Apple Vision OCR engine (Swift glue behind Rust; off-macOS stub; needs Xcode)
├── shrike-schemas/               # serde+schemars wire types (CANONICAL; schemas.py binds)
├── shrike-ffi/                   # the shared error taxonomy
└── shrike-py/                    # the pyo3 binding (the ONLY pyo3 crate) + shrike_native package
docs/
└── mcp-tools.md                  # Tool documentation (human-readable; machine schema served live by the server)
```

## Development setup

```bash
scripts/dev-setup.sh        # creates .venv, installs ".[dev]", builds the native extension
source .venv/bin/activate
```

One step, idempotent (re-run it any time as a repair button). It picks the
pinned interpreter (`.python-version`, via pyenv if present else `python3.12`),
installs the editable package + dev tooling, and builds the Rust `shrike_native`
extension — `pip install` alone does **not** build it (that's a separate cargo
step, #573). Refreshes after that are automatic if you use direnv (`.envrc`
rebuilds a stale extension on `cd`); without direnv, `pytest` fails loud if the
extension is stale rather than silently importing a stale `.so` (see Tests
below). Python 3.12 is used (managed via pyenv; `.python-version` is at repo
root). The `anki` package requires Python >= 3.11.

**Cacheable dev artifacts go in the repo-root `.cache/`, not `~/.cache`.**
Downloaded toolchains, test models, and build inputs cache under `<repo>/.cache/`
(gitignored) so they stay with the checkout instead of polluting your home or
colliding across checkouts. Use this rule for any new cacheable dev/build work,
and don't redirect a build system's output directory elsewhere just to manage
disk (bound disk via controlled parallelism + cleanup instead). Two
intentional exceptions live under `~/.cache`: the `./bazel`/bazelisk launcher
cache and the shared test-model cache (`SHRIKE_TEST_MODEL_DIR`, default
`~/.cache/shrike-test-models`). This is dev/build caching; it is separate from
the *application's* runtime cache dir (the platform `Cache` directory under
Platform directories below).

## Running commands

### Tests

```bash
pytest tests/unit -v                           # Unit tests (fast, no server)
pytest tests/integration -v -m integration     # Integration tests (starts a server)
```

#### Native (Rust) workspace

The Rust workspace lives in `native/` (run `cargo` from there); the Python
extension is rebuilt into the venv with `scripts/build-native.sh` (the fast
pip-lane inner loop). You don't have to remember to run it: with direnv the
`.envrc` rebuilds a stale extension on `cd`, and either way `pytest` aborts loud
(a `pytest.UsageError`, before any test imports the extension) if the `.so` is
stale — `pip install` does not rebuild it, so the old silent-stale-`.so` footgun
is gone (#573). A staleness check (`scripts/native-stale.sh`, well under 100ms of
git plumbing) keyed to a per-venv stamp drives both. `SHRIKE_SKIP_NATIVE_STALE_CHECK=1`
bypasses the pytest backstop (Bazel sets it — that lane builds the extension
hermetically). **Bazel is NOT on PATH** — use the
committed `./bazel` launcher at the repo root (it bootstraps bazelisk + the
pinned Bazel from `.bazelversion`; same entry point CI uses). The full local
gate for a native change:

```bash
(cd native && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd native && cargo test --workspace)
scripts/build-native.sh && pytest tests/unit tests/native -q
./bazel test //...      # the authoritative CI lane: all crate tests + layering check + py suites
                        # (CI also names the `manual` embedding halves — see test.yml)
```

The Bazel operational guide (targets, locks, caching, the polyglot seam) is
[`docs/build-bazel.md`](docs/build-bazel.md); the rationale is the "Bazel as
the polyglot build system" ADR in [`docs/decisions.md`](docs/decisions.md).

#### Coverage

Coverage lives in its own workflow (`.github/workflows/coverage.yml`), **off the
per-PR path and never a CI gate** — reported, never enforced (runs on `rc`-labelled
PRs, 3x/week on `main`, and on demand; main/scheduled runs publish the README badge
to the orphan `badges` branch). The `fail_under` in `[tool.coverage.report]` is the
**target**, enforced only locally by `scripts/coverage.sh`. **Run coverage locally**
to keep the number healthy:

```bash
scripts/coverage.sh            # full suite; prints report, exits non-zero below fail_under
scripts/coverage.sh --html     # also writes htmlcov/index.html
```

A plain `pytest tests/unit --cov=shrike` reads ~18 points lower because it can't see
the spawned server subprocess. The script's hook (a `.pth` that imports coverage only
when `COVERAGE_PROCESS_START` is set) is what captures it; `coverage.yml` and
`scripts/coverage.sh` run this identical command, so the numbers are comparable. By
hand it's:

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo 'import os; os.getenv("COVERAGE_PROCESS_START") and __import__("coverage").process_startup()' > "$SITE/coverage_subprocess.pth"
export COVERAGE_PROCESS_START="$PWD/pyproject.toml"
coverage run --parallel-mode -m pytest tests/unit tests/integration tests/native -q -m "not embedding" -n auto
coverage combine && coverage report      # exits non-zero below fail_under
```

A Bazel-lane equivalent (`scripts/coverage-bazel.sh`) captures the server subprocess
too and lands within a point; report-only until it proves out (mechanics in
`docs/build-bazel.md` § Coverage).

**`-n auto`** (pytest-xdist) parallelizes across cores — the server-spawn-bound
integration suite roughly halves (each server gets its own free port + temp dirs).
The coverage `.pth` fires for each xdist worker *and* each spawned server, so
`coverage combine` merges to one total. CI runs `-n auto`; locally it's opt-in (the
default stays serial so `-x`, `-s`, `pdb` keep working).

**Integration tests share one server, with a per-test reset.** All non-embedding
integration tests share a single session-scoped `server` (one boot per xdist
worker); an autouse fixture (`_reset_shared_collection`) resets the collection to
its pristine baseline after each test. The `mcp`/`runner` clients record what a test
mutated (a `_ResetTracker`) so the reset is cheap (created notes deleted by tracked
id; one `collection_info` catches anything untracked — the safety net). So a test
always starts clean and even collection-wide assertions (`total_notes == 0`) hold
regardless of run order. **You don't need to clean up after yourself** — just don't
assume the collection is empty mid-suite, and prefer asserting on your own deck/tag.
Affordances: `scoped_collection(url)` (snapshots + unrolls a sub-section explicitly)
and `isolated_server`/`isolated_mcp`/`isolated_runner` (spawn a *dedicated*
collection for the rare test needing an exclusive un-reset one — e.g. collection-wide
tag counts, which the reset can't restore since Anki keeps the tag registry).
Embedding tests use their own `collection_server` and are untouched by the reset.

### Linting

```bash
ruff check src/shrike/ tests/ native/shrike-py/python/           # Lint
ruff format --check src/shrike/ tests/ native/shrike-py/python/  # Format check
mypy src/shrike/                                                 # Type check
```

(`native/shrike-py/python/` is the extension's Python shim — outside `src/`, so
it must be named explicitly or it sits in no lint scope at all, #437.)

All three must pass cleanly. CI (`.github/workflows/test.yml`) **always runs on every PR** — there is no `ci`-label gate (retired in #678). On every push (Linux x64): a `lint` job and a `tests` job — ONE `bazel test` invocation over the full graph plus the `manual` embedding halves; Bazel's dependency analysis + the disk cache decide what re-executes (an unchanged target replays as "(cached) PASSED"). The `ci-ok` job emits the single required status check (`CI passed`); it always runs and is success iff every lane passed or legitimately skipped (it must stay a real *declared* job — a ruleset check pinned to the GitHub Actions integration is **not** satisfied by an API-posted check run, #664). Because CI actually runs, a PR is pending/blocked until it goes green — combined with **drafts-by-default** (see the PR loop below) nothing is mergeable before CI has run. The expensive **cross-platform ARM legs** stay opt-in by label: `rc` selects **all legs** (apply before tagging a release — it subsumes the per-leg labels), **`macos`** selects **only the macos-latest leg** (Apple-Silicon/ARM macOS — Swift/Vision glue, the PyO3 link path), and **`linux-arm`** selects **only the ubuntu-24.04-arm leg**. None of the ARM legs run on a plain PR or on merge to `main`.

The bazel/cross-platform lanes **cache** the pinned llama-server and GGUF test model (`actions/cache`). But a cache entry is only restorable from the run's own branch or the **default branch**, and `test.yml` runs on PRs only — so a **cache-warmer** (`.github/workflows/warm-cache.yml`) runs the *same* `bazel test` invocation on `main` (daily + on lock-changing merges + on demand), seeding the compiled graph, **test results**, and embedding externals into `main`'s scope for every PR to restore (also warms the cargo-side `rust-lint`/`rust-coverage` caches). Same composite/cache keys as the tests lane (`.github/actions/bazel-setup`); llama-server pinned via `scripts/llama-server.lock`, the model via `EMBEDDING_MODEL_*` in `tests/integration/model_cache.py` (both bumped manually). The fixture's `download_with_retry` (backoff on `429`/5xx) is the backstop for a cold/evicted run.

A separate **release workflow** (`.github/workflows/release.yml`) fires on `push` of a `v*` tag: it runs the same single `bazel test` invocation on all three platforms, builds the artifacts — **platform-tagged `shrike-mcp` wheels** (cp312-abi3 with `shrike_native` inside, one per platform; the linux pair auditwheel-repaired to manylinux tags), the sdist (version stamped from the tag, so build/test jobs check out `fetch-depth: 0`), the `anki-cards.skill` bundle (`scripts/package-skill.py`), and a `SHA256SUMS` — and cuts a GitHub Release. Release notes come from the matching `## [X.Y.Z]` section of `CHANGELOG.md` (falling back to auto-generated notes); a pre-release tag (`vX.Y.Z-rc.N`) publishes as a GitHub pre-release. A final (non-rc) release also publishes to PyPI as `shrike-mcp` via trusted publishing — no separate `shrike-native` distribution (the extension ships inside the platform wheel).

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

The `anki` pip package provides a headless Python API to Anki's SQLite database (no Qt/GUI). It acquires an exclusive write lock, so only one process can have a collection open at a time. `CollectionWrapper` handles lifecycle (open/close via atexit).

**Cooperative locking (opt-in).** By default the daemon holds the exclusive lock for its whole life, so you can't launch Anki desktop against the same collection while it runs. `--cooperative-lock` (config `server.cooperative_lock`, env `SHRIKE_COOPERATIVE_LOCK`) makes `CollectionWrapper` open on demand and **release the collection after a short idle window** (`--lock-hold-seconds`, config `server.lock_hold_seconds`, env `SHRIKE_LOCK_HOLD_SECONDS`, default 5 s), so an *idle* daemon no longer blocks launching Anki. It's cooperative *time-slicing*, not concurrent sharing (Anki desktop never releases mid-session). On each **re-acquire** an `on_acquire` hook runs a cheap `index.check_drift(col.mod)` and rebuilds off-lock only on real drift (an external edit during the idle gap). Default (permanent) mode leaves this inert. `server.lock` (daemon liveness) is a **separate** lock from the collection lock — `server status` / `/status` report both (`locking`, `collection_held`). The contention failure mode is a clean SQLite "database is locked" busy error, not corruption.

**Busy-acquire surface.** When a re-acquire can't open the collection (another process holds it), `_locked` catches Anki's `DBError` and raises `CollectionBusyError` **immediately** (no retry — the caller decides). It's a typed error, *not* a per-tool response variant (busy is orthogonal to every tool's response — the op never ran). It rides the existing two-layer error split: the server-side `CollectionBusyError` (`collection.py`) carries a message prefixed with the `COLLECTION_BUSY_CODE` sentinel (`"collection_busy"`, defined in `schemas.py`); `_safe_tool` logs it at WARNING and re-raises so FastMCP emits an `isError`; `ShrikeClient._call` detects the prefix and raises the **client-side** `CollectionBusyError(ShrikeError)` (a sibling of `ServerError`, so callers catch-and-retry instead of parsing a string); the CLI's root handler renders the human message with no stack trace. Cooperative-only — permanent mode never re-opens.

### MCP transport

The server uses FastMCP with streamable HTTP transport (`stateless_http=True`, `json_response=True`). It listens on `http://127.0.0.1:8372/mcp` by default. All communication is JSON-RPC 2.0: clients POST to the endpoint with `method: "tools/call"` and receive structured JSON responses.

**Trust boundary.** Every endpoint is unauthenticated, so the server binds loopback by default; binding a non-loopback host requires `--allow-remote` (it refuses to start otherwise) and llama-server stays pinned to `127.0.0.1` regardless. DNS-rebinding/CSRF protection (`_build_transport_security()`) validates `Host`/`Origin`, applied to the MCP endpoint *and* — via the `_guard` wrapper in `_register_custom_routes` — to the custom routes (`/status`, `/media/{name}`, `/shutdown`, `/index/rebuild`, `/index/save`, `/embedding/*`, `/reload`), which bypass MCP middleware (every route's guard is asserted in `tests/integration/test_security.py`). The guard is **independent of the bind address**: a loopback bind allow-lists loopback `Host`/`Origin`; `--allowed-host`/`--allowed-origin` (config `server.allowed_hosts`/`allowed_origins`, env `SHRIKE_ALLOWED_HOSTS`/`SHRIKE_ALLOWED_ORIGINS`, comma-separated) *add* trusted values for a reverse-proxy or VPN hostname — note a proxy forwards `Host: name:port`, so use the SDK's `name:*` port-wildcard form; and `--no-dns-rebinding-protection` (config `server.no_dns_rebinding_protection`) turns the guard off entirely where the network is the trust boundary (behind Caddy, on a tailnet, firewalled). A non-loopback bind with no explicit allow-list also leaves the guard off (the original `--allow-remote` behaviour). **A non-loopback bind given *only* `--allowed-origin` (no `--allowed-host`) builds a guard whose Host allow-list is empty → every request is rejected 421** (fail-closed, a config footgun, not a hole; `_build_transport_security` logs a startup warning). In every mode the endpoints stay unauthenticated — the guard is anti-CSRF/DNS-rebinding, not authentication.

**OAuth is required for native connectors (deferred).** Claude Desktop / claude.ai *URL connectors* require OAuth 2.1 + Dynamic Client Registration and fail against an unauthenticated endpoint, so the "behind a reverse proxy / on a VPN as a connector" story depends on implementing MCP server auth (`mcp.server.auth`, OAuth 2.0 + PKCE) — intentionally not started. Until then a native client reaches Shrike through the **`mcp-remote` stdio bridge** (`npx mcp-remote http://127.0.0.1:8372/mcp --allow-http --transport http-only`), which connects without auth because Shrike demands none.

### MCP tools (26 total)

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
| `collection_prune` | Working | Cleanup: unused tags + empty notes + empty cards + unused media; `dry_run` default **false** (applies; `dry_run: true` previews) |
| `collection_check` | Working | Read-only media diagnostics: unused/missing media, missing-media notes, trash state |
| `store_media` | Working | Store media (1-10) from base64 `data`, `url` (SSRF-guarded), or server-local `path` (off by default; opt-in repeatable `--media-path-root` on a purely-local daemon); per-item results; dedup/collision via Anki |
| `fetch_media` | Working | Locate media (1-10); per-item `found`(url+path+mime+size)/`missing` union — **never returns bytes** (GET the `url`) |
| `list_media` | Working | List media filenames (+ `media_dir`), optional glob `pattern`; each with `url`/`mime`/`size_bytes` |
| `delete_media` | Working | Delete media by name → Anki's recoverable trash (no ref-check); `deleted`/`not_found` |
| `list_profiles` | Working | Enumerate the collection/profile registry (#66): each profile's `name`/`path`, the active default; read-only |
| `export_package` | Working | Export the collection (or a deck/`note_ids`) to an `.apkg`/`.colpkg` (#71); returns a download `url` (GET it; never base64) or a gated server-local `path` |

**Duplicate handling lives *inside* `upsert_notes`, not in a separate pre-check.** Before each new note is written, Anki's own add-note validation (`note.fields_check()`, via `CollectionWrapper._check_new_note`) runs. A first-field duplicate (same first field as an existing note of that type — Anki's rule, collection-wide and **deck-independent**) is governed by `on_duplicate`: `error` (**default** — reported, not written), `skip` (`status: skipped`), or `allow` (written anyway). Structurally invalid notes — empty first field, broken cloze — are *always* reported as errors with a `reason` regardless of policy, and never written. `dry_run: true` runs the same validation but writes nothing (every result `ok`/`skipped`/`error`, response echoes `dry_run: true`). The per-item result union (`UpsertNoteResult` in `schemas.py`) has four `status`-discriminated variants (`UpsertNoteOk`/`UpsertNoteValidated`/`UpsertNoteSkipped`/`UpsertNoteError`; `.reason` is the machine-readable `NoteValidationReason`). This is Anki's *exact* first-field rule, distinct from the *semantic* near-duplicate signal the returned `neighbors` provide. **A standalone `canAddNotes`-style tool was rejected** (racy check-then-write, only actionable by a follow-up call — see `docs/decisions.md`). Validation applies to **creates**; updates are validated for existence/fields but not duplicate/empty. `dry_run` does not catch intra-batch duplicates (it writes nothing, so two identical new notes both validate clean — a real run catches the second).

**Note-type field/template edits come in two flavours, both data-safe.** `upsert_note_types` replaces the whole `fields`/`templates` list **by position** (the field/template at each position keeps its note data/cards even when renamed; only shortening drops the tail). `update_note_type_fields` and `update_note_type_templates` are the **by-identity** counterparts: a sequence of `add`/`remove`/`rename`/`reposition` ops addressed by *name*, so they express a true move, an insert, or a non-trailing remove that position-replace can't. They delegate to Anki's data-safe primitives (`rename_field`/`reposition_field`/`add_field`/`remove_field` and the `*_template` equivalents), which migrate note data/cards by identity (a template `rename` is a pure label change — cards key by ordinal). Each call is **atomic** (the op sequence is validated against a simulated name list before any primitive runs) and persists with a single `update_dict`. No inline index maintenance — the `col.mod` bump triggers a drift-rebuild on next startup (correct for a removed field; conservative for the rest). All live natively in `shrike-collection/src/note_types.rs`, sharing `simulate_struct_op` and the soundness check; bad ops raise `NativeInputError` → `ToolInputError`. The position vs identity tools are **reconciled** so they can't overlap dangerously: the positional replace *rejects* any update where an existing name moves to a different position (a reorder/insert/non-trailing-remove — which would silently re-label note data/cards), erroring with a pointer to the identity tool (`reject_unsound_positional_replace`); it keeps only the unambiguous positional edits (rename/edit-in-place, append, trailing-remove). A third tool, `find_replace_note_types` (anki-connect's `findAndReplaceInModels`), does a literal-or-regex rewrite over one model's `qfmt`/`afmt`/shared `css` (selected by `front`/`back`/`css`), returning a count — it touches *template text*, not fields, so it migrates nothing and bumps `col_mod` without a re-embed. Literal mode escapes the pattern and inserts verbatim (no `\1`); `match_case` defaults **true** (template/CSS is code).

The tag tools, deck tools, `find_replace_note_types`, **and** `update_note_type_field_metadata` are a derived-index-aware special case: tags, deck names, template/CSS text, and per-field editor metadata (font/size/description) are **not** part of a note's embedding text, so these ops leave every vector valid but bump `col.mod`. Each advances the stored `index.col_mod` (and requests a debounced save) **without** re-embedding — via the shared `_bump_col_mod_after_metadata_change` helper in `tools.py` — so a metadata-only change doesn't force a spurious full rebuild on next startup. Full-replace of tags lives in both `update_note_tags` (`set`) and incidentally in `upsert_notes` (`{id, tags}`); the additive/subtractive logic lives only in `update_note_tags`. `upsert_decks` mirrors `upsert_notes` (id = rename, absent = create); **decks never merge** — renaming onto an existing deck name is an error, and `delete_decks` is **empty-only** (move notes out first), so deck deletion can never delete a note.

**Collection maintenance is one tool, `collection_prune`, not scattered cleanups.** It runs any of four cleanups — clear **unused tags** (`tags.clear_unused_tags`), remove **empty notes**, remove **empty cards** (`get_empty_cards` → `remove_cards_and_orphaned_notes`), trash **unused media** (`col.media.check().unused` → `trash_files`); selecting none runs all. It **applies by default**: `dry_run` defaults **false** (unified with `find_replace_notes`/`migrate_note_type` in #686 — the CLI previews, confirms, then applies unless `--dry-run`, with `--yes` to skip the prompt; pass `dry_run: true` to the MCP tool to preview). This reverses an earlier preview-by-default safety choice (see `docs/decisions.md`); the destructiveness is now guarded by the CLI confirm gate rather than the default. An **empty note** is one whose every field is blank by `embed_text.field_is_blank` — no text *and* no media (`<img|audio|video|object|embed|source>`/`[sound:]`), so an image- or audio-only note is never deleted. Index handling is **mixed** (so it's not a pure metadata-bump op): empty-note/empty-card removal **delete notes**, so their vectors leave via `index.remove` like `delete_notes`; clearing tags and trashing media are vectors-unchanged. On apply the order is empty notes → empty cards → unused tags → unused media (so tags/media freed by the deletions get cleared in the same call); the dry-run previews each independently (an apply may clear a few more). Logic in `CollectionWrapper._prune`/`_find_empty_notes`/`_find_unused_media`; the CLI `collection` group (`cli/collection_cmd.py`) also houses `collection query`/`check`/`reload`. The read-only sibling **`collection_check`** wraps `col.media.check()` to report `unused`/`missing`/`missing_media_notes`/`have_trash` without mutating.

**Media files are first-class, via five tools, and orthogonal to the vector index.** `store_media`/`fetch_media`/`list_media`/`delete_media` wrap `col.media`; media ops **never touch `col.mod` or embedding text**, so they need *no* `index.col_mod` bump. The authoring write path `store_media` is a bulk (1-10) per-item batch where each item is base64 `data` (requires a `filename` with extension — bytes are opaque) **or** a `url` the server fetches; one item failing doesn't sink the batch (per-item `StoreMediaOk`/`StoreMediaError` union).

A server-local **`path`** source stores a file zero-copy via `col.media.add_file`, **off by default**, honored only when **all three** hold: (1) the operator set one or more **`--media-path-root DIR`** (repeatable; config `server.media_path_roots`, env `SHRIKE_MEDIA_PATH_ROOTS` `os.pathsep`-separated); (2) the server is **purely-local** (`_server_is_purely_local`: loopback bind, no `--allow-remote`, guard on, no added `--allowed-host`/`--allowed-origin`); (3) the path is **contained in one of those roots** after `..`/symlink resolution. The two gates **compose** — purely-local stops a remote/proxied caller *reaching* it, the roots bound *what* a permitted caller may read — so roots set on a non-purely-local server are refused (warn), not half-enabled. `main()` validates **each** root once at startup (`_validate_media_path_root`: realpath, reject the filesystem root via `dirname(p)==p`, require an existing dir — per element, since the disjunction means the weakest root governs), passing the list into `register_tools` only when purely-local; the containment check is `os.path.commonpath([root, realpath(src)]) == root` over the roots (`_path_within_any_root` — `commonpath` not `startswith`, on `realpath`'d sides, so `..`/symlink-escape/prefix-bug are closed; a check→`add_file` TOCTOU is an accepted local-trust residual). A blocked `path` is a per-item `StoreMediaError`, not a whole-batch rejection. It's a **process-level capability gate, no per-request peer check** (redundant under the gate, would couple to MCP-SDK internals). The CLI offers `shrike collection media store PATH` (reads + uploads bytes, any daemon) and `--server-path` (server reads it, requires a configured root). Within a root `path` is an **arbitrary read of those files at the server user's privileges** — a deliberate, documented part of the unauthenticated-loopback trust model for single-user/local use; narrow roots keep the blast radius small. Anki resolves collisions (identical → same name, `deduped`; different → hashed suffix), so callers must use the *returned* filename.

**URL fetch is SSRF-guarded** (`_fetch_media_url`): http/https only; the host is resolved and refused unless every address is **globally routable** (`ipaddress.is_global` — an *allowlist*, so it also rejects CGN 100.64/10, `192.0.0.0/24`, benchmarking ranges a denylist misses, plus loopback/RFC1918/link-local/metadata/reserved/multicast) — unless `--allow-private-media-fetch` / `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH` / config `server.media_allow_private_fetch`. **Redirects are followed manually, re-running the guard on every hop's host** (capped at `MAX_MEDIA_REDIRECTS`), because httpx's `follow_redirects` would jump to an attacker-chosen private/metadata address unchecked. **The connection is pinned to the vetted IP:** `_resolve_public_ip` returns one validated address, the request URL's host is that IP (httpx connects there and never re-resolves the name at connect time — closing the DNS-rebinding TOCTOU), the `Host` header carries the original name for routing, and for HTTPS the `sni_hostname` extension carries it so TLS SNI + cert validation verify against the name, not the IP. Items are prepared **concurrently and off the worker thread** (`asyncio.gather` over `to_thread`; the size cap applies to the *encoded* length before decoding); honors httpx proxy env (SOCKS via optional `httpx[socks]`); capped at `MEDIA_MAX_BYTES`. (`--allow-private-media-fetch` turns off both the guard and the pinning.)

`fetch_media` **never returns bytes** — base64 in a tool response is useless to a model (it can't render it) and wrecks context. Each present file returns as `MediaFile` (`status: "found"`) carrying a **`url`** (the server's `GET /media/<name>`) + server-side `path`; a missing file is `MediaMissing`. There is **no inline/base64 option** — the only way to bytes is the url (or path). The `url` is built in the tool layer (`_media_url` from `media_base_url`); `media_base_url` defaults to the bind host, overridable for reverse-proxy via `--public-url` / `SHRIKE_PUBLIC_URL` / config `server.public_url`. The **`GET /media/{filename}`** custom route (behind the same `_guard` check, `FileResponse`) serves the bytes and resolves the media dir **lock-free and absolutized** via `CollectionWrapper.media_dir`. The standalone client has `ShrikeClient.read_media(name) -> bytes`. `list_media` defaults to 100 files. fetch/delete and the route sanitize filenames to a basename inside the media dir (`_safe_media_name`, traversal guard). Logic in `CollectionWrapper.store_media/fetch_media/list_media/delete_media/media_check`; CLI in `cli/media_cmd.py`.

**Three retrieval surfaces, by intent.** `list_notes` is structured filters (deck/tags/type/ids/date, ANDed); `search_notes` is meaning + exact text (semantic + substring, annotated), **fused by Reciprocal Rank Fusion** (`rrf_fuse`, native in `shrike_kernel::fusion`, with `search_fusion.py` as the frozen Python reference spec the parity suite pins): each signal ranks its own candidates, a note's fused score is `Σ w_s·1/(k+rank_s)`, a missing signal contributes nothing (graceful degradation), ordering stable across queries; an **exact-match override** tiers literal hits above the rest (the one place RRF's blindness to magnitude is wrong). The **per-modality semantic rankers**: `index.search_by_modality` ranks notes separately per modality, so `text` and `image` enter the fusion as **distinct RRF signals** alongside `exact` — a rank-based combiner makes the CLIP modality gap's constant cosine offset invisible, so a text query that matches a card's *image* surfaces it (the payoff a single deduped cosine ranking couldn't deliver). The `image` ranking is **not** subject to the text-calibrated cosine `threshold` (meaningless across the gap); instead an offline-calibrated intra-modal **activation gate** floors it — a non-text modality joins the fusion only when its best match clears `mean + ACTIVATION_MARGIN·std` of that modality's *typical* best match, estimated by sampling stored text vectors as pseudo-queries (`index._calibrate_activation`, self-matches excluded; stored in `index.meta.json` under `activation`, surfaced in `server status`). So an off-topic query no longer injects weak image cards; text-only collections are unaffected (no image sub-index → nothing to calibrate). The same signal-agnostic seam is what the **`fuzzy` signal** plugs into by just "producing a ranking": it's the **derived-text store's** trigram/typo ranking (`DerivedTextStore.search_fuzzy`), weighted **below** the rest (`SEARCH_WEIGHTS["fuzzy"]=0.5` — a near-miss is weaker evidence), surfacing near-misses an exact search misses (`protien` → `protein`). The **same store feeds the substring (`exact`) candidates** — `search_notes` prefers `DerivedTextStore.search_substring` (a fast FTS5 trigram pre-filter) over the linear `find_notes` scan, falling back to `find_notes` when the store is unavailable/unbuilt or the query is sub-trigram (`<3` chars); either way `substring_info` stays the **authority** that confirms + annotates each candidate, so the exact tier is unchanged. Both lexical hits carry **source-aware provenance** (`SubstringInfo`/`FuzzyMatch` `.source`/`.ref`), today always `source="field"` but seamed for `ocr`/`asr`. `collection_query` is the **raw Anki escape hatch** — the string goes straight to `col.find_notes`, so the full expression language works (`is:due`, `prop:`, `added:`, `rated:`, `flag:`, `nid:`/`cid:`, `OR`/`-`/brackets). It is **read-only** (only *finds* notes — `is:due`/`rated:` filters perform no review or scheduling, so the full grammar is allowed without a whitelist). It reuses `list_notes`' serialization and `ListNotesResponse`; a malformed expression raises `anki.errors.SearchError` → `ToolInputError` (stripping Anki's U+2068/U+2069 isolation marks). Lives in `CollectionWrapper._query`.

`find_replace_notes` edits **field bodies**, which *are* embedding text, so it re-embeds the changed notes via the `upsert_notes` index path (not the col_mod-only helper). The edit runs Anki's `col.find_and_replace` (Rust regex, undo-able); the changed set is detected by diffing `notes.flds` before/after (note `mod` is only second-resolution, so a same-second edit shows no bump) and exactly those are re-embedded. The dry-run preview is computed in Python (`apply_replacement`): exact for literal, illustrative for regex. A scope (`deck`/`tags`/`note_type`/`ids`) is required.

**Changing a note's note type is `migrate_note_type`, a dedicated tool — `upsert_notes` hard-refuses a type change.** It wraps Anki's `col.models.change(source, nids, target, fmap, cmap)`: a history-safe migration (note IDs preserved; card scheduling carried across mapped templates). The user-facing `field_map`/`template_map` are by **name**; `_migrate_note_type` translates them to the ordinal `fmap`/`cmap` Anki wants. It's **data-affecting and explicit**: `field_map` is required and non-empty, a source field not in it is *dropped* (reported in `dropped_fields`, content lost), target fields nothing maps into are reported (`new_empty_fields`), and unknown names / two-sources-to-one-target / mixed source types / target==source all raise `ValueError` → `ToolInputError` (no guessing). All `note_ids` must share one source type. Like `find_replace_notes` it edits embedding text, so on apply it re-embeds the changed notes (IDs unchanged). Applies by default with a `dry_run` preview.

Every tool request/response shape — plus the server-status shapes — is a Pydantic model in `shrike/schemas.py` (the binding; `shrike-schemas` in Rust is canonical). Tool functions in `tools.py` return the response models, so FastMCP emits an `outputSchema` per tool, and `_safe_tool` runs each docstring through `inspect.cleandoc`. The standalone `ShrikeClient` exposes a typed per-tool method for each (`list_notes(...) -> ListNotesResponse`) that validates the wire response into the model; `ShrikeClient._call()` is the untyped escape hatch. There is no checked-in schema file: the authoritative machine schema is whatever the running server advertises via `tools/list`. `docs/mcp-tools.md` is the human-readable companion.

**Make illegal states unrepresentable — the schemas.py house style.** When a field's presence is *correlated* with another (a hidden state), model it as a **discriminated union**, never a bag of optionals. The pattern is an `Annotated` type alias: each variant is a `BaseModel` with a `Literal` discriminator field, and `Thing = Annotated[A | B, Field(discriminator="status")]`. Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field* typed as `Thing` validates automatically). Examples: per-item results (`UpsertNoteResult = UpsertNoteOk | UpsertNoteError` — success has `id`+`neighbors`, error has `index`+`error`), `IndexStatus` (`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored` — `progress` only on building, `error` only on errored), and the `/index/rebuild` + `/embedding/*` endpoint responses (unions on `status`). Two fields that always appear or vanish *as a pair* are the same smell at smaller scale — group them into a nested sub-model (`NoteTypeInfo.detail: NoteTypeDetail | None`), not two optionals. A bare `X | None` is reserved for *genuinely independent* optionality (a datum absent on its own — `col_mod` before the index is built, a field omitted from a partial update, a caller-selected `collection_info` section); annotate why so it reads as deliberate. Response models carry **no `error` field**: a whole-call failure (bad input, unhandled exception) is raised in the tool and surfaces as an MCP `isError` result, which `ShrikeClient._call` turns into a `ServerError`. Expected bad input raises `ToolInputError` (logged without a traceback); genuine bugs log with one. The only optional advisory on a success response is `message` (e.g. index-building notice, neighbor-retry hint).

Input bounds (e.g. `limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as `Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so FastMCP **rejects** out-of-range input rather than silently clamping. Optional list filters use `Field(default_factory=list)` (keyword-only params) so they render as a plain array in the schema, not a noisy `anyOf:[array, null]`.

### CLI structure

The CLI uses Click with rich for output formatting. Command hierarchy (the
top-level groups and per-group subcommand order are the §G canonical order from
epic #682, applied via `OrderedGroup.list_commands` in `cli/groups.py` — it
drives both `--help` and shell completions):

```
shrike [--config PATH] [--url URL] [--profile/--collection NAME] [--json] [--pretty/--no-pretty]
├── collection info|check|export|import|prune|reload
│   ├── tag rename
│   └── media store|fetch|list|delete
├── search <query>|query|coverage        # default-command group: `search <query>` searches
├── server start|stop|status|logs
│   ├── embedding start|stop|status
│   └── index rebuild|save|status
├── note create|update|delete|list|show|tag|replace|migrate-type
├── deck create|rename|delete
├── type create|update|delete|list|show
├── profile add|remove|default|list
└── completion {bash,zsh,fish}
```

The command-surface rehome (#683) moved each top-level command under its natural
parent: `note search` → `search`, `collection query` → `search query`, `info`/
`export`/`import`/`tag rename`/`media` → under `collection`, `embedding`/`index`
→ under `server` — a clean break (the old top-level commands are removed and
error). `search` is a default-command group (`SearchGroup` in `cli/groups.py`):
its `parse_args` injects the hidden default `search` command when the first token
isn't a known subcommand (so `search <query>` and `search --similar-to N` search,
while `search query`/`search coverage` dispatch).

The CLI talks to the MCP server over HTTP — it can target a remote server via `--url` or `SHRIKE_URL`.

`--json` and `--pretty/--no-pretty` are global options but also accepted on every leaf command (via the `@output_options` decorator in `output.py`), so both `shrike --json collection info` and `shrike collection info --json` work. `--json` implies `--no-pretty`; combining `--json --pretty` is an error.

**Identifier resolution** — `type show`/`update`/`delete` accept a name or numeric ID; note commands accept IDs with an optional `#` prefix (`note show #123`; `NoteIDType` in `output.py` strips it). `note show` is sugar for `note list --ids ID`; `type show` for `type list IDENTIFIER`. **Decks** are referenceable by name, numeric ID, or `#id` wherever a deck is *taken* — `#id` is always an ID, a bare number is tried as an ID then falls back to a literal name. Resolution is server-side in `CollectionWrapper._resolve_deck_ref`; the CLI passes refs through untouched except `deck rename`, which resolves to an ID client-side (`_match_deck` in `deck_cmd.py`).

**Output conventions** (styling lives in `output.py`; gh-CLI-like): cyan names/paths/URLs, green `#ID`, yellow tags, dim labels; flush-left borderless tables with dim underlined headers; Rich `Panel` detail views; `+`/`~`/`!` (green/yellow/red) for created/updated/errors; `output.spinner()` is a no-op under `--no-pretty`. Validation errors use `click.UsageError` (shows usage line); runtime errors use `click.ClickException`.

### Daemon management

`shrike server start` spawns the server as a background process (lifecycle in `shrike/daemon.py`).

**Liveness detection** uses file locks via `filelock` (fcntl/msvcrt): the server holds an exclusive lock on `server.lock` for its lifetime, the OS releases it on exit, and clients probe by a non-blocking acquisition — avoiding PID-recycling issues entirely.

**Shutdown** is cross-platform via `POST /shutdown`. The CLI's `stop_server()` escalates: (1) `POST /shutdown` (clean, all platforms) → (2) SIGTERM (Unix, if HTTP is unresponsive) → (3) SIGKILL/TerminateProcess (hung). Signal handlers (SIGTERM, SIGINT) remain a secondary path for `kill` and Ctrl+C in foreground.

**HTTP endpoints** beyond MCP (custom routes, all behind the `_guard` Host/Origin check):
- `GET /status` — JSON: pid, url, collection, log_level/dir, uptime, embedding, index, recognition, and the collection-lock state (`locking`: permanent/cooperative, `collection_held`: bool). Backs `shrike server status` + auto-start health checks.
- `GET /media/{filename}` — streams a media file (`FileResponse`); read-only, filename basename-sanitized, media dir resolved lock-free (`wrapper.media_dir`); 404 for a missing/escaping name.
- `POST /shutdown` — graceful shutdown.
- `POST /index/rebuild` — full rebuild (returns immediately with status/progress); requires the embedding service running.
- `POST /index/save` — immediate flush off the event loop; returns `saved` (with `size`/`pending`), `empty`, or `building` (refused mid-rebuild). Backs `shrike server index save`.
- `POST /embedding/start` — starts the embedding service (JSON body overrides model/port/etc., else the boot params); attaches to the index and rebuilds if the model changed or the index drifted. Returns `started`/`already_running`/400 if no model configured.
- `POST /embedding/stop` — saves the index, stops the service, marks the index `unavailable` (server + collection stay up).
- `POST /reload` — closes and re-opens the collection (picks up a restored backup / file-level sync-swap) and re-checks drift, rebuilding in background if `col.mod` moved. Returns `{status: "reloaded", col_mod, rebuilding}`. Backs `shrike collection reload`. The reopen primitive (`CollectionWrapper.reopen`/`_do_reopen` + reading `self.col` at execution time in `run`/`run_sync`) is shared with the cooperative-lock open-on-demand lifecycle. A control endpoint, not an MCP tool.

State files live in the platform state directory (`shrike/paths.py`): `server.lock` (exclusive lock), `server.pid` (diagnostics only, not liveness), `server.json` (URL, port, collection path, start time, log dir).

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

YAML at the platform config directory (`config.yml`). **User-managed: `shrike server start` never writes it unless `--save-config` is passed**; with it, start persists the resolved *operational* flags; without it, start is a no-write operation reflecting exactly the flags it was given.

**The capability sections are config-file-only** (see `docs/distribution.md`). `embedders:` (vector-space entries: `modalities` + `runtime: onnx|remote`, with `endpoint`/`api_key_env`/`pooling`/`providers`/`batch_size` per entry), `recognizers:` (keyed `ocr`/`asr`/`describe`), and `managed:` (`llama_server` with `manage: auto|attach|off`; `sync_server`) have **no flag or env spelling** — structured config has one home. `shrike/profiles.py` is the model: `parse_capabilities` (parse/validate; inapplicable knobs are structural errors), `resolve_profile` (intersect with `shrike_native.build_features()` — an uncompiled runtime or unwired capability is a config error naming its issue), `plan_to_runtime_params` (the bridge onto `EmbeddingRuntime`'s param shape). The CLI validates at spec-build time and passes `--config` to the daemon, which re-resolves the file itself (structured entries can't ride flags); under a v2 config the legacy flags are rejected and the legacy env twins warn as ignored.

**The legacy cascade survives one release as warn-and-map (#523 is the removal).** A config *without* v2 sections runs the old resolution: config defaults → config values → env vars (`SHRIKE_URL`, `SHRIKE_COLLECTION`, the `SHRIKE_EMBEDDING_*` family, `LLAMA_SERVER_PATH`, `SHRIKE_CACHE_DIR`, `SHRIKE_INDEX_SAVE_DELAY`, `SHRIKE_INDEX_SAVE_THRESHOLD`, `SHRIKE_COOPERATIVE_LOCK`, `SHRIKE_LOCK_HOLD_SECONDS`, `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH`, `SHRIKE_PUBLIC_URL`, `SHRIKE_MEDIA_PATH_ROOTS`) → CLI flags, with a legacy `embedding:`/`recognition:` section warn-mapped onto the v2 model (`_migrate_legacy`). Operational settings (cache dir, index flush tuning, transport, locking) keep the cascade permanently via `config.resolve_cache_dir()` / `resolve_index_save()` / `resolve_transport()` / `resolve_locking()`. `save_config` persists `collection`, `cache_dir`, non-default `server.*`, legacy `embedding.*` (legacy setups only), and any set `index.save_delay` / `index.save_threshold`; **logging overrides are read from config but never written by `--save-config`** — set `logging.level` / `logging.dir` in `config.yml` directly. The `index.*` flush knobs and `cache_dir` resolve to `None` in config = "use the server's built-in defaults" (the kernel saver's 60s/100, the platform cache dir); the numeric defaults live in `shrike.index`, not duplicated in config.

### Embedding service lifecycle

The embedding service can be cycled independently of the Shrike server. `EmbeddingRuntime` (`embedding.py`) owns the current backend (or `None`), the params needed to (re)start it, and the binding to the index; it serializes start/stop under a lock. The `VectorIndex` is **always** created at boot, even with no embedder — it loads on-disk vectors and reports `unavailable` until a backend is attached.

**Pluggable backends behind one protocol, native underneath.** The *operational* surface depends only on the minimal `EmbedderBackend` protocol (`embedding_base.py`): `embed_texts`, `embedding_dim`, `model_fingerprint`, `health`, lifecycle, and `modalities` — but every production backend also hands the kernel a native composition (`native_embedder()`), so `embed_texts` serves direct callers (the probe, tests) while kernel embeds run crate-side end-to-end. The implementations: `LlamaServerBackend` (managed llama-server subprocess; GGUF/MLX), `RemoteBackend` (an endpoint Shrike does not run: a v2 entry's `endpoint` with `api_key_env`, or `managed.llama_server.manage: attach`; start proves connectivity with one embed, never spawns or stops the server), `OnnxBackend` (`embedding_onnx.py`, in-process onnxruntime + tokenizers, text-only), and `ClipBackend` (`embedding_clip.py`, in-process CLIP dual-encoder for image↔text, `modalities={text,image}`). `EmbeddingService` is a back-compat alias of `LlamaServerBackend`. The runtime constructs one by *kind*, driven by the config's `embedders:` entry (the `remote` kind is config-only — no flag spells it; the deprecated `--embedding-backend {llama|onnx|clip}` flag + `SHRIKE_EMBEDDING_*` env twins still select on legacy setups for one release, #523). `_make_backend` imports onnxruntime lazily (a **hard dependency** of the published wheel; a missing install surfaces a clean `ImportError` only when that backend is selected, caught on both `/embedding/start` and boot so it degrades to no-embedding rather than crashing). The kernel's drift/hash/persistence machinery is backend-agnostic — only the embed contract is called. Guidance: **in-process ONNX for text-only collections** (no subprocess/port/orphan-reaping, ~1 ms single-note upserts, small/quantized-friendly); **llama-server** for GGUF/MLX and GPU offload; the **CLIP shape** (a `[text, image]` onnx entry) is the multimodal path — a dual text+vision ONNX encoder + PIL/numpy image preprocessing (no torch) embeds text *and* images into one shared space, so a text query retrieves a card by the content of its image. It reuses `OnnxBackend`'s provider resolution + the batch-safety probe (one text-path probe governs both graphs) and adds `embed_images(...)`. The **multi-vector index** maps a note to its text vector + one vector per image under the `note_id` key; the collection extracts `<img src>` names (`embed_text.extract_image_refs` → `note_embed_inputs`) and the index reads image bytes lazily via an injected resolver. The CLIP **modality gap** (text-text cos ~0.7 vs text-image ~0.3) means a *single* deduped cosine ranking buries image vectors under text ones — so the index is split into **per-modality sub-indexes** (`index.usearch` text, `index.image.usearch` images), each ranked separately and fed to RRF as its own signal (the gap's constant offset is invisible to a rank-based combiner), floored by the activation gate (see *Three retrieval surfaces*).
  - **`modalities` is the graceful-degradation seam.** Every backend advertises a `frozenset[str]` of what it can embed. **Text-only is a permanent, first-class capability** (the suites rely on small text-only models); a multimodal backend just advertises more modalities — search over media-by-content lights up where vectors exist and quietly returns nothing where they don't, never erroring.
  - **`OnnxBackend` specifics:** model is an ONNX dir (`model.onnx` + `tokenizer.json`, or `onnx/model.onnx`) or a `.onnx` file with `tokenizer.json` beside it; pooling (`mean|cls|last`) and optional L2 normalization are done in numpy. Pooling is **vector-affecting → folded into the fingerprint**; normalization is scale-only (USearch `cos` is scale-invariant) → **deliberately not**, mirroring llama's `--embd-normalize`. Fingerprints are namespaced by family (`onnx:…` vs llama's `meta:`/`file:…`), so the same model under two runtimes never shares a vector space. onnxruntime providers via `--embedding-onnx-provider` (repeatable; `SHRIKE_EMBEDDING_ONNX_PROVIDERS`, comma-separated; default CPU) are **resolved gracefully**: intersected with `get_available_providers()`, an unavailable one dropped *with a warning*, CPU always appended as final fallback, and the **actually-loaded** provider surfaced in `health()`/`server status` so a silent CPU fallback is visible. Packaging mirrors onnxruntime's wheels: base onnxruntime (CPU + CoreML on macOS) is a hard dep, the `gpu` extra = `onnxruntime-gpu` installed *instead of* the base carrier (they conflict); DirectML is manual `onnxruntime-directml`. **Batch safety is probed at startup, not assumed:** int8 exports use dynamic quantization whose activation scales are computed over the whole batch tensor, so a *batched* embed makes a note's vector depend on its batch-mates' content — breaking the `reconcile`==full-rebuild invariant. fp32/fp16 ONNX (and llama-server, fp) are bit-exact batched, so the variance is int8-only. Rather than guess, **every backend's `start()` runs `embed_batching.probe_max_safe_batch`** — embed a magnitude-spiked probe set serially vs all in one batch, within a tolerance above float noise and below quant drift — and `embed_texts` then batches **up to the proven probe-set size** (further capped by `--embedding-batch-size`) or **serially** (batch size 1) for a batch-variant model, so "proven safe" and "what we batch" are the same size. The probe set is spiked for activation magnitude (long/numeric/symbol/repeated/mixed-script), since int8 drift is magnitude- not length-driven. Locked by exact-equality (`np.array_equal`) tests against *real* int8 (serial) and fp32 (batched==serial) models (`tests/integration/test_backends.py`, `test_onnx_models.py`).

- `shrike server start` starts embedding at boot if a model is configured, unless `--no-embedding` is passed (lets a server run with embedding deliberately off).
- `shrike server embedding start` / `shrike server embedding stop` cycle the service on a running server (for llama-server upgrades, model swaps, freeing GPU/RAM). Stopping marks the index `unavailable` but keeps the on-disk vectors; starting re-attaches and rebuilds only if needed.
- **Pooling type:** `pooling: {mean|last|cls|none}` on the embedder entry (legacy: `--embedding-pooling`/`SHRIKE_EMBEDDING_POOLING`) is passed to llama-server as `--pooling`. Allowed only where Shrike launches the server producing the vectors (onnx entries; remote under `manage: auto`) — an external/attached endpoint owns its own pooling, and declaring it there is a ProfileError. Unset = "use the model's GGUF default" (fine for BERT-family models carrying mean pooling in metadata). **Last-token models (Jina v5, Qwen3-Embedding) need `--embedding-pooling last`**: their pooling type isn't in the GGUF metadata, so without it llama-server defaults to mean and produces wrong embeddings (and may need a newer llama.cpp than the pinned `LLAMA_TAG` — see `scripts/llama-server.lock`). Pooling is folded into `model_id` so changing it forces a rebuild.
- **Generic arg passthrough:** `managed.llama_server.args` (raw token strings; legacy: `--embedding-arg` repeatable / `embedding.extra_args` / `SHRIKE_EMBEDDING_ARGS` as one shlex string) appends raw tokens for the long tail of **runtime-only** flags (`--flash-attn`, `--ubatch-size`, gpu split). Each entry is `shlex`-split and appended last. Two guardrails: (1) Shrike-owned flags (`--model`/`-m`/`--host`/`--port`/`--embeddings`/`--embedding` + value token) are stripped with a warning — `--host` especially, since **llama-server is pinned to loopback**; (2) the effective passthrough is folded into `model_id`, so **any** change forces a rebuild (Shrike can't tell a vector-affecting flag from a perf-only one in an opaque bag). **Vector-affecting flags must be typed settings** (like pooling), not buried here. Normalization is *not* such a setting: USearch's `cos` is scale-invariant, so `--embd-normalize` is moot.
- Starting llama-server blocks (model load + health wait), so the HTTP handler runs it via `asyncio.to_thread` to keep the event loop responsive.
- **Orphan reaping:** the native lifecycle manager (`shrike-llama-server`) records the llama-server PID in `<state-dir>/embedding.pid` (written after spawn, removed on clean stop). If Shrike is hard-killed, llama-server is orphaned and keeps holding its port; on the next `start()` the reap detects a recorded PID still alive *and* holding the port and terminates it (SIGTERM→SIGKILL) before binding — **both** signals required, so a recycled PID can't kill an unrelated process. **`PR_SET_PDEATHSIG` is intentionally avoided**: the parent-death signal keys on the spawning *thread*, and start runs under `asyncio.to_thread`, so a reclaimed pool thread could kill a live server.

### Recognition (OCR)

**Recognition is the kernel's second injected capability** (the slot pattern, sibling of the embed slot): an OCR engine the harness attaches at assembly turns note media into searchable text. Off by default. **The server build no longer compiles the Apple Vision engine** (platform engines — `engine-apple` — are mobile-only on every OS; the server's replacement is the remote recognizer rows): `--ocr-backend apple` (config `recognition.ocr`, env `SHRIKE_OCR_BACKEND`) degrades the recognition state to `error` without disturbing boot, like the off-macOS case always did. The engine itself (`shrike-recognize-apple`, in mobile/`engine-apple` builds) is native; the platform glue is **Swift bolted behind Rust** (`swift/Recognize.swift` exports a 3-function C ABI driving Apple's Swift-only `RecognizeTextRequest` API, macOS 15+; Vision + the Swift runtime ship with the OS, but **building it on macOS needs full Xcode** — swiftc/the bazel genrule invoke via xcrun; the server build no longer pays that). Fingerprint `apple-vision-swift:{revision}:macos{X.Y.Z}`; off-macOS an unavailable stub. The Python contract is `RecognizerBackend` (`recognition.py`): a *blocking* `recognize(items: list[bytes]) -> list[tuple[str, float, str]]` (text, confidence, segments-JSON) plus `model_fingerprint()`; `PyRecognizer.capture` bridges it to the kernel (the custom/test seam; blocking calls ride the kernel runtime's blocking pool via `spawn_blocking` + GIL attach).

**One pass, many consumers** (the load-bearing rule): the kernel's `recognize_pending(max_items)` sweeps bounded batches of pending (note, image) pairs — pending = a resolvable image with no OCR row *and no below-gate marker* (a gate-dropped item is recorded in the derived store's `gated` table, so it's judged once, not re-OCR'd every sweep), or everything after the recognizer *fingerprint* changes (an OS upgrade re-derives rows AND markers, like a model change rebuilds vectors) — and persists BOTH the flattened text (derived rows, `source='ocr'` → substring/fuzzy search + provenance) and the per-segment structure (the `segments` table; boxes today). Gating (`RecognitionGate` kernel-side): confidence + substance to store at all, a higher substance bar to mint a vector. Gated text embeds via the TEXT encoder as extra vectors under the note key in the `text` space (no modality gap; max-over-items ranking falls out), and the per-note fingerprint folds the OCR text — byte-identical with none, so upgrades never spuriously rebuild. The derived store's drift rebuild is **field-source-scoped** (schema v2): a boot drift never discards recognition rows. The harness drives sweeps in the background (`recognition_sweep`, one batch per executor occupancy); `/status` carries `recognition: {state, backend}`.

### Vector index and consistency

The vector index is a **derived cache**, not a co-equal store. The Anki collection (SQLite) is always the source of truth. SQLite handles its own crash recovery via WAL/journal, so the collection is self-consistent after any crash. If the index is stale, corrupt, or missing, it can be rebuilt from the collection by re-embedding all notes.

**Consistency model:** the index may lag behind the collection (notes added/modified/deleted without the index being updated), but the collection never lags behind the index. This means search results may be stale, but data operations (upsert, delete, list) are always correct.

**Drift detection:** the index metadata (`index.meta.json`) stores `col_mod` (the `col.mod` at last build) and `model_id` (a fingerprint of the embedding model). On startup (and whenever the embedding service is attached), compare both. Match → skip. `model_id` differs (model changed → every vector lives in a different space) → full rebuild. Only `col.mod` differs (Anki GUI, sync, imports) → the index **reconciles incrementally** rather than re-embedding everything: a per-note embedding fingerprint sidecar (`index.hashes.json`, `{note_id: blake2b}`) lets `VectorIndex.reconcile` diff current notes against indexed ones and re-embed only changed ones, add new, drop deleted — **the end state is identical to a full rebuild** (verified), but a drift touching a handful re-embeds a handful (~87× on a 1K-note index with 5 edits). The fingerprint is **media-aware**: for an image-capable backend it folds in the sorted filenames of a note's images **that actually resolve** (`_note_hash`), so adding/removing/swapping — or *later storing* — an image re-embeds it, while a referenced-but-never-stored image stays out of the hash (no re-embed loop); for a text-only backend it's the text hash, *byte-identical to the pre-media scheme* (text-only users pay no spurious rebuild on upgrade). The fingerprint hashes *names of present images* via a cheap presence `stat` (no byte read); image bytes are read only for a note actually being embedded, lazily and lock-free via the index's image resolver. Folding in only *resolvable* names is what keeps reconcile == a full rebuild even for a note authored before its media landed. A `col.mod` bump from a non-embedding edit finds no fingerprint changes and just advances the watermark. `reconcile` **falls back to a full `rebuild`** when the model changed, there's no prior hash state (first build, or an index predating the sidecar), or an image-capable backend meets a **pre-split (v1) single-index layout** that can't be split into per-modality sub-indexes (detected by the `schema` marker; a text-only v1 index loads losslessly as the v2 text sub-index, so text-only users never rebuild on upgrade). The hash map is maintained incrementally by `add`/`remove` (so Shrike's own upserts don't re-embed on next reconcile) and persisted alongside the index; a missing/corrupt sidecar safely degrades to a full rebuild. **Explicit** rebuilds (`shrike server index rebuild`, `POST /index/rebuild`) stay full — reconcile is only the automatic drift path (`_maybe_rebuild`).

**Model fingerprint:** `model_id` comes from llama-server's `GET /v1/models` `meta` block (`n_params`, `n_embd`, `n_vocab`, `n_ctx_train`, `size`) — fast, describes the *loaded* model; falls back to model filename + on-disk size if absent. The model *name* is deliberately excluded (it would force needless rebuilds on rename and miss same-name re-quantizations the numeric fields catch). An explicitly-set pooling type is appended (`…:pool=last`) since it changes every vector but isn't in the metadata; omitted when unset so older indexes still match. The note-text normalization version (`…:textprep=N`, `EMBED_TEXT_VERSION` in `embed_text.py`) is appended **unconditionally**: the cleaned text is as much a part of the vector space as the model, so changing how notes are rendered for embedding must invalidate the index (unlike pooling, never omitted — an index built under the prior scheme *should* rebuild). `EmbeddingService.embed()` also pins `"model": <id>` in the request body against a future external multi-model endpoint.

**Note text for embedding:** `embed_text.normalize_for_embedding()` turns each raw Anki field value into stable plain text. It operates on field *values*, not rendered cards — a note (not a card) is the embedding unit; templates add presentational scaffolding (`{{FrontSide}}`, `<hr id=answer>`, the hidden cloze `[...]`) that is noise. The HTML→text + entity step **delegates to Anki's own `strip_html`** (Rust-backed, robust on malformed markup, leaves an encoded `&lt;tag&gt;` as literal `<tag>`); around it we reveal cloze (`{{c1::France}}` → `France`, hint dropped), drop MathJax/LaTeX wrappers keeping the inner source (`\(…\)`, `$$…$$`, `[latex]…[/latex]`), drop `[sound:…]`, and convert block tags to spaces *before* the stripper (which otherwise glues `a<br>b` → `ab`). The module lazily `set_lang("en")`s once so the stripper works headless. The result is a function of the field value + the pinned Anki version's stripper — identical whether freshly upserted or re-read during a rebuild, independent of which card a cloze generates. Both embed call sites (`upsert_notes`, rebuild) route through `CollectionWrapper.note_texts()`, so consistency is structural. **Bump `EMBED_TEXT_VERSION` whenever the normalized output changes** — including an Anki upgrade whose stripping differs.

**Implementation:**

1. **Startup check** — compare `col.mod` against the stored value. Match → load. `col.mod` mismatch → incremental reconcile in a background thread; missing/corrupt index or model change → full rebuild. The server accepts requests immediately; `search_notes` returns actionable status ("building 2847/5000 notes, try again shortly") until ready.
2. **Incremental updates** — `upsert_notes`/`delete_notes` update the index in the same call (`index.add()`/`remove()`) and advance stored `col_mod`. An index update failure logs a warning but doesn't fail the tool call — the next startup's `col.mod` mismatch rebuilds.
3. **Persistence** — saved on graceful shutdown (signal handler and `POST /shutdown`), at the end of a rebuild, and via a **debounced flush** during normal operation. The kernel's `DebouncedSaver` (`native/shrike-kernel/src/index_orchestrator.rs`) writes either **`save_delay` seconds after the last change** (idle debounce, default 60s) **or immediately once `save_threshold` unsaved changes accumulate** (burst cap, default 100), whichever comes first — riding the kernel runtime's `tokio::time` timers + blocking pool, driven by edit activity (no polling, no asyncio). This bounds what a hard kill discards: once a flush lands and the server idles, the on-disk index is current and reloads without a rebuild; edits since the last flush trigger a `col.mod`-mismatch rebuild (correct either way, at the cost of a re-embed). Configurable via config `index.*` / env / `--index-save-*`; cache location is `cache_dir`/`SHRIKE_CACHE_DIR`/`--cache-dir`. (Tombstone compaction is unnecessary on the pinned USearch — see the index code comments.)
4. **Full rebuild** — `shrike server index rebuild` / `POST /index/rebuild`: drops the index and re-embeds all notes; progress via CLI + `/status`.
5. **State machine** — `ready`, `building` (with progress), `unavailable` (embedding service not running), `error` (build failed). Exposed via `/status`, `search_notes` responses, and `shrike server status`.

Both reconcile and full rebuild run in a background thread, so the server is never blocked. A full rebuild is seconds for ~1K notes, minutes for 10K+.

### Derived-text store

**Derived/computed data wants one home — a sidecar SQLite `shrike.db` in `cache_dir()`, separate from Anki's synced collection.** `DerivedTextStore` (`derived.py`) is that store; its first artifact is an **FTS5 trigram index** over note text, backing the `search_notes` substring (`exact`) candidates and the **`fuzzy`** RRF signal. It's the natural relay sync target, so it's designed for more than the trigram index from day one.

**Source-seamed.** Every indexed row is keyed `(note_id, source, ref)` — `source` is *where* the text came from, `ref` the field name or media filename. Today the only source is `field` (raw field text, via `CollectionWrapper.derived_field_rows`/`note_field_map`); the seam is for `ocr`/`asr` recognized text (→ trigram **and** the text-embedding space, provenance-tagged so a result can say "matched the OCR text of diagram.png — here's the window"). **A future VLM image-describe goes to the embedding space only, never the trigram index** — a literal-search hit on invisible VLM metadata can't be cleanly explained to a user. A row stores the text (so FTS5 `snippet()` gives the window); a two-table layout (`idx` FTS5 + `rowmap(rowid → note_id, source, ref)` with an index on `note_id`) keeps incremental delete-by-note cheap (FTS5 has no secondary indexes).

**Why a sidecar, not tables in `collection.anki2`** (settled, `docs/decisions.md`): Anki's sync, "Check Database", media check, and version-upgrade migrations own that schema — foreign tables get dropped or error, and it would ship rebuildable derived data over sync. The community norm is add-ons keep their own files; a sidecar in our cache dir is safe and the correct home for derived/rebuildable data.

**Same derived-cache contract as `VectorIndex`, with two deliberate divergences.** It's rebuildable from the collection, detects `col.mod` drift, updates incrementally on upsert/delete (`ingest`/`remove` alongside `index.add`/`remove` at the five tool hook sites — guarded on `derived.available`, independent of the vector index so it works with embeddings off), and surfaces a `ready/building/unavailable/error` state. The boot/`/reload`/cooperative-reacquire paths build it in the background on drift (cheap text-only, no models). Divergences: (1) **no debounced saver** — persistence is inherent to the SQLite file (transactional/durable); incremental writes just advance the stored `col_mod` watermark so the next boot sees no drift. (2) **Graceful absence is first-class**: if the runtime's SQLite lacks FTS5/the trigram tokenizer (probed at construction), the store reports `unavailable` and every lookup signals the caller to fall back to the linear `find_notes` scan; likewise a `<3`-char query (FTS5 trigram can't match it) returns `None` → fallback. All MATCH expressions are FTS5-quoted (`_fts_quote`) so query punctuation can't be parsed as FTS5 syntax (**injection-safe**). Metadata-only edits advance the store's `col_mod` *without* re-ingesting, via the shared `_bump_col_mod_after_metadata_change`, like the index.

## Code style and conventions

- **Type annotations** on all functions (enforced by mypy with `disallow_untyped_defs`)
- **Ruff** for linting (rules: E, F, W, I, UP, B, SIM) and formatting, line length 100
- **Error handling:** batch operations (upsert_notes, upsert_note_types) use per-item try/except so one failure doesn't block the batch. Results include `status: "created"|"updated"|"error"` per item.
- **`raise ... from err`** in except blocks (enforced by ruff B904)
- **`contextlib.suppress`** instead of bare `try/except/pass`
- **`datetime.UTC`** not `timezone.utc` (ruff UP017)

### Performance conventions (the #445 audit's distilled rules)

The full kernel performance audit (issue #445, closed) carries the checkpoint→PR
map and an explicit **"not-worth-fixing" list** — re-finding those is wasted work.
Ground performance decisions at a **100k-note collection**. The failure modes that
recurred, and the house rules that prevent them:

- **No collection reads inside per-item loops.** The N+1 is the repeat
  offender: a singleton `note_dicts`/`note_texts` per candidate pays two SQL
  queries plus a *full* `deck_names`/notetype enumeration each, serialized on
  the collection actor. Discover the id set first, then ONE batched read
  (`read_notes_batch`, `note_dicts(&ids)`, `texts_for_source_for_notes`), and
  assemble from the map. **When porting policy between layers (Python →
  kernel), port its batching with it** — #456 reintroduced the exact N+1 #454
  had removed because the port kept the policy but dropped the batched read
  (caught by re-auditing post-audit PRs; fixed in #476).
- **Read only what the op needs.** Prefer scoped variants over
  full-collection renders: `note_image_refs` (the recognition sweep needs
  image names, not rendered text), the `any_tagged` probe, the notes-scoped
  derived reads. Push a pre-filter into SQL only when its semantics match the
  Rust side *exactly* (`instr(lower(flds), '<img')` — SQLite's `lower()` is
  ASCII-only, identical to the extractor's byte probe).
- **Per-op tails do no O(collection) work.** Derived signals (tag centroids)
  refresh in a coalescing background task behind a cheap relevance probe; the
  op tail only *requests*. Boot/rebuild paths keep synchronous refreshes so
  "ready" means ready.
- **Never hold a lock across file writes or compute.** Snapshot the small
  shared state under the lock, write outside it (`IndexOrchestrator::save`);
  serialize savers with a dedicated guard; blocking fs work rides
  `spawn_blocking`, never an op tail or a runtime worker.
- **One transaction per batch; prepared statements in row loops.** A journal
  commit (fsync) per item is the hidden cost — `ingest_many` and
  `set_note_tags_bulk` (anki's `UpdateNotes` is bulk) batch it away;
  `Connection::execute` re-prepares per call, so loops use `prepare_cached`.
- **Skip provably-identity work — but prove it from the pinned source.** The
  strip-skip (no `<` and no `&` → anki's stripper is byte-identity) was
  verified against anki's own `HTML` regex + `decode_entities` gate and is
  pinned by a panicking-stripper test. A skip predicate justified only
  empirically is a future correctness bug.
- **Hand out views and `Arc`s, not clones; bound unbounded expansions.**
  Arc'd per-notetype field-name lists, `Cow` pass-throughs for the common
  case (`fill_clozes`), move-out assembly instead of re-cloning, per-batch
  lookup memos (`UpsertMemo`), and ceilings with deterministic sampling where
  an input scales with the collection (`MEMBER_SCORE_CEILING`).
- **When an audit completes, re-audit what landed around it.** PRs merged
  outside the audit's snapshot carry un-analyzed code; the post-#445 delta
  pass found the #476 regression and cleared everything else (recorded on
  #445 so it isn't re-found).

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
  pushes to `main`. **Open every PR as a draft** (`gh pr create --draft`); take it
  out of draft (`gh pr ready`) only once it is complete *and* past initial review.
  A draft can't be merged or auto-merged, so the draft discipline is what stops a
  PR landing before it's ready and reviewed (a PR slipped in pre-CI when it didn't,
  #678). What's *structurally* enforced is narrower — CI always runs, so an
  un-tested PR can't merge; the ruleset requires no approvals, so review-before-
  merge rides on this draft convention, not a hard gate. Keep PRs drafts until reviewed.
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

- **PR at each natural checkpoint, as a DRAFT.** Open every PR with
  `gh pr create --draft` (CI runs on it immediately — there is no `ci` gate).
  Don't contort in-progress work to make it mergeable when going a bit further
  lands a larger, coherent section — but don't hoard mergeable work either.
- **Self-review while it's still a draft.** Run a self-check code review against
  the requirements — via a **subagent (prefer the latest Opus model)** — for
  substantive changes (skip it for mechanical PRs). Keep working while the review
  is in flight.
- **Mark ready, then auto-merge, move on.** Once review findings are addressed and
  CI is green, take it out of draft (`gh pr ready`) and set it to merge when green
  (`gh pr merge --auto --squash`) — don't poll; move on. **Never enable auto-merge
  on a draft or before CI has run** (that's how a PR slipped in pre-CI, #678). Add a
  cross-platform label (`rc`, or `macos`/`linux-arm`) when the change warrants that
  coverage.
- **Subagents assist with research, orientation, and review — never
  authorship.** All code and tests are developed by the agent itself; use
  subagents wherever they speed up or improve the work (codebase orientation,
  API research, parallel fact-finding, the self-check review above).

This composes with the gates above: the user-triggered billed passes
(`ultra`/cloud review, security/perf audits) stay user-triggered; the agent's
self-review is the floor, not a replacement, for anything crossing those
thresholds.

### Multi-agent team development — orchestrating a milestone or issue set

When the user points you at a milestone or set of issues to develop in parallel,
you act as **team lead**: decompose, dispatch workers, keep the boundaries intact,
review, drive to a user-gated merge — you orchestrate, you don't implement the
issues yourself. The full playbook is the dedicated team-development skill; the
load-bearing rules:

- **Cap at 3–6 concurrent agents per wave**, sized to the set's *natural*
  parallelism (disjoint surfaces, no inter-dependency). More ready work than the
  cap → run in **waves**, each ending in a joint review + user-gated merge before
  the next; blocked issues wait for the merge that unblocks them.
- **Pre-flight:** read the governing design doc, fix the hard boundaries, build the
  real dependency graph (**verify cross-stream state live with `gh`**), and assign
  **one owner per shared surface + one lockfile owner per wave** so two agents never
  edit the same file.
- **Workers** run in their own worktree under **`bypassPermissions`** (a background
  agent can't answer a permission prompt — without it it silently stalls on the
  first write/network call), branch `‹type›/‹issue#›-‹slug›` off `origin/main`, work
  in slices, run the full local gate, open a PR, report **"READY FOR REVIEW"**, then
  **HOLD**. In team mode workers **do not** self-review and **never self-merge**.
  Coordination is two-way: resume a worker by its agent id; **workers are expected to
  ask the lead** for clarification/re-partition/blockers. Keep the milestone
  checklist current (it's the handoff).
- **Lead incremental review (per PR, directly — not via a subagent):** review each
  as it lands for **alignment to the plan** + a general correctness pass, verifying
  load-bearing claims against the code. Hold for the joint review.
- **Joint cross-review — the gate before any merge:** resume all authoring agents to
  peer-review each other's PRs directly (a research subagent only, never delegating
  the review). Four axes: correctness/plan, **performance**, **security** (their
  dedicated rigorous pass), and **cross-PR alignment** (integration conflicts visible
  only when two merge).
- **Consolidate → user signoff → batch merge:** fold every review into one report
  (per-PR verdicts, must-fixes + owners, rebases, merge order); **nothing merges
  until the user's signoff** — a hard gate. On the go, batch-merge in dependency
  order (rebase onto latest `main` → `ci` label → `--auto --squash`), then unblock
  downstream / start the next wave.

This joint review is the team review **floor**; it does not replace the
user-triggered `ultra`/cloud review or the pre-release security/performance audits.

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
