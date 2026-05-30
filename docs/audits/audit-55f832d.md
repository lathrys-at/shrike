# Shrike Code Review & Security Audit

- **Commit:** `55f832d` (v0.2.1)
- **Date:** 2026-05-28
- **Scope:** all of `src/shrike/` (~10K LoC incl. tests), `docs/`, CI workflow, packaging, git hygiene, and the v0.3–v0.6 roadmap (skill client extraction, sync + credential storage, hosted relay with OAuth).
- **Verified against:** `mcp` 1.27.1 (installed), `anki` API usage, and the embedded MCP schema the LLM client actually sees.

## Overall assessment

Well-structured, genuinely clean codebase for an early-stage project. Type coverage, logging discipline, the derived-cache consistency model, and the daemon lock design are all thoughtful and above average. The findings below are real but mostly *latent* — they don't bite today because the server binds to `127.0.0.1` and tool calls happen to serialize, but several become serious the moment the roadmap (remote relay, network binding, stored credentials) lands. One correctness bug bites today (`collection_info` default) and one bites in an error path (`upsert_notes` neighbor attach).

The theme: lock down the trust boundary and serialize the one shared mutable resource before the network-facing roadmap arrives. Nothing here suggests the architecture is wrong.

## Suggested priority order

1. **3.3** (false-failure upsert) and **2.1** (`collection_info` contract) — correctness bugs affecting users/LLM today; small fixes. ✅ done
2. **7.1** (nested-deck double-count) — confirmed counting bug. ✅ done
3. **3.1** (serialized collection access) — corrected on second pass (no active race), then formalized: `CollectionWrapper` is now an async object with single-worker-thread serialization + backend thread affinity. ✅ done
3. **1.2** (enable DNS-rebinding protection) and the non-loopback guard in **1.1** — small config changes, real local hardening. ✅ done
4. **1.1 (auth) / 1.3 (credential storage)** — design decisions to make *before* writing relay/sync code, not after.
5. **3.2, 2.3** — responsiveness/correctness polish. ✅ done
6. Everything in §4–§6 as cleanup, with `pytest-cov` + the stale `requirements.txt` being the highest-value of those. ✅ mostly done (`pytest-cov` gate, `requirements.txt`, py.typed, hatch version, factory, supply-chain pin all landed; remaining §4–§6 items are the N+1 query perf, test-coverage additions, and relay-gated hardening).

---

## 1. Security

### 1.1 [HIGH — roadmap-critical] No authentication or authorization on any endpoint

**Where:** `server.py` (FastMCP `/mcp`, plus custom routes `/status`, `/shutdown`, `/index/rebuild`).

Every endpoint is unauthenticated. Anyone who can open a TCP connection to the port can call `delete_notes`, `upsert_notes`, dump the whole collection via `list_notes`, kill the server (`POST /shutdown`), or read the collection path/PID/log dir (`GET /status`). Today this is gated only by the `127.0.0.1` bind.

**Why it matters now:** `--host`/`server.host` is a free-form string passed straight through (`server.py:142`, `embedding.py:_build_command`). `shrike server start --host 0.0.0.0` exposes the full collection-mutation API **and** the llama-server (which inherits the same `args.host`, `server.py:248`) to the entire network with zero auth. There's no warning when binding to a non-loopback address.

**Why it's roadmap-critical:** v0.6's relay explicitly forwards MCP JSON-RPC to "a user's local Shrike instance." If the relay reaches Shrike over anything but a loopback/authenticated channel, this is a remote unauthenticated total-control hole.

**Action items:**
- [x] When `host` is not a loopback address, refuse to start unless an explicit `--allow-remote` (or similar) flag is set, and log a loud warning. (`server.py` `_is_loopback` + the guard in `main()`; `--allow-remote` threaded through the CLI `server start`, `ServerSpec`, and `build_server_spec`. Refusal exits 1; allow-remote logs a loud unauthenticated-exposure warning.)
- [x] Keep llama-server bound to `127.0.0.1` regardless of the MCP host — there is no reason to expose it. (`EmbeddingRuntime` is now constructed with a hard-coded `host="127.0.0.1"`, no longer inheriting `args.host`.)
- [ ] Design (before relay): require a bearer token / shared secret for all endpoints when bound non-locally. Build on the MCP SDK's auth framework (`mcp.server.auth`, OAuth 2.0 + PKCE) rather than rolling your own. *(Still roadmap — the non-loopback path is gated behind `--allow-remote` with a warning until this lands.)*
- [x] Put the auth check in middleware so the custom routes (`/shutdown` etc.) are covered too (they currently bypass middleware — see 1.2). *(The Host/Origin half is done — `_guard` in `_register_custom_routes` wraps every custom route with the SDK's `TransportSecurityMiddleware.validate_request`. The bearer-token half waits on the auth design above.)*
- [ ] Auth-gate `/shutdown` and `/index/rebuild` (state-changing) even in the local model if any browser-reachable surface ever exists. *(Roadmap — depends on the auth layer; CSRF/DNS-rebinding on these routes is now closed via the Host/Origin guard.)*

### 1.2 [HIGH] DNS-rebinding / CSRF protection is available but disabled

**Where:** not configured anywhere; `mcp/server/transport_security.py:41` defaults `enable_dns_rebinding_protection=False`. Custom routes (`server.py:45–126`) are registered via `@app.custom_route` and bypass the MCP transport middleware entirely regardless.

The MCP SDK has built-in Origin/Host validation precisely to stop DNS-rebinding (a malicious website the user is browsing scripting requests to `http://127.0.0.1:8372`). It's off by default and Shrike never turns it on. So a web page open in the user's browser can drive the MCP endpoint and, more easily, `POST /shutdown` (a no-body POST is a CORS "simple request" — no preflight). For a tool whose whole value is sitting on localhost next to a browser, this is a concrete local-attacker path.

**Action items:**
- [x] Pass `TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[...], allowed_origins=[...])` to FastMCP. Allow only the expected `127.0.0.1:<port>`/`localhost:<port>` Host values and reject cross-origin `Origin` headers. (`_build_transport_security(host)` builds loopback-only Host/Origin allow-lists and is passed into `create_mcp(...)`. The installed MCP SDK also now auto-enables this for loopback by default, but Shrike sets it explicitly so it matches the *actually bound* host rather than the construction-time placeholder.)
- [x] Add equivalent Host/Origin checks for the custom Starlette routes (in middleware or each handler) since they don't go through the MCP middleware. (`_guard` decorator wraps `/status`, `/index/rebuild`, `/embedding/start`, `/embedding/stop`, `/shutdown` with the same `TransportSecurityMiddleware` validation. Validated by `tests/integration/test_security.py`: forged `Origin` → 403, forged `Host` → 421, and a cross-origin `POST /shutdown` is refused without killing the server.)

### 1.3 [HIGH — design now] Sync credential storage (v0.4.0)

The roadmap calls for accepting and storing AnkiWeb / sync-server credentials. The current config path is plaintext YAML (`config.py:save_config`) written with default umask (world-readable `0644` on most systems), and `/status` already echoes config-derived data.

**Action items (decide before writing any credential code):**
- [ ] Do **not** put credentials in `config.yml`. Use the OS keyring (`keyring` package: Keychain / libsecret / Windows Credential Manager). Store only a non-secret reference in config.
- [ ] If a file fallback is ever needed, create it `0600` via `os.open(..., 0o600)` semantics (avoid chmod-after-write races). Establish a `secure_write` helper before secrets exist.
- [ ] Prefer AnkiWeb sync *tokens* over storing the raw password where the protocol allows.
- [ ] Never log credentials; scrub them from any `/status`-style introspection.

### 1.4 [LOW–MEDIUM] Information disclosure & error leakage

**Action items:**
- [ ] `GET /status` returns absolute collection path, log dir, PID, uptime to any caller (`server.py:49–78`). Gate behind auth (1.1) and/or trim before remote exposure.
- [~] `_safe_tool` returns `f"Internal error: {e}"` to the client (`tools.py:100`), and per-item errors return `str(e)` (`collection.py:259`) — can leak filesystem paths/internals. Sanitize (log full detail, return a generic message + error id) before the relay ships. *(Partial: `_safe_tool` no longer formats its own string — it `logger.exception(...)`s the full detail and re-raises, so logging is covered. The raised message still reaches the client and per-item errors still return `str(e)`; the generic-message + error-id sanitization is the relay-time remainder.)*

---

## 2. Correctness vs. documented behavior

### 2.1 [LOW — corrected from MEDIUM; stale human docs only] `collection_info` no-arg default disagrees between code and hand-written docs

> **Correction (second pass):** the first draft of this audit claimed the LLM is misled into expecting everything. That was wrong. The schema the LLM client actually receives is generated by FastMCP from the **docstring** (`tools.py:126`: *"With no arguments, returns a compact summary…"*), which correctly matches the code. The discrepancy is only with the **hand-maintained human docs**, so the model is not misled and there is no behavioral contract violation. Severity downgraded to LOW (docs accuracy). See also 7.6 on schema drift.

**Where:** `tools.py:133` (`sections = include or ["summary"]`) and `tools.py:126` docstring (correct) vs. `docs/mcp-tools.md:13` ("returns everything") and `docs/mcp-schema.json` description ("With no arguments, returns summaries of everything") — both human docs are stale. Separately, the `include` parameter docs (`mcp-tools.md:21`) omit the valid values `"summary"` and `"all"` that the code accepts (`collection.py:37–38`).

**Action items:**
- [x] Decide the intended no-arg default, then make code, docstring, `docs/mcp-tools.md`, and `docs/mcp-schema.json` all agree. (No-arg default is `["summary"]`; human docs corrected to match the code/docstring.)
- [x] Document `"summary"` and `"all"` as valid `include` values.

### 2.2 [LOW] Undocumented `search_notes` parameters

`search_notes` accepts `threshold` (`tools.py:212`) and `exclude_ids` (`tools.py:215`); `docs/mcp-tools.md` parameter table lists neither.

**Action items:**
- [x] Add `threshold` and `exclude_ids` to the `search_notes` parameter docs (and schema if missing). *(Prior commit: both are in the `search_notes` parameter table in `docs/mcp-tools.md`, and present in the live FastMCP schema.)*

### 2.3 [MEDIUM] `search_notes` deck/tag filtering can silently under-return

**Where:** `tools.py:305–339`. The index is queried for `top_k + len(exclude_set)` results, then `deck`/`tags` filters are applied *post hoc* (`tools.py:324–328`), with the loop stopping at `len(enriched) >= top_k`. The over-fetch only compensates for excludes, not for deck/tag filtering. If the nearest neighbors are mostly outside the requested deck, you can get far fewer than `top_k` matches even when plenty exist deeper in the ranking — looking like "no results" for a deck-scoped semantic search.

**Action items:**
- [x] Over-fetch more aggressively when `deck`/`tags` are set (multiple of `top_k`, or loop widening the window until satisfied), or push deck/tag constraints into the index query. *(Prior commit: `search_notes` widens the window to `max(top_k + excludes, top_k * 10)`, capped at index size, when `deck`/`tags` are set.)*
- [x] If the heuristic is kept, document the limitation. *(Prior commit: `docs/mcp-tools.md` notes that deeply-ranked in-scope notes can still under-return; widen with a higher `top_k`.)*

### 2.4 [LOW] `_note_to_dict` reports only the first card's deck

**Where:** `collection.py:229–232` (`cards[0].did`). A note whose cards live in different decks reports just one.

**Action items:**
- [x] Document the assumption (or handle multi-deck notes) since Anki permits per-card decks. (Documented: a code comment at `collection.py:_note_to_dict`, and a user-facing note in `docs/mcp-tools.md` — `deck` is the first card's deck; Shrike treats notes as single-deck.)

---

## 3. Robustness & concurrency

> Dispatch note: FastMCP runs **sync** tool functions inline on the event-loop thread (`func_metadata.py` non-async branch calls `fn(**kwargs)` directly), so tool calls are effectively *serialized* — that rules out tool-vs-tool collection corruption. The remaining concurrency risk is the background rebuild thread (3.1).

### 3.1 [DONE — formalized via async serialization] Collection access was not thread-safe by construction

**Correction (second pass):** the original claim — that the background rebuild thread races the request thread on the collection — was **inaccurate**. On re-tracing: `rebuild_in_background` (`index.py`) never touches the collection; it only consumes the `(ids, texts)` already gathered. Those gathering reads (`find_notes` + `note_texts_for_embedding`) run on the *calling* thread (startup main thread, or the event-loop thread for `/index/rebuild`) **before** the daemon thread is spawned. Combined with the dispatch note above (sync tools ran inline on the event-loop thread, so they were already serialized), there was **no active data race** in the shipped code.

What remained true is that the invariant — "only one thread ever touches `anki.Collection`" — was *implicit* and easy to violate as the roadmap adds concurrency (async tools, sync, the relay). Rather than leave it to convention, it is now **enforced structurally**.

**Resolution (implemented):** `CollectionWrapper` owns a single dedicated worker thread (`ThreadPoolExecutor(max_workers=1)`); the collection is opened on, and every operation is dispatched to, that one thread. Public operations are now `async` (`await wrapper.list_notes(...)`, etc.), scheduled via `run_in_executor`, so the event loop stays responsive while the collection is busy; `run_sync`/`close` provide synchronous entry points for the startup/shutdown paths. The MCP tools and custom routes were converted to `await` the wrapper. This gives true serialization *and* single-thread affinity for the Rust backend, eliminating the entire class of concern by construction rather than guarding individual methods with a lock.

**Action items:**
- [x] Serialize all `anki.Collection` access through a single owner thread (chose a dedicated worker thread + async API over a `threading.RLock`, per the discussion above — it adds thread affinity and keeps the event loop non-blocking).

### 3.2 [LOW–MEDIUM] Blocking I/O inside async custom routes

`GET /status` calls `embedding_service.health()` → synchronous `httpx.get(timeout=2.0)` (`embedding.py:211`) on the event-loop thread; a slow/hung embedding server stalls all request handling up to 2s. `POST /index/rebuild` runs `find_notes("deck:*")` + `note_texts_for_embedding` over the whole collection synchronously before returning (`server.py:96–101`). Both handlers are `async def`, so the blocking is on the loop.

> **Update (prior commit): the loop-blocking concern is resolved**, via `to_thread` / the collection worker thread rather than the literal bullets below. `GET /status` now does `await asyncio.to_thread(runtime.health)`; `POST /index/rebuild` gathers ids+texts via `await wrapper.run(_collect_for_rebuild)` (on the dedicated collection thread) before kicking the background rebuild. The event loop no longer blocks. The remaining nuances are cosmetic.

**Action items:**
- [x] Use an async httpx client (`await`) for the health probe. *(Addressed differently: the sync probe runs via `asyncio.to_thread`, so the loop isn't blocked. A native async client would drop the extra thread but isn't required.)*
- [x] Move the "gather all note ids + texts" work for rebuild into the background thread; have the route return immediately. *(The gather runs off the event loop on the collection worker thread (`await wrapper.run(...)`); the route then returns once the background rebuild is launched. Loop stays responsive.)*
- [x] Use `asyncio.to_thread` for any unavoidable sync collection access in async handlers. *(Done — custom routes use `asyncio.to_thread` / `await wrapper.run(...)`.)*

### 3.3 [MEDIUM — bites in error path] `upsert_notes` reports false failure when embedding throws

**Where:** `tools.py:394–405`. If `note_texts_for_embedding` raises, `texts` is never bound; the `except` swallows it, then `_attach_neighbors(..., texts, ...)` raises `NameError`, which escapes to `_safe_tool` and returns `{"error": ...}`. The notes were **already written** to the collection (`collection.upsert_notes` ran first), so the client is told the upsert failed when it succeeded — and may retry, creating duplicates.

**Action items:**
- [x] Move `_attach_neighbors` inside the `try` (or initialize `texts = []` and guard).
- [x] Ensure index-maintenance failures never convert a successful collection mutation into an error response (matches the documented best-effort index model).

### 3.4 [LOW] Deprecated `asyncio.get_event_loop()` in shutdown handler

**Where:** `server.py:125`, inside a running coroutine — deprecated in 3.12.

**Action items:**
- [x] Replace with `asyncio.get_running_loop()`. *(Prior commit: the shutdown handler was rewritten to `asyncio.create_task` + `asyncio.sleep`; no `get_event_loop()` remains.)*

### 3.5 [LOW] Intentionally leaked log file handles

**Where:** `server_cmd.py:186,360` and `embedding.py:135` open log files (`# noqa: SIM115`) handed to `subprocess.Popen` and never closed; the bootstrap log handle leaks for the CLI process lifetime.

**Action items:**
- [x] Review file-handle lifetime for spawn log targets; close handles that outlive their need. (Reviewed: the CLI bootstrap-log handles (`client._spawn`, `server_cmd` daemon spawn) use `with open(...)`; `_tail_follow` closes in `finally`. The last leak — `embedding.py`'s llama-server stderr handle — is now closed in the parent right after `Popen` (the child keeps its dup'd fd).)

---

## 4. Python / code-quality

**Action items:**
- [x] Promote `wrapper._note_to_dict` (called across module boundary from `tools.py:320,433`) to a public method — matters more once the client is extracted into a library (v0.3.0). *(Prior commit: public async `CollectionWrapper.note_to_dict` exists.)*
- [x] Construct FastMCP inside `main()`/a factory rather than as an import-time module global mutated in `main()` (`server.py:25`) — improves testability/in-process reuse. *(Prior commit: `create_mcp()` factory, now also taking host/port/transport_security.)*
- [x] Fix or delete the stale `requirements.txt`: it's missing `usearch`, `numpy`, `filelock`, `platformdirs` and has looser pins than `pyproject.toml` — a `pip install -r requirements.txt` yields a broken install. *(Prior commit: `requirements.txt` deleted; `pyproject.toml` is the single source.)*
- [x] Use Hatch dynamic version (`[tool.hatch.version] path = "src/shrike/__init__.py"`) so `pyproject.toml` and `__init__.py` versions stay in lockstep. *(Prior commit: `[tool.hatch.version]` configured.)*
- [x] Add `src/shrike/py.typed` (and include in the wheel) since typing is enforced and `shrike.client` is slated to ship as a library. *(Prior commit: `py.typed` present; `packages = ["src/shrike"]` ships it.)*
- [x] Log the swallowed per-item failures in search/neighbor loops (`tools.py:321,434`, `index.py:326`) at `debug` instead of silently dropping. (Done: `search_notes`/neighbor per-note lookups log at `debug` with `exc_info`; index-maintenance failures log at `warning`. The one remaining `contextlib.suppress` in `index.rebuild_in_background._run` is over an error that `rebuild()` already logs at `error` and records as `IndexState.ERROR` — commented to say so.)
- [x] Add `tests/` to the CI lint job (`test.yml:21` currently lints only `src/shrike/`, despite commit messages claiming `tests/` is linted). *(Prior commit: CI runs `ruff check src/shrike/ tests/` and `ruff format --check src/shrike/ tests/`.)*
- [ ] Consider date-filtering server-side for `list_notes`/info queries: `_get_decks`/`_get_stats` run a `find_notes` per deck (`collection.py:112,138`), and `list_notes` with only `modified_since` loads every note via `get_note` (`collection.py:208`) — N+1 over the collection. Fine at hundreds of notes; noticeable at tens of thousands.

---

## 5. Test coverage analysis

Breadth is genuinely good: ~365 tests (218 unit / 147 integration), real-server HTTP integration tests, semantic tests gated behind a llama-server fixture, multi-OS + arm CI matrix.

**Action items:**
- [x] Add `pytest-cov` and a coverage gate so untested-branch regressions are visible. (`[tool.coverage]` in `pyproject.toml` with `branch=true`, `fail_under=70`; a `coverage` CI job runs unit + non-embedding integration under `coverage run --parallel-mode`, with a `coverage_subprocess.pth` + `COVERAGE_PROCESS_START` so the `python -m shrike.server` subprocess is counted too — combined coverage measured ~73%.)
- [ ] Test daemon failure paths: `stop_server` SIGTERM→SIGKILL escalation (`daemon.py:221–238`), stale-state cleanup, autostart-on-ConnectError retry (`client.py:50–62`).
- [ ] Add a concurrency test: fire upserts while a background rebuild thread runs (covers 3.1).
- [ ] Test the 3.3 error path: simulate `note_texts_for_embedding` raising and assert the upsert still reports `created`/`updated`.
- [ ] Test `search_notes` deck/tag under-return (2.3): assert result counts when nearest neighbors are filtered out.
- [~] After 1.1/1.2 land, add tests for rejected Origin/Host and missing-token responses. *(Origin/Host done — `tests/integration/test_security.py` asserts 403 on forged Origin, 421 on forged Host, and a refused cross-origin `/shutdown`. Missing-token tests wait on the auth layer.)*

---

## 6. Git / repo hygiene

Clean: `.gitignore` correctly excludes `.cache/`, caches, `.DS_Store`; the llama binaries and GGUF model on disk are **not** tracked (verified). Only `scripts/fetch-llama-server.sh` is committed.

**Action items:**
- [x] `fetch-llama-server.sh` (`scripts/fetch-llama-server.sh:53`) and the CI equivalent (`test.yml:80`) download release tarballs over HTTPS with no checksum/signature verification and always take `releases/latest`. Pin a known release tag + verify a SHA256 for supply-chain hygiene. (Done: tag + per-platform SHA256 pinned in `scripts/llama-server.lock`, sourced by both the script and the CI embedding job, which now download to a file and verify the checksum before extracting. `scripts/update-llama-lock.sh [TAG]` regenerates the lock; no more unverified `releases/latest`.)

---

## 7. Second-pass findings

Added after a deeper review of the CLI command modules, the tests, the embedding/index lifecycle, and empirical verification against a live `anki.Collection`. (No shell-injection vectors found: every `subprocess` call uses list args, no `shell=True`/`eval`/`exec`.)

### 7.1 [MEDIUM — confirmed bug] Due/new card totals are double-counted for nested decks

**Where:** `_get_summary` via `_walk_due` (`collection.py:59–85`) and `_get_stats` via `walk` (`collection.py:122–154`).

Anki's `deck_due_tree()` returns nodes whose `new_count`/`review_count`/`learn_count` are **already rolled up to include all subdecks**. Both methods then recurse and add every node's count on top of its parent's, so descendants are counted twice (or more, for deeper nesting).

**Verified empirically** on a temp collection: parent deck `Lang` (2 own notes) + child `Lang::Japanese` (3 notes). The parent node reports `new=5` (rolled up), the child `new=3`; the recursive sum yields **8 for 5 actual cards**. So `summary.due_today`, `stats.cards_due_today`, and `stats.new_cards` are all inflated whenever nested decks exist. The per-deck `decks_summary` entries are individually correct — only the grand totals are wrong.

**Action items:**
- [x] Compute totals from top-level nodes only (each `tree.children` node already includes its descendants): `sum(top.new_count for top in tree.children)`, etc. Don't recurse for the total.
- [x] Add a unit test with a nested deck asserting the totals match the real card count (`test_collection_info.py::test_nested_decks_not_double_counted`).

### 7.2 [MEDIUM] First-run config auto-save drops the embedding model (and logging) settings

**Where:** `config.py:save_config` (only writes `collection` + `server.host/port`) and `server_cmd.py:396` (saves config only when the file doesn't yet exist).

`shrike server start --embedding-model X` uses the model for that run but `save_config` never persists `embedding.*` (or `logging.*`). So the auto-created `config.yml` omits the model, and the next bare `shrike server start` comes up with **no embedding service and no semantic search**, silently. The user must hand-edit `config.yml` or pass `--embedding-model` every time.

**Action items:**
- [x] Persist the embedding settings (at least `model`, and any explicitly-set port/threads/gpu_layers) in `save_config` when provided. *(Prior commit: `save_config` writes `embedding.model` plus non-default `port`/`context_size`/`threads`/`gpu_layers`/`llama_server`; `shrike server start` seeds them on first-run auto-save.)*
- [x] Decide whether logging overrides should round-trip too, or document that they're file-only. (Decided: file-only — documented in CLAUDE.md's "Config file" section. `save_config` does not write `logging.*`.)

### 7.3 [MEDIUM] CLAUDE.md claims "Periodic index save" is done, but it isn't implemented

**Where:** no timer/thread does periodic saves; `index.save()` is only called at the end of `rebuild()` (`index.py:305`), in the signal handler (`server.py:280`), and in `POST /shutdown` (`server.py:114`). CLAUDE.md's v0.2.0 roadmap lists "Periodic index save and graceful shutdown persistence ✓".

Correctness is not at risk — `col.mod` drift detection forces a rebuild after any non-graceful exit, so the index self-heals. But: (a) the roadmap doc is inaccurate, and (b) every hard kill / crash / power loss discards all in-memory incremental updates and forces a **full re-embed of the whole collection** on next start (minutes for large collections), even though the data was fine.

**Action items:**
- [ ] Either implement a real periodic/debounced save (e.g., mark dirty on `add`/`remove`, flush on a timer or after N changes), or
- [ ] Correct CLAUDE.md to say persistence is shutdown-only and rebuild-on-crash is the accepted cost.

### 7.4 [MEDIUM] llama-server is orphaned on a hard kill

**Where:** `embedding.py:137` spawns llama-server with no `start_new_session`/process-group isolation and no `PR_SET_PDEATHSIG`. Cleanup relies on `EmbeddingService.stop()` running during graceful shutdown.

If the shrike server is `SIGKILL`ed — including by its **own** force-kill path (`daemon.py:_force_kill` → SIGKILL when graceful stop times out) — `stop()` never runs and llama-server is reparented to init and keeps running, holding port 8373. On the next start, the health check may then talk to / collide with the stale process.

**Action items:**
- [ ] On Linux, set `PR_SET_PDEATHSIG` (via `preexec_fn` or `prctl`) so the child dies with the parent; on macOS/Windows, detect and kill a stale llama-server on startup (by recorded PID or by probing the embedding port).
- [ ] Consider recording the llama-server PID in state so a later `shrike server stop` / startup can reap an orphan.

### 7.5 [LOW–MEDIUM] `ShrikeClient.call()` doesn't handle HTTP error status

**Where:** `client.py:100` calls `resp.raise_for_status()` but `call()` only catches `ConnectError`/`TimeoutException` (`client.py:92–98`). A 4xx/5xx (server 500, or — once 1.2 lands — a DNS-rebinding/auth rejection) raises `httpx.HTTPStatusError` that escapes to the CLI as an unhandled traceback instead of a clean `ClickException`.

**Action items:**
- [x] Catch `HTTPStatusError` in `call()` and surface a friendly message (status + hint), especially anticipating auth/Origin rejections. *(Prior commit: `_call`/`_request` go through `_raise_for_status`, raising a typed `ServerHTTPError` the CLI renders cleanly.)*

### 7.6 [LOW] `docs/mcp-schema.json` is hand-maintained and already drifting

CLAUDE.md calls `docs/mcp-schema.json` "the authoritative schema," but it's a separate hand-written artifact from the FastMCP-generated schema the client actually receives. It has already drifted (root cause of 2.1, and `threshold`/`exclude_ids` missing per 2.2).

**Action items:**
- [x] Generate the schema from the live server (`tools/list`) in CI and fail on diff, or stop hand-maintaining it and point docs at the generated output. (Resolved via the second option: `docs/mcp-schema.json` was deleted; the authoritative schema is whatever the running server advertises, and CLAUDE.md / `docs/mcp-tools.md` say so. No hand-maintained artifact left to drift.)

### 7.7 [LOW] HNSW update churn accumulates soft-deleted vectors

**Where:** `index.add` does `remove`-then-`add` for existing keys (`index.py:163–167`); USearch HNSW `remove` is a soft delete. A long-running server with heavy update churn accumulates tombstones, gradually degrading search quality and inflating the file until the next full rebuild.

**Action items:**
- [ ] Periodically compact/rebuild (tie into 7.3's dirty-tracking, or rebuild after a churn threshold). Low priority until real usage shows churn.

### 7.8 [LOW — relay-relevant] Unbounded `queries`/`ids` in semantic operations

`upsert_notes`/`delete_notes`/`list_notes.limit`/`top_k` are all capped, but `search_notes.queries`, `search_notes.ids`, and `note_texts_for_embedding(ids)` are not. A client can submit thousands of query strings, each triggering an embedding call — a cheap DoS against the embedding server. Fine under local trust; a hardening item for the relay.

**Action items:**
- [ ] Cap `queries` and `ids` lengths in `search_notes` (and document the cap), consistent with the other tools.

### 7.9 [LOW] Embedding start failure is reported as "not configured"

**Where:** `server.py:257–259` sets `embedding_service = None` when `start()` fails, so `/status` and `shrike server status` render "Embedding: not configured" (`server_cmd.py:94`) even though it *was* configured and *failed*. Misleading when debugging a bad model path or an occupied port.

**Action items:**
- [x] Distinguish "not configured" from "configured but failed to start" in the status output / `/status` payload. *(Prior commit: `EmbeddingRuntime.state` returns `running`/`failed`/`not_configured`/`stopped`, surfaced via `health()`.)*

### 7.10 [INFO — design + legal, for the roadmap]

- **`query` is not a security boundary.** The raw Anki `query` param is AND-joined with the structured filters by string concatenation (`collection.py:172–198`); a `query` containing Anki's `or` operator can broaden results past the other filters. Harmless for a trusted local LLM, but in a multi-tenant relay, structured filters must not be relied on to scope/authorize what a caller can see — enforce scoping server-side, separately.
- **AGPL §13 network clause.** The hosted relay (v0.6) serves users over a network, which triggers AGPL's requirement to offer corresponding source to those users; `anki` itself is AGPL too. Worth a deliberate compliance plan before the hosted/relay business model, and consider adding per-file license headers (currently none).
