# Shrike Code Review & Security Audit

- **Commit:** `55f832d` (v0.2.1)
- **Date:** 2026-05-28
- **Scope:** all of `src/shrike/` (~10K LoC incl. tests), `docs/`, CI workflow, packaging, git hygiene, and the v0.3–v0.6 roadmap (skill client extraction, sync + credential storage, hosted relay with OAuth).
- **Verified against:** `mcp` 1.27.1 (installed), `anki` API usage, and the embedded MCP schema the LLM client actually sees.

## Overall assessment

Well-structured, genuinely clean codebase for an early-stage project. Type coverage, logging discipline, the derived-cache consistency model, and the daemon lock design are all thoughtful and above average. The findings below are real but mostly *latent* — they don't bite today because the server binds to `127.0.0.1` and tool calls happen to serialize, but several become serious the moment the roadmap (remote relay, network binding, stored credentials) lands. One correctness bug bites today (`collection_info` default) and one bites in an error path (`upsert_notes` neighbor attach).

The theme: lock down the trust boundary and serialize the one shared mutable resource before the network-facing roadmap arrives. Nothing here suggests the architecture is wrong.

## Suggested priority order

1. **3.3** (false-failure upsert) and **2.1** (`collection_info` contract) — correctness bugs affecting users/LLM today; small fixes.
2. **3.1** (collection lock) — quiet corruption risk; small, clearly-correct fix.
3. **1.2** (enable DNS-rebinding protection) and the non-loopback guard in **1.1** — small config changes, real local hardening.
4. **1.1 (auth) / 1.3 (credential storage)** — design decisions to make *before* writing relay/sync code, not after.
5. **3.2, 2.3** — responsiveness/correctness polish.
6. Everything in §4–§6 as cleanup, with `pytest-cov` + the stale `requirements.txt` being the highest-value of those.

---

## 1. Security

### 1.1 [HIGH — roadmap-critical] No authentication or authorization on any endpoint

**Where:** `server.py` (FastMCP `/mcp`, plus custom routes `/status`, `/shutdown`, `/index/rebuild`).

Every endpoint is unauthenticated. Anyone who can open a TCP connection to the port can call `delete_notes`, `upsert_notes`, dump the whole collection via `list_notes`, kill the server (`POST /shutdown`), or read the collection path/PID/log dir (`GET /status`). Today this is gated only by the `127.0.0.1` bind.

**Why it matters now:** `--host`/`server.host` is a free-form string passed straight through (`server.py:142`, `embedding.py:_build_command`). `shrike server start --host 0.0.0.0` exposes the full collection-mutation API **and** the llama-server (which inherits the same `args.host`, `server.py:248`) to the entire network with zero auth. There's no warning when binding to a non-loopback address.

**Why it's roadmap-critical:** v0.6's relay explicitly forwards MCP JSON-RPC to "a user's local Shrike instance." If the relay reaches Shrike over anything but a loopback/authenticated channel, this is a remote unauthenticated total-control hole.

**Action items:**
- [ ] When `host` is not a loopback address, refuse to start unless an explicit `--allow-remote` (or similar) flag is set, and log a loud warning.
- [ ] Keep llama-server bound to `127.0.0.1` regardless of the MCP host — there is no reason to expose it.
- [ ] Design (before relay): require a bearer token / shared secret for all endpoints when bound non-locally. Build on the MCP SDK's auth framework (`mcp.server.auth`, OAuth 2.0 + PKCE) rather than rolling your own.
- [ ] Put the auth check in middleware so the custom routes (`/shutdown` etc.) are covered too (they currently bypass middleware — see 1.2).
- [ ] Auth-gate `/shutdown` and `/index/rebuild` (state-changing) even in the local model if any browser-reachable surface ever exists.

### 1.2 [HIGH] DNS-rebinding / CSRF protection is available but disabled

**Where:** not configured anywhere; `mcp/server/transport_security.py:41` defaults `enable_dns_rebinding_protection=False`. Custom routes (`server.py:45–126`) are registered via `@app.custom_route` and bypass the MCP transport middleware entirely regardless.

The MCP SDK has built-in Origin/Host validation precisely to stop DNS-rebinding (a malicious website the user is browsing scripting requests to `http://127.0.0.1:8372`). It's off by default and Shrike never turns it on. So a web page open in the user's browser can drive the MCP endpoint and, more easily, `POST /shutdown` (a no-body POST is a CORS "simple request" — no preflight). For a tool whose whole value is sitting on localhost next to a browser, this is a concrete local-attacker path.

**Action items:**
- [ ] Pass `TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[...], allowed_origins=[...])` to FastMCP. Allow only the expected `127.0.0.1:<port>`/`localhost:<port>` Host values and reject cross-origin `Origin` headers.
- [ ] Add equivalent Host/Origin checks for the custom Starlette routes (in middleware or each handler) since they don't go through the MCP middleware.

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
- [ ] `_safe_tool` returns `f"Internal error: {e}"` to the client (`tools.py:100`), and per-item errors return `str(e)` (`collection.py:259`) — can leak filesystem paths/internals. Sanitize (log full detail, return a generic message + error id) before the relay ships.

---

## 2. Correctness vs. documented behavior

### 2.1 [LOW — corrected from MEDIUM; stale human docs only] `collection_info` no-arg default disagrees between code and hand-written docs

> **Correction (second pass):** the first draft of this audit claimed the LLM is misled into expecting everything. That was wrong. The schema the LLM client actually receives is generated by FastMCP from the **docstring** (`tools.py:126`: *"With no arguments, returns a compact summary…"*), which correctly matches the code. The discrepancy is only with the **hand-maintained human docs**, so the model is not misled and there is no behavioral contract violation. Severity downgraded to LOW (docs accuracy). See also 7.6 on schema drift.

**Where:** `tools.py:133` (`sections = include or ["summary"]`) and `tools.py:126` docstring (correct) vs. `docs/mcp-tools.md:13` ("returns everything") and `docs/mcp-schema.json` description ("With no arguments, returns summaries of everything") — both human docs are stale. Separately, the `include` parameter docs (`mcp-tools.md:21`) omit the valid values `"summary"` and `"all"` that the code accepts (`collection.py:37–38`).

**Action items:**
- [ ] Decide the intended no-arg default, then make code, docstring, `docs/mcp-tools.md`, and `docs/mcp-schema.json` all agree.
- [ ] Document `"summary"` and `"all"` as valid `include` values.

### 2.2 [LOW] Undocumented `search_notes` parameters

`search_notes` accepts `threshold` (`tools.py:212`) and `exclude_ids` (`tools.py:215`); `docs/mcp-tools.md` parameter table lists neither.

**Action items:**
- [ ] Add `threshold` and `exclude_ids` to the `search_notes` parameter docs (and schema if missing).

### 2.3 [MEDIUM] `search_notes` deck/tag filtering can silently under-return

**Where:** `tools.py:305–339`. The index is queried for `top_k + len(exclude_set)` results, then `deck`/`tags` filters are applied *post hoc* (`tools.py:324–328`), with the loop stopping at `len(enriched) >= top_k`. The over-fetch only compensates for excludes, not for deck/tag filtering. If the nearest neighbors are mostly outside the requested deck, you can get far fewer than `top_k` matches even when plenty exist deeper in the ranking — looking like "no results" for a deck-scoped semantic search.

**Action items:**
- [ ] Over-fetch more aggressively when `deck`/`tags` are set (multiple of `top_k`, or loop widening the window until satisfied), or push deck/tag constraints into the index query.
- [ ] If the heuristic is kept, document the limitation.

### 2.4 [LOW] `_note_to_dict` reports only the first card's deck

**Where:** `collection.py:229–232` (`cards[0].did`). A note whose cards live in different decks reports just one.

**Action items:**
- [ ] Document the assumption (or handle multi-deck notes) since Anki permits per-card decks.

---

## 3. Robustness & concurrency

> Dispatch note: FastMCP runs **sync** tool functions inline on the event-loop thread (`func_metadata.py` non-async branch calls `fn(**kwargs)` directly), so tool calls are effectively *serialized* — that rules out tool-vs-tool collection corruption. The remaining concurrency risk is the background rebuild thread (3.1).

### 3.1 [MEDIUM] Background rebuild thread races the request thread on the Anki collection

`rebuild_in_background` spawns a daemon thread (`index.py:329`) that calls `wrapper.note_texts_for_embedding(...)` → `col.get_note(...)` (`collection.py:357–371`) while the event-loop thread can simultaneously service an `upsert_notes`/`delete_notes` calling `col.add_note` / `col.remove_notes`. Two threads touching one `anki.Collection`, which is not thread-safe (single Rust backend handle + DBProxy). Reachable: startup drift triggers a background rebuild (`server.py:268–272`) and the server accepts tool calls immediately. The vector-index `self._lock` does not cover the collection reads done to *feed* the index.

**Action items:**
- [ ] Serialize all `anki.Collection` access behind a single `threading.RLock` owned by `CollectionWrapper` (wrap the public read/write methods).

### 3.2 [LOW–MEDIUM] Blocking I/O inside async custom routes

`GET /status` calls `embedding_service.health()` → synchronous `httpx.get(timeout=2.0)` (`embedding.py:211`) on the event-loop thread; a slow/hung embedding server stalls all request handling up to 2s. `POST /index/rebuild` runs `find_notes("deck:*")` + `note_texts_for_embedding` over the whole collection synchronously before returning (`server.py:96–101`). Both handlers are `async def`, so the blocking is on the loop.

**Action items:**
- [ ] Use an async httpx client (`await`) for the health probe.
- [ ] Move the "gather all note ids + texts" work for rebuild into the background thread; have the route return immediately.
- [ ] Use `asyncio.to_thread` for any unavoidable sync collection access in async handlers.

### 3.3 [MEDIUM — bites in error path] `upsert_notes` reports false failure when embedding throws

**Where:** `tools.py:394–405`. If `note_texts_for_embedding` raises, `texts` is never bound; the `except` swallows it, then `_attach_neighbors(..., texts, ...)` raises `NameError`, which escapes to `_safe_tool` and returns `{"error": ...}`. The notes were **already written** to the collection (`collection.upsert_notes` ran first), so the client is told the upsert failed when it succeeded — and may retry, creating duplicates.

**Action items:**
- [ ] Move `_attach_neighbors` inside the `try` (or initialize `texts = []` and guard).
- [ ] Ensure index-maintenance failures never convert a successful collection mutation into an error response (matches the documented best-effort index model).

### 3.4 [LOW] Deprecated `asyncio.get_event_loop()` in shutdown handler

**Where:** `server.py:125`, inside a running coroutine — deprecated in 3.12.

**Action items:**
- [ ] Replace with `asyncio.get_running_loop()`.

### 3.5 [LOW] Intentionally leaked log file handles

**Where:** `server_cmd.py:186,360` and `embedding.py:135` open log files (`# noqa: SIM115`) handed to `subprocess.Popen` and never closed; the bootstrap log handle leaks for the CLI process lifetime.

**Action items:**
- [ ] Review file-handle lifetime for spawn log targets; close handles that outlive their need.

---

## 4. Python / code-quality

**Action items:**
- [ ] Promote `wrapper._note_to_dict` (called across module boundary from `tools.py:320,433`) to a public method — matters more once the client is extracted into a library (v0.3.0).
- [ ] Construct FastMCP inside `main()`/a factory rather than as an import-time module global mutated in `main()` (`server.py:25`) — improves testability/in-process reuse.
- [ ] Fix or delete the stale `requirements.txt`: it's missing `usearch`, `numpy`, `filelock`, `platformdirs` and has looser pins than `pyproject.toml` — a `pip install -r requirements.txt` yields a broken install.
- [ ] Use Hatch dynamic version (`[tool.hatch.version] path = "src/shrike/__init__.py"`) so `pyproject.toml` and `__init__.py` versions stay in lockstep.
- [ ] Add `src/shrike/py.typed` (and include in the wheel) since typing is enforced and `shrike.client` is slated to ship as a library.
- [ ] Log the swallowed per-item failures in search/neighbor loops (`tools.py:321,434`, `index.py:326`) at `debug` instead of silently dropping.
- [ ] Add `tests/` to the CI lint job (`test.yml:21` currently lints only `src/shrike/`, despite commit messages claiming `tests/` is linted).
- [ ] Consider date-filtering server-side for `list_notes`/info queries: `_get_decks`/`_get_stats` run a `find_notes` per deck (`collection.py:112,138`), and `list_notes` with only `modified_since` loads every note via `get_note` (`collection.py:208`) — N+1 over the collection. Fine at hundreds of notes; noticeable at tens of thousands.

---

## 5. Test coverage analysis

Breadth is genuinely good: ~365 tests (218 unit / 147 integration), real-server HTTP integration tests, semantic tests gated behind a llama-server fixture, multi-OS + arm CI matrix.

**Action items:**
- [ ] Add `pytest-cov` and a coverage gate so untested-branch regressions are visible.
- [ ] Test daemon failure paths: `stop_server` SIGTERM→SIGKILL escalation (`daemon.py:221–238`), stale-state cleanup, autostart-on-ConnectError retry (`client.py:50–62`).
- [ ] Add a concurrency test: fire upserts while a background rebuild thread runs (covers 3.1).
- [ ] Test the 3.3 error path: simulate `note_texts_for_embedding` raising and assert the upsert still reports `created`/`updated`.
- [ ] Test `search_notes` deck/tag under-return (2.3): assert result counts when nearest neighbors are filtered out.
- [ ] After 1.1/1.2 land, add tests for rejected Origin/Host and missing-token responses.

---

## 6. Git / repo hygiene

Clean: `.gitignore` correctly excludes `.cache/`, caches, `.DS_Store`; the llama binaries and GGUF model on disk are **not** tracked (verified). Only `scripts/fetch-llama-server.sh` is committed.

**Action items:**
- [ ] `fetch-llama-server.sh` (`scripts/fetch-llama-server.sh:53`) and the CI equivalent (`test.yml:80`) download release tarballs over HTTPS with no checksum/signature verification and always take `releases/latest`. Pin a known release tag + verify a SHA256 for supply-chain hygiene.

---

## 7. Second-pass findings

Added after a deeper review of the CLI command modules, the tests, the embedding/index lifecycle, and empirical verification against a live `anki.Collection`. (No shell-injection vectors found: every `subprocess` call uses list args, no `shell=True`/`eval`/`exec`.)

### 7.1 [MEDIUM — confirmed bug] Due/new card totals are double-counted for nested decks

**Where:** `_get_summary` via `_walk_due` (`collection.py:59–85`) and `_get_stats` via `walk` (`collection.py:122–154`).

Anki's `deck_due_tree()` returns nodes whose `new_count`/`review_count`/`learn_count` are **already rolled up to include all subdecks**. Both methods then recurse and add every node's count on top of its parent's, so descendants are counted twice (or more, for deeper nesting).

**Verified empirically** on a temp collection: parent deck `Lang` (2 own notes) + child `Lang::Japanese` (3 notes). The parent node reports `new=5` (rolled up), the child `new=3`; the recursive sum yields **8 for 5 actual cards**. So `summary.due_today`, `stats.cards_due_today`, and `stats.new_cards` are all inflated whenever nested decks exist. The per-deck `decks_summary` entries are individually correct — only the grand totals are wrong.

**Action items:**
- [ ] Compute totals from top-level nodes only (each `tree.children` node already includes its descendants): `sum(top.new_count for top in tree.children)`, etc. Don't recurse for the total.
- [ ] Add a unit test with a nested deck asserting the totals match the real card count (the suite currently has no nested-deck due-count coverage).

### 7.2 [MEDIUM] First-run config auto-save drops the embedding model (and logging) settings

**Where:** `config.py:save_config` (only writes `collection` + `server.host/port`) and `server_cmd.py:396` (saves config only when the file doesn't yet exist).

`shrike server start --embedding-model X` uses the model for that run but `save_config` never persists `embedding.*` (or `logging.*`). So the auto-created `config.yml` omits the model, and the next bare `shrike server start` comes up with **no embedding service and no semantic search**, silently. The user must hand-edit `config.yml` or pass `--embedding-model` every time.

**Action items:**
- [ ] Persist the embedding settings (at least `model`, and any explicitly-set port/threads/gpu_layers) in `save_config` when provided.
- [ ] Decide whether logging overrides should round-trip too, or document that they're file-only.

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
- [ ] Catch `HTTPStatusError` in `call()` and surface a friendly message (status + hint), especially anticipating auth/Origin rejections.

### 7.6 [LOW] `docs/mcp-schema.json` is hand-maintained and already drifting

CLAUDE.md calls `docs/mcp-schema.json` "the authoritative schema," but it's a separate hand-written artifact from the FastMCP-generated schema the client actually receives. It has already drifted (root cause of 2.1, and `threshold`/`exclude_ids` missing per 2.2).

**Action items:**
- [ ] Generate the schema from the live server (`tools/list`) in CI and fail on diff, or stop hand-maintaining it and point docs at the generated output.

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
- [ ] Distinguish "not configured" from "configured but failed to start" in the status output / `/status` payload.

### 7.10 [INFO — design + legal, for the roadmap]

- **`query` is not a security boundary.** The raw Anki `query` param is AND-joined with the structured filters by string concatenation (`collection.py:172–198`); a `query` containing Anki's `or` operator can broaden results past the other filters. Harmless for a trusted local LLM, but in a multi-tenant relay, structured filters must not be relied on to scope/authorize what a caller can see — enforce scoping server-side, separately.
- **AGPL §13 network clause.** The hosted relay (v0.6) serves users over a network, which triggers AGPL's requirement to offer corresponding source to those users; `anki` itself is AGPL too. Worth a deliberate compliance plan before the hosted/relay business model, and consider adding per-file license headers (currently none).
