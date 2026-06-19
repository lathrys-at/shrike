# The server runtime

This covers how the server process runs: opening the collection, the network
trust boundary, daemon lifecycle, where files live, and configuration.

## The Anki collection and locking

The `anki` pip package provides a headless Python API to Anki's SQLite database
(no Qt, no GUI). It takes an **exclusive write lock**, so only one process can
have a collection open at a time. `CollectionWrapper` (`harness/collection.py`)
handles the open/close lifecycle.

### Permanent vs cooperative locking

By default the daemon holds the exclusive lock for its whole life. This is ideal
for the heavy single-collection embedding workflow — no acquire latency, no
contention — but it means you cannot launch Anki desktop against the same
collection while the daemon runs.

`--cooperative-lock` (config `server.cooperative_lock`, env
`SHRIKE_COOPERATIVE_LOCK`) opens the collection on demand and **releases it after
a short idle window** (`--lock-hold-seconds`, default 5 s). An *idle* daemon then
no longer blocks launching Anki. This is cooperative *time-slicing*, not
concurrent sharing: Anki desktop never releases mid-session, so the win is
precise — an idle daemon steps aside, not that both operate at once.

On each re-acquire an `on_acquire` hook runs a cheap drift check
(`index.check_drift(col.mod)`) and rebuilds off-lock only on real drift (an
external edit during the idle gap). Reopening starts a fresh readiness generation:
the harness re-establishes its barrier so a reconcile rides the ingest actor's
bulk-op path and `await_ready` resolves only once the reopened index has settled.
In default (permanent) mode this path is inert.

The daemon-liveness lock (`server.lock`) is **separate** from the collection
lock. `server status` and `GET /status` report both (`locking`,
`collection_held`).

### Busy is a typed error

When a re-acquire can't open the collection because another process holds it,
`CollectionWrapper` catches Anki's `DBError` and raises `CollectionBusyError`
**immediately** — no retry, the caller decides. Busy is orthogonal to every
tool's response (the op never ran), so it is modelled as an error class with a
stable wire code, not as a per-tool response variant.

It rides the two-layer error split: the server-side `CollectionBusyError`
carries a message prefixed with the `COLLECTION_BUSY_CODE` sentinel
(`"collection_busy"`, defined in `schemas.py`); `_safe_tool` logs it at WARNING
and re-raises so FastMCP emits an `isError`; `ShrikeClient._call` detects the
prefix and raises the client-side `CollectionBusyError(ShrikeError)`, so callers
catch-and-retry rather than parse a string. Permanent mode never re-opens, so it
never produces this error.

## MCP transport

The server uses FastMCP with streamable HTTP transport (`stateless_http=True`,
`json_response=True`), listening on `http://127.0.0.1:8372/mcp` by default. All
communication is JSON-RPC 2.0: clients POST `tools/call` and receive structured
JSON.

### Two planes: data and control

The server runs **two listeners**. The **data plane** (the FastMCP app) serves
`/mcp`, `/actions/{name}`, `/media/{name}`, `/export/{token}`, and a minimal
`/health`; it binds loopback by default and honors the exposure flags below. The
**control plane** is a separate Starlette app serving the privileged routes —
`/shutdown`, `/reload`, `/index/*`, `/embedding/*`, and the full `/status`
diagnostics — on its own **always-local** listener: a Unix-domain socket
(`<state_dir>/control.sock`, POSIX) or an ephemeral loopback-TCP port (Windows,
which lacks asyncio Unix sockets). The control plane **never** honors
`--allow-remote`/`--allowed-host`/`--no-dns-rebinding-protection`, so the
privileged surface is unreachable from the network no matter how the data plane is
exposed. Its address is recorded in `server.json`; the CLI/client discover it
there (`daemon.control_channel`). Rationale and the rejected alternatives are in
[`decisions.md`](decisions.md) ("The control plane is a separate, always-local
listener").

### Trust boundary (data plane)

Every endpoint is unauthenticated, so the data plane binds loopback by default.
Binding a non-loopback host requires `--allow-remote` (the server refuses to
start otherwise), and llama-server stays pinned to `127.0.0.1` regardless.

DNS-rebinding/CSRF protection (`_build_transport_security`) validates `Host` and
`Origin` headers. It applies to the MCP endpoint *and*, via the `_guard` wrapper,
to the data custom routes (`/health`, `/media/{name}`, `/export/{token}`,
`/actions/{name}`), which bypass MCP middleware. The control plane carries its own
fixed-local guard (none needed for the filesystem-gated UDS; loopback-only for the
TCP fallback). Every route's guard is asserted in
`tests/integration/test_security.py`.

The guard is independent of the bind address:

- A loopback bind allow-lists loopback `Host`/`Origin`.
- `--allowed-host` / `--allowed-origin` (config `server.allowed_hosts` /
  `allowed_origins`) *add* trusted values for a reverse-proxy or VPN hostname. A
  proxy forwards `Host: name:port`, so use the SDK's `name:*` port-wildcard form.
- `--no-dns-rebinding-protection` turns the guard off entirely where the network
  is the trust boundary (behind a reverse proxy, on a tailnet, firewalled).
- A non-loopback bind with no explicit allow-list also leaves the guard off (the
  original `--allow-remote` behaviour).

One config footgun, which fails closed: a non-loopback bind given *only*
`--allowed-origin` (no `--allowed-host`) builds a guard whose Host allow-list is
empty, so every request is rejected with 421. `_build_transport_security` logs a
startup warning.

In every mode the endpoints stay unauthenticated — the guard is
anti-CSRF/DNS-rebinding, not authentication. OAuth (required for native
connectors like Claude Desktop URL connectors) is intentionally not implemented
yet; until then a native client reaches Shrike through the `mcp-remote` stdio
bridge.

## Daemon management

`shrike server start` spawns the server as a background process; the lifecycle
lives in `platform/daemon.py`.

- **Liveness** uses file locks (`filelock`, fcntl/msvcrt). The server holds an
  exclusive lock on `server.lock` for its lifetime; the OS releases it on exit;
  clients probe by a non-blocking acquisition. This sidesteps PID-recycling
  issues entirely.
- **Shutdown** is cross-platform via `POST /shutdown` on the control plane (one
  request drains both listeners). The CLI's `stop_server` escalates: clean HTTP
  shutdown → SIGTERM (Unix, if HTTP is unresponsive) → SIGKILL/TerminateProcess
  (hung). Signal handlers (SIGTERM, SIGINT) remain a secondary path for `kill` and
  Ctrl+C.

### HTTP endpoints beyond MCP

Each custom route sits behind its plane's `_guard` check.

**Data plane** (FastMCP listener, honors the exposure flags):

| Route | Purpose |
|-------|---------|
| `GET /health` | Minimal liveness: `running` + wire-protocol version. Leaks nothing sensitive; backs `client.ping()`. |
| `GET /media/{filename}` | Streams a media file (`FileResponse`); read-only, basename-sanitized, media dir resolved lock-free. 404 for missing/escaping names. |
| `GET /export/{token}` | Streams a pending export package by its one-shot token; reaped after the stream. |
| `POST /actions/{name}` | The actions-over-HTTP edge — the UI mirror of the MCP tool catalog. |

**Control plane** (always-local listener — UDS or loopback TCP, never `--allow-remote`):

| Route | Purpose |
|-------|---------|
| `GET /status` | Full diagnostics: pid, url, collection, uptime, embedding/index/recognition state, lock state, per-collection rows. Backs `shrike server status`. |
| `POST /shutdown` | Graceful shutdown of both listeners. |
| `POST /index/rebuild` | Full rebuild (returns immediately with status/progress); requires embedding running. |
| `POST /index/save` | Immediate flush off the event loop. |
| `POST /embedding/start` / `/embedding/stop` | Cycle the embedding service on a running server. Execution-shaping overrides in the start body (`backend`/`model`/`llama_server`/`extra_args`/`onnx_providers`) are accepted only when the control transport confines callers to the operator — i.e. the filesystem-gated UDS (POSIX). On the Windows loopback-TCP fallback (reachable by any local user) they are refused and the daemon uses its boot-configured settings. Runtime knobs (port/threads/…) pass through. |
| `POST /reload` | Close and re-open the collection (picks up a restored backup or sync swap) and re-check drift. Backs `shrike collection reload`. |

`/reload` shares its reopen primitive (`CollectionWrapper.reopen` plus reading
`self.col` at execution time) with the cooperative-lock open-on-demand
lifecycle. It is a control endpoint, not an MCP tool.

State files live in the platform state directory: `server.lock` (the exclusive
lock), `server.pid` (diagnostics only, not liveness), `server.json` (the data URL
+ port, the `control` channel address, collection path, start time, log dir), and
— on POSIX — `control.sock` (the control-plane Unix socket, owner-only).

## Platform directories

All paths resolve through `platformdirs` in `platform/paths.py`:

| Purpose | macOS | Linux (XDG) | Windows |
|---------|-------|-------------|---------|
| Config | `~/Library/Application Support/shrike/` | `~/.config/shrike/` | `%APPDATA%\shrike\` |
| State | `~/Library/Application Support/shrike/` | `~/.local/state/shrike/` | `%LOCALAPPDATA%\shrike\` |
| Logs | `~/Library/Logs/shrike/` | `~/.local/state/shrike/log/` | `%LOCALAPPDATA%\shrike\Logs\` |
| Cache | `~/Library/Caches/shrike/` | `~/.cache/shrike/` | `%LOCALAPPDATA%\shrike\Cache\` |

On Linux the XDG environment variables (`XDG_CONFIG_HOME`, `XDG_STATE_HOME`, …)
are respected.

## Configuration

Config is YAML at the platform config directory (`config.yml`). It is
**user-managed**: `shrike server start` never writes it unless `--save-config` is
passed. With that flag, start persists the resolved operational flags; without
it, start reflects exactly the flags it was given and writes nothing.

The **capability sections are config-file-only** — they have no flag or env
spelling, because structured config has one home:

- `embedders:` — vector-space entries (`modalities` + `runtime: onnx|remote`,
  with per-entry `endpoint` / `api_key_env` / `pooling` / `providers` /
  `batch_size`);
- `recognizers:` — keyed `ocr` / `asr` / `describe`;
- `managed:` — `llama_server` (`manage: auto|attach|off`) and `sync_server`.

`harness/profiles.py` is the model. `parse_capabilities` parses and validates
(an inapplicable knob is a structural error); `resolve_profile` intersects the
config with the compiled build features (an uncompiled runtime or unwired
capability is a config error that names its issue); `plan_to_runtime_params`
bridges onto the embedding runtime's parameter shape. The CLI validates at
spec-build time and passes `--config` to the daemon, which re-resolves the file
itself.

Operational settings keep a permanent resolution cascade
(config defaults → config values → env vars → CLI flags) via
`config.resolve_cache_dir()` / `resolve_index_save()` / `resolve_transport()` /
`resolve_locking()`. The numeric defaults for the index-flush knobs and the cache
dir live in `harness/index.py`, not duplicated in config — a `None` in config
means "use the built-in default". Logging overrides are *read* from config but
never written by `--save-config`; set `logging.level` / `logging.dir` in
`config.yml` directly.
