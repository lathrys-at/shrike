# SPA stack spike (#506)

Evidence-only skeleton for the desktop + web client stack decision. The verdict
is recorded in [`docs/decisions.md`](../../docs/decisions.md) — *"The desktop +
web SPA stack is Rust-wasm (Leptos)"*. This directory is the supporting
skeleton, not shippable app code: two evaluation screens proving the stack
works, deliberately minimal.

It is a **standalone Cargo project** with its own `Cargo.toml`/`Cargo.lock`,
outside `native/` so the wasm frontend's dependency graph never touches the
kernel workspace's resolution or the `crate_universe` lock. It depends on
`shrike-schemas` *by path* purely to prove the zero-codegen import that drove
the decision.

## What's here

```
eval/spa-spike/
├── Cargo.toml          # standalone workspace; the Leptos CSR SPA (a bin crate)
├── Trunk.toml          # the Leptos-CSR bundler config (trunk)
├── index.html          # trunk entry; carries the card-frame CSP note
├── src/
│   ├── main.rs         # mount to <body>
│   ├── app.rs          # nav + client-side router over the two screens
│   ├── api.rs          # STUBBED /actions/* + /status edge (#505 builds the real one)
│   ├── browser.rs      # Screen 1: collection browser + sandboxed card iframe
│   └── status.rs       # Screen 2: server status pane + modality coverage matrix
└── src-tauri/          # the Tauri v2 desktop shell (serves dist/ as frontendDist)
    ├── Cargo.toml
    ├── build.rs
    ├── tauri.conf.json
    ├── capabilities/default.json
    └── src/main.rs     # window + tray icon
```

The two screens import canonical `shrike_schemas` types directly
(`Note`, `ServerStatus`, `EmbeddingStatus`, `IndexStatus`, …) — the whole point
of the spike. The internally-tagged struct-variant enums
(`IndexStatus::Ready { base }`, `EmbeddingStatus::Running { model, .. }`) are
exactly the shapes a TS codegen path flattens into a non-discriminated bag; here
they are the type, verbatim.

## Step zero — verify shrike-schemas compiles to wasm32

This is the gate the Leptos verdict hangs on. Run it first:

```sh
rustup target add wasm32-unknown-unknown          # once
cd native
cargo build -p shrike-schemas --target wasm32-unknown-unknown
```

Expected: a clean build (the crate is pure serde/serde_json/schemars over
BTreeMap+Cow, no fs/net/time/thread/process/FFI). It passed at the time of the
spike (~6.9s cold). If it ever fails on an un-gateable transitive non-wasm
dependency, the verdict's fallback trigger fires — see the ADR.

> Follow-up the real client should take (recorded in the ADR): feature-gate the
> `schemars` `JsonSchema` derive off in the wasm build — the browser needs only
> serde de/serialize, so the derive is dead weight in the bundle. The crate does
> not yet expose that split; this spike depends on it whole.

## How to run

Prerequisites:

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk            # the Leptos CSR bundler
cargo install tauri-cli        # only for the desktop shell
```

### The SPA in a browser (screens run against in-memory fixtures, no daemon)

```sh
cd eval/spa-spike
trunk serve                    # http://localhost:8080
```

`api.rs` ships fixtures so both screens are live with no server. The real client
flips `list_notes`/`search_notes`/`status` to `gloo-net` fetches against the
daemon's same-origin `/actions/*` and `/status` — the deserialize targets
(shrike-schemas types) do not move.

### The desktop shell (Tauri v2)

```sh
cd eval/spa-spike
cargo tauri dev                # builds the SPA (beforeDevCommand) + opens the window
```

The shell serves the same `dist/` bundle the browser gets (`frontendDist`),
stands up a tray icon, and is where the real app would own the daemon lifecycle
and native dialogs. Desktop is a *packaging* of the server build + a webview
around the same bundle — not a separate frontend.

## The card frame (the one security-critical integration detail)

Anki card HTML is untrusted (shared decks carry arbitrary CSS/JS/MathJax), so it
renders in an `<iframe sandbox="allow-scripts" srcdoc=...>`. The
**security-critical line**: `allow-scripts` **without** `allow-same-origin` —
with both, the sandbox is defeated and card JS shares the host origin;
`allow-scripts` alone gives the frame a unique opaque origin, so scripts run but
cannot script the host. The spike wires no `postMessage` listener (no inbound
surface); when the real client adds one it must gate on `event.source` being
this frame's `contentWindow` and treat the payload as untrusted (a sandboxed
opaque-origin frame posts with `origin: "null"`). See `browser.rs` (`CardFrame`,
with the SECURITY-CRITICAL annotation) and the host-page CSP in
`tauri.conf.json`. This is framework-agnostic; it would be identical under a TS
stack.
