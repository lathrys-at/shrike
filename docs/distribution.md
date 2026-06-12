# Distribution profiles

Settled 2026-06-12 (epic #496 tracks the work; the milestone is *Distribution &
packaging*). This document is the canonical answer to "how is Shrike built,
configured, and shipped" — all packaging and feature-configuration decisions
defer to it. The principle it enforces: **there are exactly four ways to use
Shrike, each with one kernel build, one config story, and no arbitrary
mix-and-match between them.**

## Why this exists

Before this plan, backend selection conflated three orthogonal axes in one
knob: `--embedding-backend {llama|onnx|clip}` mixed *runtime* (in-process vs
subprocess), *model format* (ONNX vs GGUF), and *capability* (text vs
text+image). "Choosing CLIP" was presented as a peer of choosing llama, when
image search is a capability you add, not a backend you marry. llama was
never an engine at all — `shrike-llama-server` is a *lifecycle manager* that
produces a local endpoint the remote engine talks to. Meanwhile the published
wheel was pure-Python and could not run (the native extension was published
nowhere), and the minimal-core feature gating had been deliberately relaxed
by the tokio pivot "until a real lean consumer exists." The distribution
profiles are that consumer.

## The two-layer model

Every decision lives in exactly one of two layers:

- **Build layer** — cargo features on the *binding crates* select which
  engine/manager/store crates exist in the binary (the established mechanism:
  engine crates compiled into one cdylib via features, never trait objects
  across `.so` boundaries). Two builds exist; a third is staged.
- **Config layer** — within what is compiled, the user declares *capabilities*
  (an embedder set, a recognizer map), never backends. A capability that
  fails to load degrades gracefully (absent + logged, #235 semantics); a
  `runtime` that is not compiled into the build is a **config error naming
  the profile** — never a silent no-op.

### The build matrix

| feature | server binding (`shrike-py`, PyO3 → daemon) | mobile binding (C ABI → xcframework/AAR, #504) |
|---|---|---|
| `anki-core` (incl. client sync) | ✅ | ✅ |
| `engine-ort` (text-ONNX + CLIP) | ✅ | ❌ |
| `engine-remote` (embed + describe over HTTP) | ✅ | ✅ (relay offload) |
| `engine-apple` / `engine-android` (platform models) | ❌ **never, on any OS** | ✅ |
| `manage-llama` (llama-server lifecycle) | ✅ | ❌ |
| `manage-syncserver` (anki's sync server as a child, #36) | ✅ | ❌ |
| runtime profile (#393) | multi-thread | `current_thread`, suspension-aware teardown |

A `thin` (wasm) build is a later stage gated on #389 (store traits), #390
(runtime-free engine contracts), and #392 (wire versioning); nothing in the
near-term plan depends on it.

**Client sync vs the sync server are different concerns.** Syncing *with* a
server (#33/#362) is part of `anki-core`, always compiled — there is no other
way to move collection data. *Running* a sync server (#34/#36) is a
manage-class component: a separate anki-shipped process the server profile
can spawn and supervise, exactly like llama-server. Mobile never manages
servers. (The anki-internal-runtime question this raises is #503: patch anki
onto the kernel's runtime handle so the one-runtime invariant becomes
unconditional.)

### The config model

```yaml
embedders:                  # the set of vector spaces (#229/#235); RRF fuses across spaces
  - modalities: [text]                    # a plain text embedder
    runtime: onnx | remote                # the mobile build adds: platform
    model: <path or name>
    endpoint: <url>                       # remote only; omit = the managed llama-server below
    api_key_env: <ENV_VAR>                # uniform on EVERY remote entry; secrets are
    pooling: last                         #   referenced, never inline
  - modalities: [text, image, audio]      # jina-v5-omni as ONE entry — either runtime:
    runtime: remote                       #   remote → llama-server + mmprojs (#501)
    model: jina-embeddings-v5-omni-small  #   onnx → in-process (#237's serving path)

recognizers:                # the #485 engine map, keyed by source
  ocr:      { runtime: onnx | remote | platform, ... }
  asr:      { runtime: onnx | remote | platform, ... }
  describe: { runtime: remote, endpoint: ..., api_key_env: ... }

managed:                    # manage-class components — orthogonal to engines
  llama_server:
    manage: auto | attach | off   # auto = spawn/own a child (today's behavior);
    binary: <path>                # attach = use an existing server; off = cloud/tailnet
    args: [...]
  sync_server:
    manage: auto | off            # server profile only (#36)
```

- Multiple embedder entries are multiple vector spaces, fused by RRF (#229);
  the single-space case is N=1, unchanged. `/status` reports per-space state
  and the modality coverage matrix (#235).
- Recognition rows by profile: `platform` is mobile (Vision, SpeechAnalyzer);
  `remote` serves DIY/desktop (VLM-OCR over chat-completions, ASR over
  `/v1/audio/transcriptions` — #502; describe is #436); `onnx` rows are
  future eval-gated engines.
- Vector-affecting knobs (pooling, mmproj set) are scoped to their entry and
  fold into that entry's fingerprint; a changed entry rebuilds its space.

**The cascade is gone.** The config file is the *only* home for `embedders`/
`recognizers`/`managed` (structured data has no sane flag or env encoding).
The CLI keeps operational flags only (`--config`, `--collection`,
`--host`/`--port`, `--foreground`, `--log-level`, daemon verbs). Environment
variables shrink to `SHRIKE_CONFIG`, `SHRIKE_URL` (client side), and secrets
referenced via `api_key_env`. The Docker story is "mount a config file."
Migration: one release of warn-and-map from the old keys, then removal (#498).

**Deleted by this plan:** `--embedding-backend` and its aliases; every
`--embedding-*` model knob and its env twin; the `onnx`/`clip` extras
(`onnxruntime` becomes a hard dependency of the platform wheel — it is the
dylib carrier for `ort load-dynamic`; a `gpu` extra swaps the carrier); the
silent flag cross-talk (inapplicable knobs become structurally
inexpressible); the README's "choosing a backend" chapter.

## The four profiles

### 1. DIY home server

Source checkout, the platform-tagged `shrike-mcp` wheel (#497 — the published
pure-Python wheel currently cannot run; that fix leads the milestone), or
Docker/brew layered over the same wheel. The `server` kernel build. BYOM:
runtimes `onnx` (in-process, CPU/GPU — the Raspberry Pi case) and `remote`
(managed child llama-server, *attached* existing server, or cloud endpoint
with `api_key_env`), blendable per entry. OCR/ASR/describe via the `remote`
recognizer rows. **No platform engines, even on Apple hardware** — installing
the wheel on a Mac does not change what is compiled. Serves MCP for LLM
clients, `shrike bridge` for stdio/relay (#52), optionally a sync server
(#36) and, later, the web frontend (with the documented reverse-proxy/auth
literacy assumptions). Relay sign-in is supported but manual (headless).

### 2. Desktop app (macOS / Linux / Windows)

One codebase across the three OSes (no app-store distribution → unified is
the obvious choice): a Tauri v2 shell around the SPA (#506 decides the
stack), owning the daemon lifecycle invisibly (logs reachable for
troubleshooting). **The same `server` kernel build as DIY** — desktop is a
*packaging* of the server profile, not a third build. Bundles llama-server +
default models (the default-model choice is #237's eval — both omni paths
are expressible, the plan does not pre-decide). Cloud models by API key.
Anki-desktop coexistence (collection discovery, cooperative locking) is
surfaced by the app. Relay sign-in in-app; optional background sync server.
**On-device platform models arrive only as Tauri sidecars** — e.g. a Swift
binary wrapping NL/Vision/Speech that exposes a remote-protocol endpoint the
kernel calls like any other `remote` entry (sidecars are the third
manage-class member). The kernel build never grows a platform dependency for
desktop, by default or otherwise — which keeps the relay-offload story
symmetric for cross-ecosystem users (macOS desktop, Android phone).

### 3. Mobile app (iOS / Android)

App stores. Fully native, idiomatic UIs — SwiftUI and Jetpack Compose
respectively; **not** Tauri-mobile (good-citizen-of-the-platform is the
explicit call; it buys lifecycle, share sheets, background sync, and
accessibility correctness). The `mobile` kernel build embedded in-process via
the action-exchange C ABI (#504) — **no Python on mobile**: since the kernel
inversion, anki is a Rust dependency and the kernel links with zero CPython
(#226's embedded-CPython sketch is superseded). Platform models by default
on a clean install (the "this is the only installation" assumption), with a
config switch to subordinate to a desktop/DIY install over the relay
(offload via `engine-remote`); sync sign-in obtains the collection (AnkiWeb,
the user's desktop via relay-automagic, or their DIY server). The index is a
per-device cache, rebuilt locally, never synced (#38). The #225 platform
agent adapters (Apple Foundation Models Tool, Android function calling) live
natively in the app.

### 4. Web client (browser)

The SPA (one codebase with the desktop UI), served by a DIY daemon
(same-origin, behind the guard; reverse-proxy/auth or tailnet at the user's
literacy) or — the stretch stage — through the relay (auth + tunneling,
priced after traffic is understood). It speaks **actions over HTTP** (#505),
not MCP: MCP is the agent edge, actions are the UI edge, both adapters over
the same catalog (#225). The wire contract is shrike-schemas verbatim; #392's
versioning becomes mandatory the day a client ships separately from its
daemon. The Rust-wasm-vs-TS stack decision is #506's spike (the wasm
*thin-client kernel* is a separate, later question and not required for any
of this).

## Supersessions and alignments

- **#226** (mobile epic): the embedded-CPython architecture section is
  superseded by the native-app-over-C-ABI shape (#504); the rest stands.
- **#224**: the serialization *property* (one owner, natively GIL-free)
  lives on in the task-actor; the PyO3-embedding mechanism is obsolete for
  mobile.
- **#183** (web frontend): scope narrows to desktop+web (mobile went native);
  the stack spike is #506; actions-over-HTTP (#505) replaces "the existing
  HTTP API" as the client surface.
- **#235** (capability-driven profiles): realized by the config model here.
- **#340** (composable wheels): the hard-dep + platform-wheel decisions
  (#497) absorb the near-term need; the plugin-extension-wheel idea stays
  recorded there for the day install-size pressure returns.
- **#338**: the feature matrix in this document is its revival, scoped to
  the two real consumers.

## Sequencing

#497 (runnable wheel) leads — it is a bug fix against the README's promise.
#498 (config v2) + #499 (build matrix) are the core and land adjacent. #500
(docs rewrite) follows them. #501–#506 are independent of each other; #485
(recognizer map) and #391's long tail are prerequisites where noted. Q-C's
web-stack conversation resolved into #506; the desktop default model stays
an eval (#237), deliberately outside this plan.
