# Architecture

Shrike has two halves that meet at a single boundary:

- A **Rust compute core** (`shrike-core/`) does the heavy lifting: it opens the
  Anki collection, embeds notes, and maintains the search indexes.
- A **Python harness** (`shrike-py/`) assembles the core, exposes it as an MCP
  server and a CLI, and handles process and transport concerns.

They are bridged by one compiled extension, `shrike_native`, built from the
`shrike-pyo3` crate. Everything below the binding is Rust; everything above is
Python.

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (server/ = the host; api/ = the verb surface)
                                      │
                                      └──▶ Harness (harness/harness.py: assembly + verbs)
                                              │
                                              └──▶ AsyncKernel (Rust shrike-kernel via shrike-pyo3)
                                                      ├──▶ collection.anki2 (anki protobuf services)
                                                      ├──▶ IndexOrchestrator (per-modality USearch HNSW)
                                                      │       └──▶ index.usearch (+ index.image.usearch) + index.meta.json
                                                      ├──▶ DerivedEngine (FTS5 trigram sidecar, shrike.db)
                                                      ├──▶ EmbedService slot ◀── shrike-engine
                                                      └──▶ RecognizeService slot ◀── shrike-engine
```

## The kernel and its plugins

The center of the Rust side is the **kernel** (`shrike-kernel`). It owns:

- the Anki collection — reached *only* through Anki's protobuf service layer,
  which is the single point of coupling to Anki;
- the vector index (per-modality HNSW indexes, drift detection, debounced
  saves);
- the derived-text store (the FTS5 sidecar);
- search fusion.

The kernel is a **plugin host**. It never names or imports a specific embedding
model, OCR engine, or transport. Instead it exposes two service slots — an
*embedder* slot and a *recognizer* slot — and runs whatever implementations it
is handed at startup. A layering check enforces this structurally: `shrike-kernel`
may not depend on any engine crate.

The contracts those implementations satisfy live in **`shrike-engine-api`**:

- the async traits the kernel calls (`Embedder`, `ImageEmbedder`, `Recognizer`);
- the sync compute traits an engine implements (`EmbedText`, `EmbedImages`,
  `RecognizeMedia`);
- one `Blocking` adapter that runs a sync engine on the runtime's blocking pool;
- the batch-safety probe.

An engine can conform two ways, picked by its natural shape. A sync engine (ONNX
inference, a blocking HTTP client) implements the sync compute traits and is
bridged by the `Blocking` adapter. A naturally-async engine implements the async
traits directly. Either way the kernel sees only the async traits.

Concrete engines live in one crate, **`shrike-engine`**, feature-gated by family:

- `onnx` — in-process ONNX text encoders and CLIP, via `ort`;
- `remote` — any OpenAI-compatible embeddings endpoint (`remote::embed`) and a
  VLM image→text describer (`remote::describe`), sharing one SSRF-pinned HTTP
  client (`remote::http`);
- `apple` (feature `engine-apple`) — the Apple Vision/Speech recognizers, which
  parse the raw JSON produced by the Swift glue in `shrike-platform`.

`shrike-llama-server` is a *lifecycle manager*, not an engine: it spawns and
supervises a local `llama-server` process that the `remote` engine then talks
to. Talking to an endpoint and launching one are deliberately separate concerns —
a cloud deployment uses the remote engine with no manager at all.

Because the kernel only sees the slot contracts, the same kernel code serves a
laptop running ONNX, a server talking to llama-server, and (in mobile builds) a
phone running Apple Vision. The difference is entirely in what the harness
attaches.

## The harness assembles the kernel

`shrike-py/src/shrike/harness/harness.py` is the assembling layer. At startup it:

1. opens one `AsyncKernel` (the asyncio-bound kernel from `shrike-pyo3`) on the
   event loop;
2. builds the engines named in the config;
3. attaches them to the kernel's slots (`attach_embedder` / `attach_recognizer`);
4. starts serving.

Every production backend attaches **natively**: once attached, an embed or a
recognition runs entirely in Rust and never crosses back into Python. The Python
facades (`OnnxBackend`, `ClipBackend`, `LlamaServerBackend`) keep only the
construction-time work — file and provider resolution, the batch probe,
fingerprint assembly, `health()` — and hand the kernel a native composition. The
`PyEmbedder` / `PyRecognizer` capture path remains as an escape hatch for custom
and test backends; no production path uses it.

Write operations route through a fixed set of kernel ops — `upsert_notes_json`,
`delete_notes`, `reindex_notes`, `forget_notes`, `metadata_changed` — so the
index and derived store stay consistent with the collection by construction.

## The runtime

The kernel owns its async runtime — ordinary async Rust on a `current_thread` tokio
runtime in `shrike_kernel::runtime` — but it spawns **none of its own threads**. The
harness owns every thread the kernel uses; shrike-core's job is to define the threads'
work and hand them out, not to provision them. The harness installs the runtime via
`init_driven_runtime` before any kernel op, commits **N + 2** threads to the kernel
for its life, and submits work on its own request threads.

(This replaced an earlier model in which the kernel self-drove a lazy multi-thread
tokio runtime and engine/sync dispatch bottomed out in `spawn_blocking`; that model is
gone — its last transitional remnant, the lazy multi-thread default, was removed, so
the driven `current_thread` runtime is now the only model. An op before the harness
installs the runtime panics; there is no fallback.)

The N + 2 committed threads come from three drive entries:

- **`drive_io` ×1** owns and drives tokio's IO + timer drivers via the runtime's
  `block_on` until shutdown, and runs the async executor: the collection actor's
  dispatch, the debounced saver's timers, every spawned op. The first `block_on`
  caller takes ownership of the drivers and the rest hook into it, so this thread
  must win that race (see the startup barrier below).
- **`drive_sync` ×1** is the `SerializedCollection` actor's execution thread —
  serialized anki / collection (SQLite) work and the anki client-sync
  release-run-reopen. One thread is a consequence of anki's single-writer
  collection, not a tuning choice.
- **`drive_compute` ×N** runs CPU-bound engine compute (ort/CLIP, apple via
  `Blocking<E>`) and blocking-fs leaves. This is the only place real parallelism
  lives, so the engine search/batch overlap property is **"N ≥ 2"**; N is the
  harness's choice, sized to its cores.

A few rules keep it correct:

- **The collection is a task-actor.** One spawned task owns the collection and
  processes jobs from an mpsc queue in order, its work executing on `drive_sync`.
  Serialization comes from the task's sequential loop, not from thread affinity —
  there is no `block_in_place`.
- **Engine compute is dispatched eagerly, never inline.** `dispatch_compute` (the
  `Blocking<E>` adapter's seam) schedules each engine call onto `drive_compute` inside
  the call, which preserves the search/batch overlap. Independent batch futures are
  `try_join`ed.
- **Each public op is spawned, then awaited across the binding.** The binding calls
  `spawn_op` to put an op on the runtime and observes a oneshot-backed completion
  future. The server awaits it through the **asyncio bridge** (its MCP handlers are
  coroutines on one loop and must not block it); a threaded host (cabi, tests) uses
  `submit_blocking` to submit and block its request thread. Dropping the future
  stops *observing* the op; it never aborts the work, because a half-applied
  collection write would be corruption.
- **Startup driver-ownership barrier.** tokio gives IO/timer-driver ownership to the
  first `block_on` caller, which MUST be `drive_io`. The harness spawns `drive_io`
  first, then runs `runtime_probe` — schedule a trivial executor-only op and block
  until it completes, proving the IO thread owns the drivers — *before* spawning the
  `drive_sync`/`drive_compute` threads. Without it a leaf could win ownership and
  timers/IO would advance only while that leaf parks in `recv`.
- **Invariant: a sync op never runs on a runtime worker thread.** Anki's sync code
  paths call `block_on`, which panics from inside a runtime worker. Every
  kernel-side sync op routes through `dispatch_sync`, which enqueues onto the
  non-runtime `drive_sync` thread — so anki's `block_on` is legal there *by
  construction*. The invariant is structural, not a dispatch discipline a caller
  must remember. The `sync_dispatch_pin` test in `shrike_kernel::runtime` pins it.
  (Anki keeps its own internal runtime, cold today because Shrike dispatches none of
  its sync/AnkiWeb services; client sync will wake it, at which point two runtimes
  coexist and this rule is what keeps sync safe.)
- **Deadlock leaf-invariant.** Every pool job is a leaf: an enqueued
  `drive_sync`/`drive_compute` job never enqueues-and-awaits further pool work. The
  read→compute→write orchestration fans out and awaits on the async side
  (`drive_io`), and compute is handed its inputs after the actor reads, so a fixed
  pool can't exhaust itself. A debug-build thread-local tripwire asserts it.

**Per-binding thread provisioning.** The binding *exposes* the drive entries; the
harness *above* it owns the threads. The server's `driven_runtime.py` spawns N + 2
`threading.Thread`s into the GIL-releasing pyo3 `drive_io`/`drive_sync`/`drive_compute`
entries — GIL-released for each thread's life, so native compute runs with real
parallelism (the `PyEmbedder`/`PyRecognizer` capture seam re-acquires the GIL only on
the test/custom path). `shrike-cabi` exposes blocking C entries
(`shrike_drive_io`/`shrike_drive_sync`/`shrike_drive_compute` + `shrike_runtime_probe`)
and the native Swift/Kotlin host spawns and joins the OS threads; cabi spawns nothing,
because it *is* shrike-core. Shutdown is host-shaped: the server drains its ops through
the bridge and then closes the pools, while cabi folds a bounded in-flight drain into
`shrike_runtime_shutdown` (its detached, callback-only ops have no bridge to await).

The remote engines are **async-direct**: `embed-remote`/`describe-remote` implement the
async `Embedder`/`Recognizer` traits on `drive_io` (with `media_fetch`) behind an
SSRF-pinned async HTTP client, so `drive_compute` stays pure CPU and engine-api's two
conformance routes map onto the two pools (sync-compute-behind-`Blocking<E>` →
`drive_compute`, async-direct → `drive_io`).

## The action exchange is the host boundary

The unit of work between a host and the kernel is an **action**: conceptually
`async fn(action_request) -> response`. This exchange is the boundary every host
adapts to. Today the only host is the Python MCP server; a future thin client,
relay, or native (Swift/Kotlin) app would adapt the same exchange rather than
reach into the kernel.

The exchange evolves additively. A breaking change to an action ships as a *new
action name* (e.g. `upsert_notes_v2`) carrying its own schema types, while the old
action keeps serving its old types. A single `WIRE_PROTOCOL_VERSION` constant is
the backstop, bumped only when the exchange fabric itself breaks (envelope
semantics, the error taxonomy). The MCP tool surface follows the same rule: a
breaking tool change is a new tool name, because external clients discover tools
via `tools/list` and cannot be handshaken.

## See also

- [`layout.md`](layout.md) — where each crate and package lives on disk.
- [`decisions.md`](decisions.md) — why the kernel owns its runtime, why engines
  are plugins, and the alternatives that were rejected.
