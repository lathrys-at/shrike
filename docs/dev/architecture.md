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

The kernel owns its async runtime: ordinary async Rust on a process-global tokio
runtime (`shrike_kernel::runtime`). Only the `Handle` escapes; `init_runtime` is
the seam that can install a single-threaded runtime (the kernel runs end-to-end
on one thread in that mode, which is what keeps the design honest about not
relying on thread affinity).

A few rules keep it correct:

- **The collection is a task-actor.** One spawned task owns the collection and
  processes jobs from an mpsc queue in order. Serialization comes from the task's
  sequential loop, not from thread affinity — there is no `block_in_place`.
- **Engine compute runs on the blocking pool.** The `Blocking<E>` adapter wraps
  each engine call in `spawn_blocking` (eagerly, inside the call, which preserves
  the search/batch overlap). Independent batch futures are `try_join`ed.
- **Each public op is spawned, then awaited across the binding.** The binding
  calls `spawn_op` to put an op on the runtime and awaits a oneshot-backed
  completion future through the asyncio bridge. Dropping that future stops
  *observing* the op; it never aborts the work, because a half-applied collection
  write would be corruption.
- **Invariant: a sync op never runs on a runtime worker thread.** Anki's sync
  code paths call `block_on`, which panics when invoked from inside a runtime
  worker. Any kernel-side sync op must therefore dispatch via `spawn_blocking` —
  a blocking-pool thread is not a runtime context, so `block_on` is legal there.
  This is the same seam the Python captures use. The `sync_dispatch_pin` test in
  `shrike_kernel::runtime` pins it. (Anki keeps its own internal runtime, which
  stays cold today because Shrike dispatches none of its sync/AnkiWeb services;
  client sync will wake it, at which point two runtimes coexist and the rule above
  is what keeps sync safe.)

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
