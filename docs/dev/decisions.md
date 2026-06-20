# Design decisions

The "why" behind non-obvious choices in shipped work — reasoning not recoverable from
the code (the "what") or the issue tracker (future work). How each piece works today is
in the other `docs/dev/` files. New entries go on top of their section.

## Semantic search and the vector index

### CLIP gives image-by-text retrieval in one shared space

A CLIP dual-encoder puts text and images in one space, so a text query can retrieve a
card by the content of its image. The embedding unit is a note's text vector plus one
vector per image, all under the `note_id` key (USearch `multi=True`): removing a note
drops all its vectors, and results dedup to the best vector per note. The **modality
gap** is the design driver — in one cosine index a text query sits closer to text
vectors than image vectors, so image hits are additive, not rank-dominant. That is why
each modality ranks as its own RRF signal behind an activation gate, not as one deduped
cosine ranking (a rank combiner is blind to the gap's constant offset).

### The activation gate floors a modality by its own typical match

Separate RRF signals neutralize the gap's constant offset but are blind to magnitude
*within* a modality, so an unfloored image signal injects a collection's top image cards
into every query. The **activation gate** lets a non-text modality contribute only when
its best match clears `mean + margin·std` of that modality's typical best match. It is
calibrated **offline from the index, not query traffic** — typical-best-match is a
property of the embedding space, so `_calibrate_activation` samples stored text vectors
as pseudo-queries and stores `{n, mean, std}` per modality in `index.meta.json`
(cold-start safe, recomputed on a model change, no search logs). The gate is binary per
modality (RRF already down-weights lower ranks), and **text is never gated** (it has its
own `threshold`; the gap is cross-modal, so a text-only collection has nothing to
calibrate). Pure addition: an absent `activation` key means no floor until the next
rebuild.

### Per-modality retrieval splits the index, not the score

A USearch hit returns the `note_id` and distance but **not which vector matched**, so a
single `multi=True` index yields only one deduped ranking — and across the gap a text
query ranks every text vector above every image vector, leaving image-only cards
unreachable. The fix is **per-modality sub-indexes** (`index.usearch` text,
`index.image.usearch` images); separate indexes are the only way to recover a
per-modality ranking, and each enters RRF as its own signal. The image ranking is **not**
thresholded (the `threshold` knob is a text-cosine floor that would kill every
gap-depressed image hit — that's the activation gate's job). Migration is a one-way schema
marker, not a converter: a text-only pre-split index loads losslessly; an image-capable
backend meeting an old mixed index rebuilds once.

### Search fuses signals by rank (RRF), not normalized score

`search_notes` blends signals on incommensurable scales (cosine in a narrow band, exact
near-binary, cross-modal cosine offset below within-modal). Normalize-and-sum inherits
every pathology and makes a note's order depend on what else was retrieved. **Reciprocal
Rank Fusion** instead ranks each signal independently and scores a note `Σ wₛ·1/(k+rankₛ)`
(k=60): rank discards magnitude, a missing signal contributes nothing (graceful
degradation), and orderings are stable across queries. The one thing it loses is
magnitude — so a **priority tier** floats literal exact hits above the rest. Because RRF
fuses rank positions, the CLIP gap's constant offset is invisible, which is what lets the
separate text/image rankings neutralize it with no normalization. The combiner
(`shrike_kernel::fusion`, with `search_fusion.py` as the frozen reference) is pure and
reports which signals contributed at what rank.

### Derived data lives in a sidecar `shrike.db`, not in `collection.anki2`

Locally-derived data (a trigram lexical index now; OCR/ASR/describe text later) lives in a
sidecar SQLite file in the cache dir, **not** in `collection.anki2`: Anki's sync, "Check
Database", media check, and migrations own that schema, so a foreign table risks being
dropped — and it would ship rebuildable data over sync. A sidecar is safe and rebuildable
from the source of truth, and is the natural relay sync target, so rows are source-seamed
`(note_id, source, ref)` from day one. Two consequences: it **feeds two signals** — a
quoted-phrase trigram MATCH is the `exact` candidate source, a trigram-OR MATCH ranked by
bm25 is `fuzzy` (both fall back to a linear `find_notes` scan without FTS5 or below trigram
length) — and **provenance is source-aware**, so a hit can report which derived text
matched. A future VLM describe source goes to the embedding space only, never the trigram
index: a literal hit on metadata the user never sees can't be cleanly explained.

### USearch stays the index

USearch (HNSW + quantization) is the vector index, and the cross-platform plan keeps it.
No native option beats it for a zero-server, local-first design (Apple offers only
brute-force kNN, Android no first-party ANN, and the portable alternatives don't win on
portability *and* performance), and it runs natively on desktop and both mobile platforms.
Revisit only on a specific measured trigger — SQLite-co-located vectors (`sqlite-vec`) or
Android object-DB ergonomics (ObjectBox) — and on a demonstrated win.

### Embedding backends are pluggable behind one small protocol

Going multimodal means swapping models and runtimes wholesale, and text-only embedding
must stay first-class forever (the suites depend on small text-only models). So a minimal
`EmbedderBackend` protocol sits in front of `LlamaServerBackend`, `OnnxBackend`,
`ClipBackend`, and `RemoteBackend`; the index never learns which backend it has, so drift,
hashing, and persistence stay backend-agnostic. Three choices:

- **The fingerprint is namespaced by family** (`onnx:…` vs llama's `meta:`/`file:…`), so
  the same model under two runtimes never shares a space, and an existing index isn't
  forced to rebuild on upgrade.
- **`modalities` is a declared capability**, so media-by-content search lights up where
  vectors exist and silently returns nothing where they don't.
- **Pooling is folded into the fingerprint; normalization is not** — pooling changes a
  vector's direction, while L2 normalization changes only magnitude and USearch's `cos`
  metric is scale-invariant.

The non-obvious operational rule: **batch safety is probed, not assumed.** int8
dynamic-quant exports compute activation scales over the whole batch, so a batched embed
would make a note's vector depend on its batch-mates and break the
reconcile-equals-rebuild invariant (fp is bit-exact batched). So every backend's `start()`
embeds a magnitude-spiked probe set serially and batched, compares within a tolerance above
float noise and below quant drift, then batches only up to the proven size. Details in
`embedding-and-recognition.md`.

### The index is a derived cache, never a co-equal store

The collection (SQLite) is always the source of truth; the index is a rebuildable
projection. It may lag the collection (stale results) but the collection never lags it
(data ops are always correct) — which is what lets drift detection be a single `col.mod`
comparison plus a background rebuild. Scheme and fingerprint details in
`indexing-and-search.md`.

### `upsert_notes` is write-only; neighbours come from the search path

`upsert_notes` returns per-item `status` + `id` and nothing more — it does not attach the
written notes' nearest neighbours. The neighbour list was never a read-after-write on the
index (it embedded each written note's text as a *query* and searched with that note
excluded), so it was a query-embed plus a search bolted onto every write's response path —
a multi-second remote-embed N+1 on the latency-sensitive write. The same information is a
`search_notes` of a card's content, which the caller runs *before* writing (the authoring
dup-check) or keyed on the id *after* (`search_notes(ids=[…])`) — a read on the search path
where it belongs, not a cost every write pays. The dedup/activation calibration sampler that
fed off the neighbour search's scores re-sources to the ordinary search path.

### Semantic duplicate detection is not a separate feature

There's no dedicated semantic-duplicate endpoint. A high similarity score in `search_notes`
*is* the soft-duplicate signal, and the caller sets its own threshold; a second code path
with a built-in cutoff would be redundant with a worse interface.

### Anki's exact duplicate rule lives inside `upsert_notes`

Anki's precise rule — a note duplicates another sharing its first field with an existing
note of the same type — is folded into `upsert_notes` as an `on_duplicate` policy (default
`error`) plus `dry_run`, not a standalone `canAddNotes`: a separate check is racy
(check-then-write TOCTOU), only actionable by another call, and overlaps the per-item result
union. `dry_run` covers the one thing a checker offers (a zero-write preview). Structurally
invalid notes (empty first field, broken cloze) are always errors; the default is `error`
because silently writing a duplicate is almost always a mistake.

### One query, many retrieval mechanisms, annotated results

`search_notes` runs one `queries` input through every retrieval mechanism and folds the
evidence into one list, rather than a param/tool per mechanism: each match carries a `score`
when semantically ranked and a `substring` annotation when it occurs literally. A
`--substring` flag was rejected — it reads as a filter and forces a union-vs-intersection
decision, whereas "matched every way we can, results tell you how" needs no mode. It
degrades gracefully (index down → exact matches plus an advisory), and the optional evidence
fields are where a future fuzzy/n-gram signal plugs in.

### The raw Anki query is its own tool, with the full grammar

The raw Anki search escape hatch is `collection_query` / `shrike search query` — a dedicated
tool, not a `query` param on `list_notes`. **Full grammar, no whitelist**: the string goes
straight to `col.find_notes`, so every operator works including `is:due` / `prop:` /
`rated:`. That doesn't brush the non-review stance, which is about not *performing* review
operations — this is read-only, returning notes and reviewing nothing; whitelisting would
mean re-implementing Anki's parser. It reuses `list_notes`' shape so all three retrieval
surfaces return one note shape; a malformed expression is a `ToolInputError`.

### Find-and-replace edits via Anki's engine; a scope is required

`find_replace_notes` runs through Anki's `col.find_and_replace` (Rust regex — linear-time,
undo-able), not a re-implementation, and **requires a scope**
(`deck`/`tags`/`note_type`/`ids`) so a collection-wide edit is always explicit. It changes
field bodies, so changed notes are re-embedded (found by diffing `notes.flds` before/after,
since `note.mod` is only second-resolution). The dry-run preview is rendered in Python
(exact for literal, illustrative for regex); the apply is authoritative.

## Tags

### Setting tags is a full replace; add/remove is a separate operation

*Setting* tags (`upsert_notes` `{id, tags}`, `note update --tags`, `update_note_tags --set`)
leaves the note with exactly the set sent — replace never merges. An additive/subtractive
`mode` on the `tags` field was rejected as the bag-of-optionals smell the schema house style
warns against; additive editing is its own tool with non-overlapping fields (`set` exclusive
with `add`/`remove`). Tags aren't embedding text, so these ops bump `col.mod` and advance the
index watermark without re-embedding.

### Tag rename matches exactly

`rename_tag` matches the tag *exactly* (find notes carrying it, then swap), not a substring
find/replace, so renaming `jp` never touches `jp-verbs`.

## Decks

### Deck deletion is empty-only; decks never merge

`delete_decks` refuses unless the deck and every subdeck is empty — empty it first by moving
its notes elsewhere. The payoff: **deck deletion can never delete a note**, so it never
touches the note set or the index. Renaming a deck onto an existing name is an **error**, not
a merge (Anki would silently disambiguate `B` → `B+`); `upsert_decks` mirrors `upsert_notes`
(id = rename, absent = create) with no hidden merge. Like tags, deck ops bump `col.mod`
without changing a vector.

### Deck references accept name, numeric ID, or `#id`

Anywhere a deck is *referenced*, the value may be a name, a bare numeric ID, or a
`#`-prefixed ID (resolved in `CollectionWrapper._resolve_deck_ref`): `#<id>` is always an ID;
a bare integer is tried as an ID then falls back to a literal name (so a deck named `123` is
still reachable); anything else is a name. On create, an unknown name is auto-created but an
unknown `#id` is an error.

## Note types

### Field and template updates are applied by position, preserving note data

`upsert_note_types` replaces a note type's whole `fields`/`templates` list on update, and
Anki migrates field values and template cards by **ordinal** — the entry at position N keeps
its data as long as N survives. (An early version rebuilt the lists from fresh ordinal-less
objects, so Anki saw every field as removed and silently blanked all content or deleted all
cards; the fix reuses the existing dicts in place.) So a whole-list replace is data-safe:
rename/edit-in-place preserve data, lengthening appends, shortening drops only trailing
entries. A real reorder is necessarily separate — a positional name swap reads as two renames.

### Identity-based field/template ops are separate tools

The move/insert/non-trailing-remove that position-replace can't express lives in
`update_note_type_fields` / `update_note_type_templates` — `add`/`remove`/`rename`/`reposition`
ops addressed by **name**. They're separate from `upsert_note_types` because the contracts
differ: declarative "the fields are now exactly this list" (position-keyed) vs imperative
"move X to 0" (identity-keyed); conflating them makes "is `["B","A"]` a reorder or two
renames?" ambiguous. They delegate to Anki's data-safe primitives and are atomic (validate the
whole op sequence against a simulated name list, then apply with one `update_dict`). With real
movers available, the two are **reconciled**: the positional replace refuses any update where
an existing name changes position (which would silently re-label note data), pointing at the
identity tool.

### `find_replace_note_types` rewrites template text, not fields

A different operation from `find_replace_notes`: it rewrites one note type's template HTML
(`qfmt`/`afmt`) and CSS, touching no field values, so the two share no code. Templates/CSS
aren't embedding text, so it bumps `col.mod` without re-embedding; `match_case` defaults
**true** because template/CSS is code; literal mode inserts the replacement verbatim
(`re.escape`d, capture refs only under `regex`). It doesn't rename a field — Anki's
`rename_field` already rewrites template references for that.

### Field editor metadata: getter in `collection_info`, dedicated setter

Per-field `font`/`size`/`description` ride the existing `note_type_details` block on
`collection_info` (the getter — it's just more of the type's definition), but the **setter is
dedicated** (`update_note_type_field_metadata`) rather than an op on `update_note_type_fields`,
because the index policies are opposite: a structural field op lets a drift-rebuild happen (a
removed field changes embedding text), while metadata changes none and wants the
col_mod-bump-no-re-embed path. One tool, one index policy.

### Changing a note's type is a dedicated tool

`upsert_notes` refuses a type change; `migrate_note_type` wraps Anki's `col.models.change`. The
point is **preserving history** — it keeps note IDs and carries each card's scheduling across
mapped templates, which delete-and-recreate would lose. Folding it into `upsert_notes` would
make `field_map`/`template_map` a conditional sub-mode with an ambiguous interaction with the
item's own `fields`, burying a destructive migration in a routine bulk write. The **map is
explicit** — a source field absent from it is dropped (reported), and unknown names /
two-to-one / mixed source types all error (auto-mapping was rejected because the whole risk is
silent content loss). Applies by default with a `dry_run` preview; re-embeds the migrated notes
since the remap changes embedding text.

## Collection maintenance

### One `collection_prune` tool, not scattered cleanups

The tidy-up chores — clear unused tags, remove empty notes/cards, trash unused media — live
behind one `collection_prune` with opt-in flags (none → run all), since they're all
whole-collection maintenance passes. Three decisions:

- **Apply by default with a `--dry-run` preview**, unified with `find_replace_notes` and
  `migrate_note_type`; the CLI's destructive blast radius is contained by the
  preview-and-confirm gate, not by a preview-by-default value.
- **"Empty" is media-safe** — a note is empty only if every field is blank with no text *and*
  no media, so an image-/audio-only card is never pruned (Anki's "generates no cards"
  definition was rejected as silent data loss).
- **Order: notes → cards → tags → media**, so anything orphaned by the deletions is cleared in
  the same call.

Index handling is mixed (empty-note/card removal drops vectors; clearing tags/media leaves them
valid), which is why prune isn't a plain metadata-bump op.

## Collection lifecycle

### Busy is a typed error, not a per-tool response variant

Under cooperative locking a re-acquire can fail because Anki holds the collection. "database is
locked" is orthogonal to every tool's response (the op never ran), so a discriminated per-tool
`CollectionBusy` variant was rejected; it's an **error class with a stable wire code**
(`COLLECTION_BUSY_CODE`), surfaced as an MCP `isError` and mapped by `ShrikeClient` to a
catchable `CollectionBusyError`. It returns busy **immediately, no server-side retry** — the
dominant case is Anki open for a whole session, where a retry just adds latency before the
inevitable busy.

### Cooperative locking is opt-in, time-sliced, with a 5 s idle hold

By default the daemon holds Anki's exclusive lock for life — ideal for the embedding workflow
but blocking Anki desktop's launch. `--cooperative-lock` opens on demand and releases after an
idle window.

- **Opt-in, default off** until proven.
- **Time-slicing, not sharing** — Anki holds the collection for its whole runtime, so the win
  is precise (an idle daemon stops blocking launch); contention is a clean SQLite busy error.
- **5 s idle hold** — SQLite's conventional `busy_timeout`, and the held resource actively
  blocks a human while re-acquiring is a cheap local open. Tunable via `--lock-hold-seconds`.
- **Drift re-checked per re-acquire, col_mod-only** — the acquire hook rebuilds on a `col.mod`
  mismatch but skips the model fingerprint (handled by `/embedding/start`), avoiding a
  llama-server round-trip each time.
- **`server.lock` and the collection lock are distinct** — "daemon alive" vs "collection held",
  both on `/status`.

### Reload is a control endpoint sharing the cooperative-lock primitive

`shrike collection reload` / `POST /reload` closes and re-opens the collection and re-checks
drift. Its value while the lock is held permanently is narrow (only a file-level swap — restored
backup, sync — changes the file underneath), but it introduces the primitive cooperative locking
needs: `reopen` plus `run`/`run_sync` reading `self.col` **at execution time on the worker
thread**, so an op queued after a reopen runs against the new handle. Control endpoint and CLI,
not an MCP tool; it touches only the collection lock.

## Tag-vector namespacing

Tag centroids live in the **same index engine as the notes**, under a distinct named space
(`tag.text`), not a separate file — the per-modality split already built the named-space
abstraction. One engine means one `model_id`/`ndim`/`metric` by construction and one
persistence path, and no-leakage is **structural**: note searches are scoped to the note
modalities, so a tag key can't surface from a note query (no post-filter; note ids are epoch-ms
with no safe disjoint range for a key-range trick). Keys are `blake2b-8(tag)` masked positive;
the key→tag map is **in-memory only**, rebuilt each recompute, so a stale file can never
mislabel a key. A centroid is a pure function of member text vectors and one membership pass over
`notes.tags` (hierarchy rolled up by `::`-prefix), so there's no watermark and no incremental
diff — the whole set recomputes at the tail of every index-changing op, best-effort (it never
fails the op it rides on). Hygiene before vectors: a member floor, a coverage cap, a meta-tag
blocklist.

## Architecture

### The serial ingest drain has no per-embed timeout

The ingest drain is single-flight: `process_batch` awaits `embedder.embed` with no
`tokio::time::timeout` on the path, and the drain's panic boundary
(`AssertUnwindSafe(work).catch_unwind().await`) does not cover a hang — a `Pending`-forever
future never returns control to be caught. So an `embed` future that never resolves (and
never errors) wedges the sole writer: the watermark never advances and `flush`/`shutdown`
block behind it.

We deliberately do **not** add a per-embed timeout. Every shipping embedder resolves in
bounded time — ONNX is bounded local compute; the llama-server backend has a 60s HTTP
timeout plus bounded retry, so a wedged server surfaces as a transport *error*, not a hang —
which makes the unbounded-hang case unreachable through any shipping backend. It needs a
custom embedder whose future never resolves *and* never errors — no shipping backend does;
the kernel's `GatedEmbedder` test double synthesizes exactly this. A drain-side timeout would
be an untunable guess (set
too low it self-wedges under load on a slow large batch) and could not reclaim a route-1 job
already running on the compute pool; it would add complexity to the most ordering-critical
path to defend a non-production hazard. The contract is documented on the `Embedder` trait
instead (#886).

The only thing that recovers a *real* wedge is the recovery side — watch `is_settled()`
staying false with outstanding writes, surface it as a degraded `/status`, and let the host
restart the embedder. That is filed as low-priority defense-in-depth (#891), not built here.

### Code-bearing package `__init__.py` files need their own marker target

Bazel's `legacy_create_init` (on by default — `--incompatible_default_to_explicit_init_py` is
unset) synthesizes an EMPTY `__init__.py` into a target's runfiles for any package dir that
holds a `.py` but no `__init__.py`. Harmless for the plain-marker subpackages, but
`server/__init__.py` and `recognition/__init__.py` carry real code, so they live in their own
targets (`:server`, `:recognition`), not the `:pkg` marker.

The trap behind #872: `:export_store` (`server/export_store.py`) materializes the `server/`
package dir but is dep-able WITHOUT `:server` (e.g. `//shrike-py/tests/unit:tools`). Bazel then
synthesizes an EMPTY `server/__init__.py` for that target. Under `bazel test //...` on Linux —
where runfiles are symlinks into one shared, writable execroot source — the bazel server's
non-atomic empty-init write races concurrent reads of the real `server/__init__.py` by the
`:server`-dependent targets, so they import an empty `shrike.server` (`main` absent) →
intermittent `ImportError`/`AttributeError`. macOS never reproduced (APFS COW isolates each
runfiles tree); per-action sandboxing (#894) couldn't fix it because the writer is the bazel
server itself, not a sandboxed spawn.

Fix: a `:server_pkg` marker carrying the real `server/__init__.py`, depended on by BOTH `:server`
and `:export_store`, so every runfiles tree that materializes `server/` also carries the real
init and `legacy_create_init` never synthesizes an empty one. The blanket
`--incompatible_default_to_explicit_init_py=true` was rejected: it strips auto-init for the
genuinely marker-only packages too, breaking `tests.*` imports. Any future code-bearing
`__init__.py` pulled in by a target that does not dep its owner needs the same split.

### The engine-plugin architecture: a pure kernel

The kernel composes engines it is *given* — never naming one. Contracts live in the leaf crate
`shrike-engine-api`; concrete engines are feature-gated families in `shrike-engine`; hosts build
engines from config and attach them to named slots; a layering check forbids `shrike-kernel`
depending on any engine crate. Decisions:

- **Two conformance routes by the engine's shape** — the kernel sees only async traits; a
  naturally-sync engine implements sync compute traits bridged by an adapter, a naturally-async
  one implements the async traits directly. Topology stays kernel-owned (independent engine
  futures `try_join`ed); a host-described execution graph was rejected as pushing the kernel's
  invariants into a meta-layer every host re-implements.
- **Named slots, not a registry** — two slots compose cleanly; a keyed registry waits for a third
  capability kind (ASR).
- **Identity and batch policy are host-assembled** — fingerprint strings fold host policy,
  `safe_batch` comes from the host probe, `WithPolicy` carries them onto a pure engine.
- **Engines and managers are distinct** — talking to an endpoint (`remote`) is separate from
  launching one (`shrike-llama-server`, excluded from mobile).
- **Engine crates link in via cargo features**, never trait objects across `.so` boundaries (no
  stable Rust ABI).
- **The Python facades stay, as assembly** — they keep construction-time work and hand the kernel
  a native composition; `PyEmbedder`/`PyRecognizer` remain the test/custom escape hatch, on no
  production path.

### The kernel owns its runtime (tokio)

The kernel installs and owns a process-global tokio runtime and is plain async Rust — no executor
traits. tokio targets every platform here and was already in the tree (via anki), so owning it
added nothing and deleted a large adaptation layer (a hand-rolled executor, a timer host over the
asyncio loop, a polling bridge) that bought runtime-portability the platforms never demanded. The
shape:

- Runtime in `shrike_kernel::runtime`; only the `Handle` escapes. The kernel spawns no threads of
  its own — the harness drives a `current_thread` runtime and donates every thread (see the
  driven-model decision below).
- **The collection is a task-actor** — one spawned task owns the core and runs jobs off an mpsc,
  FIFO by construction (serialization from the loop, not thread affinity; no `block_in_place`).
- **The action exchange is the edge** — `spawn_op` returns a oneshot-backed future pollable
  anywhere; dropping it **detaches** (never a `JoinHandle` abort, which could corrupt a
  half-applied write).
- **Timers** ride `tokio::time`; **engine execution** is one eager dispatch through `Blocking<E>`,
  routed to the driven compute pool.

### anki keeps its sync runtime; the kernel pins sync off the runtime worker

The invariant is "**sync ops never run on a runtime worker**," not "one runtime per process."
anki's internal runtime stays cold today (Shrike dispatches none of its sync/AnkiWeb services,
pinned by `runtime_singularity`), but client sync will wake it, so two runtimes will coexist.
Patching anki to keep one runtime was rejected because (1) the patch mechanism is **Bazel-only**,
so it would make `cargo test` and Bazel disagree on a correctness invariant unless we fork anki,
and (2) the panic hazard is **runtime-agnostic** — `block_on` panics from inside *any* runtime
worker, so injecting the kernel's handle wouldn't remove it. The invariant is now **structural**:
every kernel-side collection op routes through `dispatch_collection`, which enqueues onto the non-runtime
`drive_collection` thread (see the driven-model decision below), where anki's `block_on` is legal by
construction — not by a `spawn_blocking` discipline a caller must remember. The `collection_dispatch_pin`
test pins the hazard and the safe site.

### One threading model across both bindings: the harness drives, shrike-core spawns nothing

The two host bindings drove the kernel with divergent threading. The PyO3 server rode the kernel's
lazy **multi-thread** runtime (tokio owned and spawned its own worker + blocking-pool threads) and
bridged completion to the asyncio loop; `shrike-cabi` built a **`current_thread`** runtime with one
dedicated thread parked in `block_on` to drive it, firing detached ops whose completion arrived via
a C callback, with a bespoke iOS suspend-drain on top. So shrike-core dictated thread *policy* and
each binding reinvented how the runtime was driven.

The north star unifies them: **the harness owns every thread; shrike-core spawns none** (tokio's
`spawn_blocking` pool included). shrike-core keeps owning its runtime, now a **`current_thread`**
runtime it drives no threads for. The harness commits **N + 2** threads to the kernel for its life,
through three drive entries, and submits work on its own request threads:

- **`drive_io` ×1** — owns and drives tokio's IO + timer drivers via the runtime's `block_on` until
  shutdown, and runs the async executor (the collection actor's dispatch, the debounced saver's
  timers, every spawned op). Per tokio, the first `block_on` caller takes ownership of the drivers
  and the others hook into it, so this thread must win that race (see the barrier below).
- **`drive_collection` ×1** — the `SerializedCollection` actor's execution thread: serialized anki /
  collection (SQLite) work and the anki client-sync release-run-reopen. One thread is a consequence
  of anki's single-writer collection, not a tuning choice. Because it is never a runtime context,
  anki's own `block_on` is legal here by construction — which makes the sync-dispatch invariant
  (§ above) **structural** instead of a `spawn_blocking` discipline. `collection_dispatch_pin` is
  re-expressed against this dispatch site.
- **`drive_compute` ×N** — CPU-bound engine compute (ort/CLIP, and apple via the `Blocking<E>`
  adapter) plus blocking-fs leaves. This is the only place real parallelism lives, so the engine
  search/batch overlap property becomes **"N ≥ 2"**. N is the harness's choice, sized to its cores.

Submission differs by host. The **server keeps the asyncio bridge** (`spawn_op` + an awaited
`asyncio.Future`): its MCP handlers are coroutines on one loop and must not block it. **Threaded
hosts** (cabi, synchronous hosts, tests) use `submit_blocking` — submit a unit of work and block
the request thread on a completion channel. The bridge is preserved, not retired; what changed
underneath is invisible to a handler (`drive_io` drives the runtime instead of tokio workers, and
`Blocking<E>` dispatches to `drive_compute` instead of `spawn_blocking`).

This **replaces** the prior model, in which the kernel self-drove a lazy multi-thread tokio runtime
(tokio spawning its own worker + blocking-pool threads) and `dispatch_collection`/`dispatch_compute`
bottomed out in `spawn_blocking`. The driven `current_thread` runtime is the only runtime; the
transitional multi-thread default that let the bindings cut over incrementally has been removed,
leaving the driven model alone. The harness installs the runtime via `init_driven_runtime` before
any kernel op — an op before that panics, as there is no fallback.

Decisions and invariants:

- **Startup driver-ownership barrier.** tokio gives IO/timer-driver ownership to the first
  `block_on` caller, which MUST be `drive_io`. The harness spawns `drive_io` first, then a
  `runtime_probe` — schedule a trivial executor-only op and block until it completes, proving the
  IO thread owns the drivers — BEFORE spawning `drive_collection`/`drive_compute`. Without it a leaf could
  win ownership and timers/IO would advance only while that leaf parks in `recv`, a flaky
  starvation (#836).
- **Per-binding thread provisioning.** The binding (`shrike-pyo3`, `shrike-cabi`) *exposes* the
  drive entries; the harness *above* it owns the threads. The server's `driven_runtime.py` spawns
  N + 2 `threading.Thread`s into the GIL-releasing pyo3 `drive_io`/`drive_collection`/`drive_compute`
  entries (GIL-released for the thread's life, so native compute gets real parallelism). cabi
  exposes blocking C entries `shrike_drive_io`/`shrike_drive_collection`/`shrike_drive_compute` +
  `shrike_runtime_probe`; the native host (Swift/Kotlin) spawns and joins the OS threads. cabi
  spawns nothing — it is shrike-core. `shrike_runtime_init` installs the runtime only.
- **Shutdown is host-shaped.** The server drains via the bridge (`manager.close` awaits each
  `kernel.close`) and then calls `drive_pools_shutdown`. cabi has no bridge-await — its ops are
  detached fire-and-forget with a C callback as their only observation — so its shutdown entry
  retains a thinned admit/inflight gate: set a terminal shutdown flag (new ops fast-fail), drain
  in-flight ops while `drive_io` is still live (bounded by `DRAIN_TIMEOUT`, mirroring the server's
  committed-thread join timeout), then `shutdown_driven_pools` closes the queues so the host can
  join its threads. This subsumes the iOS-runtime lift (#714).
- **Remote engines went async.** `embed-remote`/`describe-remote` implement the async
  `Embedder`/`Recognizer` traits directly (reqwest IO on `drive_io`) behind an SSRF-pinned async
  HTTP client; `media_fetch` too. So `drive_compute` stays pure CPU, and engine-api's two
  conformance routes map cleanly onto the two pools: sync-compute-behind-`Blocking<E>` →
  `drive_compute`, async-direct → `drive_io`.
- **Deadlock leaf-invariant.** Every pool job is a leaf: an enqueued `drive_collection`/`drive_compute`
  job never enqueues-and-awaits further pool work — the read→compute→write orchestration fans out
  and awaits on the async side (`drive_io`), and compute is handed its inputs after the actor reads
  (the "discover ids → one batched read → compute" pattern keeps compute collection-free). A fixed
  pool can't exhaust itself. A debug-build thread-local tripwire asserts it.

Read concurrency: the single `drive_collection` thread serializes all collection access, so reads and
writes do not run concurrently, and a background rebuild/reconcile competes with data-plane ops on
that one thread.

Done / follow-ups:

- **#840 — removed the transitional multi-thread fallback.** The set-once seam carrying both the
  driven and the prior multi-thread default existed only to let the bindings cut over incrementally;
  #840 deleted the default path so the driven `current_thread` model is the sole runtime, completing
  this decision (an op before the harness installs the runtime now panics — there is no fallback).
- **#832 — chunk/stream the derived rebuild**, so a large rebuild does not monopolize `drive_collection`.
- **#833 — a boot/drift "maintenance window"** returning try-again-later on data-plane routes during
  a rebuild/reconcile; #833 owns the concurrent-reads-vs-modifications decision.

### Wire-protocol versioning: name-versioned actions, one backstop

The exchange evolves additively, and a breaking change to an action ships as a **new action name**
(`upsert_notes_v2`) alongside the old. This matters because unknown-key tolerance covers added
*fields* but a new *variant* in a tagged union breaks any client that parses the union
exhaustively. `WIRE_PROTOCOL_VERSION` is the backstop, bumped only when the exchange fabric itself
breaks. The MCP tool surface follows the same rule (external clients can't be handshaken — a
breaking tool change is a new tool name), and since shrike-schemas types are both the payloads and
the MCP models, the two layers version in lockstep.

### Bazel as the polyglot build system

One hermetic, cache-first build graph instead of N build systems with hand-rolled CI glue — the
roadmap spans Rust, a PyO3 binding, Swift glue, mobile hosts, and a wasm frontend, each with its
own toolchain.

- **Upstream native packages stay pip-consumed** (`anki`, `usearch`, `onnxruntime`, `tokenizers`
  as hashed wheels); Bazel builds *our* code and never rebuilds upstream native deps from source.
  The same logic pins llama-server and the fixture models via `http_archive`/`http_file`, which
  removed the flaky-download failure mode.
- **Two lanes** — the pip lane (`build-native.sh` + `pytest`) is the fast loop; the Bazel lane is
  what CI enforces (`bazel test //...`). Coverage stays on pip because the spawned server
  subprocess is invisible to `bazel coverage`.
- **Cache** is free `--disk_cache` via `actions/cache` with a daily warm-cache job seeding `main`'s
  scope; the upgrade when the 10 GB budget bites is a one-flag swap to a remote cache.
- **Bootstrap** is a committed `./bazel` wrapper (pinned bazelisk → pinned Bazel), so local and CI
  are the same build. Operational guide in `build-bazel.md`.

### The desktop and web frontend is Rust-wasm (Leptos)

One SPA serves both the desktop app (a Tauri v2 shell) and the browser client: Rust → wasm32 on
Leptos, over a TypeScript control and over Dioxus. Mobile is out of scope (fully native), and both
consumers speak the actions-over-HTTP edge, never MCP.

- **The decisive factor is the shared schema** — the client imports `shrike-schemas` as a Rust
  dependency (zero codegen, zero drift). Every TS generator falls down on the internally-tagged
  discriminated unions the house style is built on: `typeshare` doesn't support them, `ts-rs` is a
  second drift-prone derive, and JSON-Schema→TS emits non-discriminated unions.
- **Leptos over Dioxus is a Tauri-alignment call** — Dioxus is diverging toward its own desktop
  renderer, fighting the one-codebase-two-vehicles boundary; Leptos is plain client-side wasm that
  wraps in a Tauri webview and ships identically to a browser tab.
- **The card frame is the security-critical surface** — untrusted Anki card HTML renders in an
  `<iframe sandbox srcdoc>` with `allow-scripts` but **not** `allow-same-origin` (both together
  defeat the sandbox); a `postMessage` channel gates on `event.source`, not origin.
- **Accepted cost** — Leptos has no list virtualization, so the collection browser needs a
  hand-rolled windowing component at the 100k-note baseline. Revisit only if `shrike-schemas` stops
  compiling to wasm32, or the component-ecosystem gap proves a sustained drag.

### Read-phase parallelism is foreclosed by anki's exclusive lock (the #853 spike)

The pipeline-sanity epic asked whether the read phase of a bulk op (the 100k boot scan, search
candidate hydration) could shard across K read connections in parallel, since SQLite WAL supports
concurrent readers. The spike's answer is **no, definitively** — not because anki lacks a read-only
handle, but because anki opens the collection with `locking_mode = exclusive` (rslib
`storage/sqlite.rs`) on top of `journal_mode = wal`. In exclusive locking mode SQLite never releases
its file lock for the life of the connection, so **no second connection can read the database while
anki holds it** — WAL's concurrent-reader property does not apply. anki's backend is a single
exclusive owner by construction (its own single-process model depends on it), and it exposes no
read-only / second-connection API; the `db_rows` DB-proxy is read-only SQL but still serializes
through the one backend.

Consequence: the read phase of a bulk op cannot be parallelised without either patching anki to drop
exclusive locking (which would break its own assumptions) or replicating anki's field/notetype
decoding against a raw read-only `rusqlite` connection (fragile, and still blocked by the exclusive
lock while anki is open). Theme B's streamed read — the single `drive_collection` thread, pipelined
against the embed (`read(k+1) ‖ embed(k)`) — is therefore the read-phase design, and the
downstream read-parallelism work the spike gated is closed as not-feasible against the current anki.

### One structured-maintenance primitive — but the saver's debounce is not just burst-coalescing (Theme G)

Two background coalescers survived the ingest-actor consolidation (Theme A absorbed the
derived-claim and recognition-sweep coordinators): the index `DebouncedSaver` and the
`TagRefresher`. Both are now expressed over one primitive, `maintenance::Maintenance` — a coalescing
single-flight job with a uniform `request` / coalesce / `cancel` / `shutdown` lifecycle and a single
status counter (`pending`). The win is uniformity and one place to later hang the deferred #797/#800
instrumentation, not throughput.

The non-obvious part is *why the primitive carries two pacing knobs rather than one*. The proposal
framed both jobs as burst-coalescers whose threshold "can be trivial under rare concurrency." That is
true for the tag refresh (a cheap, pure recompute — `delay = 0`: run immediately, coalesce concurrent
requests into one re-run paced by `window`). It is **not** true for the index saver. The saver's
`delay` is a *re-arming time-window debounce*: it batches a stream of **spaced** writes — an
interactive add-N-notes session, one note every few seconds — into a single large index file write
per quiet period. Collapsing it to immediate-coalesce would issue one full-index write per note in
exactly that common case, since spaced requests never overlap a run to coalesce. So `Maintenance`
keeps both modes: a re-arming `delay` (saver: batch spaced writes), a burst `threshold` (cap a flood
so it doesn't sit behind the debounce), and a coalesced re-run `window`. `delay = 0, threshold = 0`
recovers the pure coalesce-loop. The synchronous shutdown write stays the host's own path
(`DebouncedSaver::flush` calls `cancel` then writes inline), so `close()` returns only after the
write lands.

## Performance engineering

### Stub-vs-real embedding is a profile choice, via a first-class `synthetic` runtime (#865)

The performance harness runs each gold workflow in two modes: against the real embedding
backends (end-to-end, including inference) and against a deterministic stub (isolating the
kernel/IO/orchestration cost). Both modes boot the **same** real `Harness` from a config profile;
the only difference is which profile. The stub is not attached out-of-band by the harness — that
would bypass the real config→boot path it exists to measure. Instead it is a first-class embedder
runtime, `runtime: synthetic`, selected like any other backend.

Isolating the kernel is the primary instrument because the failure modes a perf harness must catch
— per-op full-collection scans, lock-held file writes, N+1 hydration (the #445 findings) — are all
kernel-side and model-independent. A stub whose own cost is negligible makes those costs the signal
instead of drowning them in model-inference noise. The real-backend mode then measures the
end-to-end number a user sees.

The synthetic embedder is a **native** engine (`shrike-engine`, behind the `engine-synthetic`
feature), not a Python shim: it maps an input's bytes to a deterministic unit vector (a hashed
`splitmix64` stream, L2-normalized) with no model and negligible cost, and composes through the
exact `NativeEmbedder` attach path the ort engines use — so the stub mode exercises the real native
embed/attach machinery, isolating only inference. The vectors carry no semantics; it is never a
search-quality backend.

Gating keeps it out of production by construction. `engine-synthetic` is a **non-default** Cargo
feature; the release wheel and the per-PR `//...` lane build the lean extension, so
`shrike_native.build_features()` does not list it and profile resolution refuses a config naming
`runtime: synthetic` with the same two-layer capability error as any uncompiled runtime — never a
silent no-op. A dev/benchmark build opts in with `--define shrike_synthetic=on`
(`scripts/build-native.sh --synthetic`), which the Bazel target selects into the extension's
features and swaps the linked engine variant for the one built with synthetic (Bazel feature sets do
not unify across targets). A dev-only *second* extension target was rejected as heavier and against
the single-build-graph principle; one toggled target is enough. The capability is reusable beyond
perf — a config-selectable deterministic embedder is what fast, model-free tests want too.

### The perf harness measures distributions through the real stack, off the per-PR lane (#865)

The harness boots the **real** `Harness`/kernel from a config profile and times whole
workflows against a deterministic corpus — not a microbenchmark of a function in isolation.
The unit under test is the system (the #864 framing: a regression must point at a workflow,
then a line, not at a synthetic loop). Both embedder modes run the identical runner; only
the profile differs (`perf-stub` → the synthetic kernel-isolation run, `perf-real` →
end-to-end with onnx/CLIP).

Three deliberate choices:

- **Distributions, not a number.** Every workload reports p50/p90/p99/max over N repeats
  with an explicit warmup discard, *and* the conditions it was taken under (machine, build,
  native-extension version, corpus size+variant, embedder mode). A stored run is a
  comparable artifact; the baseline diff refuses to compare across mismatched invariant
  conditions rather than report a meaningless cross-context delta. A regression is read off
  the tail, not the mean.
- **Deterministic, production-shaped corpora at 500/5k/50k.** Built through the real
  `upsert_notes` write path (synthetic *content*, real *path*), seeded so a size/variant is
  byte-identical across runs, cached + gitignored — no binary fixture in git. 50k is the
  heaviest standard rung (revised down from the audit's 100k); 500/5k are the fast-feedback
  rungs.
- **Optimized builds, recorded.** The runner times whatever extension is staged, so it
  must be built `-c opt` (`scripts/build-native.sh --release`) — the default fastbuild is
  meaningless for perf. The extension reports its build profile (`debug-assertions` present
  on a non-opt build), the runner captures `optimized` as an **invariant** condition, warns
  on a debug run, and the baseline diff refuses to compare a debug run with a release one —
  so a fastbuild number can never be mistaken for a real one.
- **A manual lane, defended on demand — not gated.** The benchmark runs (the corpus
  build, the boot+drive) are `manual` Bazel targets, off the per-PR critical path. Perf is
  a measured property checked by hand against a stored baseline, **not** a per-change or
  scheduled gate. An automated regression gate was deliberately rejected: it would tax every
  change for a property better watched on demand, and the harness already supports the
  workflow — run, store the artifact, `--baseline`-diff a prior run (which refuses to
  compare across mismatched conditions). The pure pieces (the distribution math, the
  artifact, the diff) are unit-tested on the per-PR lane, since that logic is where a silent
  measurement bug hides.

### Hotspot profiling: a pluggable `--instrument=<tool>` seam, py-spy `--native` the default

Distributions say *which workflow* is slow; attribution needs *which line*. `run.py
--instrument[=<tool>]` re-execs the run under a sampling profiler. The default is **py-spy
`--native`**, the only profiler that merges Python-level frames **and** native Rust frames into
one flamegraph — so a hotspot is attributable whether it lives in the harness glue or the kernel
(`run.py → search_notes → kernel.search → usearch…` in a single view). The cross-boundary view
is the whole point: the kernel/harness split is exactly where "is this slow in Rust or Python?"
gets asked, so the profiler has to see both sides.

But py-spy's native unwinding is **Linux/Windows-only** — on macOS `--native` is rejected
outright, not a permissions issue `sudo` can lift. Rather than dead-end the Mac dev there, the
seam is a small registry (`instrument.py`) selected by `--instrument=<tool>`, with
`--instrument-arg` passing options through to the chosen tool. py-spy stays the default and
drops `--native` on macOS (Python-only flamegraph); the two native-detail tools are first-class
fallbacks for Rust hotspots:

- **samply** (`--instrument=samply`) — excellent native detail (pleasant on macOS-arm64), but
  renders the Python side as opaque CPython interpreter C frames, so it can't separate glue cost
  from kernel cost. Writes a Firefox-profiler JSON.
- **xctrace** (`--instrument=xctrace`) — Apple Instruments' Time Profiler, macOS only; same
  native-detail / opaque-Python tradeoff as samply. Writes a `.trace` bundle.
- **austin / cProfile / viztracer** — Python-only; blind to the kernel. Not wired.
- **cargo-flamegraph / perf / dtrace** — native-only; blind to the Python frames. Not wired.

So the merged Python+Rust view remains py-spy's, and on a Mac it wants a Linux container; samply
and xctrace cover local Rust-only attribution without one.

Two consequences. An optimized build drops frame pointers, which degrades native unwinding, so
the profiling build forces them across the Rust crates (`-Cforce-frame-pointers=yes`, opt-in
via `build-native.sh --frame-pointers`) — never in a clean-timing build, where a reserved
register would skew the distribution. And py-spy attaches via the OS process-inspection API,
which usually needs root, so this is a manual-lane tool, never CI (samply/xctrace launch the
target themselves and need no elevation).

Numeric per-span durations — capturing the kernel's existing `tracing` spans through a real
subscriber and a cross-FFI `traceparent` for a parse→write→derive→embed→index table — are
deferred to the observability work (#800). The flamegraph gives the visual breakdown now; #800
later adds the complementary numeric attribution.
