# Pipeline sanity: dataflow & compute-model proposals

A design audit of how data moves through the asynchronous kernel, cross-referenced
with the flake history and the standing perf rules.

The thesis: the kernel coordinates asynchronous work through **many
locally-invented, ad-hoc mechanisms**, and its multi-stage operations are expressed
as bare `async`/`await` control flow with correctness *retrofitted* (watermark
over-certification, poison floors, converge loops, claim flags). Making the dataflow
**explicit as staged pipelines with declared ordering** would fold those mechanisms
into a single, observable, testable shape — and close a family of load-sensitive
flakes at the root rather than per-symptom.

This is a proposals document, not a plan of record. Each item states the problem
(grounded in code/issues), the proposed change, the payoff, and the effort/risk.

## Operating assumption (the sharpening lens)

Shrike manages **one user's single Anki collection**. Two consequences are taken as
design axioms:

1. **Inter-operation concurrency is rare.** Two *write* operations almost never
   overlap in time. We must remain **correct** if they do (the transport allows it,
   and our own background maintenance can collide with a user write), but we must
   **not optimize for it** or pay standing complexity to make the overlapping case
   fast. Reads (searches) *do* run while a write or a background rebuild is in
   flight — so the rare case is specifically **write-vs-write overlap**, not
   read-vs-write.

2. **Intra-operation parallelism is *the* optimization target.** The operations that
   matter are large **batch/bulk** ones — a 100-note `upsert`, a 100k-note boot
   reindex, a full derived rebuild, a backlog recognition sweep. The goal is to
   minimize the **latency of one such operation** by fanning its internal stages
   across `drive_compute`, not to maximize throughput under concurrent load.

This re-weights everything below. The headline shift:

> **Funnel all index/derived writes through one persistent consumer (serialization
> where it buys correctness) but decouple the fast collection write from the slow
> embed with a queue (so rare concurrent writers never block each other). Spend the
> freed complexity budget on parallelizing the *interior* of a single bulk
> operation.**

Concretely: the elaborate inter-op machinery (`watermark.rs`'s in-flight tokens, the
register-inside-the-actor-job happens-before argument, the derived converge loop, the
`claim_external_build` dance) is **defending the rare case at standing cost to the
common one** — it is the hardest code in the kernel to reason about, and it exists
only because op tails interleave and multiple writers touch the index/derived stores.
A single persistent ingest actor with an embed queue (Theme A) makes the ordering
structural and the writer singular, so the *cross-task* half of that machinery goes (a
poison floor remains) — *without* parking a collection write behind a multi-second
embed. Meanwhile the bulk paths that dominate real latency (boot, rebuild, large
upsert) are still written as **serial phases** (read *then* embed *then* add) on
under-exploited parallelism (Theme B).

## The unifying diagnosis

Three structural facts, re-read under the assumption:

1. **Real parallelism lives only on `drive_compute` (N≥2).** `drive_io` is a single
   `current_thread` executor running *everything* async. "Spawn a background task"
   buys interleaving, not parallelism; the only way to fan work out is to push it to
   `drive_compute`. So intra-op parallelism = *decompose the op into compute jobs and
   `try_join` them*, not *spawn more tasks*.

2. **The collection actor (`drive_collection`, one thread, forced by anki's single-writer
   lock) serializes all collection access.** This is the floor on the read phase of
   every bulk op. It is acceptable for serialization *between* ops (rare), but it is
   the main obstacle to parallelizing the *interior* of one op (the 100k read can't
   currently overlap itself). The read-only-connection spike (Theme F) targets
   exactly this.

3. **The actor serializes jobs, not op transactions** — which is *why* `watermark.rs`
   exists. Under the assumption, this is the prime candidate for simplification
   (Theme A): a single persistent consumer makes ordering structural and the
   cross-task interleave-reconciliation half disappears (the failure poison floor
   stays).

---

## Theme A — A persistent ingest actor with an embed queue (the single index/derived writer)

**Problem.** A maintained write is `write-job → (off-actor embed/index/ingest) →
advance` (`lib.rs:1951`, `upsert_notes_wire` at `lib.rs:2110`). Two costs:

- Because the actor serializes *jobs* not *op transactions*, another op's write-job
  can interleave between this op's write-job and its advance. `watermark.rs` (~460
  lines + a large test battery) reconstructs the lost happens-before with a monotonic
  in-flight-token set, an `any(other ≤ captured)` gate, a poison floor, and the
  subtle requirement that registration happen **inside the actor job** (the module's
  most-defended argument).
- Multiple producers write the index and derived stores (upsert tails, the
  recognition sweep, the boot rebuild, the Python derived facade), which is the
  source of the #828/#650/#644 races and the two-writer `SQLITE_BUSY` problem.

A naive fix — one lock held across the whole op (capture → **embed** → index → ingest
→ advance) — is wrong *here specifically*: a remote embed or a multimodal/image embed
can take tens of seconds, and holding the write critical section across it would park
an unrelated user write for that whole time. Embed is exactly where serialization
must loosen.

**Proposal.** A **persistent ingest actor**, alive for the kernel's lifetime, that is
the *sole writer* of the vector index, the derived store, and the watermark. Put a
**queue of maintenance work** between *capture* and *embed*:

```
collection write (fast)          ingest actor (single consumer)
  commit to anki @ col.mod V  ─▶  drain a batch ─▶ batched re-read note content
  enqueue {ids, V, kind}          (drive_collection)
  return immediately           ─▶ embed batch (drive_compute ×N)
                               ─▶ atomic per-note index add
                               ─▶ derived ingest (one txn)
                               ─▶ advance watermark
```

- **The collection write no longer waits on embed.** It commits, enqueues
  `{note_ids, captured col.mod, kind}`, and returns. Concurrent writers don't block
  each other. The queue is **unbounded with a depth gauge** (#797): it carries note
  ids, not content (the actor re-reads at drain), so a pathological burst grows
  cheaply, consistent with the runtime's unbounded-channel doctrine (`runtime.rs`:
  "backpressure is the harness's committed pool size, not the channel"). A real bound
  can be added later if the gauge shows it's needed — cheap to revisit.

- **A single FIFO consumer makes the watermark ordering structural — precisely.**
  - **Deleted:** the cross-task in-flight token set, the `any(other ≤ captured)`
    scan, and the multi-task happens-before reasoning — with one consumer, tails no
    longer complete out of order.
  - **New load-bearing invariant:** the enqueue happens *inside the collection-write
    job*, so queue order = `col.mod` order (`col.mod` is monotonic; the collection
    actor is FIFO). This replaces the old register-in-job argument with a one-line
    ordering rule — but it is just as load-bearing: an enqueue slipping into a
    post-await *continuation* silently decouples queue order from `col.mod` order, the
    same bug class the current module-doc warns about. State and defend it explicitly.
  - **Survives:** a **poison floor** for after-commit tail failure (the failure policy
    is *skip-and-keep-going* so a down embed backend doesn't stall derived/later-note
    indexing — which needs the floor), plus a within-batch *"advance to the highest
    `col.mod` whose every earlier item succeeded"* rule (a linear pass over one
    drained batch — no cross-task state). That's roughly a quarter of today's
    `SpaceTracker`, not most. (A *retain-and-retry* policy — leave the failed item at
    the queue head — would eliminate even the floor, but it stalls all maintenance
    behind one dead backend; noted as a future option, not the baseline.)

- **The `col.mod` watermark and recognition presence are orthogonal — state it.** The
  `col.mod` watermark certifies collection-*text* indexing only. Recognition vectors
  are derived from media (in `shrike.db`, not anki) and don't bump `col.mod`, so
  advancing the watermark never claims they're present; their consistency is the
  sweep's own per-purpose fingerprint meta + pending-media derivation, advanced only
  when those vectors land. The actor advances each on its own terms. Making this
  explicit defuses the #828/#650 confusion (and the "a sweep item shares a `col.mod`
  and certifies a write prematurely" worry — under FIFO the sweep item enqueues
  *after* the write whose `col.mod` it read, so it drains after).

- **One writer ⇒ the race family dissolves.** The recognition sweep and the boot
  rebuild *produce* work (recognizer/embed compute, wherever) but hand their *writes*
  to this actor. Nothing else writes the index or `shrike.db`, so #828 (rebuild vs
  sweep), #650 (OCR add vs reindex), and the two-writer `SQLITE_BUSY` problem are
  gone by construction. This **subsumes Theme E's coordination surface and Theme D's
  *visibility* race**; D's loop-locality and H's add-shape remain independent items.

- **Drain-merge across the op cross-product is specified, not free.** Re-read-at-drain
  reflects a note's state *at drain*: two edits coalesce to one embed
  (last-writer-wins); an id queued but *absent* at re-read ⇒ remove its vectors.
  Deletes/removes don't go through the embed queue at all — they hit the actor's
  remove path directly (no embed needed), and a queued remove for id X invalidates any
  pending embed for X. find/replace and note-type migration (`reindex_notes`,
  `lib.rs:1387`) are embed-requiring items like upsert. The {upsert, reindex, migrate}
  × {delete} merge is pinned with the rigor today's `delete_notes` sidecar tail gets
  (`lib.rs:2453`) — it is correctness-critical, not "free parallelism."

- **Batching is the intra-op win.** The actor drains *all* available items per cycle,
  dedupes ids, and the merged batch's embed fans across `drive_compute` (Theme B), so
  a burst becomes one embed pass.

**Neighbors were never coupled to the ingest queue (a correction).** `upsert_notes`
returns semantic neighbors, but `_attach_neighbors` (`actions.py:1214`) embeds the
written notes' text as a *query* (`embed_queries`, `actions.py:1257`) and searches the
corpus with those ids *excluded* (`actions.py:1280`) — it never needs those notes in
the index. So the write path is a pure fire-and-forget enqueue *regardless of
neighbors*. Dropping neighbors (Theme J) is an orthogonal cost cleanup, not what makes
A's enqueue pure.

**Crash and durability ordering.** The queue is in-memory and the watermark lags until
drain, so a crash before drain just means more next-boot reconcile (the existing
"index may lag the collection" contract). The requirement Theme A must hold: the
*durable* watermark (`index.meta.json`, written by the debounced `save()`) advance
must strictly *follow* the durable index/FTS write — never precede it — or a crash
could leave the watermark durably ahead of a never-flushed vector add (silent loss).
`save()` already orders engine → hashes → meta (`index_orchestrator.rs`); the actor
must preserve that end-to-end, and the advance-then-debounced-save gap must stay
behind-is-safe. The actor also widens the committed-but-not-yet-visible window
(visibility now at drain, not op-return); state that lag in the index status. No read
path assumes synchronous visibility — neighbors excludes the new notes, and the dedup
calibration sampler moves to the search path (Theme J).

**No new deadlock (stated against the actual invariant).** The leaf-invariant
(`runtime.rs:52`) is specifically that a `drive_collection`/`drive_compute` *pool job* must
not enqueue-and-await further pool work. The ingest actor runs its orchestration on
`drive_io` (async) and awaits its collection reads (`drive_collection`) and embeds
(`drive_compute`) as leaves — the existing read→compute→write pattern — so it does not
violate it. Add the actor to the set the debug tripwire and the shutdown reasoning
cover.

**Shutdown drain ordering.** The actor holds an in-memory queue of un-drained writes,
so close must drain in a fixed order: **ingest drain → watermark durable → collection
actor close → driven-pool close** (`shutdown_driven_pools` must run after kernel work
quiesces, `runtime.rs:380`). With #344 (shutdown drain race) on the books, this
ordering is part of the design, not an afterthought.

**Payoff.** *Robustness:* deletes/shrinks the highest-risk code in the kernel and
closes #828/#650/#628/#644 + the two-writer class at the source. *Latency:* the
collection write returns immediately; concurrent writers never block. *Efficiency:*
batched drain + re-read coalescing turns a burst into one embed pass. **Effort/risk.**
Medium–high — the central structural change. The things to get right: the
enqueue-in-job ordering invariant, the drain-merge semantics, the durability and
shutdown ordering, and the queue-depth gauge.

---

## Theme B — Intra-op parallelism: pipeline, chunk, and fan a single bulk op across compute

**This is the optimization centerpiece.** The operations whose latency matters are
bulk, and they are written as **serial phases on parallel-capable hardware**.

**Problem.**

- `reindex_if_needed` / `rebuild_index` (`lib.rs:1231`, `:1325`) do
  `find_notes("")` → one whole-collection `note_embed_inputs` read → `compose` →
  reconcile/rebuild. The read, the embed, and the index add are *phases*: the embed
  doesn't start until the entire 100k read finishes; nothing overlaps.
- `rebuild_derived_once` (`lib.rs:1497`) materializes the whole collection into one
  `Vec` and hands it to a single blocking `build` (#832, #445 P2).
- The two per-space reconciles run in series (`reindex_if_needed` then
  `reconcile_image_route`, `lib.rs:1279`) though they're independent.
- A 100-note upsert reads all, embeds all, then adds all — three phases, no overlap.

**Proposal.** Make a bulk op a **streaming pipeline whose stages run on different
pools concurrently**, with the CPU stage fanned across `drive_compute`:

```
read chunk k        (drive_collection)        ─┐
   embed chunk k    (drive_compute ×N)   ├─ overlapped: read(k+1) ‖ embed(k) ‖ add(k-1)
      index add k   (drive_io/engine)   ─┘
```

- **Pipeline the phases — the genuinely new win.** The per-batch embed is *already*
  fanned (`try_join` over text/ocr/image, `index_orchestrator.rs:1165`); the new
  latency win is **cross-chunk pipelining** (read(k+1) ‖ embed(k) ‖ add(k−1)), bounded
  by the single `drive_collection` read thread (no read overlap without Theme F) and the
  FTS5 single-writer floor. Scope the claim accordingly: pipelining stages, not
  re-fanning an already-fanned embed.
- **One read pass feeds two consumers at boot — for the content read.** The derived
  rebuild's dead-note prune needs the *full* live-note set, which is irreducibly
  whole-collection (the #828 prune hazard) — but that set is just note **ids** (a
  cheap `find_notes("")`), read once. So share the chunked *content/field-row* stream
  (the expensive part) between the FTS build and the embed/index; the whole-collection
  id read stays a single cheap pass. The "halving" is of field-row reads, not the id
  scan.
- **Chunk + yield the actor** (subsumes #832 / #445 P2): O(batch) peak memory, and
  the actor is released between chunks so a concurrent search's tiny read interleaves
  instead of waiting for the whole sweep.
- **Parallelize the genuinely-independent reconciles** (`try_join` the per-space
  routes) — small, immediate.

**Payoff.** *Efficiency:* the single-op latency win — boot/rebuild/large-upsert
bounded by the slowest *overlapped* stage. *Robustness:* bounded memory at 100k.
**Effort/risk.** Medium–high; the FTS5 insert is a single-writer floor (parallelize
the *strip/extract* feeding it, keep one writer), and chunked prune needs the live-set
handling #832 describes.

---

## Theme C — Boot as ordered stages with parallel interiors (one readiness barrier)

**Problem — the dominant flake family, and it is the assumption's rare case biting
us through our *own* background tasks.** `boot()` (`harness.py:425`) spawns three
fire-and-forget tasks with no ordering: derived rebuild, index reconcile, and the
recognition sweep — and they collide:

- **#828** (closed): the rebuild snapshots field rows *before* an upsert, then its
  prune drops a recognition row the sweep ingested into the window; the
  converge-on-`col.mod` loop can't detect it (a recognition ingest never bumps
  `col.mod`). #830 patched the symptom; the *structural* race — rebuild and sweep
  writing the same store concurrently — remains.
- **#650 / #628 / #644**: the sweep's OCR/ASR vector `add` races a reindex/search;
  the second vector is "absent at search" only under Bazel xdist load.
- **#250**: the cooperative-lock boot-release races the server becoming reachable.

The owner already proposed gating direction in **#833** (serve only `/status` +
control plane until quiescent).

**Proposal.** Under the assumption, the right shape is **sequential stages with
parallel interiors** — *not* concurrent stages. Don't run rebuild ∥ sweep and then
reconcile the race; **order** them (which is what removes the race), and put the
parallelism *inside* each stage (Theme B):

```
Stage 0  open collection, read shape
Stage 1  one chunked content read → derived-build ‖ index-reconcile   (interiors fanned)
Stage 2  recognition-sweep-to-quiescence                              (after Stage 1)
Stage 3  reconcile/tag-refresh to mint recognition vectors
Ready    one readiness signal  →  data plane opens (#833)
```

Boot latency still matters (time-to-ready at 100k is a single-op latency target) —
but it's won by parallelizing *within* stages, not by overlapping stages that race.
This routes through Theme A's ingest actor, so the ordering is the default rather than
a special case.

Expose **one readiness signal** the data plane and the tests await. Today tests poll
three independent signals plus `sleep(0.5)` (#214 findings 5/6, #250, #628, #644,
#650); one barrier collapses all of them to a deterministic await.

**Re-entrancy is the crux, not a footnote.** `/reload` and cooperative re-acquire
re-run the pipeline mid-flight: `_on_reacquire` (`harness.py:457`) schedules a rebuild
+ reindex on *every* re-acquire, and a `/reload` can fire mid-Stage-1. The readiness
barrier needs explicit re-entrancy — a **generation counter** — so a ready→not-ready→
ready transition under re-entry can't strand the data-plane gate (#833). Theme A
helps: a `/reload` enqueues a rebuild/reconcile item, the single consumer processes it
in order, and readiness becomes *"queue drained + no rebuild pending at the current
generation."*

**Payoff.** *Correctness:* closes the #828/#650/#628/#644/#250 class at the root.
*Test determinism:* removes the poll/sleep flake surface. **Effort/risk.** Medium; the
re-entrancy semantics (generation counter, cancel-and-restart, the cooperative-lock /
reload interaction #833 flags) are the real design work. Depends on A and B.

---

## Theme D — Pull the recognition sweep loop fully into the kernel

**Problem.** The drive loop straddles the FFI: the while-loop is in Python
(`recognition_sweep`, `harness.py:801`), re-entering `recognize_pending` per batch,
while the pending-set derivation, no-progress stop, and per-purpose abort contract
are in Rust. Python observes `total_stored` and *then* searches — and that seam is
where the #644/#650 visibility races live (`stored` is reported across the FFI before
the Rust-side vector add is observable).

**Proposal.** One kernel op `recognize_all_pending` that drives to quiescence
*inside Rust* (keeping the bounded-batch yield so other actor jobs interleave) and
returns when drained. The harness awaits it as one boot stage (Theme C). One FFI
crossing; the "stored ⇒ visible" boundary lives inside the runtime that owns both
stores. Internally, the per-batch recognize calls are the intra-op parallelism point
(fan recognizer calls across `drive_compute`, Theme B).

With Theme A in place the *visibility* race is already closed (the ingest actor is the
sole writer); what Theme D adds on top is loop-locality — one FFI crossing instead of
one per batch, and the batch-counting/no-progress logic in one language.

**Payoff.** *Correctness:* removes the cross-FFI observe-then-search seam behind
#644/#650 (belt-and-braces with A). *Simplicity:* the loop logic stops being split
across two languages. **Effort/risk.** Medium; preserve the per-purpose abort and
no-progress stop verbatim (load-bearing, tested).

---

## Theme E — Make the kernel the sole writer of `shrike.db` (subsumed by Theme A)

**Problem.** The derived sidecar has two would-be writers — the kernel
`DerivedEngine` and the Python `DerivedTextStore` facade — on one SQLite file with
`busy_timeout=0` (#445 flag). Note the *rebuild* path is **already** single-writer
kernel-side: `claim_external_build` (`derived.py:374`) is a *no-write* claim — the
kernel builds against its own engine and "the rows never enter Python" (`derived.py`).
So the "two writer connections" framing is partly stale; what remains dual is the
**per-op ingest facade** path, plus the claim/`_external_build`/`_state_lock` dance
that exists to coordinate it.

**Proposal.** Finish #278's direction: the kernel is the **sole writer**; the facade
goes read-only or retires. This is the **natural consequence of Theme A** — once the
ingest actor is the only thing that writes `shrike.db`, the claim machinery and the
`SQLITE_BUSY`→watermark-lag→rebuild-churn class have nothing left to coordinate. Kept
distinct because the audit is *which Python write paths actually remain* and what the
facade's *read* paths and `/status` state machine need once they do.

**Payoff.** *Robustness/simplicity:* deletes a whole coordination surface.
**Effort/risk.** Low once Theme A lands; the remaining work is the facade-read audit.

---

## Theme F — Read-only collection path: parallelize the read phase of one op (spike)

**Problem.** The single `drive_collection` actor serializes *all* collection access. Under
the assumption this is fine *between* ops (rare overlap), but it is the hard floor on
the **read phase of one bulk op** — the 100k boot read cannot overlap itself, and it
is often the longest stage in Theme B's pipeline.

**Proposal (spike-gated).** anki holds an exclusive *write* lock, but SQLite in WAL
mode supports concurrent *readers*. If the anki crate exposes a **read-only
connection/snapshot**, the read phase of a bulk op (boot scan, and search candidate
hydration, #445 P1.6) could be **sharded across K read connections in parallel** —
direct single-op latency, exactly the optimization target. The write path stays on
the one actor. This is justified here *not* by concurrency support but by **intra-op
read parallelism**.

**Payoff.** *Efficiency:* potentially the largest read-phase win, and it composes
with Theme B's pipeline. **Effort/risk.** Research spike with a real chance anki
won't expose a clean read-only handle; gate the rest on the spike's result.

---

## Theme G — One structured-maintenance primitive (now mostly lifecycle + observability)

**Problem.** Five hand-rolled coordinators, none sharing a model:

| Mechanism | Where | Shape |
|-----------|-------|-------|
| `DebouncedSaver` | `index_orchestrator.rs:682` | idle-timer + burst-threshold, abort-on-rearm |
| `TagRefresher` | `tag_centroids.rs:248` | `running`/`dirty`/`window` coalescing actor |
| Derived rebuild claim | `harness/derived.py:374` | `claim_external_build`/`settle` + `_state_lock` + watchdog |
| Recognition sweep guard | `harness/harness.py:926` | `_ensure_recognition_sweep` + Python while-loop |
| `_spawn_bg` set | `harness/harness.py:487` | fire-and-forget set drained by `settle_background()` poll |

**The assumption demotes this from "substrate" to "tidy-up."** Most of these are
*coalescing* primitives — dedupe a *burst* of N ops into one recompute. Under rare
concurrency there is no burst, so the windowed pacing (`TAG_REFRESH_WINDOW`, the
saver's threshold) can be **trivial** ("one running? mark dirty; else go"), not a
tuned debounce. What stays valuable is a **uniform lifecycle and shutdown** (the
`_release_stuck_claim` watchdog and `TagRefresher::shutdown` exist only because each
invented its own) and a **single instrumentation point** for #797/#800.

The one thing that still genuinely matters: a single bulk op must not pay an
O(collection) tag-centroid recompute *inline* on its tail — that's an intra-op
latency hit (Theme B). So tag refresh stays off-tail on `drive_compute`, but its
*coalescing-against-other-ops* logic can be minimal.

**Payoff.** *Simplicity/observability*, not throughput. **Effort/risk.** Low–medium;
pure refactor behind one type, with simpler semantics than today.

---

## Theme H — Atomic per-note multi-vector add

**Problem.** A note's field-text vectors and its OCR/ASR vectors land in **two
separate `engine.add` calls** (`index_orchestrator.rs:1178`/`:1180`, named in #650 as
the leading root-cause for the "2nd text-vector absent at search" flake family). A
note can be half-indexed.

**Proposal.** Carry *all* of a note's vectors in one `add` so a note is never
half-present. Theme A closes the concurrent-search-observes-a-half-add race (sole
writer), but the **crash-between-the-two-adds** half-indexed-on-disk case is *not*
subsumed by serialization — so H is worth doing on its own merits: it's cheap and
makes the invariant structural rather than timing-dependent. Needs the
`reconcile == rebuild` STOP-and-decide #650 flags.

**Payoff.** *Correctness:* removes the #650 class structurally. **Effort/risk.**
Low–medium, invariant-sensitive.

---

## Theme J — Drop `attach_neighbors`; make `upsert_notes` write-only (cost cleanup)

**Problem.** `upsert_notes` returns semantic neighbors of the written notes
(`_attach_neighbors`, `actions.py:1214`). The cost is real but it is **not** a
read-after-write on the index: it embeds the written notes' text as a *query*
(`embed_queries`, `actions.py:1257`) and searches with those ids *excluded*
(`actions.py:1280`) — it never needs the new notes indexed. The cost is the #214 N+1:
an extra query-embed (possibly a multi-second remote call) plus a search on every
upsert's response path. That latency is worth removing on its own merits — not because
it couples to Theme A's queue (it doesn't).

`attach_neighbors` feeds **three** consumers; all have a staleness-tolerant home or
move cleanly off the write path:

1. **Authoring near-dup check.** Already handled upstream: the card-authoring skill
   instructs the model to `search_notes` with its planned notes *before* calling
   `upsert_notes`. A pre-write read, no coupling.
2. **Richer tag suggestion / difficulty-vs-interference.** What the neighbor signal
   was really kept for; #184 (concept-weakness epic) names it: *"the upsert path
   already computes a near-duplicate signal (`neighbors`); the
   difficulty-vs-interference workstream reuses it rather than reinventing."* The
   **persisted kNN graph (#185) is that same relation as a derived structure**, and
   the Leiden/CPM communities + concept hierarchy (#196, sub-epic #200) are the saner
   basis. #184 itself states the structure *"only needs the tree as group structure,
   so an approximate hierarchy is fine"* — staleness-tolerant by design.
3. **The dedup / activation calibration sampler.** `dedup_stats.record(...)`
   (`actions.py:1320`) feeds the activation-floor calibration from the neighbor
   search's semantic scores. This one is **live and in-process**, not a future epic —
   removing neighbors removes its sample source. Calibration sampling must move to the
   ordinary search path (`search_notes` produces the same semantic scores) or be
   re-sourced explicitly. Theme J must carry this, not drop it silently.

Distinct from exact-duplicate policy: `on_duplicate` (Anki first-field exact match) is
a cheap collection check with no embed — it stays synchronous on the write path.

**Proposal (v0.N, no users — break freely).**

- **`upsert_notes` becomes write-only:** commit + enqueue + return ids / per-item
  status. No query-embed + search on the response path.
- **Drop `attach_neighbors` outright** — not an opt-in flag. Authoring dup-check is
  the pre-write `search_notes`; tag suggestion moves to the kNN-graph/community
  substrate (#185/#196); the calibration sampler re-sources to the search path.
- **Cross-epic note:** #184 §4 currently sources its near-dup signal from the upsert
  `neighbors`; with this change it sources from #185's persisted kNN graph — same
  information, derived and staleness-tolerant. Worth a one-line update to #184.

**Payoff.** *Latency:* removes the per-upsert query-embed + search from the response
path; *simplicity:* deletes `_attach_neighbors` and its N+1 (#214); *architecture:*
the neighbor consumer becomes a reader of eventually-consistent derived data, the
pipeline philosophy applied end-to-end. **Note:** orthogonal to Theme A (A's enqueue
is pure regardless) — sequenced alongside it only as the upsert-path cleanup.
**Effort/risk.** Low–medium — a deletion plus re-sourcing the calibration sampler and
the documented cross-epic re-source. Breaking change to the upsert response + MCP tool
shape; fine at v0.N (post-1.0 the additive rule in `architecture.md` would make it a
new tool name).

---

## Theme I — Build A–J with the observability seam in mind (don't bolt it on)

You cannot tune intra-op parallelism you cannot measure. Pool occupancy, queue
depth, and the search/batch overlap factor (#797) are unmeasured; the kernel's
`tracing` spans are written but discarded at the pyo3-log bridge (#800/#796); the
#650 repro burned real effort precisely because the pipeline wasn't observable under
load. The ingest actor (A), the bulk pipeline (B), the boot stages (C), and the
maintenance primitive (G) are the exact points #797/#800 need — give each stage a
span and the standard gauges (incl. the queue-depth gauge from A) *as they're built*.
This is the feedback loop that makes every efficiency claim above verifiable at 100k.

---

## Enabling cleanup — collapse to the single runtime model (#840)

**#840** removes the transitional `RuntimeMode::Default` now that the server (#834)
and cabi (#835) run on the driven `current_thread` runtime. It halves the dispatch
reasoning (one `dispatch_collection`/`dispatch_compute` path) and is worth landing first —
every proposal here is easier against a single runtime model.

---

## Suggested sequencing

1. **#840** — single runtime model; removes a reasoning fork.
2. **Theme A** (persistent ingest actor + embed queue) — the central structural change
   and the direct cash-out of the rare-concurrency assumption: one index/derived
   writer, structural watermark ordering (cross-task machinery deleted, poison floor +
   enqueue-in-job invariant retained), embed decoupled from the collection write.
   Subsumes E's coordination and D's visibility race; boot ordering rides on it.
   **Theme J** lands alongside (it's independent — A's enqueue is pure regardless —
   but it's the natural upsert-path cleanup, and it must re-source the dedup
   calibration sampler).
3. **Theme B** (intra-op pipeline + chunk + fan) — *the* latency target; subsumes
   #832/#445 P2; the ingest actor's batched drain is its natural home.
4. **Theme C / #833** (ordered boot stages + one readiness barrier, with re-entrancy)
   — closes the flake class and kills the test poll/sleep surface. Depends on A and B.
5. **Theme D** (sweep loop into the kernel) + **Theme H** (atomic add) — A subsumes
   D's visibility race and the half-add race-under-concurrency; what remains is D's
   loop-locality (one FFI crossing) and H's crash-between-adds atomicity.
6. **Theme E** (facade-read audit) — the residue of A once the kernel is sole writer.
7. **Theme G** (maintenance primitive) — lifecycle + observability tidy-up for the
   non-ingest tasks (saver, tag refresh).
8. **Theme F** (read-only/parallel-read spike) — highest-leverage read-phase win,
   gated on the anki-API spike.

Observability (Theme I) is a *constraint applied throughout*, not a step.

## Cross-reference index

- Boot rebuild/sweep race: #828 (root cause), #830 (symptom fix), #833 (gate
  direction), #832 (chunked rebuild).
- Recognition-vector flake family: #650 (root-cause write-up), #628, #644.
- Boot/lock readiness flake: #250; shutdown drain race: #344; teardown finalization:
  #435.
- Per-op scans / N+1 / lock-held writes / missing index: #445 (kernel perf audit),
  #214 (read-path N+1 audit).
- `attach_neighbors` (Theme J): `_attach_neighbors`/`dedup_stats` at `actions.py:1214`
  /`:1320`; #184 (concept-weakness epic; names `neighbors` as the §4 input), #200
  (clustering sub-epic), #185 (kNN graph), #196 (Leiden/CPM communities + hierarchy).
- Runtime model: #840; `docs/dev/architecture.md` (runtime) and
  `shrike-kernel/src/runtime.rs`.
- Observability: #796 (epic), #797 (metrics incl. pools), #800 (OTel + async-tracing
  hygiene).
