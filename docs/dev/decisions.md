# Design decisions

The "why" behind non-obvious design choices in shipped work — the reasoning that
isn't reconstructable from the code, kept out of both the code (which shows the
"what") and the issue tracker (which holds future work). For how each piece works
today, see the developer docs alongside this one under `docs/dev/`. New entries go
on top of their section.

## Semantic search and the vector index

### CLIP gives image-by-text retrieval in one shared space

A CLIP dual-encoder puts text and images in one vector space, so a text query can
retrieve a card by the content of its image. Three facts shaped the design:

- **Image-by-text works.** An index holding only image vectors retrieves the right
  image from a text query well above a blind-text floor, so the signal is the image
  *content*, not leaked labels.
- **The embedding unit is a note's text vector plus one vector per image, all under
  the `note_id` key** (USearch `multi=True`), not a single fused vector: removing a
  note drops all its vectors, search returns the note id, and results dedup to the
  best vector per note.
- **The modality gap is the design driver.** In one cosine index a text query sits
  closer to text vectors than image vectors, so image hits are additive rather than
  rank-dominant. That is why each modality is ranked as its own RRF signal behind an
  activation gate, rather than as one deduped cosine ranking — a rank combiner is
  blind to the gap's constant offset.

### The activation gate floors a modality by its own typical match

Making `image` a separate RRF signal isn't enough on its own: RRF neutralizes the
modality gap's constant offset but is blind to magnitude *within* a modality, so an
unfloored image signal injects a collection's top image cards into every query. The
fix is an intra-modal **activation gate**: a non-text modality contributes only when
its best match for the query clears `mean + margin·std` of that modality's typical
best match. Three choices make it work:

- **Calibrate offline from the index, not from query traffic.** Typical-best-match is
  a property of the embedding space, not of who queries, so `_calibrate_activation`
  samples stored text vectors as pseudo-queries and records each non-text modality's
  best non-self match as `{n, mean, std}` in `index.meta.json`. This is cold-start
  safe, recomputed on a model change (which forces a rebuild anyway), and needs no
  search logs.
- **The gate is binary per modality, keyed on the best match** — it answers "should
  this modality speak at all for this query"; RRF already down-weights lower ranks.
- **Text is never gated.** Text is the always-relevant primary signal with its own
  user-facing `threshold`; the gap is a cross-modal artifact, so a text-only
  collection has nothing to calibrate and the gate is inert there.

It is a pure addition: an absent `activation` key (a text-only collection, or an
older index) means no floor until the next rebuild computes one.

### Per-modality retrieval splits the index, not the score

A USearch hit returns the matched `note_id` and distance but **not which vector
matched**. With a note's text and image vectors in one `multi=True` index, that
yields only a single deduped ranking, and across the modality gap a text query ranks
every note's text vector above every image vector — so an image-only card is
unreachable by text. The fix is **per-modality sub-indexes**: one USearch index per
modality (`index.usearch` text, `index.image.usearch` images). Separate indexes are
the only way to recover a per-modality ranking given that limitation. Each modality
then enters RRF as its own signal, and because RRF fuses rank positions the gap's
constant offset is invisible.

The image ranking is **not** thresholded — the `threshold` knob is a text-cosine
floor that would floor every gap-depressed image hit; the activation gate above is
what decides when an image modality is good enough to contribute. Migration is a
one-way schema marker, not a converter: a text-only pre-split index is byte-identical
to the text sub-index and loads losslessly, while an image-capable backend meeting an
old mixed index rebuilds once.

### Search fuses signals by rank (RRF), not normalized score

`search_notes` blends signals on incommensurable scales — cosine clusters in a narrow
band, exact match is near-binary, a cross-modal cosine sits a constant offset below
within-modal. Normalize-and-sum inherits every pathology: min-max stretches cosine's
narrow band, the binary signal dominates or vanishes with its weight, and normalized
scores depend on what else was retrieved, so order wobbles between queries.

**Reciprocal Rank Fusion** avoids all of it: each signal ranks its own candidates and
a note's score is `Σ wₛ·1/(k+rankₛ)` (k=60). Rank position discards magnitude; a note
absent from a signal contributes nothing (the graceful degradation we want); and
orderings are stable across queries. What RRF gives up is magnitude, which matters in
exactly one place — a literal exact hit should outrank a merely-similar note
regardless of rank gap — so the combiner carries a priority tier that floats exact
hits above the rest. Because RRF fuses rank positions, the per-modality cosine offset
(the CLIP gap) is invisible to it, which is what lets separate text/image rankings
neutralize the gap with no normalization. The combiner (`shrike_kernel::fusion`, with
`search_fusion.py` the frozen reference) is pure and reports which signals contributed
at what rank.

### Derived data lives in a sidecar `shrike.db`, not in `collection.anki2`

Shrike derives data locally from notes — a trigram lexical index now, OCR/ASR text
next, VLM describe-text later — and that derived data wants one home separate from
Anki's synced collection. The choice is a sidecar SQLite file (`shrike.db` in the
cache dir), not new tables in `collection.anki2`: Anki's sync, "Check Database", media
check, and version migrations own that schema, so a foreign table risks being dropped
or erroring, and it would ship rebuildable derived data over sync. A sidecar is safe
and rebuildable from the source of truth, so a corrupt or missing one is never data
loss. It is also the natural sync target for the relay (offload heavy compute to a
desktop instance, sync the artifacts), so rows are source-seamed
`(note_id, source, ref)` from day one even though `field` is the only source today.

Two design choices: it **feeds two signals** — a quoted-phrase trigram MATCH is a
literal substring match (the `exact` candidates), a trigram-OR MATCH ranked by bm25 is
fuzzy/typo matching (the `fuzzy` signal), both degrading to the linear `find_notes`
scan when SQLite lacks FTS5 or the query is sub-trigram. And **provenance is
source-aware** so a lexical hit can report which derived text matched (a field today,
"the OCR text of diagram.png" tomorrow). A future VLM describe source goes to the
embedding space only, never the trigram index: a literal hit on metadata the user
never sees can't be cleanly explained.

### USearch stays the index

USearch (HNSW + quantization) is the vector index, and the cross-platform plan keeps
it. No native option beats it for a zero-server, local-first design: Apple offers only
a brute-force kNN primitive, Android has no first-party ANN, and the portable
third-party alternatives don't win on portability *and* performance together. USearch
runs natively on desktop and both mobile platforms, so a future port carries the same
index. Revisit only on a specific measured trigger — SQLite-transactional co-location
of vectors (evaluate `sqlite-vec`) or Android object-DB ergonomics (evaluate
ObjectBox) — and only on a demonstrated win.

### Embedding backends are pluggable behind one small protocol

The embedder used to *be* llama-server. Going multimodal means swapping models and
runtimes wholesale, and two nearer-term needs pushed the same way — an ONNX runtime
for deployments where a pinned llama.cpp binary is wrong, and a guarantee that
text-only embedding stays first-class forever (the suites depend on small text-only
models). So the seam landed first: a minimal `EmbedderBackend` protocol behind which
`LlamaServerBackend`, `OnnxBackend`, `ClipBackend`, and `RemoteBackend` sit. The index
never learns which backend it has, so drift, the per-note hash sidecar, and
persistence stay backend-agnostic.

Three choices worth recording:

- **The fingerprint is namespaced by family** (`onnx:…` vs llama's `meta:`/`file:…`),
  so the same model under two runtimes never shares a vector space, while an existing
  index isn't forced to rebuild on upgrade.
- **`modalities` is a declared capability, not a config flag.** It makes text-only a
  named, permanent capability that a multimodal backend extends — media-by-content
  search lights up where vectors exist and silently returns nothing where they don't.
- **Pooling is folded into the fingerprint; normalization is not.** Pooling changes a
  vector's direction (vector-affecting); L2 normalization changes only magnitude, and
  USearch's `cos` metric is scale-invariant, so it never changes ranking — the same
  reasoning that makes llama's `--embd-normalize` moot.

The one non-obvious operational rule is that **batch safety is probed empirically at
startup, not assumed**: int8 dynamic-quant exports compute activation scales over the
whole batch tensor, so a batched embed makes a note's vector depend on its batch-mates
— which would break the invariant that a reconcile produces the same vectors as a full
rebuild. fp32/fp16 and llama (fp) are bit-exact batched, so the variance is int8-only.
Rather than guess from the quantization scheme, every backend's `start()` embeds a
magnitude-spiked probe set serially and batched and compares within a tolerance above
float noise and below quant drift; `embed_texts` then batches up to the proven size or
serially for a variant model. Operational specifics are in
`embedding-and-recognition.md`.

### The index is a derived cache, never a co-equal store

The Anki collection (SQLite) is always the source of truth; the USearch index is a
rebuildable projection of it. The index may lag the collection (stale search results)
but the collection never lags the index (data ops are always correct). This is what
lets drift detection be a single `col.mod` comparison with a background rebuild rather
than per-note reconciliation. The drift scheme and model-fingerprint design are in
`indexing-and-search.md`.

### Contextual upsert returns neighbours; it makes no suggestions

`upsert_notes` returns, for each created/updated note, the k most similar existing
notes — literally the result of running the same fused `search_notes` pipeline over
the upserted note's own content, with the batch's own notes excluded from each other's
results. There is no bespoke `neighbor_threshold`: an absolute cosine cutoff doesn't
map onto the RRF ranking. Instead a holistic similarity gate admits a neighbour only
when a genuine content signal backs it — a semantic match clearing the search's own
floor, or an exact-text overlap — so a fuzzy-trigram or tag coincidence never surfaces
as a neighbour.

It returns raw neighbour data and stops there: the server suggests no tags, flags no
duplicates, makes no decisions. It can't know the caller's intent, and baking in a
policy would be a guess that's wrong half the time and impossible to override cleanly.
Handing back the neighbours lets an LLM caller ground new cards in the existing
taxonomy and spot near-duplicates, while the policy for what to do lives in the skill.

### Semantic duplicate detection is not a separate feature

There is no dedicated semantic duplicate-detection endpoint, and there won't be one. A
high similarity score in `search_notes` or in upsert neighbours is the soft duplicate
signal; the caller applies its own threshold. A second code path re-implementing the
same lookup with a built-in cutoff would be redundant surface area with a worse
interface.

### Anki's exact duplicate rule lives inside `upsert_notes`

Distinct from the semantic signal, Anki has a precise duplicate rule: a note duplicates
another if it shares the first field with an existing note of the same type
(collection-wide, deck-independent). Rather than a standalone `canAddNotes` checker
(anki-connect's shape), this is folded into `upsert_notes` as an `on_duplicate` policy
(default `error`) plus `dry_run`:

- a separate check is **racy** (a check-then-write TOCTOU); folding it into the write
  makes creation atomic;
- its result is only **actionable by another call** — a checker just sends you to
  `upsert_notes` anyway;
- it **overlaps the existing per-item result union**.

The one thing a standalone checker offers — a zero-write preview — is covered by
`dry_run`, which runs the identical validation and writes nothing. Structurally invalid
notes (empty first field, broken cloze) are always errors regardless of policy. The
default is `error` because silently writing a duplicate or empty note is almost always
a mistake; callers who want duplicates opt in with `allow`.

### One query, many retrieval mechanisms, annotated results

`search_notes` takes a single `queries` input and runs it through every retrieval
mechanism, folding the evidence into one result list rather than a param/tool per
mechanism. Each match carries a `score` when semantically ranked and a `substring`
annotation when the text occurs literally — both when both apply. This was chosen over
a `--substring` parameter: a separate flag reads as a filter and forces a
union-vs-intersection decision, whereas "the query is matched every way we can, and
results tell you how" needs no mode. It also degrades gracefully — with the index down,
a query still returns its exact matches plus an advisory message — and the optional
match-evidence fields are the extension point a future fuzzy/n-gram backend plugs into.

### The raw Anki query is its own tool, with the full grammar

The raw Anki search escape hatch is `collection_query` / `shrike search query`, a
dedicated tool rather than a `query` parameter bolted onto the structured `list_notes`
(text search lives in `search_notes`; deck/tag/type are structured filters). Two scope
decisions:

- **Full grammar, no whitelist.** The string is passed straight to `col.find_notes`, so
  every operator works, including review/scheduling predicates (`is:due`, `prop:`,
  `rated:`). This looks like it brushes Shrike's non-review stance, but that stance is
  about not performing review *operations*; `collection_query` is read-only — filtering
  by `is:due` returns notes, it reviews nothing. Whitelisting a "non-review subset"
  would mean re-implementing Anki's grammar parser, defeating the point of an escape
  hatch.
- **Its own tool, reusing `list_notes`' shape**, so callers get one note shape across
  all three retrieval surfaces. A malformed expression surfaces as `ToolInputError`.

### Find-and-replace edits via Anki's engine; a scope is required

`find_replace_notes` / `shrike note replace` runs the edit through Anki's
`col.find_and_replace` (Rust regex — linear-time, undo-able) rather than
re-implementing replacement. A **scope is required** (`deck`/`tags`/`note_type`/`ids`)
so a collection-wide edit is always explicit. Two subtleties: it changes field bodies,
which *are* embedding text, so changed notes are re-embedded (found by diffing
`notes.flds` before/after, since `note.mod` is only second-resolution); and the dry-run
preview is rendered in Python (exact for literal, illustrative for regex), with the
apply authoritative.

## Tags

### Setting tags is a full replace; add/remove is a separate operation

When you *set* tags — `upsert_notes` `{id, tags}`, `note update --tags`,
`update_note_tags --set` — the note ends up with exactly the set you sent; replace
never silently merges. An additive/subtractive `mode` parameter on the `tags` field was
rejected as the bag-of-optionals/hidden-state smell the schema house style warns
against. Additive editing is its own tool, `update_note_tags`, with non-overlapping
fields: `set` (full replace) is mutually exclusive with `add`/`remove`, which combine
freely. There is no default mode — the caller picks one. Tags aren't embedding text, so
these ops bump `col.mod` and advance the index watermark without re-embedding.

### Tag rename matches exactly

Collection-wide and note-scoped tag rename (`rename_tag` / `shrike collection tag
rename`) matches the tag *exactly* (find notes carrying it, then swap), not a substring
find/replace, so renaming `jp` never touches `jp-verbs`.

## Decks

### Deck deletion is empty-only; decks never merge

`delete_decks` / `shrike deck delete` refuses unless the deck and every subdeck is
empty. There is deliberately no "delete the cards too" or "move to Default" mode:
emptying a deck is a separate composable step (move its notes elsewhere), after which
the deck is deletable. The payoff is that **deck deletion can never delete a note**, so
it has no bearing on the note set or the index. Renaming a deck onto a name another deck
already uses is an **error**, not a merge — Anki's backend would silently disambiguate
(`B` → `B+`), which litters the tree. So `upsert_decks` mirrors `upsert_notes` (id =
rename, absent = create) with no hidden merge. Like tags, deck ops bump `col.mod`
without changing a vector.

### Deck references accept name, numeric ID, or `#id`

Anywhere a deck is *referenced* (not created), the value may be a deck name, a bare
numeric ID, or a `#`-prefixed ID, resolved server-side in
`CollectionWrapper._resolve_deck_ref`:

- `#<id>` is **always** an ID (resolves to that deck's name, or "not found");
- a **bare integer** is tried as an ID first, then falls back to a literal name — so a
  deck genuinely named `123` is still reachable;
- anything else is a name.

This mirrors note IDs' `#`-prefix handling. On note create, a name that doesn't exist is
still auto-created; only an explicit unknown `#id` is an error.

## Note types

### Field and template updates are applied by position, preserving note data

`upsert_note_types` replaces a note type's whole `fields`/`templates` list on update.
Anki migrates note field *values* and template *cards* by **ordinal** — the entry at
position N keeps its data as long as position N survives. (An early implementation
rebuilt the lists from fresh objects carrying no ordinal, so Anki saw every field as
removed and every incoming one as new, silently blanking all note content or deleting
all cards — even on an identical re-send. The current code reuses the existing dicts in
place.) So a whole-list replace is data-safe and matches Anki's rule: rename-in-place
and edit-in-place preserve data, lengthening appends, shortening discards only the
trailing entries. A genuine reorder is necessarily a separate operation — a positional
name swap reads as two renames.

### Identity-based field/template ops are separate tools

The genuine move/insert/non-trailing-remove that position-replace can't express lives in
`update_note_type_fields` and `update_note_type_templates` — sequences of
`add`/`remove`/`rename`/`reposition` ops addressed by **name**. They're separate tools
rather than another shape of `upsert_note_types`' params because the contracts differ: a
*declarative* "the fields are now exactly this list" (position-keyed) versus an
*imperative* "move X to 0, rename Y" (identity-keyed). Conflating them makes "is
`["B","A"]` a reorder or two renames?" ambiguous.

They delegate to Anki's data-safe primitives (`rename_field`/`reposition_field`/etc.),
which migrate data and cards by identity. Each call is atomic: the op sequence is
validated against a simulated name list first, so an invalid op changes nothing; only
then are the primitives applied and persisted with one `update_dict`. With real movers
in hand, the two families are **reconciled** so they can't overlap dangerously: the
positional replace refuses any update where an existing name lands at a different
position (a reorder/insert/non-trailing-remove, which would silently re-label note
data), pointing the caller at the identity tool. So `upsert_note_types` keeps only the
unambiguous positional edits; everything else goes through the identity tools.

### `find_replace_note_types` rewrites template text, not fields

This is a different operation from the find-and-replace over note *content*: it searches
one note type's card-template HTML (`qfmt`/`afmt`) and shared CSS and touches no field
values. So the two find-and-replaces share no code — they operate on different layers.
It scopes to one model per call with `front`/`back`/`css` booleans and returns a
replacement count. Three decisions: templates/CSS aren't embedding text, so it bumps
`col.mod` without re-embedding (unlike the structural field ops, whose removes *do*
change embedding text); `match_case` defaults **true** because template/CSS is code; and
literal mode inserts the replacement verbatim (`re.escape`d, so `\1`/`$2` are characters,
not backreferences — capture refs need `regex`). It does not rename a field — Anki's
`rename_field` already rewrites template references for that.

### Field editor metadata: getter in `collection_info`, dedicated setter

Per-field `font`/`size`/`description` (Anki's edit-time cosmetics) get two surface calls.
The **getter is folded into `collection_info`** — the metadata is just more of a note
type's definition, so it rides the existing `note_type_details` block the caller already
requests. The **setter is dedicated** (`update_note_type_field_metadata`), not an op on
`update_note_type_fields`, because the index policies are opposite: a structural field op
lets a drift-rebuild happen (a removed field changes embedding text), while metadata
changes no embedding text and wants the col_mod-bump-no-re-embed treatment. Folding it in
would mix two policies in one tool.

### Changing a note's type is a dedicated tool

`upsert_notes` hard-refuses a type change; `migrate_note_type` is where it lives,
wrapping Anki's `col.models.change`. The point is **preserving history** — a type change
keeps note IDs and carries each card's scheduling across mapped templates, which
delete-and-recreate would throw away. Folding it into `upsert_notes` would make
`field_map`/`template_map` a conditional sub-mode with an ambiguous interaction with the
item's own `fields`, and would bury a destructive migration inside a routine bulk
create/update. Shape decisions: the **map is explicit, nothing guessed** — `field_map` is
required and non-empty, a source field absent from it is dropped (reported in
`dropped_fields`), and unknown names / two-to-one / mixed source types / target==source
all error (auto-mapping same-named fields was rejected because the whole risk is silent
content loss). It applies by default with a `dry_run` preview, and re-embeds the migrated
notes in place since the remap changes embedding text.

## Collection maintenance

### One `collection_prune` tool, not scattered cleanups

The small "tidy up" chores — clear unused tags, remove empty notes, remove empty cards,
trash unused media — live behind one `collection_prune` tool rather than a verb each:
they're all maintenance passes over the whole collection, so one entry point with opt-in
flags (none selected → run all) beats N one-off verbs cluttering the surface. Three
decisions:

- **Apply by default, with a `--dry-run` preview**, unified with `find_replace_notes` and
  `migrate_note_type` so every mutating verb shares one predictable model (the CLI
  previews → confirms → applies unless `--dry-run`; the MCP tool applies, `dry_run: true`
  previews). On the CLI the destructive blast radius is contained by the
  preview-and-confirm gate rather than by a preview-by-default value.
- **"Empty" is media-safe.** A note is empty only if every field is blank by
  `embed_text.field_is_blank` — no text *and* no media. This is stricter than the
  embedding normalizer (which drops media to `""`): an image- or audio-only card has real
  content and must never be pruned. Anki's "generates no cards" definition was rejected —
  it would delete a note whose only content sits in a field no template renders.
- **Apply ordering: notes → cards → tags → media**, so anything orphaned by the deletions
  is cleared in the same call; the dry-run previews each independently, so an apply can
  clear a few more than the preview showed.

Index handling is mixed — empty-note/empty-card removal drops vectors like
`delete_notes`, while clearing tags and trashing media leave vectors valid — which is the
whole reason prune isn't a plain metadata-bump op.

## Collection lifecycle

### Busy is a typed error, not a per-tool response variant

Under cooperative locking a re-acquire can fail because Anki holds the collection, so
"database is locked" is an expected outcome every tool needs a clean path for. A
discriminated `CollectionBusy` *response* variant was rejected: busy is orthogonal to
every tool's response (the op never ran), so bolting the same variant onto every response
model is the wrong kind of union. It's modelled as an **error class with a stable wire
code**, riding the two-layer split: a server-side `CollectionBusyError` prefixed with
`COLLECTION_BUSY_CODE`, surfaced as an MCP `isError`, mapped by `ShrikeClient` to a
client-side `CollectionBusyError(ShrikeError)` callers catch-and-retry. It returns busy
**immediately, with no server-side retry**: the dominant case is Anki open for a whole
session, where a retry just adds latency before the inevitable busy.

### Cooperative locking is opt-in, time-sliced, with a 5 s idle hold

By default the daemon holds Anki's exclusive lock for its whole life — ideal for the
heavy embedding workflow (no acquire latency, no contention) but blocking Anki desktop's
launch. `--cooperative-lock` opens on demand and releases after an idle window.

- **Opt-in, default off** until proven; whether it becomes the default is deferred.
- **Time-slicing, not concurrent sharing.** Anki desktop holds the collection for its
  entire runtime, so the win is precise: an idle daemon stops blocking Anki's launch, not
  that both operate at once. Contention surfaces as a clean SQLite busy error.
- **5 s idle hold.** SQLite's conventional `busy_timeout` is ~5 s, and unlike a DB
  connection pool our held resource actively blocks a human launching Anki while
  re-acquiring is a cheap local open — both forces push short. Tunable via
  `--lock-hold-seconds` (a human-free deployment can raise it).
- **Drift is re-checked per re-acquire, col_mod-only.** The acquire hook compares
  `col.mod` to the index's stored value and rebuilds on a mismatch, reusing the boot
  machinery. It deliberately doesn't re-fetch the model fingerprint (a model change is
  handled by `/embedding/start`), avoiding a llama-server round-trip every re-acquire.
- **`server.lock` and the collection lock are distinct** — "daemon alive" and "collection
  held" are different facts, both reported by `/status`.

### Reload is a control endpoint sharing the cooperative-lock primitive

`shrike collection reload` / `POST /reload` closes and re-opens the `anki.Collection` and
re-checks index drift. Its honest value while the daemon holds the lock permanently is
narrow — only a file-level replacement (a restored backup, a sync swap) can change the
file underneath it — but it introduces the primitive cooperative locking needs:
`CollectionWrapper.reopen` plus `run`/`run_sync` reading `self.col` **at execution time on
the worker thread** rather than capturing it when called, so an op queued after a reopen
runs against the new handle. It's a control endpoint and CLI, not an MCP tool, and it
touches only Anki's collection lock, never `server.lock`.

## Tag-vector namespacing

Tag centroids live in the **same index engine as the note items**, under a distinct named
space (`tag.text`, file `index.tag.text.usearch`), rather than a separate index file — the
per-modality split already built the named-vector-space abstraction. One engine means one
`model_id`/`ndim`/`metric` by construction (a tag centroid is only meaningful in the
notes' space) and one persistence path, and the no-leakage property is **structural**:
note searches are scoped to the note modalities, so a tag key can't surface from a note
query (no post-filter, no key-range trick — note ids are epoch-ms timestamps with no safe
disjoint range).

Keys are `blake2b-8(tag)` masked positive; the key→tag map is **in-memory only**, rebuilt
on every recompute, so a stale persisted file can never mislabel a key — the signal is
simply off until the first recompute. A centroid is a pure function of the member notes'
text vectors and one membership pass over `notes.tags` (hierarchy rolled up by
`::`-prefix), so there is **no separate watermark and no incremental diff**: the whole set
recomputes at the tail of every index-changing op, best-effort (the tag layer never fails
the op it rides on). Hygiene before vectors: a member floor, a structural-coverage cap,
and a meta-tag blocklist (`leech`, `marked`).

## Architecture

### The engine-plugin architecture: a pure kernel

The kernel composes engines it is *given* — it never names one. The contracts live in
`shrike-engine-api` (a leaf crate), and concrete engines are feature-gated families in
`shrike-engine`. Hosts construct engines from config and attach them to the kernel's named
slots; a layering check enforces that `shrike-kernel` depends on no engine crate, ever.

Decisions worth recording:

- **Two conformance routes, chosen by the engine's natural shape.** The kernel sees only
  the async traits. A naturally-sync engine implements chunk-level sync compute traits and
  is bridged by an adapter; a naturally-async engine implements the async traits directly.
  Pipeline topology stays kernel-owned, with independent engine futures `try_join`ed — a
  host-described execution graph was rejected because it would push the kernel's
  consistency invariants into a meta-layer every host re-implements.
- **Named slots, not a registry** — two slots (embed, recognize) compose cleanly; a keyed
  modality→engine registry is the step to take when a third capability kind (ASR/audio)
  actually lands, not before.
- **Identity and batch policy are host-assembled, not engine-known.** Fingerprint strings
  fold host policy (`pool=`/`args=`/`textprep=`); `safe_batch` comes from the host-run
  probe; `WithPolicy` carries them onto a pure-compute engine.
- **Engines and managers are different concerns.** Talking to an embeddings endpoint
  (`remote`) is separate from launching one (`shrike-llama-server`, a manage-class
  capability mobile builds never include).
- **Engine crates link into the single binding cdylib via cargo features**, never trait
  objects across `.so` boundaries (there is no stable Rust ABI).
- **The Python facades stay, as assembly** — they keep construction-time work and hand the
  kernel a native composition; the `PyEmbedder`/`PyRecognizer` capture seam remains as the
  custom/test escape hatch, on no production path.

### The kernel owns its runtime (tokio)

The kernel installs and owns a process-global tokio runtime and is written as ordinary
async Rust — no executor traits, no runtime-agnostic gymnastics. tokio supports every
platform this project targets and was already in the dependency tree (transitively via
anki), so owning it added nothing and removed a large custom adaptation layer (a
hand-rolled executor, a timer host over the asyncio loop, a polling bridge) that had bought
runtime-portability the platforms never demanded — at the cost of real bridge complexity
and lock-contention hazards.

The shape:

- A process-global runtime owned by `shrike_kernel::runtime`; only the `Handle` escapes.
  `init_runtime` is the builder seam — a degenerate `current_thread` mode runs the whole
  kernel on one thread, which keeps "no `block_in_place`" honest.
- **The collection is a task-actor**: one spawned task owns the core and runs jobs inline
  off an mpsc — FIFO by construction, serialization from the task's loop rather than thread
  affinity.
- **The action exchange is the edge**: `spawn_op` spawns each public op and returns a
  oneshot-backed future pollable from any context. Dropping it **detaches** (the task
  completes) — never a `JoinHandle` abort, since a half-applied write would be corruption.
  The asyncio bridge survives, shrunk to a one-wake completion handoff.
- **Timers** ride `tokio::time`; **engine execution** is one `Blocking<E>` adapter (an
  eager `spawn_blocking`, pinned by test).

### anki retains its sync runtime; the kernel pins sync off the runtime worker

The invariant is "**sync ops never run on a runtime worker**," not "one runtime in the
process." anki's rslib owns an internal lazy runtime whose only consumers are the
sync/AnkiWeb services; Shrike dispatches none of them today (pinned by the
`runtime_singularity` test), so anki's runtime stays cold. But client sync is exactly the
path that wakes those services, at which point two runtimes coexist. The question of
whether to preserve a literal one-runtime invariant by patching anki was settled **no**,
for two reasons:

- **The anki-patch mechanism is Bazel-only.** Source patches ride `MODULE.bazel`
  annotations and apply only on the Bazel lane; the cargo inner loop builds unpatched anki.
  A patch that changed anki's runtime accessor would make `cargo test` and Bazel disagree on
  a correctness invariant. Closing that would mean owning a `[patch.crates-io]` fork of anki
  — a standing cost on a fast-moving upstream.
- **The panic hazard is kernel-side and runtime-agnostic.** anki's sync paths call
  `block_on`, which panics from inside *any* runtime worker regardless of which runtime owns
  it — so injecting the kernel's handle wouldn't even remove the hazard.

The fix is a dispatch discipline: kernel-side sync ops that may `block_on` **must dispatch
via `spawn_blocking`** (a blocking-pool thread isn't a runtime context, so `block_on` is
legal there) — the same seam the Python captures use. It is pinned by the
`sync_dispatch_pin` panic-repro test, which shows the same call panicking on a worker
thread and completing on a blocking-pool thread. The patch mechanism stays reserved for
source patches identical on both lanes by construction (a version string, a build flag).

### Wire-protocol versioning: name-versioned actions, one backstop constant

The action exchange evolves additively, and a breaking change to an action ships as a
**new action name** (`upsert_notes_v2`) carrying its own schema types alongside the old.
This matters more than it sounds with union-heavy schemas: unknown-key tolerance covers
added *fields*, but a new *variant* in a tagged union breaks any client that parses the
union exhaustively — so name-versioning is what actually keeps the exchange additive.
`WIRE_PROTOCOL_VERSION` (in shrike-schemas, mirrored in `schemas.py`, pinned equal by a
test) is the backstop, bumped only when the exchange fabric itself breaks (envelope
semantics, the error taxonomy). The MCP tool surface follows the same rule from the other
side: external clients can't be handshaken (tools are discovered via `tools/list`), so a
breaking tool change is a new tool name. Because shrike-schemas types are both the exchange
payloads and the MCP response models, the two layers stay versioned in lockstep.

### Bazel as the polyglot build system

One hermetic, cache-first build graph instead of N build systems with hand-rolled CI glue.
The roadmap is polyglot — Rust crates with a PyO3 binding, Swift platform glue, mobile
hosts, a wasm frontend — and each drags in its own native toolchain that must interoperate
and ship as coherent artifacts. Bazel is the standard answer: one dependency graph, one
test invocation, content-addressed caching that makes "nothing changed → nothing re-runs"
hold across all of it.

- **Upstream native packages stay pip-consumed, always.** `anki`, `usearch`,
  `onnxruntime`, `tokenizers` are consumed as PyPI wheels via a hashed universal lock.
  Bazel builds *our* code and orchestrates; it never rebuilds upstream native deps from
  source — that would trade a solved packaging problem for an unbounded toolchain one. The
  same logic gave the pinned llama-server and the fixture models to
  `http_archive`/`http_file` with pinned sha256s, which removed the flaky-download failure
  mode rather than relocating it.
- **Two lanes, deliberately.** The pip lane (`scripts/build-native.sh` + `pytest`) is the
  fast iteration loop; the Bazel lane is what CI enforces (one `bazel test //...`).
  Coverage stays on the pip path because the integration suite's spawned server subprocess
  is invisible to `bazel coverage`.
- **Cache choice: free `--disk_cache` via `actions/cache`**, with a daily warm-cache
  workflow seeding `main`'s scope (entries are only restorable from the same branch or the
  default branch, and the test workflow runs on PRs only). The named upgrade path when the
  10 GB budget bites is a real remote cache, a one-flag swap.
- **Dev bootstrap is a committed `./bazel` wrapper** (pinned, sha-verified bazelisk →
  pinned Bazel), so "works locally" and "works in CI" are the same build, with zero
  contributor install. See `build-bazel.md` for the operational guide.

### The desktop and web frontend is Rust-wasm (Leptos)

The single SPA that serves both the desktop app (a Tauri v2 shell) and the browser client
is Rust compiled to wasm32, on Leptos — chosen over a TypeScript control and over the other
Rust-wasm finalist (Dioxus). Mobile is explicitly out of scope (it went fully native,
SwiftUI/Compose), so Tauri-mobile viability was never a criterion; both consumers speak the
actions-over-HTTP edge, never MCP.

- **The decisive factor is the shared schema.** The client imports `shrike-schemas` as a
  Rust dependency — zero codegen, zero drift — against a catalog full of internally-tagged
  discriminated unions (the schema house style). Every concrete TS generator falls down on
  exactly that: `typeshare` has no internally-tagged enum support; `ts-rs` is a second
  drift-prone derive on every wire type; the JSON-Schema→TS path emits non-discriminated
  unions, throwing away the tagging the house style exists to preserve. Importing the
  canonical types directly makes a new action or changed variant a recompile, not a
  regenerate-and-reconcile.
- **Leptos over Dioxus is a Tauri-alignment call.** Dioxus 0.7 is diverging toward its own
  desktop renderer, which fights the "one codebase shipped to both a Tauri shell and a plain
  browser tab" boundary; Leptos is pure client-side wasm that wraps cleanly in a Tauri v2
  webview and ships the identical bundle to a browser.
- **The card frame is the one security-critical surface, and it's framework-agnostic.**
  Anki card HTML is untrusted (arbitrary CSS/JS/MathJax from shared decks), so it renders in
  an `<iframe sandbox srcdoc>` granting `allow-scripts` but **deliberately not**
  `allow-same-origin` — with both, the sandbox is defeated and card JS shares the host
  origin; `allow-scripts` alone gives a unique opaque origin. Any `postMessage` channel
  treats `event.origin` as untrusted and gates on `event.source` being the frame's
  `contentWindow`.
- **Accepted cost:** Leptos has no list-virtualization equivalent of TanStack Virtual, so
  the collection browser needs a hand-rolled windowing component to render a large note list
  at the 100k-note baseline — a known, bounded widget cost, the honest price of the
  zero-drift schema win. Revisit the stack only if `shrike-schemas` stops compiling to
  wasm32 on an un-gateable dependency, or the component-ecosystem gap proves a sustained
  drag.
