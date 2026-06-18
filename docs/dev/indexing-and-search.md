# Indexing and search

This covers how vectors are stored and kept consistent with the collection, the
derived-text sidecar, and how `search_notes` combines its signals.

## The vector index is a derived cache

The Anki collection (SQLite) is **always** the source of truth. The USearch index
is a rebuildable projection of it. The consistency model is one-directional:

- the index may lag the collection (search results can be stale);
- the collection never lags the index (data ops — upsert, delete, list — are
  always correct).

If the index is stale, corrupt, or missing, it can be rebuilt from the collection
by re-embedding. This is what lets drift detection be a single comparison plus a
background rebuild rather than per-note reconciliation.

### Drift detection

`index.meta.json` stores `col_mod` (the `col.mod` at last build) and `model_id`
(the embedding-model fingerprint). On startup, and whenever the embedding service
attaches, both are compared:

- **Both match** → load the index as-is.
- **`model_id` differs** (model changed; every vector now lives in a different
  space) → full rebuild.
- **Only `col.mod` differs** (Anki GUI, sync, imports) → **incremental
  reconcile**, not a full re-embed.

Reconcile uses a per-note embedding fingerprint sidecar (`index.hashes.json`,
`{note_id: blake2b}`). `VectorIndex.reconcile` diffs current notes against indexed
ones and re-embeds only changed notes, adds new ones, and drops deleted ones. The
end state is identical to a full rebuild, but a drift touching a handful of notes
re-embeds only a handful.

The fingerprint is **media-aware**. For an image-capable backend it folds in the
sorted filenames of a note's images *that actually resolve* — so adding, removing,
swapping, or later storing an image re-embeds the note, while a
referenced-but-never-stored image stays out of the hash (no re-embed loop). For a
text-only backend it is the text hash, byte-identical to the pre-media scheme, so
text-only users pay no spurious rebuild on upgrade. It hashes *names of present
images* via a cheap `stat` (no byte read); image bytes are read only for a note
actually being embedded, lazily and lock-free through the index's image resolver.

`reconcile` falls back to a full `rebuild` when the model changed, when there is
no prior hash state (first build, or an index predating the sidecar), or when an
image-capable backend meets a pre-split single-index layout. The hash map is
maintained incrementally by `add`/`remove`, so Shrike's own upserts don't
re-embed on the next reconcile. **Explicit** rebuilds
(`shrike server index rebuild`, `POST /index/rebuild`) stay full; reconcile is
only the automatic drift path.

### Persistence

The index is saved on graceful shutdown, at the end of a rebuild, and via a
**debounced flush** during normal operation. The kernel's `DebouncedSaver`
(`shrike-kernel/src/index_orchestrator.rs`) writes either `save_delay` seconds
after the last change (idle debounce, default 60 s) or immediately once
`save_threshold` unsaved changes accumulate (burst cap, default 100), whichever
comes first. It rides the kernel runtime's timers and blocking pool, driven by
edit activity — no polling. This bounds what a hard kill discards: edits since the
last flush trigger a `col.mod`-mismatch rebuild on next boot (correct, at the cost
of a re-embed). The knobs are configurable; the numeric defaults live in
`harness/index.py`.

### State machine

The index reports `ready`, `building` (with progress), `unavailable` (embedding
service not running), or `error` (build failed). It is exposed via `/status`,
`search_notes` responses, and `shrike server status`. Both reconcile and full
rebuild run in a background thread, so the server is never blocked; the server
accepts requests immediately and `search_notes` returns an actionable status
("building 2847/5000 notes, try again shortly") until ready.

### Per-modality sub-indexes

A note's embedding unit is its text vector plus one vector per image, all stored
under the `note_id` key (USearch `multi=True`): `remove(note_id)` drops all of a
note's vectors, search returns `note_id`, and results dedup to the best vector per
note.

The CLIP **modality gap** (text-text cosine ~0.7 vs text-image ~0.3) means that in
a single cosine ranking a text query ranks every note's text vector above every
image vector, so image hits could never reach rank 1. To fix this the index is
**split per modality**: one USearch index per modality (`index.usearch` for text,
`index.image.usearch` for images). `index.search_by_modality` ranks notes
separately within each modality (max-sim over a note's items of that modality), and
each modality enters fusion as its own signal. Because rank fusion compares rank
*positions*, the gap's constant offset becomes invisible.

## The derived-text store

Derived data — text Shrike computes locally from notes — wants one home, separate
from Anki's synced collection. `DerivedTextStore` (`harness/derived.py`) is that
home: a sidecar SQLite file (`shrike.db`) in the cache directory. Its first
artifact is an **FTS5 trigram index** over note text, which backs the substring
(`exact`) candidates and the `fuzzy` signal of `search_notes`.

It lives in a sidecar rather than in `collection.anki2` because Anki's sync,
"Check Database", media check, and version migrations own that schema — a foreign
table risks being dropped or erroring, and it would ship rebuildable derived data
over sync. A sidecar in the cache dir is safe and rebuildable from the source of
truth.

Every row is keyed `(note_id, source, ref)` — `source` is *where* the text came
from, `ref` the field name or media filename. Today the only source is `field`;
the seam is for `ocr`/`asr` recognized text (which goes to both the trigram index
and the text-embedding space, provenance-tagged). A future VLM image-describe
source would go to the embedding space *only*, never the trigram index — a
literal-search hit on metadata the user never sees can't be cleanly explained. A
two-table layout (`idx` FTS5 + `rowmap`) keeps incremental delete-by-note cheap.

The store follows the same derived-cache contract as `VectorIndex`, with two
deliberate divergences:

- **No debounced saver** — SQLite writes are durable per commit; incremental
  writes just advance the stored `col_mod` watermark so the next boot sees no
  drift.
- **Graceful absence is first-class** — if the runtime's SQLite lacks FTS5 or the
  trigram tokenizer (probed at construction), the store reports `unavailable` and
  every lookup signals the caller to fall back to the linear `find_notes` scan.
  Likewise a `<3`-character query (which trigram can't match) falls back. All
  MATCH expressions are FTS5-quoted (`_fts_quote`), so query punctuation can't be
  parsed as FTS5 syntax.

The store is independent of the embedder (it builds and ingests with embeddings
off) and updates incrementally on upsert/delete alongside the vector index.

## The three retrieval surfaces

`search_notes`, `list_notes`, and `collection_query` are three surfaces chosen by
intent:

- **`list_notes`** — structured filters (deck, tags, type, IDs, date), ANDed.
- **`search_notes`** — meaning plus exact text, fused (below).
- **`collection_query`** — the raw Anki escape hatch. The string goes straight to
  `col.find_notes`, so the full expression language works (`is:due`, `prop:`,
  `added:`, `rated:`, `flag:`, `nid:`/`cid:`, `OR`/`-`/brackets). It is read-only:
  it only *finds* notes, so the full grammar is allowed without a whitelist. A
  malformed expression raises `SearchError` → `ToolInputError`.

All three reuse the same `Note`/`ListNotesResponse` shape.

## Search fusion (RRF)

`search_notes` blends several retrieval signals that live on incommensurable
scales — semantic cosine (per-modality `text` and `image`), exact substring, and
trigram `fuzzy`. Normalizing and summing them inherits every scale pathology and
makes a note's order depend on what else was retrieved.

Instead, signals are combined by **Reciprocal Rank Fusion**: each signal ranks its
own candidates, and a note's fused score is `Σ wₛ·1/(k+rankₛ)` (k=60). The
properties that matter:

- rank position discards raw magnitude, so cosine-0.7 and a binary hit are never
  reconciled directly;
- a note absent from a signal is rank-∞ and contributes nothing — the graceful
  degradation we want;
- orderings are stable across queries.

What RRF gives up is magnitude, which matters in exactly one place: a literal exact
hit should outrank a merely-similar note regardless of rank gap. So the combiner
carries a **priority tier** that floats exact hits above the rest, RRF-ordered
within. The combiner (`shrike_kernel::fusion`, with `harness/search_fusion.py` as
the frozen Python reference the parity suite pins) is pure: ints in, ranked ints
out, plus which signals contributed at what rank.

Two signals deserve note:

- **The `image` signal is not thresholded.** The user-facing `threshold` is a
  *text*-cosine floor (~0.5); applied to gap-depressed image cosines it would floor
  every image hit. Instead an offline-calibrated **activation gate** floors a
  non-text modality: it contributes only when its best match clears
  `mean + ACTIVATION_MARGIN·std` of that modality's *typical* best match, estimated
  by sampling stored text vectors as pseudo-queries (`index._calibrate_activation`,
  self-matches excluded; stored in `index.meta.json` under `activation`). So an
  off-topic query no longer injects weak image cards, and text-only collections
  are unaffected (no image sub-index → nothing to calibrate).
- **The `fuzzy` signal** is the derived store's trigram/typo ranking
  (`DerivedTextStore.search_fuzzy`), weighted below the rest
  (`SEARCH_WEIGHTS["fuzzy"]=0.5` — a near-miss is weaker evidence), surfacing
  near-misses an exact search misses (`protien` → `protein`).

Both lexical hits carry source-aware provenance (`SubstringInfo`/`FuzzyMatch`
`.source`/`.ref`), today always `source="field"` but seamed for `ocr`/`asr`. The
substring (`exact`) candidates come from the same store
(`DerivedTextStore.search_substring`, a fast FTS5 pre-filter), falling back to
`find_notes` when the store is unavailable or the query is sub-trigram; either way
`substring_info` stays the authority that confirms and annotates each candidate.

## Which ops touch the index

Tool operations fall into three index-handling categories:

- **Re-embed** — anything that changes a note's *field bodies* (embedding text):
  `upsert_notes`, `find_replace_notes`, `migrate_note_type`. Changed notes are
  re-embedded via the `upsert_notes` index path.
- **Remove vectors** — `delete_notes`, and the empty-note/empty-card paths of
  `collection_prune`, drop vectors via `index.remove`.
- **Metadata-only (no re-embed)** — tags, deck names, template/CSS text, and
  per-field editor metadata are *not* part of a note's embedding text. These ops
  leave every vector valid but bump `col.mod`. They advance the stored
  `index.col_mod` (and the derived store's watermark) without re-embedding, via the
  shared `_bump_col_mod_after_metadata_change` helper, so a metadata-only change
  doesn't force a spurious full rebuild on next startup.
