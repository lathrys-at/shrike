# Design decisions

A log of non-obvious design choices for *shipped* work — the "why", kept out of
both the code (which shows the "what") and the issue tracker (which holds *future*
work). New entries go on top of their section. For the mechanics of how a feature
works, see `CLAUDE.md`'s "Key technical details"; this file is the reasoning that
isn't reconstructable from the code.

## Semantic search & the vector index

### The activation gate floors a modality by its own typical match, calibrated offline (#201b)

#201a made `image` a separate RRF signal but fed it **unfloored**: RRF fuses rank positions, so it
neutralizes the modality gap's *constant* offset but is blind to *magnitude within* a modality — it
can't tell "this query's best image match is genuinely good" from "this query has no good image
match, here's the least-bad one." Unfloored, a multimodal collection injects its top-k image cards
into *every* query. The fix is an **intra-modal activation gate**: a non-text modality contributes
only when its best match for the query clears `mean + margin·std` of that modality's *typical* best
match. Three decisions made it tractable:

- **Calibrate offline from the index, not from query traffic.** "Typical best match per modality" is
  a property of the embedding *space* (the gap offset and its spread), not of who queries. So
  `_calibrate_activation` samples stored text vectors as pseudo-queries, searches each non-text
  modality, and records the best **non-self** match (a note's own image isn't a match) as
  `{n, mean, std}` in `index.meta.json`. This is cold-start-safe (ready the moment the index builds),
  BYO-robust (recomputed on a model change, which forces a rebuild), and independent of sparse,
  unrepresentative end-user search logs. Reading the vectors already in the text index means no
  re-embedding. Pseudo-queries are note texts rather than search strings; accepted, because the gap
  offset dominates and is query-length-insensitive.
- **Gate is binary per modality, keyed on the best match** — not a per-hit floor. It answers "should
  this modality speak at all for this query"; RRF already down-weights ranks 2+. Simpler, and it
  matches what the calibration measures (a best-match distribution). A per-hit floor is a later
  refinement if needed.
- **Text is never gated.** Text is the always-relevant primary signal with its own user-facing
  `threshold`; the gap is a *cross-modal* artifact, so calibrating-and-gating text would change
  text-only behaviour for no benefit. Off-text means a text-only collection has nothing to calibrate
  and the gate is fully inert there — byte-for-byte unchanged.

Pure addition, no schema bump: `activation` is an optional meta key, and its absence (a pre-#201b
index, or text-only) yields no floor, so the gate is simply off until the next rebuild/reconcile (or
a boot-time `ensure_calibrated`) computes it. The margin is a module constant (`ACTIVATION_MARGIN`,
like `RRF_K`) pending the search-tuning harness.

### Per-modality retrieval splits the index, not the score (#201a)

The 3c multi-vector index stored a note's text + image vectors under one `note_id` key in one
`multi=True` USearch index. The problem: a USearch hit returns the `note_id` and the distance but
**not which vector matched** — so `search_notes` could build only *one* deduped semantic ranking,
and across the CLIP modality gap (text-text cos ~0.7 vs text-image ~0.3) a text query ranks every
note's text vector above every image vector. Image hits were additive but never rank-1: a card whose
*meaning lives in its image* couldn't be found by a text query. The fix is **per-modality
sub-indexes** — `VectorIndex` keeps one USearch `Index` per modality (`text`, `image`), text vectors
in `index.usearch`, image vectors in `index.image.usearch`. Separate indexes are the *only* way to
recover a per-modality ranking from USearch given that limitation; a composite-key or filter scheme
on one index can't, because the hit still wouldn't say which modality it came from.
`search_by_modality` then ranks notes per modality (max-sim over a note's items of that modality —
late-interaction style: a note scores as its single best-matching aspect) and each modality enters
RRF as its **own signal**. Because RRF fuses rank *positions*, the gap's constant offset is invisible
(see the #180 entry) — this is what makes the split pay off rather than just re-expose the gap.

Two sub-decisions. **The image ranking is not thresholded.** The `threshold` knob is a *text*-cosine
floor (~0.5); applied to gap-depressed image cosines (~0.3) it would floor every image hit, defeating
the point. So #201a passes image hits through unfloored, and deciding *when a modality's matches are
good enough to contribute* is the job of the offline-calibrated intra-modal activation gate, **#201b**
(above) — a separate problem (it needs per-modality best-match statistics, not a global constant).
Text-only collections are unaffected either way (their image sub-index is empty → no image signal).
**Migration is a one-way schema bump, not a format converter.** `index.meta.json` gains a `schema`
marker; a pre-#201a (v1) index has none. A *text-only* v1 index is byte-identical to a v2 text
sub-index (all its vectors are text, its file already *is* `index.usearch`), so it loads losslessly
with no rebuild — text-only users pay nothing on upgrade. A *CLIP* v1 index mixed text + image vectors
under one key and **can't be unmixed**, so an image-capable backend meeting a v1 index rebuilds once
(reusing the existing drift-rebuild path). Detecting this by the schema marker rather than the old
`multi`-flag check is what lets the text-only case skip the rebuild.

### Search fuses signals by rank (RRF), not normalized score (#180)

`search_notes` blends retrieval signals — semantic cosine (now per-modality `text`/`image`, #201a) +
exact substring + trigram `fuzzy` (#98), soon tag-centroid (#179). They live on
incommensurable scales: cosine clusters in a narrow ~0.3–0.7 band, exact match is near-binary, a
cross-modal (text-query↔image-vector) cosine sits a roughly *constant offset* below within-modal.
**Normalize-and-sum inherits every pathology** — min-max stretches cosine's narrow band so trivial
gaps look huge, the binary exact signal dominates or vanishes with its weight, and normalized scores
depend on *what else* was retrieved, so a card's order wobbles between queries. We chose **Reciprocal
Rank Fusion**: each signal ranks its own candidates and a note's score is `Σ w_s·1/(k+rank_s)` (k=60).
Rank position discards raw magnitude (never reconcile cosine-0.7 vs a binary hit); a note absent from
a signal is rank-∞ → contributes nothing, which *is* the graceful degradation we want for
untagged / no-match cards; and orderings are **stable across queries** — the single biggest "feels
right" property in search UX. What RRF gives up is magnitude, which matters in exactly one place: a
literal exact hit should outrank a merely-similar note regardless of rank gap — so the combiner
carries a **priority tier** (`priority_signals`) that floats exact hits above the rest, RRF-ordered
within. The decisive property for the multimodal arc: because RRF fuses *rank positions*, the
per-modality constant cosine offset (the CLIP modality gap) is **invisible** to it — so #201a's
separate `text`/`image` rankings neutralize the gap with no normalization or calibration, which is
why the multimodal addendum landed as new signals over an unchanged fusion backbone. The combiner (`search_fusion.rrf_fuse`) is pure (ints in,
ranked ints out) and returns per-note which signals contributed at what rank — the seam provenance
(#182) reads. This first slice ships the backbone over the two existing signals (near
behaviour-equivalent today, since RRF over one semantic ranking == rank order); its worth is the
extensible architecture every later signal plugs into by just producing a ranking.

### Derived data lives in a sidecar `shrike.db`, not in `collection.anki2` (#98)

Shrike is starting to **derive data locally from notes** — a trigram lexical index now (the `fuzzy`
signal + the substring candidate source); OCR/ASR recognized text next (#199); VLM image-describe
text later. That derived/computed data wants **one home, separate from Anki's synced collection**.
The settled choice is a sidecar SQLite file (`shrike.db` in `cache_dir()`), **not** new tables in
`collection.anki2`. Anki's sync, "Check Database", media check, and version-upgrade migrations all
**own** that schema: a foreign table risks being dropped or erroring, and — worse — it would ship
*rebuildable derived data over sync*, which is exactly what a derived cache must never do. We already
time-share the collection lock with Anki desktop (#64), and the community norm is that add-ons keep
their own files beside the collection. A sidecar in our cache dir is both safe and the correct home:
rebuildable from the collection (the source of truth), so a corrupt/stale/missing sidecar is never a
data-loss event. With the relay in view (offload heavy compute to a user's desktop instance and sync
the *artifacts*, not recompute them everywhere), this is also the natural sync target — so the store
is **source-seamed** (`(note_id, source, ref)` rows) from day one, even though `field` is its only
source today.

The store is the FTS5-trigram **`DerivedTextStore`** (`derived.py`). Two design choices worth pinning:
**(1) it feeds two signals, not one.** A quoted-phrase trigram MATCH *is* a literal substring match,
so the store supplies the `exact` signal's candidates (a fast pre-filter replacing the linear
`find_notes` scan); a trigram-OR MATCH ranked by bm25 *is* fuzzy/typo matching, the new `fuzzy`
signal. Both degrade to the old `find_notes` path when the runtime's SQLite lacks FTS5 (probed at
construction) or the query is sub-trigram — `substring_info` stays the authority that confirms every
exact candidate, so the swap is behaviour-preserving. **(2) provenance is source-aware** so the
payoff #199 unlocks is designed in now: a lexical hit carries `source`/`ref`/`snippet`, so a result
can report *which* derived text matched (a field today; "the OCR text of diagram.png" tomorrow) — the
window an LLM/MCP client needs to understand an image/audio card it can't be shown. A VLM
image-describe source, when it lands, goes to the embedding space **only**, never the trigram index:
a literal-search hit on metadata the user never sees can't be cleanly explained, so it must not drive
fast lexical search. Unlike the vector index the store has **no debounced saver** (SQLite writes are
durable per-commit) and is **independent of the embedder** (it builds and ingests with embeddings
off) — the one place the derived-cache pattern deliberately diverges from `VectorIndex`.

### USearch stays the index; revisit only on a measured, specific trigger

USearch (HNSW + quantization) is the vector index, and the cross-platform plan keeps
it. No native option beats it for a zero-server, local-first design: Apple offers only
a brute-force kNN primitive (`BNNS.NearestNeighbors`), Android has no first-party ANN,
and the portable third-party alternatives don't win on portability *and* performance
together. USearch already runs natively on desktop and on both mobile platforms, so a
future port carries the same index rather than swapping it. Revisit **only** on a
specific, measured trigger — SQLite-transactional co-location of vectors (→ evaluate
`sqlite-vec`), or Android object-DB ergonomics (→ evaluate ObjectBox) — and only on a
demonstrated win, not on spec. (This is the index *engine* choice; that the index is a
per-device derived cache rebuilt from synced cards is a separate decision — see "The
index is a derived cache" below and the Sync epic, #38.)

### Embedding backends are pluggable behind one small protocol (#172)

The embedder used to *be* llama-server: `VectorIndex` and the server boot path
talked to a concrete `EmbeddingService` and nothing else. Going multimodal (#162)
means swapping models and *runtimes* wholesale, and two nearer-term needs pushed
the same way — an ONNX runtime for deployments where a pinned llama.cpp binary is
the wrong fit, and a guarantee that **text-only embedding stays first-class
forever** (the test suite depends on small/fast text-only models). So we landed
the seam first, ahead of any multimodal model or index change: a minimal
`EmbedderBackend` protocol (`embed_texts`, `embedding_dim`, `model_fingerprint`,
`health`, lifecycle, `modalities`) with `LlamaServerBackend` and `OnnxBackend`
behind it. The index never learns which backend it has — it only calls
`embed_texts` — so drift, the per-note hash sidecar, and persistence stay
backend-agnostic.

Three choices worth recording. **The fingerprint is namespaced by family**
(`onnx:…` vs llama's `meta:`/`file:…`) rather than adding a backend token to the
existing llama fingerprint: the same model under two runtimes produces different
vectors and must not share a space, but an existing llama index should *not* be
forced to rebuild on upgrade — distinct prefixes give the separation for free.
**`modalities` is a declared capability, not a config flag.** In Phase 1 every
backend is `{"text"}`, so it changes no behaviour today; its job is to make
text-only a *named, permanent* capability that a later multimodal backend extends,
so media-by-content search lights up where vectors exist and silently returns
nothing where they don't — degrading, never erroring. We deliberately did **not**
build the media-embedding branch yet (no dead code) nor decide the multi-vector-
vs-fusion index question — that needs evaluation on a real collection (#162).
**ONNX pooling is folded into the fingerprint; normalization is not.** Pooling
(mean/cls/last) changes a vector's direction → vector-affecting → must invalidate
the index; L2 normalization changes only magnitude, and USearch's `cos` metric is
scale-invariant, so it never changes ranking — the same reasoning that already
makes llama's `--embd-normalize` moot. CI runs a minimal embedding subset against
*both* backends (`test_backends.py`), plus a second architecturally-different real
model (DistilRoBERTa, `test_onnx_models.py`) — the real models keep the cheap mocked
unit tests honest (their assumed onnxruntime input-type strings, output ranks, and
tokenizer behaviour stay falsifiable instead of drifting from reality).

Three ONNX operational calls came out of the second-model work, all anchored by that
real DistilRoBERTa run:

- **Batch safety is probed empirically at startup, not assumed (#174).** This is the
  headline. int8 ONNX exports use dynamic quantization, which computes activation scales
  over the *whole batch tensor* — so a batched embed makes a note's vector depend on the
  *content* of its batch-mates (~0.06 drift; the two-different-same-length-batch-mates
  control rules out padding). Non-quantized fp32/fp16 ONNX is **bit-exact** batched, and
  llama-server (fp) is at float noise — so the variance is purely an int8-dynamic-quant
  property, not general. A batch-variant backend would break the index's core invariant
  that a `reconcile`'s end state is identical to a full rebuild (the same note would embed
  differently in a rebuild-of-64 than an upsert-of-1). Rather than guess from a model's
  quantization scheme, **every backend probes at `start()`** (`embed_batching.probe_max_safe_batch`):
  embed the probe texts serially (the reference) and **all in one batch**, and compare within a
  tolerance chosen to sit above float noise (~4e-5) and below quant drift (~0.06). Match → the
  model batches deterministically, safe **up to the probe-set size**; mismatch → embed
  **serially** (batch size 1). `embed_texts` then never batches larger than that proven size
  (further capped by `--embedding-batch-size`), so "what we proved" and "what we do" are the
  same size — no extrapolating from a small sweep to a larger runtime batch. Locked by
  exact-equality (`np.array_equal`) tests against *real* models — int8 (serial) and fp32
  (batched == serial) — not mocks, which are trivially batch-independent. Universal across
  backends (llama, ONNX, any future one) and re-confirmed every boot; it retries a transient
  embed failure before falling back to serial. **Release-ordering caveat:** batching a *safe*
  model is bit-exact, so it changes no vectors and needs no fingerprint bump; only a change to
  the serial-vs-batched *decision* would, and the probe makes that a property of the model, not
  a stored setting.

  Three design points make the empirical check trustworthy rather than wishful (#174 review):
  - **It is a heuristic for arbitrary user models, not a proof.** A model that is batch-variant
    on real notes but happens to drift `<tol` on the probe set would be misclassified safe →
    silent index non-determinism (the exact invariant this protects). So the probe set is
    **spiked for activation magnitude**: int8 drift on a text is maximized when it is *calm* and
    a batch-mate is *spiky* (drives the per-tensor activation min/max), so the set mixes calm
    anchors with deliberately spiky inputs (a long text, numeric/hex/code, symbol soup, a
    degenerate repeated token, mixed-script/emoji, ALL CAPS). An fp model has no activation
    quant, so its batched-vs-serial drift is exactly 0 *regardless of content* — spiking only
    raises sensitivity to variant models, never false-positives a safe one. The set's
    sensitivity is **pinned by a test** asserting the probe drift exceeds ~10× `tol` on the real
    int8 MiniLM and DistilRoBERTa fixtures, so a future bland-set regression fails CI.
  - **We compare against one full batch, and the probe-set size *is* the batch ceiling.** A
    single most-heterogeneous batch is the most sensitive configuration *and* matches usage:
    `embed_texts` never batches larger than the verified set — sized to **64**, the index's
    `BATCH_SIZE` chunk, so a probe-safe (fp / non-dynamic-quant) model batches at the full chunk
    a GPU favours, while `--embedding-batch-size` caps it lower and a request above the ceiling is
    honoured up to it with a one-time log. This closes the earlier gap where an escalating
    2/4/8/16 sweep could return "safe at 8" while `embed_texts` batched at 64. (Batching past 64
    would also need `index.BATCH_SIZE` raised — a later slice. A deterministic ONNX-only
    fallback — scan the graph for
    `DynamicQuantizeLinear`/`MatMulInteger` — would classify int8 variance exactly, but it needs
    graph introspection the in-process backend avoids and wouldn't generalize to llama, so the
    empirical probe stays the default.)
- **"Bit-exact" is a CPU property; accelerators and llama are search-stable float noise.** The
  `reconcile`==full-rebuild invariant is *byte-identical* only on `CPUExecutionProvider` (int8
  serial; fp32 batched at exactly 0 drift). On a GPU provider (CUDA/CoreML) an fp model's
  batched-vs-serial result differs by ~1e-5 (measured on CoreML; different matmul kernels are
  chosen per batch shape) — the same float-noise tier llama-server already occupies (~4e-5). The
  probe's `BATCH_DRIFT_TOL` (1e-3) sits above all of this, so such a model still measures
  batch-safe and batches; the difference is far below cosine-ranking resolution, so search is
  identical. The execution provider is therefore **deliberately not in `model_fingerprint`** (the
  same call as normalization and the llama-server build version): a CPU↔GPU switch produces
  vectors that differ at float noise, never enough to warrant a re-embed, and a mixed index ranks
  identically. The `np.array_equal` determinism tests assert the *CPU* bit-exact property
  specifically (CI is CPU); a future GPU test lane would assert `allclose(atol≈1e-4)` / identical
  ranking, not byte-equality. It also vindicates the empirical probe: on any provider it *measures*
  (fp→batch, int8→serial) rather than guessing what the accelerator does to quantization.
- **Pad token resolved across conventions, not hard-coded.** BERT/WordPiece names the
  pad token `[PAD]`, RoBERTa/BPE uses `<pad>`; `OnnxBackend` resolves `[PAD]` then
  `<pad>`, falling back to id 0 only if neither exists. RoBERTa derives position ids
  from *which tokens ≠ the pad id*, so padding a batch with the wrong id shifts the real
  tokens' positions and corrupts their embeddings (the real DistilRoBERTa run surfaced
  this; a BERT-tokenizer mock never reaches the `<pad>` branch). Padding is applied
  whenever a batch-safe model embeds a chunk of >1, so this resolution is load-bearing,
  not merely defensive.
- **A required input we don't supply fails loud at `start()`, not silently at first embed.**
  The backend feeds a fixed set (`input_ids`/`attention_mask`/`token_type_ids`); a model with
  a *required* input outside it — most commonly `position_ids` (some optimum/transformers.js
  exports) — would otherwise boot fine (the startup probe's failure was caught → serial) and
  then break on the first real embed, with only a generic "probe failed" line. Now an ONNX
  serial-embed failure (deterministic, unlike llama's transient HTTP) raises from `start()`,
  naming the unsupported inputs, so it surfaces at boot (ERROR + degrade-to-no-embedding) or as
  a 500 from `/embedding/start`. We deliberately do **not** auto-supply `position_ids`: correct
  positions are architecture-specific (RoBERTa offsets by `padding_idx`, BERT uses a plain
  `arange`), so guessing would silently *corrupt* embeddings — strictly worse than refusing.
  Same detect-and-refuse principle as the pad-token resolution above. (A *batch-only* failure —
  serial works, batched doesn't, e.g. a fixed batch-1 graph — is not fatal: it degrades to
  serial.)
- **`--embedding-context-size` truncates but is not clamped to the model's ceiling.**
  It sets the ONNX token-truncation length; raising it past the model's
  `max_position_embeddings` is the operator's responsibility (documented in the CLI
  help). We deliberately don't clamp: that limit isn't reliably discoverable from an
  arbitrary ONNX graph (sometimes a static input shape, sometimes only a sibling
  `config.json`, sometimes absent). A warn-only `config.json` read is a possible
  future safety net, not a clamp.
- **Execution providers resolve gracefully and the active one is visible.** A requested
  `--embedding-onnx-provider` (e.g. `CUDAExecutionProvider`) is intersected with
  onnxruntime's `get_available_providers()`; an unavailable one is dropped **with a warning**
  (not onnxruntime's silent CPU fallback), and `CPUExecutionProvider` is always appended as the
  final fallback so an absent accelerator degrades rather than hard-errors. After construction
  we read `session.get_providers()` — what *actually* loaded — and warn if a requested provider
  was available but failed to initialise. `health()`/`server status` surface the effective
  provider, so "I asked for CUDA but it's running on CPU" is visible instead of a silent
  performance cliff. Packaging mirrors how onnxruntime ships: `shrike[onnx]` is the base wheel
  (CPU + CoreML on macOS); `shrike[onnx-gpu]` is `onnxruntime-gpu` (CUDA/TensorRT) and is
  installed *instead of* `[onnx]` (the two wheels conflict); Windows DirectML is a manual
  `onnxruntime-directml`.
- **The CLIP backend is a *dual* encoder behind the same seam (#162 Phase 3b).** Multimodal
  search (a text query retrieving a card by its image) needs image and text in *one* vector
  space. `ClipBackend` (`embedding_clip.py`) loads two ONNX graphs — text (`input_ids → text_embeds`)
  and vision (`pixel_values → image_embeds`) — both projecting into the same space (L2-normalize,
  no pooling: unlike the text-only `OnnxBackend`, both graphs emit a pre-pooled projected vector).
  Image preprocessing (resize → center-crop → rescale → normalize) is read from the model's
  `preprocessor_config.json` and done in PIL + numpy — **no torch/torchvision** (the eval's
  dependency pain). It reuses `OnnxBackend`'s provider resolution and the batch-safety probe: the
  two graphs come from one export with one quantization, so **a single text-path probe governs
  both** (int8 CLIP → serial; fp CLIP → batches). It advertises `modalities={text,image}` (the
  graceful-degradation seam) and exposes an extra `embed_images()` method — text-only backends
  don't implement it, so callers gate on `IMAGE in modalities`. The GO and the chosen embedding
  unit (multi-vector per note, `multi=True`) came from the Phase-3a eval (#193). `jina-clip-v2` is
  the production-quality option; a small `clip-vit-base-patch32` is the CI fixture.
- **The multi-vector index stores a note's text + image vectors under one key (#162 Phase 3c).**
  USearch `Index(..., multi=True)` lets a `note_id` key hold several vectors (its text vector + one
  per image); `remove(note_id)` drops them all, and search dedups multi-hits back to one result per
  note (over-fetch, keep the best distance per note). The flag is always on — a text-only backend
  just stores one vector per key (identical behaviour). Three decisions are load-bearing: **(1)
  image bytes are read lazily and lock-free** — the media dir is path-derived (resolves without the
  Anki lock, #70), so the index reads bytes on its own embed thread via an injected resolver, only
  for the notes it's (re-)embedding, never pre-reading every image on a drift check. **(2) The
  reconcile fingerprint hashes the *filenames of a note's resolvable images*, not bytes** — Anki
  content-addresses media (a filename is a stable content identity), so this detects
  add/remove/swap/late-arrival cheaply (DB text + a regex + a presence `stat`, no byte read) and is
  folded in *only* for an image-capable backend, leaving text-only hashes byte-identical to the
  pre-3c scheme (no spurious upgrade rebuild). Hashing only *resolvable* names (the resolver's
  cheap `exists` half) is what keeps reconcile == a full rebuild even for a note authored before
  its media landed — the name of an unstored image isn't folded in, so the image re-embeds once it
  appears rather than being claimed-but-absent forever. **(3) No index schema marker is
  needed** — USearch persists the `multi` flag, so a pre-3c `multi=False` index is detected on load
  and rebuilt into a multi-vector one *only when* an image-capable backend attaches (`check_drift`);
  a text-only user keeps their single-vector index untouched. **Scope:** 3c indexes image vectors
  so they're retrievable and maintained, but the CLIP **modality gap** (text-text cos ~0.7 vs
  text-image ~0.3) means a text query ranks text vectors above image vectors at rank-1 — image hits
  are *additive*, not dominant. Ranking them across the gap is **rank fusion (the Search epic #180,
  Phase 3d)**, deliberately not 3c; the integration test asserts the data layer (indexed,
  reconciled, retrievable), and the eval (#193) measured the rank reality.

### The index is a derived cache, never a co-equal store

The Anki collection (SQLite) is always the source of truth; the USearch index is
a rebuildable projection of it. The index may lag the collection (stale search
results) but the collection never lags the index (data ops are always correct).
This is what lets drift detection be a single `col.mod` comparison with a
background rebuild, rather than per-note reconciliation. The full rationale,
drift-detection scheme, and model-fingerprint design live under "Vector index and
consistency" in `CLAUDE.md` — not duplicated here.

### Contextual upsert returns neighbours; it makes no suggestions

`upsert_notes` returns, for each created/updated note, the *k* most similar
existing notes (`{id, score, tags}`, defaults `top_k_neighbors=5`,
`neighbor_threshold=0.5`) — the same similarity operation `search_notes` runs,
queried by the upserted note's own content, with the batch's own notes excluded
from each other's results.

It returns **raw neighbour data and stops there.** The server suggests no tags,
flags no duplicates, makes no decisions. The reasoning: the server can't know the
caller's intent, and baking in a policy (auto-tagging, dup-rejection thresholds)
would be a guess that's wrong half the time and impossible to override cleanly.
Handing back the neighbours lets an LLM-driven caller ground new cards in the
collection's existing taxonomy (reuse tags that already exist) and spot
near-duplicates — while the *policy* for what to do with that lives in the skill,
not the server.

### Semantic duplicate detection is not a separate feature

There is no dedicated *semantic* duplicate-detection endpoint, and there won't be
one. A high similarity score in `search_notes` results or in upsert neighbours *is*
the soft duplicate signal; the caller applies its own threshold. Adding a second
code path that re-implements the same cosine-similarity lookup with a built-in
cutoff would be redundant surface area with a worse interface (a hard-coded
threshold the caller can't see or tune).

### Anki's exact duplicate rule lives inside `upsert_notes`, not a `canAddNotes` tool (#77)

Distinct from the semantic signal above, Anki has a *precise* duplicate rule:
a note duplicates another if it shares the first field with an existing note of the
same type (collection-wide, deck-independent). #77 asked for a pre-flight check for
this, mirroring anki-connect's `canAddNotes`. We folded it into `upsert_notes` (an
`on_duplicate` policy defaulting to `error`, plus a `dry_run` flag) rather than
shipping a standalone checker, because a separate check is the wrong shape for this
codebase:

- **It would be racy.** A check-then-write pair has a TOCTOU gap; the collection
  can change between the two calls, so the check can lie. Folding the rule into the
  write makes creation a single atomic, race-free operation.
- **Its result is only actionable by another call.** A checker that says "this would
  be a duplicate" just sends you to `upsert_notes` anyway — extra surface area and a
  round-trip for no committed outcome.
- **It overlaps the existing per-item result union.** `error`/`skipped`/`ok`
  variants already fit `UpsertNoteResult`; a second tool would re-model the same
  thing.

The one capability a standalone checker has that an inline policy doesn't — a pure
zero-write preview — is covered by `dry_run`, which runs the identical validation
path and writes nothing. So `dry_run` with the default `on_duplicate="error"` is a
full `fields_check`-based sanity pass over a batch, without a second tool or a
second code path. Structurally invalid notes (empty first field, broken cloze) are
always errors regardless of `on_duplicate` — they're malformed, not merely
duplicated. The default is `error` (not the old silent-create behaviour) because
silently writing a duplicate or an empty note is almost always a mistake; callers
who genuinely want duplicates opt in with `allow`.

### One query, many retrieval mechanisms, annotated results (#86)

`search_notes` takes a single `queries` input and runs it through *every*
retrieval mechanism, folding the evidence into one result list rather than
exposing a separate param/tool per mechanism. Today that's semantic similarity
(the vector index) and exact case-insensitive substring matching; each match
carries a `score` when semantically ranked and a `substring` annotation (matched
fields + snippet) when the text occurs literally — both when both apply, and a
given annotation is simply absent otherwise (a returned match always has at
least one).

This was deliberately chosen over a `--substring`/`substring` parameter: a
separate flag reads as a filter and forces a union-vs-intersection decision,
whereas "the query is matched every way we can, and results tell you how" needs
no mode. It also degrades gracefully — with the embedding index down, a query
still returns its exact matches (no `score`) plus an advisory `message`. The
optional match-evidence fields are the extension point: a future n-gram/fuzzy
backend (#98) adds another annotation, never a new param or tool. Exact matches
are **not** subject to the semantic `threshold` (a literal hit is always
relevant), and within a group literal hits are listed first, then by score.

### Raw Anki query moved from a leaky param to its own tool (#86 → #97)

`note list --query` / `list_notes.query` (the raw Anki search escape hatch) was
removed in #86: text search lives in `search_notes`, deck/tag/type are structured
`list_notes` filters, and the raw param was a leaky mode bolted onto a structured
tool. The remaining raw-Anki power (`is:due`, `prop:ivl>=30`, `added:`, `flag:`,
`OR`/`-`/brackets, …) returned in #97 as an explicit tool — `collection_query` /
`shrike collection query` — rather than that param.

Two scope decisions when building it:

- **Full grammar, no whitelist.** The query string is passed straight to
  `col.find_notes` — every operator works, including the review/scheduling
  predicates (`is:due`, `prop:`, `rated:`). This *looks* like it brushes Shrike's
  non-review stance, but that stance is about not performing review *operations*
  (answering cards, rescheduling). `collection_query` is **read-only**: filtering
  by `is:due` returns notes, it reviews nothing. Whitelisting a "non-review
  subset" would mean re-implementing a parser to police Anki's grammar (fragile)
  and would defeat the whole point of a raw escape hatch — so we don't.

- **Its own tool, reusing `list_notes`' shape.** It returns the same `Note` /
  `ListNotesResponse` as `list_notes` (same `_note_to_dict`), so callers get one
  note shape across all three retrieval surfaces. A malformed expression is a
  caller error: `find_notes` raises `SearchError`, surfaced as `ToolInputError`
  (with Anki's U+2068/U+2069 isolation marks stripped from the message). It lives
  in the `collection` CLI group introduced by #89.

### Find-and-replace edits via Anki's engine; a scope is required (#85)

`find_replace_notes` / `shrike note replace` runs the actual edit through Anki's
`col.find_and_replace` (Rust regex — linear-time, no catastrophic backtracking —
and undo-able), rather than re-implementing replacement. A **scope is required**
(`deck`/`tags`/`note_type`/`ids`) so a collection-wide edit is always an explicit
choice. The MCP tool applies by default with a `dry_run` option; the CLI previews
then confirms (`--dry-run`/`--yes`).

Two subtleties worth recording:
- **Re-embedding.** Unlike tag/deck ops, this changes field bodies, which *are*
  embedding text — so changed notes are re-embedded (the `upsert_notes` index
  path), not just `col_mod`-bumped. The changed set is found by diffing
  `notes.flds` before/after the apply; `note.mod` is only second-resolution, so a
  same-second edit wouldn't show a bump.
- **Preview faithfulness.** The dry-run preview is rendered in Python
  (`apply_replacement`): exact for literal edits, illustrative for regex (Anki's
  `$1` capture syntax vs Python's `\1`), with the apply authoritative. Acceptable
  because literal is the common case and the apply is what mutates.

## Tags

### Setting tags is a full replace; add/remove is a separate operation (#73)

When you *set* tags — `upsert_notes` partial updates (`{id, tags}`),
`shrike note update --tags`, and `update_note_tags`/`shrike note tag --set`
(`--set ""` clears) — the note ends up with *exactly* the set you sent. Replace
never silently merges.

We rejected an additive/subtractive `mode` parameter that changes the meaning of
the `tags` field on `upsert_notes` — that's the "bag of optionals / hidden state"
the `schemas.py` house style warns against, where one field's meaning depends on
another. Create-time tagging stays a full-set decision ("this card's tags are X,
Y, Z").

Additive/subtractive editing is instead its **own** tool, `update_note_tags`,
with explicit, non-overlapping fields: `set` (full replace) is mutually exclusive
with `add`/`remove`, and `add`/`remove` combine freely (e.g. add `["jp","verbs"]`
+ remove `["jp-verbs"]` swaps one tag for two). There is no default mode — the
caller picks one. So replace and merge are distinct, named operations rather than
a single overloaded field.

### Retroactive collection-wide tag curation is built (#73)

Earlier this was deliberately unbuilt "until a concrete need appears" — #73 was
that need. Bulk add/remove over existing notes (`update_note_tags`, backed by
Anki's `bulk_add`/`bulk_remove`) and collection-wide and note-scoped tag rename
(`rename_tag` / `shrike tag rename`) are available. Note-scoped rename matches the
tag *exactly* (find notes carrying it, then swap) rather than a substring
find/replace, so renaming `jp` never touches `jp-verbs`. Unused-tag cleanup
shipped briefly here too, but was pulled back out: it belongs with other
collection-maintenance chores (remove-empty-notes, etc.) under a single
`collection_prune` op rather than a one-off `tag clean` (#89).

Because tags are not part of a note's embedding text, these ops bump `col.mod`
but leave every vector valid; each advances the stored index `col_mod` without
re-embedding, so a tag-only change doesn't trigger a full rebuild on next
startup.

## Decks

### Deck deletion is empty-only; decks never merge (#74)

`delete_decks` / `shrike deck delete` refuses unless the deck *and every subdeck*
is empty. There is deliberately no "delete the cards too" or "move cards to
Default" mode. Emptying a deck is a separate, composable step — move its notes
elsewhere (`upsert_notes` with a new `deck`, `shrike note update --deck`) — and
then the deck is deletable. The payoff: **deck deletion can never delete a note**,
so it has no bearing on the collection's note set or the vector index (a deck name
isn't embedding text). A destructive cards-and-notes wipe, if ever wanted, is just
`delete_notes` on the deck's notes — no need to overload deck delete with it.

Renaming a deck onto a name another deck already uses is an **error**, not a
merge. Anki's backend would silently disambiguate (`B` → `B+`), which is
surprising and litters the tree; we reject the collision instead and tell the
caller to move notes if they meant to consolidate. So `upsert_decks` mirrors
`upsert_notes` (id present = rename the existing deck; absent = create) without
ever introducing a hidden merge.

Like tag ops, deck create/rename/delete bump `col.mod` but change no vector, so
they advance the stored index `col_mod` (shared `_bump_col_mod_after_metadata_change`
helper) without re-embedding.

### Deck references accept name, numeric ID, or `#id` (#88)

Anywhere a deck is *referenced* (not created) — `list_notes`/`note list --deck`,
`search_notes`, `upsert_notes`' `deck`, `delete_decks`, `deck rename`'s target —
the value may be a deck name, a bare numeric ID, or a `#`-prefixed ID. One rule,
applied server-side in `CollectionWrapper._resolve_deck_ref` and mirrored by the
CLI's `_match_deck` for `deck rename`:

- `#<id>` is **always** an ID; it resolves to that deck's name, or is "not found"
  if no deck has it (never silently treated as a name).
- a **bare integer** is tried as an ID first, then falls back to a literal name —
  so a deck genuinely named `123` is still reachable, while `123` meaning deck-id
  123 is the common case.
- anything else is a name.

This mirrors note IDs' `#`-prefix handling (`NoteIDType`). Resolution lives on the
server because that's where the collection is; the CLI passes references through
untouched (except `deck rename`, which must resolve to an ID client-side for the
`upsert_decks` call). On note **create**, a name that doesn't exist is still
auto-created as before; only an explicit unknown `#id` is an error.

## Note types

### Field and template updates are applied by position, preserving note data (#76)

`upsert_note_types` lets you replace a note type's whole `fields`/`templates`
list on update. Anki migrates a note's field *values* (and a template's *cards*)
by **ordinal** — the field/template at position *N* keeps its data as long as
position *N* survives. The original implementation rebuilt `flds`/`tmpls` from
fresh `new_field`/`new_template` objects, which carry no ordinal, so Anki saw
every existing field/template as removed and every incoming one as new. The
effect was catastrophic and silent: **any** update carrying a `fields` key blanked
every note's content for that type, and **any** `templates` key deleted every
card (losing all scheduling/review history) — even re-sending the *identical*
list did it.

The fix (`_set_fields` / `_set_templates` in `note_types.py`) reuses the existing
field/template dicts in place — renaming/retitling the ones whose position
survives, appending only for added positions, and dropping the tail for removed
ones. So a whole-list replace is now data-safe and matches Anki's
"keyed by position" rule: rename-in-place and edit-in-place preserve data;
lengthening appends empty fields / new cards; shortening discards only the
trailing entries (the standard meaning of removing a field/template). A genuine
*reorder* (moving a field/template to a new position while keeping its identity)
is necessarily a separate, explicit operation — a positional name swap reads as
two renames, which is non-destructive but not a move.

### Identity-based field/template ops are separate tools, not modes of `upsert_note_types` (#76)

The genuine move/insert/non-trailing-remove that position-replace can't express
lives in two tools: `update_note_type_fields` and `update_note_type_templates` —
sequences of `add`/`remove`/`rename`/`reposition` operations addressed by **name**.
They exist separately from `upsert_note_types` rather than as another shape of its
`fields`/`templates` params because the two are fundamentally different contracts —
a *declarative* "the fields/templates are now exactly this list" (position-keyed)
versus an *imperative* "move X to 0, rename Y" (identity-keyed). Conflating them in
one param would make "is `["B","A"]` a reorder or two renames?" ambiguous; keeping
them apart makes each unambiguous.

They delegate to Anki's own data-safe primitives (`rename_field` /
`reposition_field` / `add_field` / `remove_field` and the `*_template`
equivalents), which migrate note data / cards by identity — so a reposition is a
true move (the data/cards follow) and a non-trailing remove drops only that
field's data / that template's cards. (A template `rename` is a pure label change:
cards key by ordinal, not name, so no primitive and no card migration.) Each call
is **atomic**: the op sequence is validated against a simulated name list first, so
an invalid op (unknown name, clash, out-of-range position, removing the last entry)
changes nothing; only once every op is known-good are the primitives applied to one
in-memory notetype and persisted with a single `update_dict`. Like
`upsert_note_types`, they do no inline index maintenance — the `col.mod` bump
drives a drift-rebuild on next startup (correct, since a removed field changes a
note's embedding text). The two tools share `_simulate_struct_op` (the op shape is
identical) and raise `NoteTypeOpError`, which the tool layer turns into a
`ToolInputError`.

With correct movers in hand, the position vs identity tools are **reconciled** so
they don't overlap dangerously: `upsert_note_types`' positional `fields`/`templates`
replace now *refuses* any update where an existing name lands at a different
position — a reorder, an insert before another entry, or a non-trailing remove
(which shifts the names after it). Positionally those silently re-label note data /
cards (the value/cards stay in their slot while the slot's name changes), the
footgun the position-keyed contract can't avoid. The check is one shared rule
(`_reject_unsound_positional_replace`: an existing name may not change index) whose
error points at the matching identity tool. So `upsert_note_types` keeps only the
*unambiguous* positional edits — rename/edit-in-place, append, trailing-remove —
and every move/insert/non-trailing-remove goes through the identity tools. The
overlap that remains (those simple edits doable both ways) is the benign
PUT-vs-PATCH kind; the dangerous overlap is gone.

### `find_replace_note_types` rewrites template text, not fields (#76)

anki-connect's `findAndReplaceInModels` is the third note-type edit tool, and it
is deliberately a *different* operation from the find-and-replace over note
*content* (#85, `find_replace_notes`): this one searches a single note type's
card-template HTML (`qfmt`/`afmt`) and shared CSS — the model definition — and
touches no note field values. So the two find-and-replaces don't share code or a
selector vocabulary; they operate on different layers (one model's templates vs a
scoped set of notes' fields), and shipping the model one separately from the
unmerged #85 was the clean call.

It scopes to **one model per call** with `front`/`back`/`css` booleans (mirroring
anki-connect's shape) and returns a replacement count plus which templates/CSS
changed. Three decisions worth recording:

- **No data migration, so no re-embed — but a `col_mod` bump.** Templates and CSS
  aren't part of a note's embedding text, so every vector stays valid. Like the
  tag/deck metadata ops, it advances the stored `col_mod` without re-embedding
  (via `_bump_col_mod_after_metadata_change`) so the `update_dict` mod-bump
  doesn't trigger a spurious rebuild. This is the opposite of the *structural*
  field/template ops, whose removes change embedding text and correctly let a
  drift-rebuild happen.
- **`match_case` defaults to true**, unlike a note-content find-and-replace where
  prose makes case-insensitive the friendlier default. Template and CSS text is
  *code* — field names (`{{Front}}`), CSS class names, colours — where case is
  significant, so a case-sensitive default is the safe one.
- **Literal mode inserts the replacement verbatim.** Literal `search` is
  `re.escape`d and the replacement is applied through a constant function, so a
  replacement containing `\1` or `$2` is inserted as those characters, not
  interpreted as a backreference; capture refs are available only under `regex`.
  We reused Python `re` (not Anki's `find_and_replace`, which is note-scoped)
  because the substitution target is model strings we already hold in memory.

It does **not** rename a field — Anki's `rename_field` already rewrites template
references when a field is renamed via `update_note_type_fields`, so this tool is
for the cases that primitive doesn't cover (CSS edits, literal markup typos,
collapsing two field refs into one). Producing a template that references a
missing field still fails Anki's own save validation, as it should.

### Field editor metadata: getter folded into `collection_info`, dedicated setter (#119)

The last #76 item — per-field `font`/`size`/`description` (Anki's edit-time
cosmetics, no bearing on note data, cards, or search). Two surface calls:

- **Getter folded into `collection_info`**, not its own read. The metadata is just
  more of a note type's *definition*, so it rides the existing
  `note_type_details` block (`detail.fields[]`) the caller already requests for
  templates/CSS — one fetch for the whole definition, and it inherits the CLI
  surface (`type show`, `info --type-details`). A separate getter tool would be
  redundant round-trips for data that travels with the note type.
- **Dedicated setter `update_note_type_field_metadata`, not an op on `update_note_type_fields`.**
  The structural field ops deliberately let a drift-rebuild happen on next startup
  (a removed field changes embedding text). Field metadata changes *no* embedding
  text, so it wants the col_mod-bump-no-re-embed treatment (like the tag/deck/
  `find_replace_note_types` ops) — the opposite index policy. Folding it into
  `update_note_type_fields` would mix the two policies in one tool; a small
  dedicated setter keeps each tool's index behaviour single and clear. Like the
  #101/#102 structural ops it's MCP + client only (no CLI setter) and atomic
  (validate the whole batch, then one `update_dict`).

### Changing a note's type is a dedicated tool, not part of `upsert_notes` (#75)

`upsert_notes` hard-refuses a type change; `migrate_note_type` is where it lives,
wrapping Anki's `col.models.change`. The point of the operation is **preserving
history** — a type change keeps note IDs and carries each card's scheduling across
mapped templates, which is exactly what delete-and-recreate would throw away. So
it's worth a first-class, careful tool rather than a flag.

Why not fold it into `upsert_notes` (the issue floated both)? Folding makes
`field_map`/`template_map` a conditional sub-mode — meaningful only on an update
whose `note_type` differs — and creates an ambiguous interaction with the item's
own `fields` (set values on old field names or new? before or after the remap?).
That's the bag-of-optionals/hidden-state smell `schemas.py` warns against, plus it
buries a destructive migration (dropped fields, deleted cards' scheduling) inside
a routine bulk create/update. A dedicated tool keeps `upsert_notes` simple and its
"cannot change type" guard intact.

Three shape decisions:

- **The map is explicit; nothing is guessed.** `field_map` (source field *name* →
  target field name) is required and non-empty. A source field absent from it is
  *dropped* and reported in `dropped_fields` (content lost); target fields nothing
  maps into are reported in `new_empty_fields`. Unknown field names, two source
  fields mapping to one target, mixed source types across the `note_ids`, or
  target == source are all errors. We rejected auto-mapping same-named fields: the
  whole risk here is silent content loss, so the caller states intent and the
  response shows exactly what was dropped. (Maps are by name for the caller;
  `_migrate_note_type` translates to the ordinal `fmap`/`cmap` Anki's API takes.)
- **Apply-by-default with a `dry_run` preview**, CLI confirms — same posture as
  `find_replace_notes` (a targeted note-data edit with explicit inputs), not the
  preview-by-default posture of `collection_prune` (unscoped, collection-wide).
- **"migrate", not "change".** The verb names the intent — carrying content and
  scheduling to a new home — which is the feature's whole reason to exist over
  delete+recreate. The object stays in the name (`migrate_note_type` /
  `note migrate-type`) so it isn't a bare ambiguous "migrate".

Index handling matches `find_replace_notes`: the remap changes a note's embedding
text but not its ID, so on apply the migrated notes are re-embedded in place via
the `upsert_notes` index path.

## Collection maintenance

### One `collection_prune` tool, not scattered cleanups (#89)

Small "tidy up" chores — clear unused tags, remove empty notes, remove empty
cards — live behind one `collection_prune` tool / `shrike collection prune`
rather than a tool each. `clear_unused_tags` actually shipped standalone first
(`shrike tag clean`, #73) and was deliberately **removed** (#90) to fold in here,
and this supersedes a standalone remove-empty-notes (#78, closed wontfix). The
reasoning mirrors the "one query, many mechanisms" call for search: these are all
"maintenance passes over the whole collection," so one entry point with opt-in
flags (none selected → run all) beats N one-off verbs cluttering the surface. The
`collection` CLI group it introduces is also where `collection query` (#97) will
land.

Three decisions worth recording:

- **Preview by default — the opposite of `find_replace_notes`.** `dry_run`
  defaults **true**; the CLI previews unless `--apply`. The note find/replace
  applies by default (it's scoped to an explicit selector, and the edit is
  undoable in Anki). Prune is collection-wide *and* deletes notes and cards, with
  no per-call scope to contain a mistake — so it errs the safe way and makes you
  ask for the mutation. Same two primitives (`dry_run` / a confirm flow), opposite
  default, chosen per blast radius.

- **"Empty" is media-safe.** A note is empty only if *every* field is blank by
  `embed_text.field_is_blank` — no text **and** no media reference. This is
  stricter than the embedding normalizer (which drops media to `""`): an
  image-only or audio-only card has real content and must never be pruned. We did
  *not* use Anki's "generates no cards" definition, which would delete a note
  whose only content sits in a field no template renders — silent data loss.

- **Apply ordering: notes → cards → tags.** Empty notes are removed first, then
  empty cards, then unused tags, so a tag orphaned by the deletions is cleared in
  the same call. The dry-run previews each cleanup independently against the
  current state (and subtracts empty-note ids from the empty-cards list to avoid
  double-listing), so an apply can legitimately clear a few more tags than the
  preview showed — preview is advisory, apply is authoritative.

Index handling is **mixed**, which is the whole reason prune isn't a plain
metadata-bump op: empty-note/empty-card removal deletes notes, so their vectors
leave the index via `index.remove` exactly like `delete_notes`, while unused-tag
clearing leaves every vector valid. The tool does both in one pass and advances
`col_mod` once when anything changed.

## Collection lifecycle

### Busy is a typed error, not a per-tool response variant (#65)

Under cooperative locking a re-acquire can fail because Anki holds the collection
— "database is locked" is now an *expected* outcome that every tool call needs a
clean path for. The issue floated a discriminated-union `CollectionBusy` variant,
but busy is **orthogonal to every tool's response**: the op never ran, so it's
not "one of the shapes `upsert_notes` can return", it's a transport-level failure
that applies identically to all 18 tools. Bolting the same variant onto 18
unrelated response models would be the wrong kind of union. So it's modelled as an
**error class with a stable wire code**, riding the existing two-layer split that
already separates server `ToolInputError` from client `ServerError`: a server-side
`CollectionBusyError` whose message is prefixed with `COLLECTION_BUSY_CODE`
(`schemas.py`, the one source of truth), surfaced as an MCP `isError`, and mapped
by `ShrikeClient` to a client-side `CollectionBusyError(ShrikeError)` callers can
catch-and-retry. "Make illegal states unrepresentable" governs *within* a
response; a condition orthogonal to all responses is an error class.

And it **returns busy immediately**, with no server-side retry. A retry only helps
for a momentary lock (Anki mid-write), but the dominant case is Anki open for a
whole session — there a retry just adds latency before the inevitable busy. Fail
fast and let the caller decide; the cooperative idle-hold (#64) already smooths the
*daemon's own* open/close churn, which is the churn worth smoothing.

### Cooperative locking is opt-in, time-sliced, with a 5 s idle hold (#64)

By default the daemon holds Anki's exclusive collection lock for its whole life,
which is ideal for the heavy single-collection embedding workflow (no acquire
latency, no contention) but blocks launching Anki desktop against the same file.
`--cooperative-lock` opens on demand and releases after an idle window.

- **Opt-in, default off.** The permanent-hold model stays the default until
  cooperative mode is proven; cooperative is a flag, not a replacement. Whether it
  ever becomes the default is deferred.
- **Cooperative time-slicing, not concurrent sharing.** Anki desktop holds the
  collection for its entire runtime — it does not cooperate. So the win is
  precise: an *idle* daemon stops blocking Anki's launch, not that both operate at
  once. Contention surfaces as a clean SQLite "database is locked" busy error
  (SQLite guarantees no corruption); making that error pretty is separate work.
- **5 s idle hold.** Sized from the parallels: SQLite's conventional
  `busy_timeout` is ~5 s, and unlike a DB connection pool (idle timeouts of
  minutes, because holding a pooled connection harms nothing and reconnecting is
  expensive) our held resource *actively blocks a human launching Anki* and
  re-acquiring is a cheap local SQLite open. Both forces push short; 5 s is the
  default, tunable via `--lock-hold-seconds` / `server.lock_hold_seconds` (a
  human-free programmatic deployment can raise it to cut re-acquire churn).
- **Drift is re-checked per re-acquire, col_mod-only.** After each idle release,
  the next op re-opens and the acquire hook compares `col.mod` to the index's
  stored value; a mismatch (an external edit during the gap) triggers a rebuild,
  reusing the boot machinery (read texts under the lock, embed off-lock). The hook
  deliberately does *not* re-fetch the model fingerprint — a model change is
  already handled by `/embedding/start`, and skipping it avoids a llama-server
  round-trip on every re-acquire.
- **`server.lock` and the collection lock are now distinct.** "Daemon alive" and
  "collection currently held" stopped being the same fact, so `/status` /
  `server status` report both (`locking`, `collection_held`).

Implementation keeps `self.col` typed `Collection` (a `Collection | None` would
ripple through every `self.col.X`); a `_open_flag` tracks held-vs-released and
`_locked` re-opens before any access, so the handle is never dereferenced while
closed. In permanent mode `_open_flag` is always True and every cooperative path
is inert. Built on #79's reopen + read-at-execution-time primitive.

### Reload is the first slice of cooperative locking, built deliberately (#79 → #64)

`shrike collection reload` / `POST /reload` closes and re-opens the
`anki.Collection` and re-checks index drift. We built it now as an explicit
**down-payment on #64** (cooperative locking), not as a fully independent feature,
because its honest value today is narrow: the daemon holds the collection's
exclusive lock for its whole life, so while it's up almost nothing can edit the
file underneath it except a *file-level* replacement (a restored backup, an
`rsync`/sync swap). The "edit in Anki desktop, then reload" story the issue
implies only works once #64 lets the daemon release the lock when idle — at which
point #64's per-acquire drift check makes reload mostly automatic and the explicit
command shrinks to "release now + re-check."

What makes it a down-payment rather than throwaway is the primitive it introduces:
`CollectionWrapper.reopen`/`_do_reopen` (close + re-open on the worker thread) and
— the subtle part — `run`/`run_sync` now read `self.col` **at execution time on
the worker thread** instead of capturing it when called. That means an operation
queued after a reopen runs against the new handle, never a closed one; `self.col`
becomes "the current handle" rather than "a handle fixed at boot." That open/swap
lifecycle on the single worker chokepoint is exactly what #64's open-on-demand,
idle-release design needs. It's a **control endpoint + CLI, not an MCP tool**
(operational, like `/index/rebuild` and `/embedding/*`), and `server.lock`
(daemon liveness) is untouched — only Anki's collection lock is released and
re-acquired.

## Tag-vector namespacing (#178): named spaces in ONE engine, not a separate index file

Tag centroids (#179) live in the **same `MultiModalIndex` engine as the note
items**, under a distinct named space (`tag.text`, file
`index.tag.text.usearch`) — option 1's separation with option 3's mechanics,
because the #201a per-modality split already built the named-vector-space
abstraction this issue asked for. One engine means one `model_id`/`ndim`/
`metric` by construction (a tag centroid is only meaningful in the notes'
space), one persistence path (the orchestrator's save/restore handles every
space's file), and the **no-leakage property is structural**: note searches are
scoped to `NOTE_MODALITIES` (`text`, `image`), so a tag key cannot surface from
a note query — no post-filter, no key-range trick (rejected: `note_id`s are
epoch-ms timestamps with no safe disjoint range).

Keys are `blake2b-8(tag string)` masked positive (tags have no Anki numeric
id); the **key→tag map is in-memory only**, rebuilt with every centroid
recompute. The persisted tag space survives restarts but is searched only
through that map, so a stale file can never mislabel a key — the signal is
simply off until the first recompute, which boot triggers.

**Consistency contract:** a centroid is a pure function of (a) the member
notes' text vectors already in the engine and (b) one membership pass over
`notes.tags` (hierarchy rolled up by `::`-prefix aggregation, so
`science::physics` members also feed `science`). So there is **no separate
watermark and no incremental diff**: the whole set (typically hundreds of
tags) recomputes at the tail of every index-changing kernel op
(reconcile/rebuild/upsert/delete/prune), best-effort — the tag layer is
conditionally-present and never fails the op it rides on. Hygiene before
vectors: a member floor (default 2), a structural-coverage cap (default 50%
of notes), and a meta-tag blocklist (`leech`, `marked`) matched per `::`
segment — `TagCentroidConfig`, the curation surface #179 calls for.

## The engine-plugin architecture: a pure kernel, per-concern engine crates (#342, June 2026)

**The kernel composes engines it is *given* — it never names one.** Since the
migration (P1–P5, PRs #368–#373) the contracts live in `shrike-engine-api` (a
leaf: shrike-ffi + futures + serde), and every concrete engine is its own
crate: `shrike-embed` (ort text + CLIP), `shrike-recognize-apple` (Vision;
Swift glue behind Rust since #398 — Apple's new Vision API is Swift-only),
`shrike-embed-remote` (any OpenAI-compatible endpoint). Hosts —
the Python server, the C-ABI surface, future Swift/Kotlin apps — construct
engines from config and attach them to the kernel's named slots. The
layering check enforces it structurally: `shrike-kernel` may depend on no
engine crate, ever.

**Two conformance routes, chosen by the engine's natural shape.** The kernel
only sees the async traits (`Embedder`/`ImageEmbedder`/`Recognizer`). A
naturally-sync engine (ort inference, a sync HTTP client) implements
chunk-level sync compute traits (`EmbedText`/`EmbedImages`/`RecognizeMedia`)
and is bridged by an adapter; a naturally-async engine implements the async
traits directly. Pipeline *topology* — what must order — stays kernel-owned,
with independent engine futures `try_join`ed (a host-described execution
graph was rejected: it would push the kernel's consistency invariants into a
meta-layer every host re-implements). *(Historical note: the original
host-injected execution machinery — `Inline`/`OnExecutor` over a
`ComputeExecutor` lane — was superseded hours later by the tokio pivot's one
`Blocking` adapter; see the #374 entry below.)*

**Named slots, not a registry — until n>2 capability *kinds*.** Two slots
(embed, recognize) compose cleanly; a keyed modality→engine registry is
recorded as the step to take when a third capability kind (ASR/audio)
actually lands, not before.

**Identity and batch policy are host-assembled, not engine-known.**
Fingerprint strings fold host policy (`pool=`/`args=`/`textprep=`);
`safe_batch` comes from the host-run probe over the loaded model (the
spiked 64-text set, ported to `shrike-engine-api::probe` as shared engine
policy — the Python host's `embed_batching` sources the set from it).
`WithPolicy` carries all three onto a pure-compute engine.

**Engines and managers are different concerns.** Talking to an embeddings
endpoint (`shrike-embed-remote`) is separate from launching one
(`shrike-llama-server`, a manage-class capability per #338 that mobile
builds never include). llama-server is just the on-device instance of the
general case; a cloud/tailnet deployment composes the remote engine with no
manager at all.

**The #340 cross-cdylib answer:** engine crates link into the single
binding cdylib via cargo features — never trait objects across `.so`
boundaries (no stable Rust ABI; one `shrike_native` build carries the
engines its features select). *(The C-ABI surface that briefly gated
engine registration the same way was retired with the tokio pivot,
#374 — see that entry.)*

**Python facades stay, as assembly.** `OnnxBackend`/`ClipBackend`/
`LlamaServerBackend` keep construction-time work (file/provider resolution,
the probe, fingerprint assembly, `health()`) and hand the kernel a native
composition (`NativeEmbedder.from_onnx/from_clip/from_remote`); the
`PyEmbedder`/`PyRecognizer` capture seam remains permanently as the
custom/test-backend escape hatch — no production path rides it.


## The tokio pivot: the kernel owns its runtime; the injected-executor model walked back (#374, June 2026)

**The decision (hours after #342's realization merged):** the
injected-runtime model is walked back — the core installs tokio and is
architected around it; tokio supports every platform this project targets.
The design's center of gravity: **the kernel is perfectly idiomatic async
Rust — no executor traits, no runtime-agnostic gymnastics muddling the
picture** — with **the exchange of actions as THE async boundary** every
host adapts (`async fn(action_request) -> response`; a future C layer would
be completion-callback shaped).

**What it replaced, and why.** The #308-era injected model (`SerialExecutor`
+ a harness worker thread, `TimerHost` over the asyncio loop, a hand-rolled
polling bridge, and #342's `ComputeExecutor`/`OnExecutor`/`AsyncioComputeLane`
engine lanes) bought runtime-portability the platforms never demanded, at
real cost: a custom bridge, eager-submit subtleties, and lock-contention
hazards (the P4b loop-stall bug came from exactly this machinery). tokio was
already in the dependency tree (transitively via anki), so owning it added
nothing and deleted ~1,500 lines of adaptation.

**The shape (PRs #375–#379):**
- A process-global runtime owned by `shrike_kernel::runtime`; only the
  `Handle` escapes. `init_runtime` is the builder seam — the degenerate
  proof installs a `current_thread` runtime and runs the whole kernel on
  one thread (no `block_in_place` anywhere is what keeps that honest).
- **The collection is a task-actor**: one spawned task owns the core and
  runs jobs inline off an mpsc — FIFO by construction, serialization from
  the task's sequential loop rather than thread affinity (a task, not a
  thread — less presumptive of threads, and what makes the single-thread
  degenerate mode work by construction).
- **The action exchange at the edge**: `spawn_op` spawns each public op
  onto the runtime and returns a oneshot-backed Send future pollable from
  any context. Dropping it DETACHES (the task completes; never a JoinHandle
  abort — a half-applied collection write would be corruption). The
  hand-rolled asyncio bridge survives, shrunk to a one-wake completion
  handoff; pyo3-async-runtimes stayed out (it would add a second runtime
  for nothing — inverting #332's rejection made it possible, not useful).
- **Timers** ride `tokio::time` (the debounced saver re-arms by aborting a
  sleeping task, under one lock — the re-arm race was caught in review).
- **Engine execution is one adapter**: `Blocking<E>`, an eager
  `spawn_blocking` (scheduled inside `embed()` itself — eagerness is what
  preserves the search/batch overlap properties and is pinned by test).
  The Python capture seam (`PyEmbedder`/`PyRecognizer`) does
  `spawn_blocking` + GIL-attach; no loop machinery anywhere.
- **Runtime singularity**: anki's internal lazy runtime is never
  instantiated — its only consumers are the sync/AnkiWeb services Shrike
  never dispatches, pinned structurally over the adapter's service-index
  table. One runtime in the process.
- **`shrike-cabi` removed** (speculative surface, no confirmed need) and
  the #338 minimal-core feature discipline relaxed — gating returns when a
  real lean consumer exists. A future C surface adapts the action exchange with
  completion callbacks instead of the calling-thread/block_on model.

**What survived from #342:** the per-concern engine crates, the contract
crate, the layering rule, native end-to-end attach, `WithPolicy` +
host-assembled identity, the batch-safety probe, and every behavioral pin.
