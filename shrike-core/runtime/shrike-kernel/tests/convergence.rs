//! Kernel integration suite: the end-to-end invariants the 150+ in-crate unit
//! tests can't reach at the composition level — that the incremental
//! drift-reconcile and a full rebuild *converge*, that the freshness signal
//! never over-certifies, and that the orchestrator/ingest/fusion seams behave
//! once stitched together behind the public kernel API.
//!
//! A SEPARATE integration binary: the driven-runtime seam is process-global, so
//! this links the public crate and drives every flow through
//! `runtime::testing::run_with_collection` (the clean harness — 1 io + 1
//! collection + N compute threads, one-time per process).
//!
//! No network, no model: a deterministic token-hash embedder stands in for a
//! real one, so a given text always maps to the same vector and search results
//! are reproducible. Synchronization is by `settle().await` only — never a
//! wall-clock sleep — so an oversubscribed host just waits; a genuine hang is
//! caught by Bazel's per-test timeout, the single global hang guard.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::future::BoxFuture;
use shrike_collection::{CreateOutcome, DuplicatePolicy};
use shrike_error::NativeResult;
use shrike_kernel::runtime::testing;
use shrike_kernel::{Embedder, Kernel, NoteSpec};

/// Deterministic embedder: a token-hash bag vector. Similar texts share tokens
/// → close vectors; no model, no network, no Python. A local copy of the
/// in-crate `HashEmbedder` (the `#[cfg(test)]` one is not visible to an
/// integration binary).
struct HashEmbedder;

impl HashEmbedder {
    fn embed_sync(texts: &[String]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; 64];
                for token in t.to_lowercase().split_whitespace() {
                    let mut h: u64 = 1469598103934665603;
                    for b in token.bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(1099511628211);
                    }
                    v[(h % 64) as usize] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / norm).collect()
            })
            .collect()
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move { Ok(Self::embed_sync(&texts)) })
    }

    fn fingerprint(&self) -> Option<String> {
        Some("hash-embedder:v1".to_string())
    }

    fn dim(&self) -> Option<usize> {
        Some(64)
    }
}

/// A fresh per-process-unique scratch dir under Bazel's `$TEST_TMPDIR` (falling
/// back to `$TMPDIR` for a bare `cargo test`). The pid+counter keys it so a
/// recycled pid can't reopen a prior run's lingering collection.
fn temp_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("TEST_TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!(
        "shrike-convergence-{}-{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Open a kernel on a fresh collection at `dir`, with the deterministic embedder
/// attached and the boot reconcile driven (an empty collection materializes an
/// empty-but-ready index).
async fn open_kernel(dir: &std::path::Path) -> Kernel {
    let kernel = Kernel::open(
        dir.join("c.anki2").to_str().unwrap(),
        dir.join("cache").to_str().unwrap(),
    )
    .await
    .unwrap();
    kernel.attach_embedder(Arc::new(HashEmbedder), None);
    kernel.reindex_if_needed().await.unwrap();
    kernel
}

/// Resolve the `Basic` notetype id once per kernel.
async fn basic(kernel: &Kernel) -> i64 {
    kernel.notetype_id("Basic").await.unwrap()
}

/// Upsert one note and return its id, asserting it was created (not a duplicate
/// rejection or error).
async fn create(kernel: &Kernel, nt: i64, front: &str, back: &str) -> i64 {
    let CreateOutcome::Created(id) = kernel
        .upsert_note(
            nt,
            1,
            vec![front.into(), back.into()],
            vec![],
            DuplicatePolicy::Error,
        )
        .await
        .unwrap()
    else {
        panic!("create of {front:?} was rejected");
    };
    id
}

/// The bank of queries the observable-state snapshot is taken over. Each touches
/// distinct tokens so the per-query hit set actually discriminates between
/// collection states (a vacuous bank would make any two states "equal").
const QUERY_BANK: &[&str] = &[
    "mitochondria",
    "krebs",
    "photosynthesis",
    "chlorophyll",
    "ribosome",
    "osmosis",
    "glycolysis",
    "helicase",
    "zzqqnonexistent",
];

/// The OBSERVABLE state of a kernel: everything a read-side caller can see that
/// must CONVERGE between the incremental and rebuild paths, so two kernels with
/// this equal are indistinguishable through the public surface. Captured AFTER
/// `settle()`, so every prior write's effects are visible.
///
/// Deliberately excludes the FUZZY lexical signal: the incremental path leaves
/// the `trigram_df` snapshot the fuzzy prune *ranks* on debounced, where a full
/// rebuild re-materializes it inline. That snapshot lag is a documented RANKING
/// drift, not a recall/content one (see `derived_refresh.rs`) — so fuzzy hit
/// sets legitimately differ between the two paths and equating them would be
/// asserting a non-invariant. The EXACT-substring signal is a pure FTS5 `MATCH`
/// with no DF dependency, so it tracks the derived store's actual row CONTENT,
/// which the replace/delete-fan-out semantics make converge exactly.
#[derive(Debug, PartialEq)]
struct Observable {
    /// Per-note STORED TEXT VECTOR (`note_id → vector`) — the index's actual
    /// content, not a search ranking. The deterministic embedder maps a note's
    /// text to a fixed vector, so two paths that indexed the same final text
    /// land bit-identical vectors; this is the strongest, RRF-independent form
    /// of "the vector index converged" (a search-hit-set observable degenerates
    /// here — with <top_k notes every query returns all of them).
    text_vectors: std::collections::BTreeMap<i64, Vec<Vec<f32>>>,
    /// Per-query EXACT-substring hit sets — the notes a fused search matched via
    /// the DF-independent `exact` lexical signal. This is the derived store's
    /// row CONTENT as the public surface exposes it (fuzzy excluded, see above).
    lexical_exact: Vec<BTreeSet<i64>>,
    /// The note-id set the vector index holds (the text modality's keys).
    index_keys: BTreeSet<i64>,
    /// The text-modality vector count (the orchestrator's reported size).
    index_size: usize,
}

/// Snapshot the observable state. `settle()` first so the index and derived
/// store reflect every committed write before any read (the deterministic
/// barrier the data plane awaits instead of polling).
async fn observe(kernel: &Kernel) -> Observable {
    kernel.settle().await;

    let mut lexical_exact = Vec::with_capacity(QUERY_BANK.len());
    for q in QUERY_BANK {
        let hits = kernel.search(q, 50).await.unwrap();
        // The EXACT lexical signal only: a note appears here iff its stored
        // derived rows literally contain the query (DF-independent), so this
        // reflects derived CONTENT, not the fuzzy DF-ranked candidate set.
        lexical_exact.push(
            hits.iter()
                .filter(|h| h.signals.iter().any(|(s, _)| s == "exact"))
                .map(|h| h.note_id)
                .collect(),
        );
    }

    let engine = kernel.index().engine_arc();
    let index_keys: BTreeSet<i64> = engine.keys().into_iter().collect();
    let index_size = engine.size();
    let text_vectors = index_keys
        .iter()
        .filter_map(|&id| engine.modality_get("text", id).map(|v| (id, v)))
        .collect();

    Observable {
        text_vectors,
        lexical_exact,
        index_keys,
        index_size,
    }
}

/// Force a FULL rebuild of BOTH derived caches from the current collection
/// contents (never the incremental path), then settle. This is the reference
/// state the incremental path must converge to.
async fn full_rebuild(kernel: &Kernel) {
    kernel.rebuild_index().await.unwrap();
    kernel.rebuild_derived().await.unwrap();
    kernel.settle().await;
}

/// The MARQUEE invariant: an incremental drift-reconcile lands the vector index
/// AND the derived (FTS5) store in the SAME observable state as a full rebuild
/// from the same final collection contents.
///
/// The convergence is over NON-TRIVIAL drift — adds, removes, AND edits all
/// applied incrementally — so the equality proves the incremental maintenance
/// path actually tracks the collection, not a vacuous "two empty stores match".
#[test]
fn incremental_reconcile_converges_to_a_full_rebuild() {
    testing::run_with_collection(async {
        let dir = temp_dir();
        let kernel = open_kernel(&dir).await;
        let nt = basic(&kernel).await;

        // ── Initial population: a varied set so search discriminates. ──
        let seed = create(&kernel, nt, "mitochondria powerhouse of the cell", "energy").await;
        let krebs = create(&kernel, nt, "krebs cycle citric acid", "biology").await;
        let photo = create(
            &kernel,
            nt,
            "photosynthesis chlorophyll absorbs light",
            "plant",
        )
        .await;
        let ribo = create(
            &kernel,
            nt,
            "ribosome assembles protein chains",
            "synthesis",
        )
        .await;
        let doomed = create(
            &kernel,
            nt,
            "glycolysis splits glucose to pyruvate",
            "energy",
        )
        .await;
        let stale = create(&kernel, nt, "placeholder text to be edited later", "old").await;
        kernel.settle().await;

        // ── Drift, applied INCREMENTALLY (each mutation drains through the
        // maintained per-op tail / ingest queue) ──
        //   ADD two new notes
        let added_osmosis = create(&kernel, nt, "membrane transport osmosis water", "cell").await;
        let added_dna = create(
            &kernel,
            nt,
            "dna replication helicase polymerase",
            "genetics",
        )
        .await;
        kernel.settle().await;

        //   REMOVE two existing notes (one seeded, one added-then-removed churn)
        let churn = create(&kernel, nt, "temporary note destined for removal", "tmp").await;
        kernel.settle().await;
        let del = kernel.delete_notes(vec![doomed, churn]).await.unwrap();
        assert_eq!(del.deleted.len(), 2, "both target notes deleted");
        kernel.settle().await;

        //   EDIT two notes' fields (re-embeds the changed text, re-derives rows)
        kernel
            .collection()
            .run(move |core| {
                core.update_note(
                    stale,
                    &[
                        "membrane transport osmosis revisited".into(),
                        "edited".into(),
                    ],
                    None,
                )
            })
            .await
            .unwrap()
            .unwrap();
        kernel
            .collection()
            .run(move |core| {
                core.update_note(
                    krebs,
                    &[
                        "krebs cycle now mentions photosynthesis too".into(),
                        "edited".into(),
                    ],
                    None,
                )
            })
            .await
            .unwrap()
            .unwrap();
        // An external-style field edit doesn't enqueue maintenance on its own —
        // request the maintained tail for the edited ids, then settle the drain.
        kernel.reindex_notes(&[stale, krebs]).await.unwrap();
        kernel.settle().await;

        // Sanity: the surviving notes are what we expect (the equality below is
        // only meaningful over a non-empty, churned collection).
        let survivors: BTreeSet<i64> = [seed, krebs, photo, ribo, stale, added_osmosis, added_dna]
            .into_iter()
            .collect();

        // ── State A: the observable state reached via the INCREMENTAL path. ──
        let incremental = observe(&kernel).await;
        assert_eq!(
            incremental.index_keys, survivors,
            "the incrementally-maintained index holds exactly the live notes"
        );
        assert_eq!(
            incremental.index_size,
            survivors.len(),
            "no stale vectors left from the removed/edited notes"
        );

        // Non-vacuity: the observable must actually DISCRIMINATE, or the
        // convergence equality below would be trivially true. Pin the drift's
        // effects on the exact-substring observable (the per-query order matches
        // QUERY_BANK):
        let q = |needle: &str| QUERY_BANK.iter().position(|x| *x == needle).unwrap();
        let lex = |needle: &str| &incremental.lexical_exact[q(needle)];
        // The EDIT landed: `krebs`'s new text mentions photosynthesis, so the
        // "photosynthesis" query now finds BOTH krebs and the original photo note.
        assert!(
            lex("photosynthesis").contains(&krebs) && lex("photosynthesis").contains(&photo),
            "the edit is visible: the edited krebs note joined the photosynthesis hits"
        );
        // The other EDIT landed: `stale`'s text now mentions osmosis.
        assert!(
            lex("osmosis").contains(&stale),
            "the edit is visible: the re-texted note is found by its new token"
        );
        // The ADDs landed: the new osmosis/dna notes are findable by their tokens.
        assert!(
            lex("osmosis").contains(&added_osmosis) && lex("helicase").contains(&added_dna),
            "the added notes are lexically findable"
        );
        // The REMOVE landed: the deleted glycolysis note left no exact-match row,
        // so no SURVIVOR is matched by its now-unique token (and the deleted id is
        // gone from the index entirely).
        assert!(
            !incremental.index_keys.contains(&doomed),
            "the deleted glycolysis note is gone from the index"
        );
        assert!(
            lex("glycolysis").iter().all(|id| survivors.contains(id)),
            "no dangling hit points at a removed note"
        );
        // The index actually holds a vector per surviving note (the vector-level
        // convergence below is over real content, not two empty maps).
        assert_eq!(
            incremental
                .text_vectors
                .keys()
                .copied()
                .collect::<BTreeSet<i64>>(),
            survivors,
            "every surviving note has a stored text vector"
        );

        // ── State B: force a FULL rebuild of both caches from the SAME
        // collection contents, then re-observe. ──
        full_rebuild(&kernel).await;
        let rebuilt = observe(&kernel).await;

        // ── The convergence claim: indistinguishable through every observable —
        // the same per-note vectors, the same exact-lexical content, the same
        // index keys/size. ──
        assert_eq!(
            incremental, rebuilt,
            "the incremental reconcile and the full rebuild converge to the same \
             observable state (per-note vectors + exact-lexical content + index \
             keys/size) over non-trivial drift"
        );

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// The BOOT-time reconcile path: a kernel opened against a collection whose
/// sidecars are STALE/ABSENT heals on open (the `reindex_if_needed` drift route
/// + a derived rebuild), reaching the same observable state as a full rebuild.
///
/// Drift is manufactured by mutating the collection through a kernel that does
/// NOT run the maintained tail for those writes, then dropping it; a fresh
/// kernel opens onto the drifted collection and must reconcile it.
#[test]
fn boot_reconcile_heals_a_drifted_collection_to_a_full_rebuild() {
    testing::run_with_collection(async {
        let dir = temp_dir();

        // First kernel: seed, settle, then apply UNMAINTAINED drift (collection
        // writes whose index/derived tail is never driven) so the sidecars lag
        // the collection — exactly the on-disk state a crash-between-write-and-
        // ingest would leave.
        let first = open_kernel(&dir).await;
        let nt = basic(&first).await;
        let keep = create(&first, nt, "mitochondria powerhouse of the cell", "energy").await;
        let edit = create(&first, nt, "krebs cycle citric acid", "biology").await;
        let drop_target = create(&first, nt, "glycolysis splits glucose", "energy").await;
        first.settle().await;

        // Unmaintained drift: write directly through the collection actor, never
        // requesting the maintained tail, so the index/derived fall behind.
        first
            .collection()
            .run(move |core| {
                // Edit an existing note's text (vector + derived row now stale).
                core.update_note(
                    edit,
                    &[
                        "photosynthesis chlorophyll light reaction".into(),
                        "edited".into(),
                    ],
                    None,
                )?;
                // Delete a note (its vector + derived row now orphaned).
                core.delete_notes(&[drop_target])?;
                // Add a brand-new note the sidecars have never seen.
                core.create_note(
                    nt,
                    1,
                    &["membrane transport osmosis".into(), "cell".into()],
                    &[],
                    DuplicatePolicy::Error,
                )?;
                Ok::<(), shrike_error::NativeError>(())
            })
            .await
            .unwrap()
            .unwrap();
        // Close WITHOUT settling the (never-enqueued) maintenance: the sidecars
        // are now genuinely behind the collection's col_mod.
        first.close().await.unwrap();
        drop(first);

        // Second kernel: opens onto the drifted collection. The boot reconcile
        // (`reindex_if_needed`) heals the vector index; rebuild the derived store
        // to bring the FTS5 sidecar in line (the harness owns that store, exactly
        // as `reload`/`import_package` do). Capture the healed observable state.
        let healed_kernel = Kernel::open(
            dir.join("c.anki2").to_str().unwrap(),
            dir.join("cache").to_str().unwrap(),
        )
        .await
        .unwrap();
        healed_kernel.attach_embedder(Arc::new(HashEmbedder), None);
        // The drift signal is the col_mod bump; reconcile must run.
        let reconciled = healed_kernel.reindex_if_needed().await.unwrap();
        assert!(
            reconciled,
            "the boot reconcile detected drift and ran (col_mod moved while the index lagged)"
        );
        healed_kernel.rebuild_derived().await.unwrap();
        let healed = observe(&healed_kernel).await;

        // The collection now holds: keep, the edited note, the new osmosis note;
        // the deleted note is gone. The healed index must reflect exactly that —
        // no orphan vector for the deleted note, the edited note re-embedded.
        assert!(
            healed.index_keys.contains(&keep) && healed.index_keys.contains(&edit),
            "the kept and edited notes are present after reconcile"
        );
        assert!(
            !healed.index_keys.contains(&drop_target),
            "the deleted note's orphan vector was dropped by the boot reconcile"
        );

        // Reference: a full rebuild from the SAME (drifted) collection contents.
        full_rebuild(&healed_kernel).await;
        let rebuilt = observe(&healed_kernel).await;

        assert_eq!(
            healed, rebuilt,
            "the boot-time reconcile heals the drifted sidecars to the same observable \
             state as a full rebuild"
        );

        healed_kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// An embedder that parks every embed until the test OPENS the gate, so the test
/// OWNS when the ingest queue drains — the only deterministic way to hold the
/// not-yet-settled window open (the `HashEmbedder` drains too fast to observe).
/// The gate is STICKY: once opened it stays open, so every embed after the open
/// (the ingest re-embed AND the later query embed in `search`) proceeds — a
/// one-shot `notify_one` would deadlock the second embedder caller.
struct GatedEmbedder {
    open: Arc<std::sync::atomic::AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
}

impl GatedEmbedder {
    fn new() -> Self {
        Self {
            open: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            wake: Arc::new(tokio::sync::Notify::new()),
        }
    }
    /// Open the gate and wake every parked embed; subsequent embeds pass straight
    /// through.
    fn release(&self) {
        self.open.store(true, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_waiters();
    }
}

impl Embedder for GatedEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let open = Arc::clone(&self.open);
        let wake = Arc::clone(&self.wake);
        Box::pin(async move {
            // Register for the wake BEFORE the open re-check, so an open() racing
            // between the check and the await can't be missed (the standard
            // tokio::Notify "register then check" idiom).
            while !open.load(std::sync::atomic::Ordering::SeqCst) {
                let notified = wake.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if open.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                notified.await;
            }
            Ok(HashEmbedder::embed_sync(&texts))
        })
    }
    fn fingerprint(&self) -> Option<String> {
        Some("gated-embedder:v1".to_string())
    }
    fn dim(&self) -> Option<usize> {
        Some(64)
    }
}

/// Watermark honesty, end-to-end: the index must never CERTIFY a just-committed
/// write as visible before it has actually caught up. The pure `SpaceFloor`
/// ordering is pinned in `watermark.rs`; here the *observable consequence* — a
/// committed-but-still-draining note is reported as not-settled AND is absent
/// from the vector index — is held open DETERMINISTICALLY with a gated embedder
/// (so the not-settled window can't race shut). Releasing the gate then proves
/// completeness: once settled, the note's vector is present.
///
/// A regression that over-certified (reported settled, or indexed the note,
/// while its embed was still in flight) would let a search trust a result it
/// can't yet serve — the stale-certified hit this guards against.
#[test]
fn freshness_signal_never_over_certifies_a_pending_write() {
    testing::run_with_collection(async {
        let dir = temp_dir();
        let kernel = Kernel::open(
            dir.join("c.anki2").to_str().unwrap(),
            dir.join("cache").to_str().unwrap(),
        )
        .await
        .unwrap();
        let gate = Arc::new(GatedEmbedder::new());
        kernel.attach_embedder(Arc::clone(&gate) as Arc<dyn Embedder>, None);
        kernel.reindex_if_needed().await.unwrap();
        let nt = basic(&kernel).await;

        // Quiescent baseline: with nothing in flight, the kernel is settled.
        kernel.settle().await;
        assert!(
            kernel.is_settled(),
            "a quiescent kernel reports settled (the honest baseline)"
        );

        // Commit a write. The collection write commits immediately and enqueues
        // the embed maintenance — which PARKS in the gated embedder, so the queue
        // genuinely cannot drain until this test releases it.
        let id = create(&kernel, nt, "unique sentinel quokka platypus", "rare").await;
        // Hand the drain task a chance to dequeue the item and reach the parked
        // embed; the outstanding count is still 1 (the embed has not returned).
        tokio::task::yield_now().await;

        // The honesty invariant: the freshness probe must NOT claim settled while
        // the embed is parked, and the note's vector must NOT yet be in the index
        // (the index has provably not caught up — over-certifying it would be the
        // stale-certified hit). Both are deterministic here: the gate holds the
        // embed, so nothing can advance.
        assert!(
            !kernel.is_settled(),
            "not settled while a committed write's embed is still in flight"
        );
        assert!(
            !kernel.index().engine().contains(id),
            "the index has NOT certified the pending note (its vector hasn't minted yet)"
        );

        // Open the gate; the parked ingest embed (and the later query embed)
        // proceed. Once the drain completes the kernel is settled and complete.
        gate.release();
        kernel.settle().await;
        assert!(
            kernel.is_settled(),
            "settled once the drain completes — the advisory clears"
        );
        assert!(
            kernel.index().engine().contains(id),
            "completeness: the note's vector is present once the drain settles"
        );
        let hits = kernel.search("quokka platypus", 10).await.unwrap();
        assert!(
            hits.iter().any(|h| h.note_id == id),
            "the committed write is complete and visible once settled (no lost note)"
        );

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// Upsert→search consistency end-to-end: every note committed in a batch is
/// findable by BOTH the semantic and the lexical signal once settled — the
/// orchestrator add and the per-note derived ingest both landed. This is the
/// composition the unit tests exercise in pieces; here it runs through the
/// public batch upsert + both public search paths at once.
#[test]
fn batch_upsert_is_findable_via_both_signals_after_settle() {
    testing::run_with_collection(async {
        let dir = temp_dir();
        let kernel = open_kernel(&dir).await;
        let nt = basic(&kernel).await;

        let specs = vec![
            NoteSpec {
                notetype_id: nt,
                deck_id: 1,
                fields: vec!["alpha mitochondria powerhouse".into(), "a".into()],
                tags: vec![],
            },
            NoteSpec {
                notetype_id: nt,
                deck_id: 1,
                fields: vec!["beta krebs cycle biology".into(), "b".into()],
                tags: vec![],
            },
            NoteSpec {
                notetype_id: nt,
                deck_id: 1,
                fields: vec!["gamma ribosome protein synthesis".into(), "c".into()],
                tags: vec![],
            },
        ];
        let outcomes = kernel
            .upsert_notes(specs, DuplicatePolicy::Error)
            .await
            .unwrap();
        let ids: Vec<i64> = outcomes
            .into_iter()
            .map(|o| match o.unwrap() {
                CreateOutcome::Created(id) => id,
                other => panic!("expected Created, got {other:?}"),
            })
            .collect();
        assert_eq!(ids.len(), 3, "three notes created in one batch");
        kernel.settle().await;

        // Each note is found by its distinctive token via BOTH the fused
        // (semantic) and the lexical-only path.
        for (id, token) in [
            (ids[0], "mitochondria"),
            (ids[1], "krebs"),
            (ids[2], "ribosome"),
        ] {
            let sem = kernel.search(token, 10).await.unwrap();
            assert!(
                sem.iter().any(|h| h.note_id == id),
                "note {id} found via the fused search for {token:?}"
            );
            let (groups, _stale) = kernel
                .search_lexical_single(token.to_string(), 10, None, vec![], vec![])
                .await
                .unwrap();
            let lex: BTreeSet<i64> = groups
                .into_iter()
                .flat_map(|g| g.matches.into_iter().map(|m| m.note.id))
                .collect();
            assert!(
                lex.contains(&id),
                "note {id} found via the lexical search for {token:?}"
            );
        }

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// Remove drops a note from BOTH sidecars in one maintained op, end-to-end: a
/// deleted note vanishes from the vector index AND the derived (lexical) store,
/// and leaves NO drift behind (the watermark advanced in-op, so the next
/// reconcile check is a no-op). This pins the prune fan-out through the public
/// delete path, not the orchestrator internals.
#[test]
fn delete_drops_from_index_and_derived_with_no_residual_drift() {
    testing::run_with_collection(async {
        let dir = temp_dir();
        let kernel = open_kernel(&dir).await;
        let nt = basic(&kernel).await;

        let keep = create(&kernel, nt, "photosynthesis chlorophyll light", "plant").await;
        let gone = create(&kernel, nt, "glycolysis pyruvate energy release", "energy").await;
        kernel.settle().await;

        // Both present in both sidecars before the delete.
        let engine = kernel.index().engine_arc();
        assert!(
            engine.contains(keep) && engine.contains(gone),
            "both indexed"
        );
        let (pre, _) = kernel
            .search_lexical_single("glycolysis".to_string(), 10, None, vec![], vec![])
            .await
            .unwrap();
        assert!(
            pre.into_iter()
                .any(|g| g.matches.iter().any(|m| m.note.id == gone)),
            "the doomed note is lexically findable before deletion"
        );

        // Delete → the maintained op drops vectors + derived rows; settle the
        // off-drain sidecar removal.
        let resp = kernel.delete_notes(vec![gone]).await.unwrap();
        assert_eq!(resp.deleted, vec![gone]);
        kernel.settle().await;

        // Gone from the vector index.
        assert!(
            !engine.contains(gone),
            "the deleted note's vector left the index"
        );
        assert!(engine.contains(keep), "the surviving note's vector stayed");

        // Gone from the derived (lexical) store.
        let (post, _) = kernel
            .search_lexical_single("glycolysis".to_string(), 10, None, vec![], vec![])
            .await
            .unwrap();
        assert!(
            post.into_iter()
                .all(|g| g.matches.iter().all(|m| m.note.id != gone)),
            "the deleted note's derived row left the lexical store too"
        );

        // No residual drift: the in-op watermark advance means the next reconcile
        // check finds nothing to do (an over-removal or a missed watermark would
        // resurface here as drift).
        assert!(
            !kernel.reindex_if_needed().await.unwrap(),
            "the delete advanced the watermark in-op — no residual drift to reconcile"
        );

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}

/// Duplicate-policy behaviour end-to-end across the three policies, through the
/// public upsert path. Anki's collection-wide first-field rule makes a
/// second note with the same first field a duplicate:
///
/// - `Error` rejects it (the original stays the only copy),
/// - `Skip` leaves it unwritten (no second copy),
/// - `Allow` writes a second note (two copies coexist).
///
/// Pinned through the kernel's batch upsert so the per-item policy outcome is
/// the composed behaviour, not a collection-layer unit.
#[test]
fn duplicate_policy_governs_a_repeated_first_field() {
    testing::run_with_collection(async {
        let dir = temp_dir();
        let kernel = open_kernel(&dir).await;
        let nt = basic(&kernel).await;

        // The original.
        let original = create(&kernel, nt, "paris capital of france", "geo").await;
        kernel.settle().await;

        // Error: a repeat of the first field is rejected. `upsert_note` is sugar
        // over a batch of one, so the single item's rejection surfaces as the
        // call's Err — nothing new is written.
        let err_out = kernel
            .upsert_note(
                nt,
                1,
                vec!["paris capital of france".into(), "again".into()],
                vec![],
                DuplicatePolicy::Error,
            )
            .await;
        assert!(
            err_out.is_err(),
            "Error policy rejects the duplicate first field"
        );

        // Skip: also leaves no second copy (the outcome distinguishes Skipped
        // from Created).
        let skip_out = kernel
            .upsert_note(
                nt,
                1,
                vec!["paris capital of france".into(), "skip".into()],
                vec![],
                DuplicatePolicy::Skip,
            )
            .await
            .unwrap();
        assert!(
            !matches!(skip_out, CreateOutcome::Created(_)),
            "Skip policy does not create a second copy (got {skip_out:?})"
        );

        // Allow: writes a real second note with the same first field.
        let CreateOutcome::Created(allowed) = kernel
            .upsert_note(
                nt,
                1,
                vec!["paris capital of france".into(), "allowed".into()],
                vec![],
                DuplicatePolicy::Allow,
            )
            .await
            .unwrap()
        else {
            panic!("Allow policy must create a second note");
        };
        assert_ne!(allowed, original, "Allow created a distinct second note");
        kernel.settle().await;

        // Observable consequence: exactly two notes share that first field now
        // (the original + the Allow copy) — Error/Skip added none.
        let dupes = kernel
            .collection()
            .run(|core| core.find_notes("paris"))
            .await
            .unwrap()
            .unwrap();
        let dupe_set: BTreeSet<i64> = dupes.into_iter().collect();
        assert_eq!(
            dupe_set,
            BTreeSet::from([original, allowed]),
            "only the original and the Allow copy exist — Error and Skip wrote nothing"
        );

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}
