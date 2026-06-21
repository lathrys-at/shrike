//! A debounced re-materialize of the derived store's per-write-stable snapshots
//! (today: the trigram document-frequency table the fuzzy prune ranks on). A full
//! (re)build refreshes them inline so "ready means ready"; this keeps them fresh
//! between rebuilds on the incremental write path, where `ingest_many`/`remove`
//! change the index but not the snapshot.
//!
//! The workload is infrequent writes in discrete batches then long read-heavy
//! quiet, so the job uses a RE-ARMING debounce (not the immediate coalesce-loop the
//! tag refresh uses): a write re-arms the timer, so a whole batch of writes
//! collapses into ONE refresh once it settles — never one per call, never a refresh
//! mid-batch (the recompute walks every trigram's doclist, so firing it before the
//! batch is done is wasted work). A burst cap bounds the staleness when a write
//! stream never quiesces (the re-arming delay would otherwise starve the refresh).
//!
//! The snapshot lag is a RANKING drift, not a recall loss: the prune scans every
//! absent (DF-0) trigram, so a trigram written since the snapshot is still scanned
//! and a match through it is still found (see `DerivedEngine::prune_to_rare_terms`).
//! Keeping the snapshot fresh keeps the prune's rarest-trigram SELECTION accurate —
//! a quality concern, not a correctness one. See `refresh_trigram_df`.

use std::sync::Arc;
use std::time::Duration;

use shrike_store::DerivedStore;

use crate::maintenance::Maintenance;

/// The re-arming debounce: a refresh fires this long after the LAST write of a
/// batch settles.
const REFRESH_DELAY: Duration = Duration::from_secs(2);
/// Pace between coalesced re-runs if a write lands while a refresh is executing.
const REFRESH_WINDOW: Duration = Duration::from_secs(2);
/// Burst cap: refresh immediately once this many writes accumulate without one, so a
/// never-quiescing sub-`REFRESH_DELAY` write stream can't starve the re-arming
/// debounce indefinitely. High enough that a normal batch quiesces (the delay fires)
/// well before reaching it; a mid-stream refresh past the cap is merely a slightly-
/// early, harmless re-materialize.
const REFRESH_BURST_THRESHOLD: u64 = 4096;

/// A coalescing, debounced refresher for the derived store's cached snapshots.
pub struct DerivedSnapshotRefresher {
    job: Arc<Maintenance>,
}

impl DerivedSnapshotRefresher {
    /// Build a refresher over `derived`. The refresh runs on the compute pool (a
    /// blocking SQLite rewrite, O(distinct trigrams)) and is best-effort: a failure
    /// leaves a stale snapshot — which degrades prune quality (a ranking drift, see
    /// the module doc), but never fails the op it rides on — so it is logged, not
    /// surfaced.
    pub fn new(derived: Arc<dyn DerivedStore>) -> Arc<Self> {
        let job = Maintenance::new(
            Box::new(move || {
                let derived = Arc::clone(&derived);
                Box::pin(async move {
                    let outcome = crate::runtime::dispatch_compute(move || {
                        derived.refresh_derived_snapshots()
                    })
                    .await;
                    if let Err(e) = outcome {
                        tracing::warn!(error = %e, "derived snapshot refresh failed");
                    }
                })
            }),
            REFRESH_DELAY,
            REFRESH_WINDOW,
            REFRESH_BURST_THRESHOLD,
        );
        Arc::new(Self { job })
    }

    /// Note that the derived index changed (an `ingest_many`/`remove` landed); the
    /// re-arming debounce refreshes once the batch settles. Never blocks or errors.
    pub fn request(&self) {
        self.job.request();
    }

    /// Disarm the debounce on kernel close — orderly shutdown of the maintenance
    /// coordinators, so no new refresh is scheduled as the kernel tears down. (A
    /// refresh already mid-flight on the compute pool finishes harmlessly: it holds
    /// its own `Arc<dyn DerivedStore>`, which `collection.close()` does not close.)
    pub fn shutdown(&self) {
        self.job.shutdown();
    }

    /// Whether the refresh job is fully quiescent (nothing armed, in flight, or
    /// pending). A test awaits this to know no late refresh can still fire.
    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.job.is_idle()
    }
}
