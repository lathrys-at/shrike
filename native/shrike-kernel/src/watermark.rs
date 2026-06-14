//! The watermark over-certification guard (#585/#590).
//!
//! The index and derived watermarks (`col_mod`) are the *only* drift signal:
//! `check_drift`/`rebuild_derived` reconcile iff `stored_watermark !=
//! live_col_mod`. So a watermark may be advanced to a `col.mod` value `V` ONLY
//! when every write whose `col.mod` is `≤ V` has actually been indexed/ingested
//! by this point. Violating that certifies an un-indexed note as current → the
//! heal gates go quiet → permanent silent search loss.
//!
//! Two facts make naive advancement unsafe:
//!
//! 1. **The actor serializes JOBS, not multi-job op TRANSACTIONS.** A maintained
//!    op is `write-job` → (await embed/index/ingest off-actor) → `advance`. A
//!    concurrent op B's write-job can interleave between op A's write-job and op
//!    A's advance, so by the time op A advances, `live col.mod` already reflects
//!    B's not-yet-indexed write. Reading the *live* `col.mod` at advance time
//!    (the pre-#585 bug) stamps B's write as certified.
//! 2. **A tail can fail after the collection write committed** (embed backend
//!    down, transient index/ingest error — #590). The note is in the collection
//!    but absent from the index/FTS5; the watermark must be left behind so boot
//!    drift heals it.
//!
//! The fix: capture `col.mod` in the SAME actor job as the write, register the
//! write as *in flight* there, then on the tail's completion advance the
//! watermark to the captured value ONLY IF
//!
//! - this op's tail SUCCEEDED (a failed/partial tail leaves the watermark
//!   behind — #590), AND
//! - no other still-in-flight write has a captured `col.mod ≤ V` (an earlier or
//!   concurrent write that hasn't been indexed yet must not be certified by us),
//!   AND
//! - no earlier FAILED write is still un-healed (the poison floor — a failed
//!   tail at `col.mod F` keeps the watermark strictly below `F` so drift stays
//!   armed until a full reconcile re-indexes that note).
//!
//! When the gate blocks an advance, the op leaves the watermark behind; the
//! lagging op's own completion — or, failing that, the next boot/reload drift
//! check — heals it. `reconcile == rebuild` is idempotent, so a later reconcile
//! that re-touches an already-indexed note is a harmless no-op. A successful
//! whole-collection reconcile/rebuild re-indexes *everything* (so every prior
//! failure is healed) and stamps its own snapshot `col.mod` directly — that is
//! where the poison floor clears ([`SpaceTracker::clear_poison`]).
//!
//! The index and derived watermarks are tracked SEPARATELY because their tails
//! fail independently (an embed failure must not block the derived watermark,
//! and vice-versa).

use std::collections::BTreeMap;
use std::sync::Mutex;

/// A monotonic token identifying one in-flight write within one watermark
/// space. Unique per registration; completion is explicit (`complete`) so
/// success/failure can be distinguished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WriteToken(u64);

/// Tracks the in-flight + un-healed-failed writes for ONE watermark space
/// (index OR derived) and decides, on each completion, whether the watermark
/// may advance to the completing write's captured `col.mod`.
#[derive(Debug, Default)]
pub struct SpaceTracker {
    next: u64,
    /// token → the `col.mod` captured with that write's job.
    in_flight: BTreeMap<u64, i64>,
    /// The lowest `col.mod` of a write whose tail FAILED and has not yet been
    /// healed by a full reconcile/rebuild. The watermark must never advance to
    /// or past this, so drift (`watermark != live_col_mod`) stays armed until
    /// the heal path clears it.
    poison_floor: Option<i64>,
}

impl SpaceTracker {
    /// Register a write that captured `col_mod` in its actor job. Call this in
    /// the SAME job as the write (or, for the no-own-write tails — recognition
    /// sweep, reindex_notes, metadata — at the point the captured `col.mod` is
    /// read), so the registration cannot miss a concurrent write that lands
    /// after us.
    fn register(&mut self, col_mod: i64) -> WriteToken {
        let token = self.next;
        self.next += 1;
        self.in_flight.insert(token, col_mod);
        WriteToken(token)
    }

    /// Complete the write for `token`. Removes it from the in-flight set and
    /// returns the `col.mod` the watermark may now advance to:
    ///
    /// - `None` when `success` is false (a failed/partial tail must leave the
    ///   watermark behind, and records a poison floor), OR when an
    ///   earlier/concurrent in-flight write — or the poison floor — has a
    ///   `col.mod ≤ V` (advancing past it would over-certify an un-indexed
    ///   note).
    /// - `Some(V)` when the captured `col.mod` is safe to certify now.
    fn complete(&mut self, token: WriteToken, success: bool) -> Option<i64> {
        let captured = self.in_flight.remove(&token.0)?;
        if !success {
            // The note is in the collection but not in this space — keep the
            // watermark strictly below it so drift re-fires until a full
            // reconcile/rebuild heals it.
            self.poison_floor = Some(match self.poison_floor {
                Some(f) => f.min(captured),
                None => captured,
            });
            return None;
        }
        // A still-in-flight earlier/concurrent write (≤ ours) is un-indexed —
        // we must not certify a watermark that would also cover it.
        if self.in_flight.values().any(|other| *other <= captured) {
            return None;
        }
        // An un-healed earlier failure at/below ours likewise forbids advancing
        // to or past it.
        if matches!(self.poison_floor, Some(f) if f <= captured) {
            return None;
        }
        Some(captured)
    }

    /// Clear the poison floor — called when a full reconcile/rebuild has
    /// re-indexed the whole collection (so every prior failed write is healed)
    /// and stamped its own snapshot `col.mod` directly.
    fn clear_poison(&mut self) {
        self.poison_floor = None;
    }
}

/// The kernel-held tracker: one [`SpaceTracker`] for the index watermark and
/// one for the derived watermark. A single maintained op registers in both at
/// write time and completes each as its respective tail finishes (which may be
/// success for one and failure for the other).
#[derive(Debug, Default)]
pub struct WatermarkTracker {
    index: Mutex<SpaceTracker>,
    derived: Mutex<SpaceTracker>,
}

/// The pair of tokens one maintained write holds — one per watermark space.
#[derive(Debug, Clone, Copy)]
pub struct WriteTokens {
    pub index: WriteToken,
    pub derived: WriteToken,
}

impl WatermarkTracker {
    /// Register an in-flight write that captured `col_mod`, returning the tokens
    /// to complete each space with. Call in the write's actor job.
    pub fn register(&self, col_mod: i64) -> WriteTokens {
        WriteTokens {
            index: self
                .index
                .lock()
                .expect("watermark tracker poisoned")
                .register(col_mod),
            derived: self
                .derived
                .lock()
                .expect("watermark tracker poisoned")
                .register(col_mod),
        }
    }

    /// Complete the INDEX-watermark side. `Some(V)` = advance the index
    /// watermark to `V`; `None` = leave it behind.
    pub fn complete_index(&self, token: WriteToken, success: bool) -> Option<i64> {
        self.index
            .lock()
            .expect("watermark tracker poisoned")
            .complete(token, success)
    }

    /// Complete the DERIVED-watermark side. `Some(V)` = advance the derived
    /// watermark to `V`; `None` = leave it behind.
    pub fn complete_derived(&self, token: WriteToken, success: bool) -> Option<i64> {
        self.derived
            .lock()
            .expect("watermark tracker poisoned")
            .complete(token, success)
    }

    /// Clear the INDEX poison floor — a full index reconcile/rebuild healed all
    /// prior index-tail failures.
    pub fn clear_index_poison(&self) {
        self.index
            .lock()
            .expect("watermark tracker poisoned")
            .clear_poison();
    }

    /// Clear the DERIVED poison floor — a full derived rebuild healed all prior
    /// derived-ingest failures.
    pub fn clear_derived_poison(&self) {
        self.derived
            .lock()
            .expect("watermark tracker poisoned")
            .clear_poison();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_inflight_advances_to_its_captured_col_mod() {
        let t = WatermarkTracker::default();
        let toks = t.register(100);
        assert_eq!(t.complete_index(toks.index, true), Some(100));
        assert_eq!(t.complete_derived(toks.derived, true), Some(100));
    }

    #[test]
    fn a_failed_tail_leaves_the_watermark_behind() {
        let t = WatermarkTracker::default();
        let toks = t.register(100);
        assert_eq!(t.complete_index(toks.index, false), None);
        // The derived side is independent — it may still succeed.
        assert_eq!(t.complete_derived(toks.derived, true), Some(100));
    }

    #[test]
    fn an_earlier_inflight_write_blocks_a_later_completion() {
        // The #585 interleave: op B writes col.mod 100 (parks, in flight); op A
        // writes col.mod 200 and completes FIRST. A must NOT certify 200,
        // because B's 100 ≤ 200 is still un-indexed.
        let t = WatermarkTracker::default();
        let b = t.register(100);
        let a = t.register(200);
        assert_eq!(
            t.complete_index(a.index, true),
            None,
            "A cannot certify 200 while B's 100 is in flight"
        );
        // B then completes (success): nothing earlier remains → it advances to
        // its own 100. Drift between 100 and the live 200 heals A's note via an
        // idempotent reconcile.
        assert_eq!(t.complete_index(b.index, true), Some(100));
    }

    #[test]
    fn an_earlier_failure_poisons_until_a_reconcile_clears_it() {
        // The #585 keystone: B (col.mod 100) FAILS its tail → its note is
        // genuinely un-indexed. A (200) completing AFTER must NOT certify 200,
        // or B's note is lost forever (watermark == live col.mod → drift quiet).
        let t = WatermarkTracker::default();
        let b = t.register(100);
        let a = t.register(200);
        assert_eq!(
            t.complete_index(b.index, false),
            None,
            "B failed → poisoned at 100"
        );
        assert_eq!(
            t.complete_index(a.index, true),
            None,
            "A cannot certify 200 past the un-healed failure at 100"
        );
        // A full reconcile re-indexes everything (incl. B's note) and stamps
        // its own col.mod → the poison floor clears.
        t.clear_index_poison();
        let c = t.register(300);
        assert_eq!(t.complete_index(c.index, true), Some(300));
    }

    #[test]
    fn same_col_mod_in_flight_blocks_until_all_complete() {
        // Two writes in the same millisecond share a col.mod. Neither may
        // certify while the other is in flight (≤ is inclusive).
        let t = WatermarkTracker::default();
        let x = t.register(100);
        let y = t.register(100);
        assert_eq!(t.complete_index(x.index, true), None);
        assert_eq!(t.complete_index(y.index, true), Some(100));
    }

    #[test]
    fn index_and_derived_floors_are_independent() {
        // An embed (index) failure must not poison the derived watermark.
        let t = WatermarkTracker::default();
        let a = t.register(100);
        assert_eq!(t.complete_index(a.index, false), None);
        assert_eq!(t.complete_derived(a.derived, true), Some(100));
        // A later index write is still poisoned; a later derived write is not.
        let b = t.register(200);
        assert_eq!(
            t.complete_index(b.index, true),
            None,
            "index poisoned at 100"
        );
        assert_eq!(
            t.complete_derived(b.derived, true),
            Some(200),
            "derived clean"
        );
    }
}
