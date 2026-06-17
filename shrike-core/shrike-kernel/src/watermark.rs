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
//! The fix: read `col.mod` AND [`WatermarkTracker::register`] the write as *in
//! flight* INSIDE the SAME collection-actor job as the write, then on the tail's
//! completion advance the watermark to the captured value ONLY IF
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
//! **Why registration must be IN the actor job (the happens-before argument).**
//! The collection actor runs jobs FIFO and inline; an op is a write job, then an
//! off-actor tail (embed/index/ingest, several awaits), then a completion. Ops
//! are independent tasks on a multi-thread runtime, so a *continuation* (the
//! code after `.await` on the job) is NOT ordered against other ops — only the
//! jobs are. If `register` ran in the continuation, an adversarial schedule
//! (op B's continuation starved across op A's whole async tail) could let op A
//! register col.mod 200 and advance to 200 BEFORE op B ever registers its
//! col.mod-100 write — then if B's tail fails, B's note is un-indexed yet the
//! watermark is 200 ≥ 100 → drift quiet → the silent loss this fix exists to
//! close. Registering INSIDE the job fixes the ordering: `col.mod` is monotonic,
//! so any write with `col.mod ≤ V` committed in a job that, by FIFO, ran (and
//! therefore registered, in-job) before the job that observes `V`. Hence at any
//! op's completion every earlier-or-equal write is already in-flight (or
//! poisoned) and the `≤ V` gate sees it. Registration order == write order, by
//! construction.
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
    /// Register a write that captured `col_mod`. MUST be called INSIDE the
    /// collection-actor job that reads `col_mod` (the write's own job, or — for
    /// the no-own-write tails: recognition sweep, reindex_notes, forget_notes,
    /// metadata — a job that reads `col.mod` and registers atomically). The
    /// actor's FIFO then orders registration with every concurrent write; see
    /// the module-level happens-before argument. Calling it from a post-await
    /// continuation reintroduces the over-certification race.
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
    /// to complete each space with. MUST be called INSIDE the collection-actor
    /// job that read `col_mod` (the tracker is held by an `Arc` so it can be
    /// cloned into the job closure) — the actor's FIFO is what orders
    /// registration with concurrent writes (see the module docs). A post-await
    /// continuation call is unordered and reintroduces the #585 race.
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

    /// Test-only: how many index-side writes are currently in flight. Used to
    /// pin the FIFO-register-in-job guarantee (both writes in flight once their
    /// actor jobs have returned).
    #[cfg(test)]
    pub fn index_in_flight(&self) -> usize {
        self.index
            .lock()
            .expect("watermark tracker poisoned")
            .in_flight
            .len()
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

    // ── Cross-review guards (#588 peer review): seams that, if they regressed,
    // would silently re-open the #585 silent-loss bug. Pure tracker-level, like
    // the rest of this module.

    /// (a') A later SUCCESS cannot jump a poison left by an EARLIER FAILURE. The
    /// failure here is the *earlier* write (200) and the success arrives *after*
    /// (300) — the watermark must stay below the un-healed 200 until a reconcile
    /// clears the floor, or 200's note is lost (watermark would reach 300 ≥ 200).
    #[test]
    fn a_later_success_cannot_jump_an_earlier_failures_poison() {
        let t = WatermarkTracker::default();
        let mid = t.register(200);
        assert_eq!(
            t.complete_index(mid.index, false),
            None,
            "200 failed → poison@200"
        );
        let later = t.register(300);
        assert_eq!(
            t.complete_index(later.index, true),
            None,
            "300 must not certify past the un-healed failure at 200"
        );
        // Only a full reconcile (re-indexes 200's note) may clear the floor.
        t.clear_index_poison();
        let next = t.register(400);
        assert_eq!(t.complete_index(next.index, true), Some(400));
    }

    /// (b) A same-`col.mod` FAILURE blocks a peer at the same `col.mod`. Two
    /// writes share col.mod 100 (a same-millisecond pair); x fails → poison@100;
    /// y succeeds but its own 100 is `≤` the poison floor of 100, so it must NOT
    /// advance — otherwise y would certify 100 while x's note (also at 100) is
    /// un-indexed.
    #[test]
    fn a_same_col_mod_failure_blocks_its_peer() {
        let t = WatermarkTracker::default();
        let x = t.register(100);
        let y = t.register(100);
        assert_eq!(
            t.complete_index(x.index, false),
            None,
            "x failed → poison@100"
        );
        assert_eq!(
            t.complete_index(y.index, true),
            None,
            "y's own 100 is ≤ the poison floor 100 — it cannot certify its peer's loss"
        );
    }

    /// (c-HARD) `clear_poison` must clear ONLY the poison floor, never drop
    /// in-flight writes. If a reconcile fires while an earlier write is still in
    /// flight, clearing the poison must not also forget that write — a later
    /// success must stay BLOCKED by the still-in-flight earlier write (not by
    /// poison). This is the exact regression that would re-open the bug if
    /// `clear_poison` ever touched `in_flight`.
    #[test]
    fn clear_poison_does_not_drop_in_flight_writes() {
        let t = WatermarkTracker::default();
        let early = t.register(100); // E: in flight, never completed
        let high = t.register(200); // H: completes after the clear
                                    // A reconcile clears the poison floor mid-flight (E is still pending).
        t.clear_index_poison();
        assert_eq!(
            t.complete_index(high.index, true),
            None,
            "H must stay blocked by the still-in-flight E@100, NOT by poison — \
             clear_poison must not have dropped E"
        );
        // Sanity: once E completes, nothing earlier remains → it advances.
        assert_eq!(t.complete_index(early.index, true), Some(100));
    }

    /// (d) On a FAILURE, the index and derived sides move independently: an
    /// index failure poisons only index; the same op's derived side still
    /// advances; and a LATER write advances derived but not (still-poisoned)
    /// index.
    #[test]
    fn a_failure_keeps_index_and_derived_independent() {
        let t = WatermarkTracker::default();
        let a = t.register(100);
        assert_eq!(
            t.complete_index(a.index, false),
            None,
            "index failed → poison@100"
        );
        assert_eq!(
            t.complete_derived(a.derived, true),
            Some(100),
            "derived side is unaffected by the index failure"
        );
        let b = t.register(200);
        assert_eq!(
            t.complete_index(b.index, true),
            None,
            "later index still blocked by the index poison@100"
        );
        assert_eq!(
            t.complete_derived(b.derived, true),
            Some(200),
            "later derived advances — derived was never poisoned"
        );
    }
}
