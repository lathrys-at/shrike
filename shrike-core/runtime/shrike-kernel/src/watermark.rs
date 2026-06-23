//! Per-space watermark poison floors.
//!
//! The index and derived watermarks (`col.mod`) are the *only* drift signal:
//! `check_drift`/`rebuild_derived` reconcile iff `stored_watermark !=
//! live_col_mod`. So a watermark may be advanced to a `col.mod` value `V` ONLY
//! when every write whose `col.mod ≤ V` has actually been indexed/ingested.
//! Violating that certifies an un-indexed note as current → the heal gates go
//! quiet → permanent silent search loss.
//!
//! # Ordering is structural now
//!
//! The single persistent [`ingest`](crate::ingest) actor is the *sole writer*
//! of the index and derived stores, so watermark ordering no longer needs the
//! cross-task in-flight-token machinery the old multi-writer model carried (the
//! monotonic token set, the `any(other ≤ captured)` scan, the
//! register-inside-the-actor-job happens-before argument). The replacement is a
//! one-line invariant defended in the ingest actor: **the maintenance item is
//! enqueued INSIDE the collection-write job, so queue order == `col.mod` order**
//! (`col.mod` is monotonic; the collection actor is FIFO), and one FIFO consumer
//! processes items in that order. Each drained batch is therefore a contiguous
//! `col.mod` prefix, and the actor advances each space's watermark to **the
//! highest `col.mod` in the batch whose every earlier item also succeeded** —
//! a linear pass over one batch, no cross-task state.
//!
//! # What remains: the poison floor
//!
//! The after-commit tail failure policy is *skip-and-keep-going* (a down embed
//! backend must not stall derived ingest or later notes), so a failed write
//! leaves its note in the collection but absent from the index/derived store.
//! The watermark must then stay STRICTLY BELOW that note's `col.mod` until a
//! full reconcile/rebuild re-indexes it — otherwise drift goes quiet and the
//! note is lost. [`SpaceFloor`] records that floor and blocks any later advance
//! to/past it until [`SpaceFloor::clear`] (a whole-collection
//! reconcile/rebuild, which re-indexes everything and stamps its own snapshot
//! `col.mod` directly).
//!
//! The index and derived floors are tracked SEPARATELY because their tails fail
//! independently (an embed failure must not block the derived watermark, and
//! vice-versa).

/// The poison floor for ONE watermark space (index OR derived): the lowest
/// `col.mod` of a write whose tail FAILED and has not yet been healed by a full
/// reconcile/rebuild. The watermark must never advance to or past it.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpaceFloor {
    poison_floor: Option<i64>,
}

impl SpaceFloor {
    /// Decide the value the watermark may advance to after processing a write
    /// (or a contiguous batch prefix) that captured `col_mod`:
    ///
    /// - `success == false` → records `col_mod` as the poison floor (lowest
    ///   wins) and returns `None` (leave the watermark behind so drift re-fires
    ///   until a heal clears the floor).
    /// - an un-healed earlier failure at/below `col_mod` → `None` (advancing
    ///   would over-certify the un-indexed note the floor guards).
    /// - otherwise → `Some(col_mod)` (safe to certify now).
    pub fn resolve(&mut self, col_mod: i64, success: bool) -> Option<i64> {
        if !success {
            self.poison_floor = Some(match self.poison_floor {
                Some(f) => f.min(col_mod),
                None => col_mod,
            });
            return None;
        }
        if matches!(self.poison_floor, Some(f) if f <= col_mod) {
            return None;
        }
        Some(col_mod)
    }

    /// Clear the floor — a full reconcile/rebuild has re-indexed the whole
    /// collection (so every prior failed write is healed) and stamped its own
    /// snapshot `col.mod` directly.
    pub fn clear(&mut self) {
        self.poison_floor = None;
    }

    /// The current poison floor, for status/observability.
    pub fn floor(&self) -> Option<i64> {
        self.poison_floor
    }
}

/// The two per-space floors a maintained write resolves — one for the index
/// watermark, one for the derived watermark.
#[derive(Debug, Default, Clone, Copy)]
pub struct WatermarkFloors {
    /// The index-watermark floor.
    pub index: SpaceFloor,
    /// The derived-watermark floor.
    pub derived: SpaceFloor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_write_advances_to_its_captured_col_mod() {
        let mut f = SpaceFloor::default();
        assert_eq!(f.resolve(100, true), Some(100));
        assert_eq!(f.resolve(200, true), Some(200));
    }

    #[test]
    fn a_failed_write_leaves_the_watermark_behind_and_floors() {
        let mut f = SpaceFloor::default();
        assert_eq!(f.resolve(100, false), None);
        // A later success cannot jump the un-healed failure at 100.
        assert_eq!(
            f.resolve(200, true),
            None,
            "200 must not certify past the un-healed failure at 100"
        );
    }

    #[test]
    fn the_lowest_failure_wins_the_floor() {
        let mut f = SpaceFloor::default();
        assert_eq!(f.resolve(200, false), None);
        assert_eq!(f.resolve(100, false), None);
        assert_eq!(f.floor(), Some(100));
        // Only a heal that re-indexes everything clears it.
        f.clear();
        assert_eq!(f.resolve(300, true), Some(300));
    }

    #[test]
    fn a_clean_write_below_an_existing_floor_still_blocks() {
        // The floor is at 100; a clean write at 50 is below it, but the floor
        // guards 100, not 50 — a write at 50 can certify only itself.
        let mut f = SpaceFloor::default();
        assert_eq!(f.resolve(100, false), None);
        assert_eq!(
            f.resolve(50, true),
            Some(50),
            "a clean write strictly below the floor certifies itself"
        );
        // But a clean write at/above the floor stays blocked.
        assert_eq!(f.resolve(100, true), None);
        assert_eq!(f.resolve(150, true), None);
    }

    #[test]
    fn clear_only_clears_the_floor() {
        let mut f = SpaceFloor::default();
        assert_eq!(f.resolve(100, false), None);
        f.clear();
        assert_eq!(f.floor(), None);
        assert_eq!(f.resolve(100, true), Some(100));
    }

    // ── Generative "never over-certifies" property ──────────────────────
    //
    // The load-bearing safety invariant (module docs): a watermark may certify
    // a col_mod V only when every write with col.mod ≤ V is indexed. Over a
    // random op sequence, no `resolve` may return `Some(v)` while an unhealed
    // failure sits at col.mod ≤ v — that is the silent-search-loss bug. Pin it
    // against an independent oracle of the floor.

    /// Seed-reproducible SplitMix64 (the inline-test generator).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn resolve_never_over_certifies_an_unhealed_failure() {
        let mut rng = Rng::new(0x7A7E_8004);
        for _ in 0..50_000 {
            let mut f = SpaceFloor::default();
            // Independent oracle of the floor (min unhealed-failure col_mod).
            let mut oracle_floor: Option<i64> = None;
            let ops = 1 + rng.next_u64() % 40;
            for _ in 0..ops {
                // Small col_mod space so failures and successes interleave at the
                // same and adjacent values (the boundary the floor guards).
                let col_mod = (rng.next_u64() % 12) as i64;
                let success = !rng.next_u64().is_multiple_of(3); // ~2/3 succeed
                let do_clear = rng.next_u64().is_multiple_of(17);
                if do_clear {
                    f.clear();
                    oracle_floor = None;
                    continue;
                }

                let got = f.resolve(col_mod, success);

                // Oracle decision, computed from the floor BEFORE this op folds in.
                let blocked = matches!(oracle_floor, Some(fl) if fl <= col_mod);
                let want = if !success || blocked {
                    None
                } else {
                    Some(col_mod)
                };
                assert_eq!(got, want, "resolve diverged from the oracle");

                // The safety invariant: a certified value is strictly below any
                // unhealed failure floor (so no un-indexed note ≤ v is certified).
                if let Some(v) = got {
                    assert!(
                        oracle_floor.is_none() || oracle_floor.unwrap() > v,
                        "over-certified {v} at/above floor {oracle_floor:?}"
                    );
                }

                // Fold the failure into the oracle AFTER the decision (a failure
                // floors itself and returns None — it can't certify itself).
                if !success {
                    oracle_floor = Some(match oracle_floor {
                        Some(fl) => fl.min(col_mod),
                        None => col_mod,
                    });
                }
                assert_eq!(f.floor(), oracle_floor, "floor tracking diverged");
            }
        }
    }

    #[test]
    fn once_floored_no_value_at_or_above_certifies_until_clear() {
        // Directed restatement of the property: after an unhealed failure at F,
        // every clean write at col.mod ≥ F is blocked, every write strictly
        // below F certifies itself, and only clear() reopens F and above.
        let mut rng = Rng::new(0x00F1_000F);
        for _ in 0..10_000 {
            let mut f = SpaceFloor::default();
            let fail_at = (rng.next_u64() % 50) as i64;
            f.resolve(fail_at, false);
            for _ in 0..10 {
                let v = (rng.next_u64() % 60) as i64;
                let got = f.resolve(v, true);
                if v < fail_at {
                    assert_eq!(got, Some(v), "below the floor must self-certify");
                } else {
                    assert_eq!(got, None, "at/above the floor must stay blocked");
                }
            }
            f.clear();
            let v = fail_at + 5;
            assert_eq!(f.resolve(v, true), Some(v), "clear reopens at/above F");
        }
    }

    #[test]
    fn index_and_derived_floors_are_independent() {
        // The two spaces fail independently: an index-tail failure must not
        // block the derived watermark, and vice versa.
        let mut wf = WatermarkFloors::default();
        assert_eq!(wf.index.resolve(100, false), None);
        // derived saw no failure → it still certifies at/above 100.
        assert_eq!(wf.derived.resolve(100, true), Some(100));
        assert_eq!(wf.derived.resolve(200, true), Some(200));
        // index is still floored.
        assert_eq!(wf.index.resolve(200, true), None);
        // Clearing one space leaves the other's floor intact.
        let mut wf2 = WatermarkFloors::default();
        wf2.index.resolve(50, false);
        wf2.derived.resolve(60, false);
        wf2.index.clear();
        assert_eq!(wf2.index.resolve(70, true), Some(70));
        assert_eq!(
            wf2.derived.resolve(70, true),
            None,
            "derived floor survives"
        );
    }
}
