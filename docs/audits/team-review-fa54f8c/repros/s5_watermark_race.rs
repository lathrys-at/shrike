//! S5-1 repro (preserved by lead; rev-S5 worktree reaped, reverted clean).
//! Concurrent ops falsely advance the index watermark → a not-yet-indexed note's
//! col_mod is certified → check_drift sees no drift → reconcile NEVER heals it →
//! permanent silent search loss. Root cause: advance_watermarks reads the LIVE
//! col.mod (lib.rs:1771-1783), not the col.mod captured with THIS op's write.
//! SHARES ROOT CAUSE with S8b-1 (half-write trigger). RED at fa54f8c.
//! Add to `mod no_cpython_smoke` in native/shrike-kernel/src/lib.rs (reuses
//! HashEmbedder/temp_dir/etc. test helpers).
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s5 cargo test -p shrike-kernel s5_interleaved -- --nocapture

struct GatedEmbedder {
    gate: tokio::sync::Notify,
    parked: tokio::sync::Notify,
    bravo_seen: std::sync::atomic::AtomicBool,
}
impl Embedder for GatedEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move {
            let is_bravo = texts.iter().any(|t| t.contains("bravo"));
            if is_bravo {
                self.bravo_seen.store(true, std::sync::atomic::Ordering::SeqCst);
                self.parked.notify_one();
                self.gate.notified().await;
                return Err(NativeError::internal("bravo embed deliberately failed"));
            }
            Ok(HashEmbedder::embed_sync(&texts))
        }
        )
    }
    fn fingerprint(&self) -> Option<String> { Some("gated-embedder:v1".to_string()) }
    fn dim(&self) -> Option<usize> { Some(64) }
}

#[test]
fn s5_interleaved_upsert_falsely_advances_watermark_losing_a_note() {
    crate::runtime::block_on(async {
        let dir = temp_dir();
        let kernel = Arc::new(
            Kernel::open(dir.join("c.anki2").to_str().unwrap(), dir.join("cache").to_str().unwrap())
                .await.unwrap(),
        );
        let embedder = Arc::new(GatedEmbedder {
            gate: tokio::sync::Notify::new(),
            parked: tokio::sync::Notify::new(),
            bravo_seen: std::sync::atomic::AtomicBool::new(false),
        });
        kernel.attach_embedder(embedder.clone(), None);
        kernel.reindex_if_needed().await.unwrap();
        let basic = kernel.notetype_id("Basic").await.unwrap();

        // Op B: writes "bravo", then PARKS in embed (in flight). Its collection
        // write completes before it parks, so col.mod already reflects bravo.
        let kb = Arc::clone(&kernel);
        let op_b = crate::spawn_op(async move {
            kb.upsert_note(basic, 1, vec!["bravo bravo bravo".into(), "b".into()],
                vec![], DuplicatePolicy::Error).await.map(|_| ())
        });
        embedder.parked.notified().await;
        assert!(embedder.bravo_seen.load(std::sync::atomic::Ordering::SeqCst));

        // Op A: writes "alpha" + runs its full index tail — advance_watermarks
        // reads the CURRENT col.mod (already includes bravo) and stamps it.
        let alpha = kernel.upsert_note(basic, 1, vec!["alpha alpha alpha".into(), "a".into()],
            vec![], DuplicatePolicy::Error).await.unwrap();
        let CreateOutcome::Created(alpha_id) = alpha else { panic!("alpha create") };

        embedder.gate.notify_one(); // release op B → it ERRORS, bravo never indexed
        let _ = op_b.await;

        let all_ids = kernel.collection().run(|core| core.find_notes("")).await.unwrap().unwrap();
        let bravo_id = *all_ids.iter().find(|id| **id != alpha_id).unwrap();
        let primary = kernel.index_set().primary();
        let col_mod = kernel.col_mod().await.unwrap();

        assert!(primary.engine().contains(alpha_id), "alpha indexed (sanity)");
        assert!(!primary.engine().contains(bravo_id), "PRECONDITION: bravo's embed failed, not indexed");
        assert_eq!(primary.col_mod(), Some(col_mod), "EVIDENCE: watermark falsely advanced to live col.mod");
        let drift_will_heal = kernel.reindex_if_needed().await.unwrap();
        assert!(drift_will_heal,
            "DEFECT: watermark == col.mod, reindex sees no drift, bravo PERMANENTLY missing (silent search loss)");

        kernel.close().await.unwrap();
        std::fs::remove_dir_all(dir).ok();
    });
}
