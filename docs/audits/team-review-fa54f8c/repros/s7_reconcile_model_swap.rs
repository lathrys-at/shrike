//! S7-1 repro (preserved by lead; rev-S7 worktree reaped).
//! reconcile != rebuild on a model swap that keeps col_mod. RED at fa54f8c:
//! model_id left stale (model-A vs rebuild model-B); 3/3 vectors diverge.
//! Place at native/shrike-kernel/tests/s7_reconcile_model_swap.rs (compiles w/ existing deps).
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s7 cargo test -p shrike-kernel --test s7_reconcile_model_swap -- --nocapture
//! (The s7_reconcile_model_swap_mixed.rs variant reuses these helpers.)

use std::sync::Arc;
use futures::executor::block_on;
use futures::future::BoxFuture;
use shrike_ffi::NativeResult;
use shrike_index::MultiModalIndex;
use shrike_kernel::index_orchestrator::{EmbedInput, IndexOrchestrator};
use shrike_kernel::Embedder;
use shrike_store_api::VectorIndex;

struct TaggedEmbedder { tag: f32 }
impl Embedder for TaggedEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let tag = self.tag;
        Box::pin(async move {
            Ok(texts.iter().map(|t| {
                let b = (t.len() as f32 % 7.0) / 7.0;
                vec![b, 1.0 - b, tag, 0.0]
            }).collect())
        })
    }
}
fn input(nid: i64, text: &str) -> EmbedInput {
    EmbedInput { note_id: nid, text: text.to_owned(), image_names: vec![], ocr_texts: vec![] }
}
fn temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!("s7-modelswap-{tag}-{}-{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&d).unwrap(); d
}
fn engine() -> Arc<dyn VectorIndex> {
    Arc::new(MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap())
}
#[test]
fn reconcile_on_model_swap_diverges_from_rebuild() {
    let model_a = TaggedEmbedder { tag: 0.10 };
    let model_b = TaggedEmbedder { tag: 0.90 };
    let inputs = vec![input(1, "one"), input(2, "two"), input(3, "three")];
    let col_mod = 7; // unchanged across the model swap
    let dir_r = temp_dir("recon");
    let reconciled = IndexOrchestrator::open(dir_r.clone(), engine());
    block_on(reconciled.rebuild(inputs.clone(), col_mod, Some("model-A".into()), &model_a, None)).unwrap();
    assert!(reconciled.check_drift(col_mod, Some("model-B"), false), "a model swap must register as drift");
    block_on(reconciled.reconcile(inputs.clone(), col_mod, Some("model-B".into()), &model_b, None)).unwrap();
    let dir_f = temp_dir("rebuild");
    let rebuilt = IndexOrchestrator::open(dir_f.clone(), engine());
    block_on(rebuilt.rebuild(inputs, col_mod, Some("model-B".into()), &model_b, None)).unwrap();
    let mut diverged = Vec::new();
    for key in rebuilt.engine().keys() {
        let r = reconciled.engine().get(key); let f = rebuilt.engine().get(key);
        if r != f { diverged.push((key, r, f)); }
    }
    let recon_model = reconciled.model_id(); let rebuilt_model = rebuilt.model_id();
    std::fs::remove_dir_all(&dir_r).ok(); std::fs::remove_dir_all(&dir_f).ok();
    assert_eq!(recon_model, rebuilt_model, "reconcile left model_id stale after a model swap (still drifting forever)");
    assert!(diverged.is_empty(), "reconcile != full rebuild after a model swap: {} notes still hold OLD-model vectors", diverged.len());
}

// --- s7_reconcile_model_swap_mixed.rs (same imports/helpers as above) ---
// #[test]
// fn reconcile_on_model_swap_plus_edit_makes_a_mixed_model_index() {
//     let model_a = TaggedEmbedder { tag: 0.10 };
//     let model_b = TaggedEmbedder { tag: 0.90 };
//     let v1 = vec![input(1, "one"), input(2, "two"), input(3, "three")];
//     let v2 = vec![input(1, "one"), input(2, "two EDITED LONGER"), input(3, "three")];
//     let dir_r = temp_dir("recon");
//     let reconciled = IndexOrchestrator::open(dir_r.clone(), engine());
//     block_on(reconciled.rebuild(v1, 1, Some("model-A".into()), &model_a, None)).unwrap();
//     block_on(reconciled.reconcile(v2.clone(), 2, Some("model-B".into()), &model_b, None)).unwrap();
//     let dir_f = temp_dir("rebuild");
//     let rebuilt = IndexOrchestrator::open(dir_f.clone(), engine());
//     block_on(rebuilt.rebuild(v2, 2, Some("model-B".into()), &model_b, None)).unwrap();
//     assert_eq!(reconciled.model_id(), Some("model-B".into())); // drift goes quiet
//     // ... assert no diverged unchanged notes (2/3 keep OLD-model vectors → RED)
// }
