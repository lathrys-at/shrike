//! The engine contract (#342): what the kernel consumes, what engine crates
//! implement, and nothing else. The kernel composes `Arc<dyn Embedder>` /
//! `Arc<dyn Recognizer>` it is *given* at assembly — it never names a
//! concrete engine, a runtime (ort), a platform (Apple/Android), or a
//! transport (HTTP, subprocess). Each concrete engine lives in its own crate
//! depending on this one; deployment hosts (Python server, Swift/Kotlin
//! apps, the C host) construct engines from configuration and attach them.
//!
//! # The two conformance routes
//!
//! The kernel only ever sees the async traits ([`Embedder`],
//! [`ImageEmbedder`], [`Recognizer`]). An engine conforms by whichever route
//! matches its underlying API's nature:
//!
//! - **Route 1 — naturally-sync compute** (ort inference, a synchronous HTTP
//!   client): implement the sync compute traits ([`EmbedText`],
//!   [`EmbedImages`], [`RecognizeMedia`]) — chunk-level, `Send + Sync`, no
//!   futures, no threads, assuming *nothing* about execution. The ONE
//!   adapter, [`Blocking`], bridges to the async traits over the owned
//!   runtime's blocking pool (#374): an eager `spawn_blocking` with the
//!   `safe_batch` chunk loop inside — batching is execution policy, not
//!   engine compute.
//! - **Route 2 — naturally-async engines** (a completion-handler platform
//!   API reached through ObjC/Swift glue; an async HTTP client): implement
//!   the async traits directly — the future suspends and completes from the
//!   engine's own completion source (callback → oneshot → waker). No ambient
//!   runtime assumed, no thread parked. Where the underlying API is async,
//!   this is the preferred shape: a blocking lane would waste a thread
//!   waiting on a callback.
//!
//! # Execution is the runtime's, topology is the kernel's
//!
//! Pipeline *topology* — what must order before what — is the kernel's
//! consistency model; independent engine futures are `try_join`ed by the
//! kernel. Execution lives on the kernel's owned tokio runtime (#374):
//! sync engines ride the blocking pool through [`Blocking`], async engines
//! complete from their own sources. Engines spawn no threads themselves and
//! never block a runtime worker.
//!
//! # Errors
//!
//! A dependency or availability failure (model missing, service down,
//! platform API unavailable) is `NativeError::unavailable`; a contract
//! violation (malformed payload, wrong arity) is `internal`.

pub mod probe;

use std::sync::Arc;

use futures::future::BoxFuture;

use shrike_ffi::{NativeError, NativeResult};

// ── media items ──────────────────────────────────────────────────────────────

/// One media payload with its type hint. `mime` comes from the media handler
/// (derived from the filename — the resolver has the name when it reads the
/// bytes), so engines that route by media kind (an omni recognizer: image →
/// OCR vs audio → ASR) or whose decoder wants a format hint get it for free.
/// It is a *hint*: decoders may still sniff magic bytes as the fallback, and
/// `None` is always legal.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaItem {
    pub bytes: Vec<u8>,
    pub mime: Option<String>,
}

impl MediaItem {
    pub fn new(bytes: Vec<u8>, mime: Option<String>) -> Self {
        Self { bytes, mime }
    }

    /// Bytes-only item (no hint — the decoder sniffs).
    pub fn untyped(bytes: Vec<u8>) -> Self {
        Self { bytes, mime: None }
    }

    /// An item whose hint is derived from a filename via [`mime_for_name`].
    pub fn from_named(name: &str, bytes: Vec<u8>) -> Self {
        Self {
            mime: mime_for_name(name),
            bytes,
        }
    }
}

/// The extension→MIME map for the media kinds notes carry. Deliberately
/// small: this exists to give engines a routing/decoding hint, not to be a
/// general MIME database.
pub fn mime_for_name(name: &str) -> Option<String> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "svg" => "image/svg+xml",
        "avif" => "image/avif",
        "heic" => "image/heic",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        "opus" => "audio/opus",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        "mov" => "video/quicktime",
        _ => return None,
    };
    Some(mime.to_string())
}

// ── the async traits the kernel consumes ─────────────────────────────────────

/// Text embedding — the kernel-facing seam. Futures are runtime-agnostic
/// (no tokio); see the crate docs for the two conformance routes.
pub trait Embedder: Send + Sync + 'static {
    /// Embed a batch of texts, order-preserving, one vector per input. The
    /// batch may be arbitrarily large — conforming implementations chunk
    /// internally (route 1 adapters do this; route 2 engines own it).
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>>;

    /// Stable engine identity (model fingerprint) — drives index drift
    /// invalidation. Read once at attach.
    fn fingerprint(&self) -> Option<String> {
        None
    }

    /// The embedding dimension, when known without a probe — lets an empty
    /// collection materialize its index without embedding anything.
    fn dim(&self) -> Option<usize> {
        None
    }
}

/// Engines share freely — and `?Sized` means an `Arc<dyn Embedder>` is
/// itself an `Embedder`, so hosts pass type-erased handles anywhere a
/// concrete engine fits (the same for every blanket impl below).
impl<T: Embedder + ?Sized> Embedder for Arc<T> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        (**self).embed(texts)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        (**self).dim()
    }
}

/// Image embedding (the CLIP-style image half).
pub trait ImageEmbedder: Send + Sync {
    /// Embed a batch of images, order-preserving, one vector per item.
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>>;
}

impl<T: ImageEmbedder + ?Sized> ImageEmbedder for Arc<T> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        (**self).embed_images(images)
    }
}

/// Media access the host injects: read bytes lazily, stat cheaply. The
/// per-note fingerprint hashes *names of present media* via `exists` (no
/// byte read); bytes are read only for items actually being processed.
pub trait ImageResolver: Send + Sync {
    fn read(&self, name: &str) -> Option<Vec<u8>>;
    fn exists(&self, name: &str) -> bool;
}

// ── recognition (#228) ───────────────────────────────────────────────────────

/// One recognized segment: a line/word for OCR (with an optional normalized
/// top-left `[x, y, w, h]` box) — the shape generalizes to time spans for
/// ASR (a future field, not a bbox reuse).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    pub text: String,
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
}

/// One media item's recognition: the flattened text (reading order), the
/// overall confidence (engine-defined aggregate), and the retained segments
/// (#228's one-pass/many-consumers rule: never flatten-and-discard).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Recognition {
    pub text: String,
    pub confidence: f64,
    #[serde(default)]
    pub segments: Vec<Segment>,
}

/// Media-to-text recognition (OCR now; ASR and VLM description are the same
/// shape) — the kernel-facing seam.
pub trait Recognizer: Send + Sync + 'static {
    /// Recognize a batch of media items, order-preserving. An unreadable
    /// item yields an empty recognition (text "", confidence 0) rather than
    /// failing the batch.
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>>;

    /// Stable engine identity (model/OS version) — a change invalidates
    /// derived text on the next pending sweep. Read once at attach.
    fn fingerprint(&self) -> Option<String> {
        None
    }
}

impl<T: Recognizer + ?Sized> Recognizer for Arc<T> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        (**self).recognize(items)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }
}

// ── route 1: sync compute traits ─────────────────────────────────────────────

/// Chunk-level text embedding, pure compute. Implementors assume nothing
/// about execution: no futures, no threads, no executors. `embed_chunk`
/// receives at most [`EmbedText::safe_batch`] texts per call — the adapter
/// owns the batch loop.
pub trait EmbedText: Send + Sync + 'static {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>>;

    /// The largest batch this engine is proven safe to embed in one call
    /// (batch-variance probing, e.g. int8 dynamic quantization). 1 = serial.
    fn safe_batch(&self) -> usize {
        1
    }

    fn fingerprint(&self) -> Option<String> {
        None
    }

    fn dim(&self) -> Option<usize> {
        None
    }
}

impl<T: EmbedText + ?Sized> EmbedText for Arc<T> {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        (**self).embed_chunk(texts)
    }

    fn safe_batch(&self) -> usize {
        (**self).safe_batch()
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        (**self).dim()
    }
}

/// Chunk-level image embedding, pure compute (the CLIP image half).
pub trait EmbedImages: Send + Sync + 'static {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>>;
}

impl<T: EmbedImages + ?Sized> EmbedImages for Arc<T> {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        (**self).embed_image_chunk(images)
    }
}

/// Chunk-level media recognition, pure compute.
pub trait RecognizeMedia: Send + Sync + 'static {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>>;

    fn fingerprint(&self) -> Option<String> {
        None
    }
}

impl<T: RecognizeMedia + ?Sized> RecognizeMedia for Arc<T> {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        (**self).recognize_chunk(items)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }
}

/// Host-assembled identity and batch policy over a pure-compute engine.
/// Fingerprint strings are host policy (they fold settings the engine can't
/// know — text-prep versions, pooling flags); `safe_batch` comes from a
/// host-run probe over the *loaded* model; `dim` may already be known from
/// the same probe. The host pins all three at composition time, so engine
/// crates carry none of them: wrap the engine in `WithPolicy` and hand the
/// result to an adapter.
pub struct WithPolicy<E> {
    engine: Arc<E>,
    fingerprint: Option<String>,
    dim: Option<usize>,
    safe_batch: usize,
}

impl<E> WithPolicy<E> {
    pub fn new(
        engine: Arc<E>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> Self {
        Self {
            engine,
            fingerprint,
            dim,
            safe_batch: safe_batch.max(1),
        }
    }
}

impl<E: EmbedText> EmbedText for WithPolicy<E> {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        self.engine.embed_chunk(texts)
    }

    fn safe_batch(&self) -> usize {
        self.safe_batch
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }

    fn dim(&self) -> Option<usize> {
        self.dim.or_else(|| self.engine.dim())
    }
}

impl<E: EmbedImages> EmbedImages for WithPolicy<E> {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        self.engine.embed_image_chunk(images)
    }
}

impl<E: RecognizeMedia> RecognizeMedia for WithPolicy<E> {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        self.engine.recognize_chunk(items)
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }
}

// ── the adapter: sync compute onto the owned runtime (#374 C) ───────────────

/// Route-1 engines become kernel-facing async engines here: each call moves
/// the chunk loop onto the runtime's blocking pool via
/// `tokio::task::spawn_blocking`. **Eager by contract**: the work is
/// scheduled inside `embed()` itself, before the returned future is first
/// polled — that is what lets the kernel build engine futures ahead of
/// lexical/sibling work and genuinely overlap them (the #342 search/add
/// overlap properties).
///
/// Must be called in runtime context (kernel ops are — the action-exchange
/// edge spawns every op onto the kernel runtime); calling an adapted engine
/// off-runtime is a contract violation and panics fast in tokio.
///
/// Dropping a returned future detaches it (a `JoinHandle` drop never aborts
/// the blocking task) — wasted compute at worst, consistent with the edge's
/// detach semantics.
pub struct Blocking<E>(pub Arc<E>);

fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> BoxFuture<'static, NativeResult<T>> {
    let handle = tokio::task::spawn_blocking(work);
    Box::pin(async move {
        handle
            .await
            .map_err(|e| NativeError::internal(format!("blocking engine task failed: {e}")))?
    })
}

impl<E: EmbedText + 'static> Embedder for Blocking<E> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.0);
        run_blocking(move || {
            let chunk = engine.safe_batch().max(1);
            let mut out = Vec::with_capacity(texts.len());
            for piece in texts.chunks(chunk) {
                out.extend(engine.embed_chunk(piece)?);
            }
            Ok(out)
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.0.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.0.dim()
    }
}

impl<E: EmbedImages + 'static> ImageEmbedder for Blocking<E> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.0);
        run_blocking(move || engine.embed_image_chunk(&images))
    }
}

impl<E: RecognizeMedia + 'static> Recognizer for Blocking<E> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        let engine = Arc::clone(&self.0);
        run_blocking(move || engine.recognize_chunk(&items))
    }

    fn fingerprint(&self) -> Option<String> {
        self.0.fingerprint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Toy {
        batch_cap: usize,
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl EmbedText for Toy {
        fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        }

        fn safe_batch(&self) -> usize {
            self.batch_cap
        }

        fn fingerprint(&self) -> Option<String> {
            Some("toy:v1".into())
        }
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap()
    }

    #[test]
    fn blocking_adapter_chunks_by_safe_batch_and_preserves_order() {
        let toy = Arc::new(Toy {
            batch_cap: 2,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = Blocking(Arc::clone(&toy));
        let texts: Vec<String> = ["a", "bb", "ccc", "dddd", "eeeee"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rt = test_runtime();
        let _guard = rt.enter();
        let out = rt.block_on(adapted.embed(texts)).unwrap();
        assert_eq!(
            out,
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            "order mirrors input across chunks"
        );
        assert_eq!(
            *toy.calls.lock().unwrap(),
            vec![2, 2, 1],
            "split by safe_batch"
        );
        assert_eq!(adapted.fingerprint().as_deref(), Some("toy:v1"));
    }

    /// The eager-embed pin (#374 C): the blocking task is scheduled inside
    /// `embed()` itself — observable as the engine running WITHOUT the
    /// returned future ever being polled. The #342 overlap properties
    /// (search embed ∥ lexical reads; orchestrator try_join) depend on this.
    #[test]
    fn blocking_embed_is_eager() {
        let rt = test_runtime();
        let _guard = rt.enter();
        let toy = Arc::new(Toy {
            batch_cap: 8,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = Blocking(Arc::clone(&toy));
        let fut = adapted.embed(vec!["scheduled before any poll".into()]);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while toy.calls.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            !toy.calls.lock().unwrap().is_empty(),
            "the engine ran without the future being polled (eager scheduling)"
        );
        drop(fut); // and dropping the unpolled future detached, not panicked
    }

    #[test]
    fn with_policy_overrides_identity_and_batch() {
        let toy = Arc::new(Toy {
            batch_cap: 64, // the engine's own answer — WithPolicy must win
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let tuned = WithPolicy::new(
            Arc::clone(&toy),
            Some("host:fp:textprep=3".into()),
            Some(384),
            2,
        );
        assert_eq!(tuned.safe_batch(), 2);
        assert_eq!(tuned.fingerprint().as_deref(), Some("host:fp:textprep=3"));
        assert_eq!(tuned.dim(), Some(384));
        // The adapter chunks by the POLICY batch, not the engine's.
        let adapted = Blocking(Arc::new(tuned));
        let texts: Vec<String> = ["a", "bb", "ccc"].iter().map(|s| s.to_string()).collect();
        let rt = test_runtime();
        let _guard = rt.enter();
        let out = rt.block_on(adapted.embed(texts)).unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(*toy.calls.lock().unwrap(), vec![2, 1]);
        // safe_batch is floored at 1 (a zero would loop forever).
        assert_eq!(WithPolicy::new(toy, None, None, 0).safe_batch(), 1);
    }

    #[test]
    fn media_item_mime_derivation() {
        assert_eq!(
            MediaItem::from_named("diagram.PNG", vec![1])
                .mime
                .as_deref(),
            Some("image/png")
        );
        assert_eq!(mime_for_name("clip.m4a").as_deref(), Some("audio/mp4"));
        assert_eq!(mime_for_name("noext"), None);
        assert_eq!(MediaItem::untyped(vec![2]).mime, None);
    }

    #[test]
    fn recognition_serde_round_trips_and_omits_absent_bbox() {
        let r = Recognition {
            text: "label".into(),
            confidence: 0.8,
            segments: vec![Segment {
                text: "label".into(),
                confidence: 0.8,
                bbox: Some([0.1, 0.2, 0.3, 0.05]),
            }],
        };
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Recognition>(&json).unwrap(), r);
        let no_box = serde_json::to_string(&Segment {
            text: "t".into(),
            confidence: 1.0,
            bbox: None,
        })
        .unwrap();
        assert!(!no_box.contains("bbox"));
    }
}
