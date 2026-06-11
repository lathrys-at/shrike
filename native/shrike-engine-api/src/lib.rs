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
//!   futures, no threads, no executors, assuming *nothing* about execution.
//!   The host bridges to the async traits with an adapter at composition
//!   time: [`Inline`] (compute on whatever thread polls — the C host's
//!   calling-thread model) or [`OnExecutor`] (submit each chunk to a
//!   host-injected [`ComputeExecutor`] lane). The adapter owns the batch
//!   loop; batching is execution policy, not engine compute.
//! - **Route 2 — naturally-async engines** (a completion-handler platform
//!   API reached through ObjC/Swift glue; an async HTTP client): implement
//!   the async traits directly — the future suspends and completes from the
//!   engine's own completion source (callback → oneshot → waker). No ambient
//!   runtime assumed, no thread parked. Where the underlying API is async,
//!   this is the preferred shape: a blocking lane would waste a thread
//!   waiting on a callback.
//!
//! # Execution is the host's, topology is the kernel's
//!
//! Pipeline *topology* — what must order before what — is the kernel's
//! consistency model. Execution *capacity and placement* are host facts,
//! handed over through adapter composition: the host assigns a
//! [`ComputeExecutor`] lane per engine (two engines sharing one GPU get the
//! same lane; a remote engine gets a wide one; a mobile host maps lanes onto
//! its own queues). Engines spawn no threads and assume no runtime — the
//! same injected-scheduling principle as the kernel's `SerialExecutor`. An
//! engine future must never submit to the kernel's collection executor
//! (re-entrancy is a deadlock by contract).
//!
//! # Errors
//!
//! A dependency or availability failure (model missing, service down,
//! platform API unavailable) is `NativeError::unavailable`; a contract
//! violation (malformed payload, wrong arity) is `internal`.

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

impl<T: Embedder> Embedder for Arc<T> {
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

impl<T: ImageEmbedder> ImageEmbedder for Arc<T> {
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

impl<T: Recognizer> Recognizer for Arc<T> {
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

impl<T: EmbedText> EmbedText for Arc<T> {
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

impl<T: EmbedImages> EmbedImages for Arc<T> {
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

impl<T: RecognizeMedia> RecognizeMedia for Arc<T> {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        (**self).recognize_chunk(items)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }
}

// ── adapters: execution policy, composed by the host ────────────────────────

/// A host-injected execution lane for engine compute: `submit` schedules the
/// job somewhere of the host's choosing and returns a runtime-agnostic
/// future for its completion — the `SerialExecutor` shape *without* the
/// serialization requirement. Submissions may be in flight concurrently
/// (one completion per submission; an implementation must not serialize
/// internally unless the lane's whole point is to serialize, e.g. one GPU).
pub trait ComputeExecutor: Send + Sync {
    fn submit(&self, job: Box<dyn FnOnce() + Send>) -> BoxFuture<'static, NativeResult<()>>;
}

/// The inline conformer: runs the job on the calling/polling thread. The C
/// host's calling-thread model; also the test default.
pub struct InlineComputeExecutor;

impl ComputeExecutor for InlineComputeExecutor {
    fn submit(&self, job: Box<dyn FnOnce() + Send>) -> BoxFuture<'static, NativeResult<()>> {
        job();
        Box::pin(futures::future::ready(Ok(())))
    }
}

/// Route-1 adapter: ready futures, compute on whatever thread polls.
pub struct Inline<E>(pub E);

/// Route-1 adapter: each call submits ONE job to the injected
/// [`ComputeExecutor`] lane; the adapter-owned batch loop (splitting the
/// kernel's batch by the engine's `safe_batch`) runs inside that job, so
/// chunk-to-chunk order is trivially preserved (output order mirrors input
/// order) and the lane sees one submission per kernel call. *Cross-call*
/// concurrency is the lane's business (the [`ComputeExecutor`] contract
/// requires concurrent in-flight submissions).
pub struct OnExecutor<E> {
    engine: Arc<E>,
    lane: Arc<dyn ComputeExecutor>,
}

impl<E> OnExecutor<E> {
    pub fn new(engine: Arc<E>, lane: Arc<dyn ComputeExecutor>) -> Self {
        Self { engine, lane }
    }
}

fn chunked<T: Clone, R>(
    items: &[T],
    chunk: usize,
    mut f: impl FnMut(&[T]) -> NativeResult<Vec<R>>,
) -> NativeResult<Vec<R>> {
    let chunk = chunk.max(1);
    let mut out = Vec::with_capacity(items.len());
    for piece in items.chunks(chunk) {
        out.extend(f(piece)?);
    }
    Ok(out)
}

impl<E: EmbedText> Embedder for Inline<E> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let result = chunked(&texts, self.0.safe_batch(), |c| self.0.embed_chunk(c));
        Box::pin(futures::future::ready(result))
    }

    fn fingerprint(&self) -> Option<String> {
        self.0.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.0.dim()
    }
}

impl<E: EmbedImages + 'static> ImageEmbedder for Inline<E> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let result = chunked(&images, usize::MAX, |c| self.0.embed_image_chunk(c));
        Box::pin(futures::future::ready(result))
    }
}

impl<E: RecognizeMedia> Recognizer for Inline<E> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        let result = self.0.recognize_chunk(&items);
        Box::pin(futures::future::ready(result))
    }

    fn fingerprint(&self) -> Option<String> {
        self.0.fingerprint()
    }
}

/// Run `compute` on the lane, delivering its output through a oneshot. The
/// generic-payload helper both `OnExecutor` impls share: the job closure
/// owns the inputs and the sender; the returned future joins the lane's
/// completion with the payload.
fn run_on_lane<R: Send + 'static>(
    lane: &Arc<dyn ComputeExecutor>,
    compute: impl FnOnce() -> NativeResult<R> + Send + 'static,
) -> BoxFuture<'static, NativeResult<R>> {
    let (tx, rx) = futures::channel::oneshot::channel::<NativeResult<R>>();
    let submitted = lane.submit(Box::new(move || {
        let _ = tx.send(compute());
    }));
    Box::pin(async move {
        submitted.await?;
        rx.await
            .map_err(|_| NativeError::internal("compute lane dropped the job"))?
    })
}

impl<E: EmbedText> Embedder for OnExecutor<E> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.engine);
        run_on_lane(&self.lane, move || {
            chunked(&texts, engine.safe_batch(), |c| engine.embed_chunk(c))
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.engine.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.engine.dim()
    }
}

impl<E: EmbedImages> ImageEmbedder for OnExecutor<E> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.engine);
        run_on_lane(&self.lane, move || engine.embed_image_chunk(&images))
    }
}

impl<E: RecognizeMedia> Recognizer for OnExecutor<E> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        let engine = Arc::clone(&self.engine);
        run_on_lane(&self.lane, move || engine.recognize_chunk(&items))
    }

    fn fingerprint(&self) -> Option<String> {
        self.engine.fingerprint()
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

    #[test]
    fn adapters_chunk_by_safe_batch_and_preserve_order() {
        let toy = Arc::new(Toy {
            batch_cap: 2,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = OnExecutor::new(Arc::clone(&toy), Arc::new(InlineComputeExecutor));
        let texts: Vec<String> = ["a", "bb", "ccc", "dddd", "eeeee"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = futures::executor::block_on(adapted.embed(texts)).unwrap();
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

    #[test]
    fn inline_adapter_matches_on_executor() {
        let toy = Arc::new(Toy {
            batch_cap: 3,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let inline = Inline(Arc::clone(&toy));
        let texts: Vec<String> = ["x", "yy"].iter().map(|s| s.to_string()).collect();
        let out = futures::executor::block_on(inline.embed(texts)).unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0]]);
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
