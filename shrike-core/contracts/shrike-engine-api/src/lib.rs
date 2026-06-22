//! The engine contract: what the kernel consumes, what engine crates
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
//!   runtime's blocking pool: an eager `spawn_blocking` with the
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
//! kernel. Execution lives on the kernel's owned tokio runtime:
//! sync engines ride the blocking pool through [`Blocking`], async engines
//! complete from their own sources. Engines spawn no threads themselves and
//! never block a runtime worker.
//!
//! # Errors
//!
//! A dependency or availability failure (model missing, service down,
//! platform API unavailable) is `NativeError::unavailable`; a contract
//! violation (malformed payload, wrong arity) is `internal`.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

pub mod probe;

use std::sync::Arc;

use futures::channel::oneshot;
use futures::future::BoxFuture;

use shrike_error::{NativeError, NativeResult};

// ── media items ──────────────────────────────────────────────────────────────

/// One media payload with its type hint. `mime` comes from the media handler
/// (derived from the filename — the resolver has the name when it reads the
/// bytes), so engines that route by media kind (an omni recognizer: image →
/// OCR vs audio → ASR) or whose decoder wants a format hint get it for free.
/// It is a *hint*: decoders may still sniff magic bytes as the fallback, and
/// `None` is always legal.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaItem {
    /// The raw media payload.
    pub bytes: Vec<u8>,
    /// MIME hint (a filename-derived guess; `None` is always legal).
    pub mime: Option<String>,
}

impl MediaItem {
    /// A media item with a MIME hint.
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
///
/// DELIBERATELY DISTINCT from `shrike_media::guess_mime`/`mime_extension`:
/// this is the engine routing-HINT (it carries `heic`/`aiff` an engine
/// may route on, omits store/response kinds like `pdf`/`txt`/`css`), while
/// shrike-media's tables are the store/response MIME the media write/fetch
/// paths serve. Keeping them apart is what keeps shrike-engine-api a LEAF (no
/// dep on shrike-media → no media-fetch/SSRF dependency in the engine
/// contract). Do NOT "consolidate" them into one table — the leaf rule
/// outranks table-count==1 (Chesterton's fence).
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
        "aiff" | "aif" => "audio/aiff",
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
    ///
    /// The call MUST eventually resolve — `Ok` or `Err` in bounded time. The
    /// kernel awaits `embed` on the single-flight ingest drain with no
    /// per-embed timeout and trusts every attached embedder to honor this. A
    /// future that never resolves (and never errors) wedges the sole writer
    /// permanently: the drain watermark never advances, and `flush`/`shutdown`
    /// block behind it. Every shipping backend honors the contract — bounded
    /// local compute, or a bounded transport timeout plus retry — and a custom
    /// embedder must too.
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

/// L2-normalize `v` to unit length in place — a no-op (to within fp error) on an
/// already-unit vector; a zero vector is left unchanged (no divide by zero).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// An [`Embedder`] decorator that L2-normalizes every output vector to unit length.
///
/// Wrapping the embedder at the engine boundary means EVERY vector the kernel and
/// vector index see — stored notes AND queries — is already unit, so the index uses
/// the cheaper inner-product metric (equal to cosine on unit vectors, but without
/// the per-comparison vector norms) and nothing downstream re-normalizes. A backend
/// that already emits unit vectors (the ONNX text/CLIP paths) sees a near-no-op; one
/// that does not (a remote service) is made conformant here, in one place.
pub struct NormalizingEmbedder(pub Arc<dyn Embedder>);

impl Embedder for NormalizingEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move {
            let mut vectors = self.0.embed(texts).await?;
            for v in &mut vectors {
                l2_normalize(v);
            }
            Ok(vectors)
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.0.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.0.dim()
    }
}

/// The [`ImageEmbedder`] counterpart of [`NormalizingEmbedder`] — unit-normalizes
/// every image vector at the engine boundary.
pub struct NormalizingImageEmbedder(pub Box<dyn ImageEmbedder>);

impl ImageEmbedder for NormalizingImageEmbedder {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move {
            let mut vectors = self.0.embed_images(images).await?;
            for v in &mut vectors {
                l2_normalize(v);
            }
            Ok(vectors)
        })
    }
}

/// Media access the host injects: read bytes lazily, stat cheaply. The
/// per-note fingerprint hashes *names of present media* via `exists` (no
/// byte read); bytes are read only for items actually being processed.
pub trait ImageResolver: Send + Sync {
    /// Read a media file's bytes by name, or `None` if it is unresolvable.
    fn read(&self, name: &str) -> Option<Vec<u8>>;
    /// Whether a media file is present (a cheap stat, no byte read).
    fn exists(&self, name: &str) -> bool;
}

// ── recognition ───────────────────────────────────────────────────────

/// Where a segment sits in its medium: a normalized top-left `[x, y, w, h]`
/// box for OCR, or a `[start_seconds, duration_seconds]` time span for ASR.
/// One enum, not two optionals — a segment can't carry both, and
/// the type makes that unrepresentable. The flattened lowercase tag keeps
/// the wire identical to the historical shape (`"bbox": [...]`), so
/// existing derived rows parse unchanged and the span variant joins as
/// `"span": [...]`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Locator {
    /// Normalized OCR box `[x, y, w, h]`.
    Bbox([f64; 4]),
    /// ASR time span `[start_seconds, duration_seconds]`.
    Span([f64; 2]),
}

/// One recognized segment: a line/word for OCR, a stretch of speech for
/// ASR — with the locator that fits the medium, or none at all (absent
/// locators stay off the wire entirely).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    /// The segment's recognized text.
    pub text: String,
    /// Engine-defined confidence for this segment.
    pub confidence: f64,
    /// Where the segment sits in its medium, if known.
    #[serde(flatten)]
    pub locator: Option<Locator>,
}

/// One media item's recognition: the flattened text (reading order), the
/// overall confidence (engine-defined aggregate), and the retained segments
/// (the one-pass/many-consumers rule: never flatten-and-discard).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Recognition {
    /// The flattened text in reading order.
    pub text: String,
    /// Engine-defined aggregate confidence for the item.
    pub confidence: f64,
    /// The retained per-segment structure (never flatten-and-discarded).
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
    /// Embed a chunk of at most [`EmbedText::safe_batch`] texts, order-preserving.
    ///
    /// # Errors
    ///
    /// Returns the engine's error if the compute fails (model unavailable,
    /// malformed input).
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>>;

    /// The largest batch this engine is proven safe to embed in one call
    /// (batch-variance probing, e.g. int8 dynamic quantization). 1 = serial.
    fn safe_batch(&self) -> usize {
        1
    }

    /// The engine's model fingerprint, or `None` if it has none.
    fn fingerprint(&self) -> Option<String> {
        None
    }

    /// The embedding dimensionality, if known up front.
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
    /// Embed a chunk of at most [`EmbedImages::safe_batch`] images,
    /// order-preserving.
    ///
    /// # Errors
    ///
    /// Returns the engine's error if the compute fails (model unavailable,
    /// undecodable image).
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>>;

    /// The largest image batch this engine is proven safe to embed in one
    /// call — the vision analogue of [`EmbedText::safe_batch`]. An int8
    /// vision graph that batches non-deterministically must be capped here so
    /// a note's image vector stays a pure function of its own image (the
    /// `reconcile`==rebuild invariant for image vectors). 1 = serial.
    fn safe_batch(&self) -> usize {
        1
    }
}

impl<T: EmbedImages + ?Sized> EmbedImages for Arc<T> {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        (**self).embed_image_chunk(images)
    }

    fn safe_batch(&self) -> usize {
        (**self).safe_batch()
    }
}

/// Chunk-level media recognition, pure compute.
pub trait RecognizeMedia: Send + Sync + 'static {
    /// Recognize a chunk of media items, order-preserving.
    ///
    /// # Errors
    ///
    /// Returns the engine's error if the compute fails (engine unavailable).
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>>;

    /// The engine's model fingerprint, or `None` if it has none.
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
    /// Wrap `engine` with its precomputed policy (fingerprint, dim, proven
    /// safe batch size — floored at 1).
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

    fn safe_batch(&self) -> usize {
        self.safe_batch
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

/// Host-assembled identity + batch policy over a route-2 ASYNC engine.
/// The async sibling of [`WithPolicy`]: the host injects the fingerprint/dim the
/// engine can't know (text-prep versions, the describe prompt version, a probed
/// dim) AND the proven-safe text batch size — exactly the three knobs sync
/// `WithPolicy` carries, minus the [`Blocking`] adapter that consumed them. A
/// route-2 engine does its own IO; this wrapper owns the same text-chunking the
/// `Blocking` adapter did (one engine `embed` call per `batch_size` chunk, in
/// order), so the host's probed batch governs request size on the async path
/// too. `recognize`/`embed_images` delegate unchanged (recognition has no batch
/// knob; image embeds chunk per-item inside the engine). Wrap the engine
/// and hand the result straight to the kernel slot — no adapter in between.
pub struct AsyncWithPolicy<E> {
    engine: Arc<E>,
    fingerprint: Option<String>,
    dim: Option<usize>,
    batch_size: usize,
}

impl<E> AsyncWithPolicy<E> {
    /// Wrap `engine` with its host-assembled policy (fingerprint, dim, proven
    /// safe text batch — floored at 1). `batch_size` chunks the text path
    /// exactly as [`WithPolicy`]'s `safe_batch` did under [`Blocking`].
    pub fn new(
        engine: Arc<E>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        batch_size: usize,
    ) -> Self {
        Self {
            engine,
            fingerprint,
            dim,
            batch_size: batch_size.max(1),
        }
    }
}

impl<E: Embedder> Embedder for AsyncWithPolicy<E> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let chunk = self.batch_size;
        Box::pin(async move {
            if texts.len() <= chunk {
                return self.engine.embed(texts).await;
            }
            let mut out = Vec::with_capacity(texts.len());
            // Owned chunks so each engine future is independent of `texts`.
            let mut iter = texts.into_iter().peekable();
            while iter.peek().is_some() {
                let piece: Vec<String> = iter.by_ref().take(chunk).collect();
                out.extend(self.engine.embed(piece).await?);
            }
            Ok(out)
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }

    fn dim(&self) -> Option<usize> {
        self.dim.or_else(|| self.engine.dim())
    }
}

impl<E: ImageEmbedder> ImageEmbedder for AsyncWithPolicy<E> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        self.engine.embed_images(images)
    }
}

impl<E: Recognizer> Recognizer for AsyncWithPolicy<E> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        self.engine.recognize(items)
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }
}

// ── the adapter: sync compute onto a blocking pool ─────────────────

/// Where [`Blocking`] runs a unit of sync engine compute.
///
/// The adapter is told *that* it should hand work to a blocking pool but never
/// *which* one — the host injects the pool. A `submit` schedules the job
/// **eagerly**: it must run (or be queued to run) by the time `submit` returns,
/// so the future [`Blocking`] hands back is already in flight before its first
/// poll. The kernel relies on this to build engine futures ahead of sibling
/// work and overlap them (the search/add overlap property).
///
/// The job is the runtime-agnostic [`tokio::task`]-style erased closure: it
/// owns the channel that carries its result back to the awaiting future, so the
/// dispatcher need only run it.
pub trait BlockingDispatch: Send + Sync {
    /// Schedule `job` on the pool, eagerly. The job runs to completion exactly
    /// once and delivers its own result.
    fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>);
}

/// The **standalone, no-kernel** dispatcher [`Blocking`] uses when the host
/// injects none: tokio's blocking pool via [`tokio::task::spawn_blocking`]. It is
/// independent of the kernel runtime — engine-api has no dependency on it. The
/// production path with the kernel always injects its own `BlockingDispatch`
/// (the binding's `KernelDispatch` → the committed `drive_compute` pool), so this
/// default serves only the compute-only build (no kernel to inject) and this
/// crate's own standalone tests. Requires an ambient tokio runtime; off-runtime
/// it panics, the contract the adapter has always had.
pub struct DefaultDispatch;

impl BlockingDispatch for DefaultDispatch {
    fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        tokio::task::spawn_blocking(job);
    }
}

/// Route-1 engines become kernel-facing async engines here: each call moves the
/// chunk loop onto a blocking pool and returns a future of the result.
/// **Eager by contract**: the work is scheduled inside the call itself, before
/// the returned future is first polled — that is what lets the kernel build
/// engine futures ahead of lexical/sibling work and genuinely overlap them (the
/// search/add overlap properties).
///
/// The pool is the injected [`BlockingDispatch`], or [`DefaultDispatch`]
/// (tokio's blocking pool) when none is given. The host wires its own pool — a
/// committed compute pool sized to its cores — at the engine's construction
/// site.
///
/// Dropping a returned future detaches it (the result channel closes; the job
/// runs to completion anyway) — wasted compute at worst, consistent with the
/// action-exchange edge's detach semantics.
pub struct Blocking<E> {
    engine: Arc<E>,
    dispatch: Arc<dyn BlockingDispatch>,
}

impl<E> Blocking<E> {
    /// Adapt `engine` over the default blocking pool ([`DefaultDispatch`]).
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            dispatch: Arc::new(DefaultDispatch),
        }
    }

    /// Adapt `engine` over a host-injected blocking pool.
    pub fn with_dispatch(engine: Arc<E>, dispatch: Arc<dyn BlockingDispatch>) -> Self {
        Self { engine, dispatch }
    }
}

/// Run `work` on `dispatch`'s pool, returning an eagerly-scheduled future of its
/// result. The result rides a oneshot the job owns; awaiting the returned future
/// yields it (or an internal error if the pool dropped the job — a vanished
/// pool, i.e. host shutdown).
fn run_blocking<T: Send + 'static>(
    dispatch: &Arc<dyn BlockingDispatch>,
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> BoxFuture<'static, NativeResult<T>> {
    let (tx, rx) = oneshot::channel();
    dispatch.submit(Box::new(move || {
        let _ = tx.send(work());
    }));
    Box::pin(async move {
        rx.await
            .map_err(|_| NativeError::internal("the blocking pool dropped an engine job"))?
    })
}

impl<E: EmbedText + 'static> Embedder for Blocking<E> {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.engine);
        run_blocking(&self.dispatch, move || {
            let chunk = engine.safe_batch().max(1);
            let mut out = Vec::with_capacity(texts.len());
            for piece in texts.chunks(chunk) {
                out.extend(engine.embed_chunk(piece)?);
            }
            Ok(out)
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.engine.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.engine.dim()
    }
}

impl<E: EmbedImages + 'static> ImageEmbedder for Blocking<E> {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let engine = Arc::clone(&self.engine);
        run_blocking(&self.dispatch, move || {
            // Chunk by the probed vision safe_batch, exactly like the
            // text path — a batch-variant int8 vision graph embeds serially so
            // an image vector never depends on its batch-mates.
            let chunk = engine.safe_batch().max(1);
            let mut out = Vec::with_capacity(images.len());
            for piece in images.chunks(chunk) {
                out.extend(engine.embed_image_chunk(piece)?);
            }
            Ok(out)
        })
    }
}

impl<E: RecognizeMedia + 'static> Recognizer for Blocking<E> {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        let engine = Arc::clone(&self.engine);
        run_blocking(&self.dispatch, move || engine.recognize_chunk(&items))
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

    /// The image analogue of `Toy`, for the vision-batching pin.
    struct ImageToy {
        batch_cap: usize,
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl EmbedImages for ImageToy {
        fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(images.len());
            Ok(images
                .iter()
                .map(|im| vec![im.bytes.len() as f32])
                .collect())
        }

        fn safe_batch(&self) -> usize {
            self.batch_cap
        }
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        // A current_thread runtime: `DefaultDispatch` only needs an ambient
        // runtime for `spawn_blocking` (the blocking pool is separate from the
        // worker), and shrike-core uses no multi-thread runtime anywhere.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn blocking_adapter_chunks_by_safe_batch_and_preserves_order() {
        let toy = Arc::new(Toy {
            batch_cap: 2,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = Blocking::new(Arc::clone(&toy));
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

    /// The vision path chunks by the (image) safe_batch too: a
    /// batch-variant int8 vision graph probed to 1 must embed images serially
    /// on the kernel path, not all-in-one — else the probe's verdict is inert
    /// where the reconcile==rebuild invariant for image vectors actually lives.
    #[test]
    fn blocking_adapter_chunks_images_by_safe_batch_and_preserves_order() {
        let toy = Arc::new(ImageToy {
            batch_cap: 2,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = Blocking::new(Arc::clone(&toy));
        // Distinct byte lengths so the per-image vector is identifiable.
        let images: Vec<MediaItem> = [1usize, 2, 3, 4, 5]
            .iter()
            .map(|&n| MediaItem::untyped(vec![0u8; n]))
            .collect();
        let rt = test_runtime();
        let _guard = rt.enter();
        let out = rt.block_on(adapted.embed_images(images)).unwrap();
        assert_eq!(
            out,
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            "image order mirrors input across chunks"
        );
        assert_eq!(
            *toy.calls.lock().unwrap(),
            vec![2, 2, 1],
            "images split by the vision safe_batch"
        );
    }

    struct NonUnitEmbedder;
    impl Embedder for NonUnitEmbedder {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            // One non-unit vector per text: [3, 4] has norm 5.
            Box::pin(async move { Ok(texts.iter().map(|_| vec![3.0f32, 4.0]).collect()) })
        }
        fn fingerprint(&self) -> Option<String> {
            Some("nonunit:v1".into())
        }
    }

    #[test]
    fn l2_normalize_makes_unit_and_leaves_zero() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0f32, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0], "a zero vector is left unchanged");
    }

    #[test]
    fn normalizing_embedder_unit_normalizes_output_and_delegates() {
        let wrapped = NormalizingEmbedder(Arc::new(NonUnitEmbedder) as Arc<dyn Embedder>);
        let rt = test_runtime();
        let _guard = rt.enter();
        let out = rt
            .block_on(wrapped.embed(vec!["a".into(), "b".into()]))
            .unwrap();
        for v in &out {
            let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "every output vector is unit, got {norm}"
            );
        }
        assert!((out[0][0] - 0.6).abs() < 1e-6 && (out[0][1] - 0.8).abs() < 1e-6);
        // Identity methods delegate to the inner embedder.
        assert_eq!(wrapped.fingerprint().as_deref(), Some("nonunit:v1"));
    }

    /// safe_batch=1 (a probed batch-variant vision graph) embeds every image
    /// alone — the kernel-path enforcement of the mixed-precision guard.
    #[test]
    fn variant_vision_safe_batch_embeds_images_serially() {
        let toy = Arc::new(WithPolicy::new(
            Arc::new(ImageToy {
                batch_cap: 64, // the engine's own answer — WithPolicy(1) must win
                calls: std::sync::Mutex::new(Vec::new()),
            }),
            None,
            None,
            1, // min(text, vision) collapsed to serial for the variant vision graph
        ));
        let adapted = Blocking::new(toy);
        let images: Vec<MediaItem> = (0..3).map(|_| MediaItem::untyped(vec![0u8; 4])).collect();
        let rt = test_runtime();
        let _guard = rt.enter();
        let _ = rt.block_on(adapted.embed_images(images)).unwrap();
        assert_eq!(
            *adapted.engine.engine.calls.lock().unwrap(),
            vec![1, 1, 1],
            "each image embedded alone"
        );
    }

    /// The eager-embed pin: the blocking task is scheduled inside
    /// `embed()` itself — observable as the engine running WITHOUT the
    /// returned future ever being polled. The overlap properties
    /// (search embed ∥ lexical reads; orchestrator try_join) depend on this.
    #[test]
    fn blocking_embed_is_eager() {
        let rt = test_runtime();
        let _guard = rt.enter();
        let toy = Arc::new(Toy {
            batch_cap: 8,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let adapted = Blocking::new(Arc::clone(&toy));
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

    /// A dispatcher that records each submission and runs the job inline, so a
    /// test can observe both that `submit` was called and that the result still
    /// plumbs back through the returned future.
    struct RecordingDispatch {
        submits: Arc<std::sync::Mutex<usize>>,
    }

    impl BlockingDispatch for RecordingDispatch {
        fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>) {
            *self.submits.lock().unwrap() += 1;
            job();
        }
    }

    /// An injected dispatcher is used (its `submit` fires eagerly, inside the
    /// `embed` call) and the result plumbs back unchanged — the same output the
    /// default dispatcher produces, so swapping the host pool is behaviour-
    /// preserving.
    #[test]
    fn injected_dispatch_is_used_eagerly_and_preserves_results() {
        let toy = Arc::new(Toy {
            batch_cap: 2,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let submits = Arc::new(std::sync::Mutex::new(0usize));
        let dispatch: Arc<dyn BlockingDispatch> = Arc::new(RecordingDispatch {
            submits: Arc::clone(&submits),
        });
        let adapted = Blocking::with_dispatch(Arc::clone(&toy), dispatch);
        let texts: Vec<String> = ["a", "bb", "ccc", "dddd", "eeeee"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // `submit` fires during `embed`, before the future is ever polled.
        let fut = adapted.embed(texts);
        assert_eq!(
            *submits.lock().unwrap(),
            1,
            "the injected dispatcher's submit ran eagerly, inside embed"
        );

        let rt = test_runtime();
        let out = rt.block_on(fut).unwrap();
        assert_eq!(
            out,
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            "the injected pool yields the same result as the default — order preserved across chunks"
        );
        assert_eq!(
            *toy.calls.lock().unwrap(),
            vec![2, 2, 1],
            "the chunk loop still runs unchanged on the injected pool"
        );
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
        let adapted = Blocking::new(Arc::new(tuned));
        let texts: Vec<String> = ["a", "bb", "ccc"].iter().map(|s| s.to_string()).collect();
        let rt = test_runtime();
        let _guard = rt.enter();
        let out = rt.block_on(adapted.embed(texts)).unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(*toy.calls.lock().unwrap(), vec![2, 1]);
        // safe_batch is floored at 1 (a zero would loop forever).
        assert_eq!(WithPolicy::new(toy, None, None, 0).safe_batch(), 1);
    }

    /// A route-2 async engine: it owns its own IO (here trivial) and returns a
    /// future directly, the `RemoteEmbedder` shape `AsyncWithPolicy` wraps. It
    /// records each `embed` call's batch size so the wrapper's chunking is
    /// observable.
    #[derive(Default)]
    struct AsyncToy {
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl Embedder for AsyncToy {
        fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            self.calls.lock().unwrap().push(texts.len());
            Box::pin(async move { Ok(texts.iter().map(|t| vec![t.len() as f32]).collect()) })
        }

        fn fingerprint(&self) -> Option<String> {
            Some("async-toy:engine".into())
        }

        fn dim(&self) -> Option<usize> {
            Some(7)
        }
    }

    /// `AsyncWithPolicy` overrides the host-injected identity (fingerprint/dim),
    /// chunks the text path by the host `batch_size`, and falls back to the
    /// engine's own dim when the host pins none — the async sibling of
    /// `with_policy_overrides_identity_and_batch`.
    #[test]
    fn async_with_policy_overrides_identity_and_chunks_by_batch() {
        let toy = Arc::new(AsyncToy::default());
        let tuned = AsyncWithPolicy::new(
            Arc::clone(&toy),
            Some("host:fp:textprep=3".into()),
            Some(384),
            2,
        );
        // The host policy wins over the engine's own answers.
        assert_eq!(tuned.fingerprint().as_deref(), Some("host:fp:textprep=3"));
        assert_eq!(tuned.dim(), Some(384));
        // embed chunks by batch_size (2): 5 inputs → 2,2,1 — order preserved.
        let rt = test_runtime();
        let _guard = rt.enter();
        let texts: Vec<String> = ["a", "bb", "ccc", "dddd", "eeeee"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = rt.block_on(tuned.embed(texts)).unwrap();
        assert_eq!(
            out,
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            "order mirrors input across chunks"
        );
        assert_eq!(
            *toy.calls.lock().unwrap(),
            vec![2, 2, 1],
            "split by batch_size"
        );

        // dim falls back to the engine's when the host pins none; batch_size 0
        // floors to 1 (no zero-length chunk loop).
        let bare = AsyncWithPolicy::new(Arc::new(AsyncToy::default()), None, None, 0);
        assert_eq!(bare.dim(), Some(7));
        assert_eq!(bare.fingerprint(), None);
        let single = Arc::new(AsyncToy::default());
        let one = AsyncWithPolicy::new(Arc::clone(&single), None, None, 0);
        let _ = rt
            .block_on(one.embed(vec!["a".into(), "bb".into()]))
            .unwrap();
        assert_eq!(
            *single.calls.lock().unwrap(),
            vec![1, 1],
            "batch_size 0 → serial"
        );
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
    fn recognition_serde_round_trips_and_omits_absent_locators() {
        let r = Recognition {
            text: "label".into(),
            confidence: 0.8,
            segments: vec![
                Segment {
                    text: "label".into(),
                    confidence: 0.8,
                    locator: Some(Locator::Bbox([0.1, 0.2, 0.3, 0.05])),
                },
                // The ASR shape: a time span, not a box.
                Segment {
                    text: "spoken".into(),
                    confidence: 0.9,
                    locator: Some(Locator::Span([1.25, 0.75])),
                },
            ],
        };
        let json = serde_json::to_string(&r).unwrap();
        // The flattened tag keeps the historical wire keys.
        assert!(json.contains(r#""bbox":[0.1"#) && json.contains(r#""span":[1.25"#));
        assert_eq!(serde_json::from_str::<Recognition>(&json).unwrap(), r);
        let bare = serde_json::to_string(&Segment {
            text: "t".into(),
            confidence: 1.0,
            locator: None,
        })
        .unwrap();
        assert!(!bare.contains("bbox") && !bare.contains("span"));
        // Older rows parse unchanged: bare segments and bbox'd segments.
        let old: Segment = serde_json::from_str(r#"{"text":"t","confidence":1.0}"#).unwrap();
        assert_eq!(old.locator, None);
        let boxed: Segment =
            serde_json::from_str(r#"{"text":"t","confidence":1.0,"bbox":[0.0,0.1,0.2,0.3]}"#)
                .unwrap();
        assert_eq!(boxed.locator, Some(Locator::Bbox([0.0, 0.1, 0.2, 0.3])));
    }
}
