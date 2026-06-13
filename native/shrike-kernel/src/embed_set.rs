//! The embed slot generalized from ONE service to an ordered SET of embedding
//! spaces (#233, the multi-space substrate's foundation — #229).
//!
//! Each [`EmbedSpace`] wraps an [`EmbedService`](crate::EmbedService) (the
//! existing text-embedder-plus-optional-image-half) with its **space key**
//! (the backend's content fingerprint — keyed by CONTENT so reordering the
//! config never re-keys or rebuilds, NOT a declaration index) and its
//! **role**: which note modalities it is PRIMARY for, and whether it is
//! text-capable for query routing.
//!
//! This PR (#233) lands the **carrier + the routing metadata** only. The
//! fan-out — index-narrow (note items → their per-modality primary space) and
//! query-wide (a query → every text-capable space, RRF-fused) — is PR-B/C
//! (#232/#234). The kernel's index/search paths still consume exactly ONE
//! engine this PR ([`EmbedSpaces::primary`]), so with one declared embedder
//! every on-disk artifact and fused ranking stays bit-identical to the
//! single-slot era: single-space is the structurally-degenerate case of the
//! general set, never a parallel branch.

use std::sync::Arc;

use crate::EmbedService;

/// One attached embedding space: the service, its content-fingerprint space
/// key, and its routing role.
///
/// The **space key** is the backend's `model_fingerprint` (a CONTENT
/// fingerprint — the loaded model's identity, reorder-stable). A space with
/// an unknown fingerprint (a backend that advertises none) carries `None`;
/// such spaces never collide by key (each `attach` with a `None` key is a
/// fresh slot — see [`EmbedSpaces::attach`]).
pub struct EmbedSpace {
    /// The content fingerprint that keys this space (`None` = the backend
    /// advertised none; treated as never-equal to any other key).
    pub key: Option<String>,
    /// The embedding service (text embedder + optional image half).
    pub service: Arc<EmbedService>,
}

impl EmbedSpace {
    /// Whether this space can embed a TEXT query — the query-routing flag.
    /// Every [`Embedder`](crate::Embedder) embeds text, so a text embedder is
    /// always text-capable; the flag exists as the explicit routing seam the
    /// query fan-out reads in PR-C.
    pub fn text_capable(&self) -> bool {
        true
    }

    /// Whether this space embeds the IMAGE modality — i.e. it carries an image
    /// half (a CLIP/omni backend with a media resolver). The per-modality
    /// primary for `image` is the first attached space for which this holds.
    pub fn image_capable(&self) -> bool {
        self.service.images.is_some()
    }
}

/// The ordered set of attached embedding spaces (#233).
///
/// Insertion-ordered, keyed by content fingerprint with replace-on-same-key
/// (an existing key's slot is updated in place, preserving its position — a
/// model swap that keeps the fingerprint is a replace, a reorder of the config
/// is invisible because the key is content not index). A `None` key never
/// collides, so a keyless backend always takes a fresh trailing slot.
///
/// The **primary text space** is the first text-capable space in insertion
/// order; the **primary image space** is the first image-capable one. With one
/// declared embedder the sole space is primary for both modalities it serves,
/// so [`primary`](Self::primary) returns it and the index/search paths are
/// byte-identical to the single-slot era.
#[derive(Default)]
pub struct EmbedSpaces {
    spaces: Vec<EmbedSpace>,
}

impl EmbedSpaces {
    /// Attach (or replace) a space. A space whose key matches an already-
    /// attached one REPLACES it in place (same position — a model swap that
    /// preserves the fingerprint, or a re-attach of the same space); a new key
    /// (or a `None`/keyless backend) is appended. Returns nothing — the
    /// orchestrator readiness flip stays the caller's (the kernel).
    pub fn attach(&mut self, key: Option<String>, service: Arc<EmbedService>) {
        if let Some(k) = key.as_deref() {
            if let Some(slot) = self.spaces.iter_mut().find(|s| s.key.as_deref() == Some(k)) {
                slot.service = service;
                return;
            }
        }
        self.spaces.push(EmbedSpace { key, service });
    }

    /// Detach the space with this key, if present. A `None` argument is a
    /// no-op (a keyless space can only be cleared via [`clear`](Self::clear)).
    /// Returns whether a space was removed.
    pub fn detach(&mut self, key: &str) -> bool {
        let before = self.spaces.len();
        self.spaces.retain(|s| s.key.as_deref() != Some(key));
        self.spaces.len() != before
    }

    /// Detach EVERY space (embedding stop) — the N=1 wrapper's whole-clear,
    /// preserving the single-slot era's `attach_embedder`-then-`detach`
    /// semantics exactly.
    pub fn clear(&mut self) {
        self.spaces.clear();
    }

    /// The PRIMARY text space — the first text-capable space in insertion
    /// order. This is the one engine the index/search paths consume this PR;
    /// with one declared embedder it is the sole space (byte-identical to the
    /// single-slot `embed_service()`).
    pub fn primary(&self) -> Option<Arc<EmbedService>> {
        self.spaces
            .iter()
            .find(|s| s.text_capable())
            .map(|s| Arc::clone(&s.service))
    }

    /// The first image-capable space's service — the per-modality primary for
    /// `image` (metadata only this PR; the index fan-out is PR-B).
    pub fn primary_image(&self) -> Option<Arc<EmbedService>> {
        self.spaces
            .iter()
            .find(|s| s.image_capable())
            .map(|s| Arc::clone(&s.service))
    }

    /// The ordered set (services), for the query fan-out / status surfaces
    /// (unused by the index path this PR).
    pub fn services(&self) -> Vec<Arc<EmbedService>> {
        self.spaces.iter().map(|s| Arc::clone(&s.service)).collect()
    }

    /// Every text-capable space's service in insertion order — what the query
    /// fan-out (PR-C) embeds the query into; the index path does not use it
    /// this PR.
    pub fn text_capable_services(&self) -> Vec<Arc<EmbedService>> {
        self.spaces
            .iter()
            .filter(|s| s.text_capable())
            .map(|s| Arc::clone(&s.service))
            .collect()
    }

    /// The number of attached spaces.
    pub fn len(&self) -> usize {
        self.spaces.len()
    }

    /// Whether any space is attached (the embedder gate — replaces the old
    /// `Option::is_none()` check).
    pub fn is_empty(&self) -> bool {
        self.spaces.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Embedder, ImageEmbedder};
    use futures::future::BoxFuture;
    use shrike_ffi::NativeResult;

    /// A do-nothing embedder — the carrier tests only exercise set bookkeeping,
    /// never an embed call.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, _texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    struct StubImages;
    impl ImageEmbedder for StubImages {
        fn embed_images(
            &self,
            _images: Vec<crate::MediaItem>,
        ) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    struct StubResolver;
    impl crate::ImageResolver for StubResolver {
        fn read(&self, _name: &str) -> Option<Vec<u8>> {
            None
        }
        fn exists(&self, _name: &str) -> bool {
            false
        }
    }

    fn text_service() -> Arc<EmbedService> {
        Arc::new(EmbedService {
            embedder: Arc::new(StubEmbedder),
            images: None,
        })
    }

    fn image_service() -> Arc<EmbedService> {
        Arc::new(EmbedService {
            embedder: Arc::new(StubEmbedder),
            images: Some((Box::new(StubImages), Box::new(StubResolver))),
        })
    }

    fn key_of(svc: &Option<Arc<EmbedService>>) -> Option<String> {
        // The carrier doesn't store the key on the service, so distinguish
        // services by pointer identity instead.
        svc.as_ref().map(|s| format!("{:p}", Arc::as_ptr(s)))
    }

    #[test]
    fn attach_is_insertion_ordered_and_keyed_by_content() {
        let mut set = EmbedSpaces::default();
        assert!(set.is_empty());

        let a = text_service();
        let b = text_service();
        set.attach(Some("text".into()), Arc::clone(&a));
        set.attach(Some("clip".into()), Arc::clone(&b));
        assert_eq!(set.len(), 2);

        // The primary is the FIRST text-capable space (insertion order), not
        // the most-recently attached.
        assert_eq!(key_of(&set.primary()), key_of(&Some(a.clone())));

        // Reusing a key REPLACES that space in place — no growth, position kept.
        let a2 = text_service();
        set.attach(Some("text".into()), Arc::clone(&a2));
        assert_eq!(set.len(), 2, "same key replaces, does not append");
        assert_eq!(
            key_of(&set.primary()),
            key_of(&Some(a2.clone())),
            "the replacement holds the first slot"
        );
    }

    #[test]
    fn keyless_spaces_never_collide() {
        let mut set = EmbedSpaces::default();
        set.attach(None, text_service());
        set.attach(None, text_service());
        assert_eq!(set.len(), 2, "two keyless attaches are two distinct slots");
    }

    #[test]
    fn primary_image_is_first_image_capable() {
        let mut set = EmbedSpaces::default();
        set.attach(Some("text".into()), text_service());
        let img = image_service();
        set.attach(Some("clip".into()), Arc::clone(&img));
        // primary() (text) is the text space; primary_image() is the clip space.
        assert!(set.primary().is_some());
        assert_eq!(key_of(&set.primary_image()), key_of(&Some(img.clone())));
        // text_capable_services() returns both (every embedder is text-capable);
        // the order is insertion order.
        assert_eq!(set.text_capable_services().len(), 2);
    }

    #[test]
    fn detach_by_key_and_whole_clear() {
        let mut set = EmbedSpaces::default();
        set.attach(Some("text".into()), text_service());
        set.attach(Some("clip".into()), text_service());

        assert!(set.detach("text"));
        assert_eq!(set.len(), 1);
        assert!(!set.detach("text"), "detaching an absent key is false");
        assert!(!set.detach("missing"));

        set.clear();
        assert!(set.is_empty());
    }
}
