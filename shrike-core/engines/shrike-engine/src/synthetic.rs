//! The synthetic engine: a deterministic, dependency-free embedder for
//! benchmarking and fast deterministic tests.
//!
//! It derives each vector purely from the input bytes — a hash seeds a
//! `splitmix64` stream that fills the components, which are then L2-normalized
//! — so the same input always yields the same unit vector and distinct inputs
//! spread across the unit hypersphere. Diverse enough that the index and the
//! search fusion do representative work, with no model to load and negligible
//! per-call cost.
//!
//! It is NOT a real embedder: the vectors carry no semantics, so neighbour
//! relationships are meaningless. It exists so a profile can attach an embedder
//! whose own cost is negligible — isolating the kernel/IO/orchestration cost a
//! workflow pays from model-inference time. Route-1 sync compute
//! ([`EmbedText`] + [`EmbedImages`]); the host wraps it in the `Blocking`
//! adapter at attach, exactly like the ort engines.

use shrike_engine_api::{EmbedImages, EmbedText, MediaItem};
use shrike_error::NativeResult;

/// FNV-1a (64-bit) over the input bytes — the per-input seed for the component
/// stream. Empty input maps to the offset basis (a fixed nonzero seed), so an
/// empty string still produces a stable, valid vector.
fn seed_for(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One step of the `splitmix64` generator: advance `state` and scramble it into
/// a well-distributed `u64`. Returns `(next_state, output)`.
fn splitmix64(state: u64) -> (u64, u64) {
    let next = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (next, z ^ (z >> 31))
}

/// A deterministic embedder producing fixed-dimensionality unit vectors as a
/// pure function of the input bytes. See the [module docs](self) for what it is
/// and is not for.
#[derive(Debug, Clone)]
pub struct SyntheticEmbedder {
    dim: usize,
}

impl SyntheticEmbedder {
    /// Construct a synthetic embedder producing `dim`-dimensional unit vectors.
    /// `dim` is floored at 1 (a zero-dimension index is meaningless).
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    /// The deterministic unit vector for an input's bytes: seed a `splitmix64`
    /// stream from a hash of the bytes, draw `dim` components uniform in
    /// `[-1, 1)`, then L2-normalize. Distinct inputs spread across the sphere;
    /// identical inputs coincide.
    fn vector(&self, bytes: &[u8]) -> Vec<f32> {
        let mut state = seed_for(bytes);
        let mut v = Vec::with_capacity(self.dim);
        let mut norm_sq = 0.0_f64;
        for _ in 0..self.dim {
            let (next, draw) = splitmix64(state);
            state = next;
            // Top 53 bits → a uniform f64 in [0, 1), remapped to [-1, 1).
            let unit = (draw >> 11) as f64 / ((1u64 << 53) as f64);
            let x = unit.mul_add(2.0, -1.0);
            norm_sq += x * x;
            v.push(x as f32);
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for c in &mut v {
                *c = (f64::from(*c) / norm) as f32;
            }
        } else {
            // Unreachable in practice (the stream never draws all-zero), but a
            // cosine index must never receive a zero vector — emit a basis one.
            v[0] = 1.0;
        }
        v
    }
}

impl EmbedText for SyntheticEmbedder {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.vector(t.as_bytes())).collect())
    }

    /// Every vector is a pure function of its own input, so there is no
    /// batch variance — any chunk size is safe.
    fn safe_batch(&self) -> usize {
        usize::MAX
    }

    fn fingerprint(&self) -> Option<String> {
        Some(format!("synthetic:v1:dim={}", self.dim))
    }

    fn dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl EmbedImages for SyntheticEmbedder {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        Ok(images.iter().map(|im| self.vector(&im.bytes)).collect())
    }

    fn safe_batch(&self) -> usize {
        usize::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f32]) -> f64 {
        v.iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn vectors_have_the_configured_dim_and_are_unit_norm() {
        let e = SyntheticEmbedder::new(384);
        let out = e
            .embed_chunk(&["hello".to_string(), "world".to_string()])
            .unwrap();
        assert_eq!(out.len(), 2);
        for v in &out {
            assert_eq!(v.len(), 384);
            assert!(
                (l2(v) - 1.0).abs() < 1e-5,
                "expected unit norm, got {}",
                l2(v)
            );
        }
    }

    #[test]
    fn embedding_is_deterministic() {
        let e = SyntheticEmbedder::new(64);
        let a = e.embed_chunk(&["the same input".to_string()]).unwrap();
        let b = e.embed_chunk(&["the same input".to_string()]).unwrap();
        assert_eq!(a, b, "identical input must yield an identical vector");
    }

    #[test]
    fn distinct_inputs_are_diverse() {
        let e = SyntheticEmbedder::new(128);
        let a = &e.embed_chunk(&["cat".to_string()]).unwrap()[0];
        let b = &e.embed_chunk(&["dog".to_string()]).unwrap()[0];
        assert_ne!(a, b);
        // Two independent random directions on a 128-sphere have cosine tightly
        // concentrated near 0; assert they are nowhere near collinear.
        let cos: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        assert!(
            cos.abs() < 0.5,
            "distinct inputs should not be near-collinear (cos={cos})"
        );
    }

    #[test]
    fn text_and_image_share_the_space_and_stay_deterministic() {
        let e = SyntheticEmbedder::new(32);
        let img = MediaItem::untyped(b"\x89PNG fake bytes".to_vec());
        let one = e.embed_image_chunk(std::slice::from_ref(&img)).unwrap();
        let two = e.embed_image_chunk(std::slice::from_ref(&img)).unwrap();
        assert_eq!(one, two);
        assert_eq!(one[0].len(), 32);
        assert!((l2(&one[0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dim_is_floored_at_one() {
        let e = SyntheticEmbedder::new(0);
        assert_eq!(e.dim(), Some(1));
        assert_eq!(e.embed_chunk(&["x".to_string()]).unwrap()[0].len(), 1);
    }

    #[test]
    fn empty_batch_is_empty() {
        let e = SyntheticEmbedder::new(16);
        assert!(e.embed_chunk(&[]).unwrap().is_empty());
        assert!(e.embed_image_chunk(&[]).unwrap().is_empty());
    }
}
