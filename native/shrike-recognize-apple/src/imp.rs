//! The macOS implementation: objc2-vision bindings, 1:1 with the retired
//! pyobjc backend's semantics. Vision/Foundation objects are `!Send` and
//! created per call inside an autorelease pool (temporaries — observation
//! arrays, strings — are reclaimed per image, not at thread exit, which
//! matters on a long-lived sweep's pool thread).

use objc2::rc::autoreleasepool;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSData, NSDictionary, NSProcessInfo};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel,
};

use shrike_engine_api::{Recognition, Segment};
use shrike_ffi::NativeResult;

use crate::{empty_recognition, round4};

/// `apple-vision:rev{N}:macos{X.Y[.Z]}` — byte-compatible with the Python
/// backend (`platform.mac_ver()` elides a zero patch), so the swap to the
/// native engine never invalidates existing derived text.
pub(crate) fn fingerprint() -> NativeResult<String> {
    let revision = objc2_vision::VNRecognizeTextRequestRevision3;
    let v = NSProcessInfo::processInfo().operatingSystemVersion();
    let macos = if v.patchVersion == 0 {
        format!("{}.{}", v.majorVersion, v.minorVersion)
    } else {
        format!("{}.{}.{}", v.majorVersion, v.minorVersion, v.patchVersion)
    };
    Ok(format!("apple-vision:rev{revision}:macos{macos}"))
}

/// One image through one accurate-level, language-corrected
/// `VNRecognizeTextRequest` (the request revision stays Vision's default,
/// like the Python backend). A failed request logs and yields the empty
/// recognition — per-item failures never sink a batch.
pub(crate) fn recognize_one(bytes: &[u8]) -> Recognition {
    autoreleasepool(|_pool| {
        let data = NSData::with_bytes(bytes);
        let handler = VNImageRequestHandler::initWithData_options(
            VNImageRequestHandler::alloc(),
            &data,
            &NSDictionary::new(),
        );
        let request = unsafe { VNRecognizeTextRequest::init(VNRecognizeTextRequest::alloc()) };
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(true);

        let requests: [&VNRequest; 1] = [request.as_ref()];
        let array = NSArray::from_slice(&requests);
        if let Err(e) = handler.performRequests_error(&array) {
            tracing::warn!("Vision request failed: {e}");
            return empty_recognition();
        }

        let mut lines: Vec<String> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let observations = request.results();
        for observation in observations.iter().flatten() {
            let candidates = observation.topCandidates(1);
            let Some(candidate) = candidates.firstObject() else {
                continue;
            };
            let text = candidate.string().to_string();
            // VNConfidence is f32; widen exactly as Python's float() does.
            let confidence = candidate.confidence() as f64;
            let bbox = unsafe { observation.boundingBox() };
            // Vision: normalized, origin bottom-left → top-left [x, y, w, h].
            let x = bbox.origin.x;
            let w = bbox.size.width;
            let h = bbox.size.height;
            let y = 1.0 - bbox.origin.y - h;
            lines.push(text.clone());
            segments.push(Segment {
                text,
                confidence,
                bbox: Some([round4(x), round4(y), round4(w), round4(h)]),
            });
        }
        if lines.is_empty() {
            return empty_recognition();
        }
        let overall = segments.iter().map(|s| s.confidence).sum::<f64>() / segments.len() as f64;
        Recognition {
            text: lines.join("\n"),
            confidence: overall,
            segments,
        }
    })
}
