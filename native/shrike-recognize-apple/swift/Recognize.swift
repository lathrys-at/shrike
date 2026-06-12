// The Swift half of the engine (#398): Apple's new Vision API
// (`RecognizeTextRequest`, macOS 15+) is Swift-only — Apple's stated
// direction is that Vision introduces new features in Swift exclusively —
// so the platform glue is Swift, bolted behind Rust via a 3-function C ABI.
// Rust stays the integration surface; this file is an implementation detail
// of `src/imp.rs`.
//
// The C contract:
//   shrike_av_recognize_one(ptr, len) -> char*   one image in, one JSON
//       Recognition out ({text, confidence, segments:[{text, confidence,
//       bbox}]}, plus an optional `error` the Rust side logs). Never null.
//   shrike_av_fingerprint() -> char*             the engine identity, or
//       null below macOS 15 (the Rust side maps null to `unavailable`).
//   shrike_av_free_string(char*)                 frees either — strings
//       allocated here must be freed here (allocators differ across the
//       boundary).
//
// Concurrency: the new API is async-only, and these entries are synchronous
// by design — each is invoked from a tokio *blocking-pool* thread (the
// kernel's `Blocking` adapter), which is disjoint from Swift's cooperative
// executor. The detached task runs the `await` on Swift's own pool while
// the semaphore parks the tokio blocking thread: two disjoint pools, no
// shared executor, no starvation — the one context where a semaphore
// bridge over async is correct.

import Foundation
import Vision

/// The wire shape of one recognized line — mirrors the Rust `Segment`
/// (`text`, `confidence`, `bbox: [x, y, w, h]` normalized top-left, 4 dp).
private struct WireSegment: Encodable {
    let text: String
    let confidence: Double
    let bbox: [Double]
}

/// The wire shape of one image's result — mirrors the Rust `Recognition`,
/// plus `error` (absent on success) for the Rust side to log.
private struct WireRecognition: Encodable {
    var text: String = ""
    var confidence: Double = 0
    var segments: [WireSegment] = []
    var error: String?
}

/// Crosses the semaphore bridge: written by the detached task before
/// `signal()`, read by the entry after `wait()` — the semaphore is the
/// happens-before edge, so the unchecked Sendable is sound.
private final class ResultBox: @unchecked Sendable {
    var json = "{\"text\":\"\",\"confidence\":0.0,\"segments\":[]}"
}

/// Round to 4 decimal places (the segments contract's box precision).
private func round4(_ v: Double) -> Double {
    (v * 10_000).rounded() / 10_000
}

private func encode(_ wire: WireRecognition) -> String {
    guard let data = try? JSONEncoder().encode(wire),
        let json = String(data: data, encoding: .utf8)
    else {
        return ResultBox().json  // the empty-recognition default
    }
    return json
}

@available(macOS 15, *)
private func recognize(_ data: Data) async -> WireRecognition {
    var request = RecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    var wire = WireRecognition()
    let observations: [RecognizedTextObservation]
    do {
        observations = try await request.perform(on: data)
    } catch {
        // Unreadable bytes / a failed request yield the empty recognition —
        // per-item failures never sink a batch. The error rides the wire for
        // the Rust side to log.
        wire.error = String(describing: error)
        return wire
    }

    var lines: [String] = []
    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else { continue }
        // Vision: normalized, origin bottom-left → top-left [x, y, w, h].
        let box = observation.boundingBox
        let x = Double(box.origin.x)
        let w = Double(box.width)
        let h = Double(box.height)
        let y = 1.0 - Double(box.origin.y) - h
        lines.append(candidate.string)
        wire.segments.append(
            WireSegment(
                text: candidate.string,
                confidence: Double(candidate.confidence),
                bbox: [round4(x), round4(y), round4(w), round4(h)]
            ))
    }
    if wire.segments.isEmpty {
        return wire
    }
    wire.text = lines.joined(separator: "\n")
    wire.confidence =
        wire.segments.map(\.confidence).reduce(0, +) / Double(wire.segments.count)
    return wire
}

/// Recognize one image: bytes in (Rust owns the buffer; copied into `Data`
/// before the call returns to async-land), JSON `Recognition` out (caller
/// frees via `shrike_av_free_string`).
@_cdecl("shrike_av_recognize_one")
public func shrike_av_recognize_one(
    _ ptr: UnsafePointer<UInt8>, _ len: Int
) -> UnsafeMutablePointer<CChar>? {
    let data = Data(bytes: ptr, count: len)
    let box = ResultBox()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        defer { semaphore.signal() }
        guard #available(macOS 15, *) else { return }
        box.json = encode(await recognize(data))
    }
    semaphore.wait()
    return strdup(box.json)
}

/// The engine identity: `apple-vision-swift:{revision}:macos{X.Y.Z}` —
/// model revision + OS version, so an OS upgrade re-derives exactly like a
/// model change rebuilds vectors. Null below macOS 15 (the API floor).
@_cdecl("shrike_av_fingerprint")
public func shrike_av_fingerprint() -> UnsafeMutablePointer<CChar>? {
    guard #available(macOS 15, *) else { return nil }
    let revision = String(describing: RecognizeTextRequest().revision)
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return strdup(
        "apple-vision-swift:\(revision):macos\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)")
}

/// Free a string returned by either entry (pairs with `strdup` here —
/// never freed on the Rust side's allocator).
@_cdecl("shrike_av_free_string")
public func shrike_av_free_string(_ ptr: UnsafeMutablePointer<CChar>?) {
    free(ptr)
}
