// The ASR half of the Swift glue (#410): Apple's SpeechAnalyzer /
// SpeechTranscriber (macOS 26+, Swift-only) transcribes audio on-device.
// Same architecture as Recognize.swift — Swift bolted behind Rust via
// `@_cdecl` C entries, the semaphore bridge over the async API (safe: the
// entries run on the kernel runtime's blocking pool, disjoint from Swift's
// cooperative executor) — this file is an implementation detail of
// `src/speech.rs`.
//
// The C contract (strings freed via `shrike_av_free_string`):
//   shrike_av_transcribe_one(ptr, len, mime, locale) -> char*  JSON
//       Recognition (segments carry `span` [start_s, duration_s]); never
//       null. Per-item failures ride the `error` field, never a throw.
//   shrike_av_speech_fingerprint(locale) -> char*  the engine identity, or
//       null when the API/locale is unavailable.
//   shrike_av_speech_ensure_assets(locale) -> char*  JSON
//       {"status": "ready"|"installed"|"unsupported", "error"?} — the ONE
//       entry allowed to drive the on-device model download.
//
// The SDK gate is double-layered, and that's load-bearing (unlike OCR):
// SpeechAnalyzer's types are ABSENT from pre-26 SDKs, so `#available`
// alone would not compile there. `#if canImport(FoundationModels)` — a
// module that exists only in the macOS-26 SDK — is the compile-time proxy
// (Swift has no SDK-version `#if`, and Speech itself is an ancient module);
// `#available(macOS 26, *)` guards at runtime. On an older SDK every
// symbol still exists and returns the unavailable sentinel.

import AVFAudio
import CoreMedia
import Foundation
import Speech

/// Map the caller's mime hint to a filename extension so `AVAudioFile`'s
/// container sniffing gets its hint; unknown/absent hints fall back to wav
/// (CoreAudio sniffs content for the common containers regardless).
private func extensionFor(mime: String?) -> String {
    switch mime {
    case "audio/mpeg": return "mp3"
    case "audio/wav", "audio/x-wav": return "wav"
    case "audio/mp4": return "m4a"
    case "audio/flac": return "flac"
    case "audio/aiff", "audio/x-aiff": return "aiff"
    // CoreAudio has no Vorbis/Opus decoder — an ogg item will open-fail
    // into the empty recognition with an error, which is the contract.
    case "audio/ogg": return "ogg"
    default: return "wav"
    }
}

#if canImport(FoundationModels)

    @available(macOS 26, *)
    private func resolvedLocale(_ identifier: String) async -> Locale? {
        await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: identifier))
    }

    /// Drain the transcriber's result stream to completion (it ends when the
    /// analyzer finalizes).
    @available(macOS 26, *)
    private func collect(_ transcriber: SpeechTranscriber) async throws -> [SpeechTranscriber
        .Result]
    {
        var out: [SpeechTranscriber.Result] = []
        for try await result in transcriber.results {
            out.append(result)
        }
        return out
    }

    @available(macOS 26, *)
    private func transcribe(_ data: Data, mime: String?, localeId: String) async -> WireRecognition
    {
        var wire = WireRecognition()
        guard let locale = await resolvedLocale(localeId) else {
            wire.error = "unsupported transcription locale: \(localeId)"
            return wire
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(extensionFor(mime: mime))
        do {
            try data.write(to: url)
        } catch {
            wire.error = "temp write failed: \(error)"
            return wire
        }
        defer { try? FileManager.default.removeItem(at: url) }

        do {
            let transcriber = SpeechTranscriber(
                locale: locale,
                transcriptionOptions: [],
                reportingOptions: [],  // final results only — no volatile stream
                attributeOptions: [.audioTimeRange, .transcriptionConfidence]
            )
            let analyzer = SpeechAnalyzer(modules: [transcriber])
            let file = try AVAudioFile(forReading: url)

            // On a throw below, structured concurrency cancels and awaits
            // this child — the no-hang guarantee at the C boundary rests on
            // `results` terminating when the analyzer/transcriber go away
            // (Apple's contract; exercised by the garbage-bytes live test).
            async let collected = collect(transcriber)
            let last = try await analyzer.analyzeSequence(from: file)
            if let last {
                try await analyzer.finalizeAndFinish(through: last)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
            let results = try await collected

            var lines: [String] = []
            for result in results {
                let text = String(result.text.characters)
                guard !text.isEmpty else { continue }
                // Confidence: mean over the attributed runs that carry one;
                // 1.0 when the model reports none (absence isn't doubt).
                var confidences: [Double] = []
                for run in result.text.runs {
                    if let c = run.transcriptionConfidence {
                        confidences.append(c)
                    }
                }
                let confidence =
                    confidences.isEmpty
                    ? 1.0 : confidences.reduce(0, +) / Double(confidences.count)
                let start = result.range.start.seconds
                let duration = result.range.duration.seconds
                lines.append(text)
                wire.segments.append(
                    WireSegment(
                        text: text,
                        confidence: round4(confidence),
                        locator: .span([
                            round4(start.isFinite ? start : 0),
                            round4(duration.isFinite ? duration : 0),
                        ])
                    ))
            }
            if wire.segments.isEmpty {
                return wire
            }
            wire.text = lines.joined(separator: "\n")
            wire.confidence =
                wire.segments.map(\.confidence).reduce(0, +) / Double(wire.segments.count)
            return wire
        } catch {
            // Undecodable bytes / a failed analysis yield the empty
            // recognition — per-item failures never sink a batch.
            return WireRecognition(error: String(describing: error))
        }
    }

    @available(macOS 26, *)
    private func ensureAssets(localeId: String) async -> String {
        guard let locale = await resolvedLocale(localeId) else {
            return #"{"status":"unsupported"}"#
        }
        do {
            let transcriber = SpeechTranscriber(
                locale: locale,
                transcriptionOptions: [],
                reportingOptions: [],
                attributeOptions: []
            )
            if let request = try await AssetInventory.assetInstallationRequest(
                supporting: [transcriber])
            {
                try await request.downloadAndInstall()
                return #"{"status":"installed"}"#
            }
            return #"{"status":"ready"}"#
        } catch {
            let detail = String(describing: error).replacingOccurrences(of: "\"", with: "'")
            return "{\"status\":\"error\",\"error\":\"\(detail)\"}"
        }
    }

#endif

/// Transcribe one audio item: bytes + mime hint + BCP-47 locale in, JSON
/// `Recognition` out (segments carry `span` locators).
@_cdecl("shrike_av_transcribe_one")
public func shrike_av_transcribe_one(
    _ ptr: UnsafePointer<UInt8>, _ len: Int,
    _ mime: UnsafePointer<CChar>?,
    _ locale: UnsafePointer<CChar>?
) -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
        let data = Data(bytes: ptr, count: len)
        let mimeHint = mime.map { String(cString: $0) }
        let localeId = locale.map { String(cString: $0) } ?? "en-US"
        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            defer { semaphore.signal() }
            guard #available(macOS 26, *) else { return }
            box.json = encode(await transcribe(data, mime: mimeHint, localeId: localeId))
        }
        semaphore.wait()
        return strdup(box.json)
    #else
        // Pre-26 SDK build: the symbol exists, the capability doesn't.
        return strdup(
            "{\"text\":\"\",\"confidence\":0.0,\"segments\":[],"
                + "\"error\":\"SpeechAnalyzer requires the macOS 26 SDK\"}")
    #endif
}

/// The engine identity: `apple-speech:{locale}:macos{X.Y.Z}` — the resolved
/// locale + OS version. There is no public model-version accessor, so the
/// OS version is the honest proxy (a speech-asset update without an OS bump
/// won't re-derive — accepted; assets ride OS point releases in practice).
/// Null when the API or the locale is unavailable.
@_cdecl("shrike_av_speech_fingerprint")
public func shrike_av_speech_fingerprint(
    _ locale: UnsafePointer<CChar>?
) -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
        guard #available(macOS 26, *) else { return nil }
        let localeId = locale.map { String(cString: $0) } ?? "en-US"
        let box = ResultBox()
        box.json = ""
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            defer { semaphore.signal() }
            if let resolved = await resolvedLocale(localeId) {
                let v = ProcessInfo.processInfo.operatingSystemVersion
                box.json =
                    "apple-speech:\(resolved.identifier(.bcp47)):"
                    + "macos\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
            }
        }
        semaphore.wait()
        return box.json.isEmpty ? nil : strdup(box.json)
    #else
        return nil
    #endif
}

/// Ensure the on-device model assets for the locale are installed — the ONE
/// entry allowed to drive a (possibly multi-hundred-MB) download. Explicit
/// by design: constructors never call this.
@_cdecl("shrike_av_speech_ensure_assets")
public func shrike_av_speech_ensure_assets(
    _ locale: UnsafePointer<CChar>?
) -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
        guard #available(macOS 26, *) else {
            return strdup(#"{"status":"unsupported"}"#)
        }
        let localeId = locale.map { String(cString: $0) } ?? "en-US"
        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            defer { semaphore.signal() }
            box.json = await ensureAssets(localeId: localeId)
        }
        semaphore.wait()
        return strdup(box.json)
    #else
        return strdup(#"{"status":"unsupported"}"#)
    #endif
}
