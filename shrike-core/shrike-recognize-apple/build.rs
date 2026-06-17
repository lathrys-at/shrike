//! Compile the Swift half (#398) into a static library and link it.
//!
//! macOS only — off macOS this is a no-op and the crate compiles to the
//! `imp_stub` with zero platform deps (the Linux lane needs no Swift
//! toolchain). On macOS, building requires full Xcode (the Swift-only
//! Vision module isn't in the Command Line Tools SDK); *running* needs
//! nothing extra — the Swift runtime ships with the OS and is linked
//! dynamically (static-stdlib was rejected: it risks a duplicate runtime
//! if anything else in the host process loads Swift). One catch the crate
//! can't fix from here: downstream FINAL links (rustc's 11.0 default
//! deployment target) fall in libswift_Concurrency's `$ld$previous`
//! back-deploy window (≤ 12.0), which remaps its install name to @rpath —
//! so every final link needs `-rpath /usr/lib/swift`, carried by
//! `native/.cargo/config.toml` (cargo) and the `swift_glue` cc_library
//! linkopts (bazel). Don't remove either.
//!
//! Bazel never runs this file (first-party targets are hand-written
//! `rust_library`s) — `BUILD.bazel`'s genrule runs the same `xcrun swiftc`
//! invocation, so the two builds produce identical Swift.

fn main() {
    // Keep the no-op lanes cheap and the rebuild triggers exact (cargo
    // doesn't glob — every Swift source is named).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=swift/Recognize.swift");
    println!("cargo:rerun-if-changed=swift/Transcribe.swift");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    build_swift();
}

fn build_swift() {
    let out_dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        other => panic!("unsupported macOS arch for the Swift glue: {other:?}"),
    };
    // Deployment floor: the build's own target if set, else 13.0 — above
    // the SDK's `$ld$previous` window for libswift_Concurrency ([10.9, 12.0]
    // links the @rpath back-deploy install name; 13.0 links the absolute
    // /usr/lib/swift one, so consumers need no rpath). The macOS-15-only
    // Vision API is weak-linked behind `#available` guards, so the floor
    // needn't (and shouldn't) be 15.
    let deployment = std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or("13.0".into());

    // Route through xcrun: bare `swiftc` from the Command Line Tools can't
    // see the Swift-only Vision module — the SDK selection matters.
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"]);
    let swiftc = xcrun(&["--sdk", "macosx", "--find", "swiftc"]);

    let lib = format!("{out_dir}/libShrikeRecognizeApple.a");
    let status = std::process::Command::new(&swiftc)
        .args([
            "-emit-library",
            "-static",
            "-parse-as-library",
            "-O",
            "-module-name",
            "ShrikeRecognizeApple",
            "-sdk",
            &sdk,
            "-target",
            &format!("{arch}-apple-macos{deployment}"),
            "swift/Recognize.swift",
            "swift/Transcribe.swift",
            "-o",
            &lib,
        ])
        .status()
        .expect("failed to spawn swiftc (full Xcode is required to build on macOS)");
    assert!(status.success(), "swiftc failed building the Vision glue");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=ShrikeRecognizeApple");
    // The archive's autolink metadata (LC_LINKER_OPTION) names the Swift
    // runtime libs and frameworks; the SDK search path lets ld resolve the
    // .tbd stubs (at runtime the absolute /usr/lib/swift install names win).
    println!("cargo:rustc-link-search=native={sdk}/usr/lib/swift");
    println!("cargo:rustc-link-lib=framework=Vision");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // The ASR half (#410): Speech (SpeechAnalyzer), AVFoundation
    // (AVAudioFile), CoreMedia (CMTime ranges).
    println!("cargo:rustc-link-lib=framework=Speech");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreMedia");
}

fn xcrun(args: &[&str]) -> String {
    let out = std::process::Command::new("xcrun")
        .args(args)
        .output()
        .expect("failed to run xcrun (full Xcode is required to build on macOS)");
    assert!(out.status.success(), "xcrun {args:?} failed");
    String::from_utf8(out.stdout)
        .expect("xcrun output is UTF-8")
        .trim()
        .to_owned()
}
