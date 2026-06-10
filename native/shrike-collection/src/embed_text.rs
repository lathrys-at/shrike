//! Stable note-text normalization for embedding — the Rust port of
//! `shrike/embed_text.py` (#278 series, step 2).
//!
//! The contract is **byte-identity with the Python normalizer**: the output is
//! part of the vector space (`EMBED_TEXT_VERSION` is folded into the index
//! fingerprint), so the port must not change a single byte or every index
//! silently mismatches its queries. Two structural guarantees make that hold:
//!
//! - the HTML→text + entity step calls the SAME service RPC
//!   (`CardRenderingService::StripHtml`, NORMAL mode) that pylib's
//!   `anki.utils.strip_html` calls — the stripper is literally the same Rust
//!   code at the same pinned tag, not a reimplementation;
//! - the around-stripper transforms (cloze reveal, MathJax/LaTeX wrappers,
//!   `[sound:…]`, block-tag→space, NBSP fold, whitespace collapse) use the
//!   IDENTICAL patterns (fancy-regex supports the cloze lookahead verbatim),
//!   and the tests/native parity corpus pins the equivalence end to end.
//!
//! `EMBED_TEXT_VERSION` therefore stays at the Python value — bump BOTH (and
//! force a rebuild) only if the output genuinely changes.

use std::sync::LazyLock;

use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use shrike_ffi::NativeResult;

/// Mirrors `shrike.embed_text.EMBED_TEXT_VERSION` — the parity test asserts
/// the two constants agree.
pub const EMBED_TEXT_VERSION: i64 = 1;

// {{c1::answer}} / {{c1::answer::hint}} — identical to the Python pattern
// (DOTALL via (?s); the tempered dot means innermost clozes match first and
// iterating flattens nesting). fancy-regex for the lookahead.
static CLOZE_RE: LazyLock<FancyRegex> =
    LazyLock::new(|| FancyRegex::new(r"(?s)\{\{c\d+::((?:(?!\{\{|\}\}).)*)\}\}").unwrap());
// [sound:file.mp3]
static SOUND_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\[sound:[^\]]*\]").unwrap());
// Legacy image-LaTeX wrappers: [latex] [/latex] [$] [/$] [$$] [/$$].
static LATEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\[/?(?:latex|\$\$?)\]").unwrap());
// MathJax delimiters: \( \) \[ \] and $$.
static MATHJAX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\[()\[\]]|\$\$").unwrap());
// Block-level tags and <br> → whitespace (Anki's stripper glues across them).
static BLOCK_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<\s*/?\s*(?:br|div|p|li|ul|ol|tr|td|h[1-6]|blockquote)\b[^>]*>").unwrap()
});
static WS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// Replace cloze deletions with their answer text (wrapper + hint dropped).
/// Iterates to flatten shallow nesting; bounded like the Python original.
fn fill_clozes(text: &str) -> String {
    let mut text = text.to_string();
    for _ in 0..10 {
        let new = CLOZE_RE
            .replace_all(&text, |caps: &fancy_regex::Captures<'_>| {
                let content = caps.get(1).map_or("", |m| m.as_str());
                content
                    .split_once("::")
                    .map_or(content, |(a, _)| a)
                    .to_string()
            })
            .into_owned();
        if new == text {
            return new;
        }
        text = new;
    }
    text
}

/// Turn one raw Anki field value into stable plain text for embedding —
/// byte-identical to `shrike.embed_text.normalize_for_embedding`. The
/// HTML-strip step is injected (the adapter's service-layer `strip_html`).
pub fn normalize_for_embedding(
    value: &str,
    strip_html: &dyn Fn(&str) -> NativeResult<String>,
) -> NativeResult<String> {
    if value.is_empty() {
        return Ok(String::new());
    }
    let text = fill_clozes(value);
    let text = LATEX_RE.replace_all(&text, " ");
    let text = MATHJAX_RE.replace_all(&text, " ");
    let text = SOUND_RE.replace_all(&text, " ");
    let text = BLOCK_TAG_RE.replace_all(&text, " ");
    let text = strip_html(&text)?;
    let text = text.replace('\u{a0}', " ");
    Ok(WS_RE.replace_all(&text, " ").trim().to_string())
}

/// Render a note's embedding text: each non-empty normalized field as
/// `Name: text`, newline-joined — mirrors
/// `CollectionWrapper._render_embed_text` (the single render both the query
/// and index paths must share).
pub fn render_embed_text(
    names: &[String],
    values: &[String],
    strip_html: &dyn Fn(&str) -> NativeResult<String>,
) -> NativeResult<String> {
    let mut parts: Vec<String> = Vec::new();
    for (name, value) in names.iter().zip(values.iter()) {
        let cleaned = normalize_for_embedding(value, strip_html)?;
        if !cleaned.is_empty() {
            parts.push(format!("{name}: {cleaned}"));
        }
    }
    Ok(parts.join("\n"))
}

// Media references that make a field non-empty even with no text (the #89
// empty-note rule) — mirrors Python's _MEDIA_RE.
static MEDIA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<\s*(?:img|audio|video|object|embed|source)\b|\[sound:").unwrap()
});

/// True if a field value carries no content — no text and no media. Mirrors
/// `shrike.embed_text.field_is_blank` (stricter than the embedding
/// normalization: media makes a field non-blank).
pub fn field_is_blank(
    value: &str,
    strip_html: &dyn Fn(&str) -> NativeResult<String>,
) -> NativeResult<bool> {
    if value.is_empty() {
        return Ok(true);
    }
    if MEDIA_RE.is_match(value) {
        return Ok(false);
    }
    let stripped = strip_html(value)?;
    Ok(stripped.replace('\u{a0}', " ").trim().is_empty())
}

/// Image filenames referenced by a field's `<img src>` attributes — in order,
/// de-duplicated; basenames only; remote (`scheme://`) srcs skipped. Mirrors
/// `shrike.embed_text.extract_image_refs`, including its parser-not-regex
/// property: attributes are tokenized, so a `data-src=` or a `src=` inside
/// another attribute's quoted value can't be mistaken for the tag's own src.
pub fn extract_image_refs(value: &str) -> Vec<String> {
    if value.is_empty() || !value.to_lowercase().contains("<img") {
        return Vec::new();
    }
    let mut names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for src in img_src_values(value) {
        let src = src.trim();
        if src.is_empty() || src.contains("://") {
            continue;
        }
        let name = src.rsplit('/').next().unwrap_or(src);
        if !name.is_empty() && !seen.contains(name) {
            seen.insert(name.to_string());
            names.push(name.to_string());
        }
    }
    names
}

/// Scan for `<img …>` start tags and collect their `src` attribute values
/// (entity-unescaped, like HTMLParser with convert_charrefs). A small
/// hand-rolled tokenizer: tag name match is case-insensitive; attributes are
/// `name`, `name=bare`, `name='…'`, or `name="…"`.
fn img_src_values(html: &str) -> Vec<String> {
    let bytes = html.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    // ASCII-case-insensitive byte search: never search a `to_lowercase()` copy
    // with offsets into the original — Unicode lowercasing can change byte
    // length (İ → i̇), misaligning every index after it.
    let find_img = |from: usize| -> Option<usize> {
        bytes
            .get(from..)?
            .windows(4)
            .position(|w| w.eq_ignore_ascii_case(b"<img"))
            .map(|p| from + p)
    };
    while let Some(pos) = find_img(i) {
        let start = pos + 4;
        // must be followed by whitespace, '>', or '/' to be the img tag
        match bytes.get(start) {
            Some(b) if b.is_ascii_whitespace() || *b == b'>' || *b == b'/' => {}
            _ => {
                i = start;
                continue;
            }
        }
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'>' {
            // skip whitespace and stray slashes between attributes
            while j < bytes.len() && (bytes[j].is_ascii_whitespace() || bytes[j] == b'/') {
                j += 1;
            }
            if j >= bytes.len() || bytes[j] == b'>' {
                break;
            }
            // attribute name
            let name_start = j;
            while j < bytes.len()
                && !bytes[j].is_ascii_whitespace()
                && !matches!(bytes[j], b'=' | b'>' | b'/')
            {
                j += 1;
            }
            let name = html[name_start..j].to_ascii_lowercase();
            // optional = value
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            let mut value: Option<String> = None;
            if j < bytes.len() && bytes[j] == b'=' {
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b'\'') {
                    let quote = bytes[j];
                    j += 1;
                    let v_start = j;
                    while j < bytes.len() && bytes[j] != quote {
                        j += 1;
                    }
                    value = Some(html[v_start..j].to_string());
                    if j < bytes.len() {
                        j += 1; // closing quote
                    }
                } else {
                    let v_start = j;
                    while j < bytes.len() && !bytes[j].is_ascii_whitespace() && bytes[j] != b'>' {
                        j += 1;
                    }
                    value = Some(html[v_start..j].to_string());
                }
            }
            if name == "src" {
                if let Some(v) = value {
                    if !v.is_empty() {
                        out.push(unescape_entities(&v));
                    }
                }
            }
        }
        i = j;
    }
    out
}

/// Minimal HTML entity unescape for attribute values (the named + numeric
/// forms HTMLParser resolves; attribute srcs in practice only carry these).
fn unescape_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let semi = tail.find(';');
        match semi {
            Some(end) if end <= 10 => {
                let entity = &tail[1..end];
                let replacement: Option<String> = match entity {
                    "amp" => Some("&".into()),
                    "lt" => Some("<".into()),
                    "gt" => Some(">".into()),
                    "quot" => Some("\"".into()),
                    "apos" => Some("'".into()),
                    "nbsp" => Some("\u{a0}".into()),
                    e if e.starts_with("#x") || e.starts_with("#X") => {
                        u32::from_str_radix(&e[2..], 16)
                            .ok()
                            .and_then(char::from_u32)
                            .map(|c| c.to_string())
                    }
                    e if e.starts_with('#') => e[1..]
                        .parse::<u32>()
                        .ok()
                        .and_then(char::from_u32)
                        .map(|c| c.to_string()),
                    _ => None,
                };
                match replacement {
                    Some(r) => {
                        out.push_str(&r);
                        rest = &tail[end + 1..];
                    }
                    None => {
                        out.push('&');
                        rest = &tail[1..];
                    }
                }
            }
            _ => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_html(s: &str) -> NativeResult<String> {
        Ok(s.to_string()) // identity stripper for stripper-independent cases
    }

    #[test]
    fn cloze_fill_matches_python_semantics() {
        assert_eq!(fill_clozes("{{c1::France}}"), "France");
        assert_eq!(fill_clozes("{{c1::France::country}}"), "France");
        assert_eq!(fill_clozes("{{c2::a {{c1::b}} c}}"), "a b c");
        assert_eq!(fill_clozes("no cloze"), "no cloze");
    }

    #[test]
    fn wrappers_and_whitespace() {
        let out =
            normalize_for_embedding("x [sound:a.mp3] \\(e=mc^2\\) [latex]y[/latex]  z", &no_html)
                .unwrap();
        assert_eq!(out, "x e=mc^2 y z");
    }

    #[test]
    fn image_refs_are_parsed_not_regexed() {
        assert_eq!(
            extract_image_refs(r#"<img data-src="lazy.png" src="real.png">"#),
            vec!["real.png"]
        );
        assert_eq!(
            extract_image_refs(r#"<img alt="src=fake.png" src='dir/pic.jpg'>"#),
            vec!["pic.jpg"]
        );
        assert_eq!(
            extract_image_refs(r#"<IMG SRC="a.png"><img src="a.png"><img src="http://x/b.png">"#),
            vec!["a.png"]
        );
        assert_eq!(
            extract_image_refs("<img src=\"a&amp;b.png\">"),
            vec!["a&b.png"]
        );
        assert!(extract_image_refs("no images").is_empty());
        // Offset-safety regression: İ's lowercase is LONGER in bytes (i + a
        // combining dot), so a lowercase-copy search would misalign every
        // index after it; the byte scanner must not.
        assert_eq!(
            extract_image_refs("İİİ before <img src=\"after.png\">"),
            vec!["after.png"]
        );
    }
}
