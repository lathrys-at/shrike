//! Whole-text similarity — a byte-faithful port of CPython's
//! `difflib.SequenceMatcher(None, a, b).ratio()` (#391 phase 1).
//!
//! The dedup near-verbatim verifier (#206) was specified against difflib, so
//! the kernel port reproduces it exactly rather than substituting a generic
//! edit distance: the Ratcliff-Obershelp recursion over
//! `find_longest_match`, INCLUDING the `autojunk` heuristic — when `b` is
//! ≥ 200 elements, elements occurring in more than 1% of `b` are barred from
//! *anchoring* a match (deleted from the index map) but still extend one
//! (with `isjunk=None`, difflib's bjunk set is empty, so the junk-extension
//! passes are no-ops and popular elements take part in the plain extension).
//! Elements are Unicode scalar values (`char`), matching Python's
//! code-point-indexed `str`.
//!
//! The parity corpus in the tests below is generated from CPython's difflib
//! and pins the port — including the 199/200 autojunk boundary.

use std::collections::HashMap;

/// `difflib.SequenceMatcher(None, a, b).ratio()`.
pub fn sequence_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (la, lb) = (a.len(), b.len());
    if la + lb == 0 {
        return 1.0; // difflib's _calculate_ratio: empty vs empty is identical
    }

    // __chain_b: b2j maps each element of b to its (ascending) indices —
    // minus the autojunk "popular" elements when b is large.
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, ch) in b.iter().enumerate() {
        b2j.entry(*ch).or_default().push(j);
    }
    if lb >= 200 {
        let ntest = lb / 100 + 1;
        b2j.retain(|_, idxs| idxs.len() <= ntest);
    }

    // get_matching_blocks, iteratively; only the match-size sum feeds ratio.
    let mut matches = 0usize;
    let mut queue: Vec<(usize, usize, usize, usize)> = vec![(0, la, 0, lb)];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (besti, bestj, bestsize) = find_longest_match(&a, &b, &b2j, alo, ahi, blo, bhi);
        if bestsize > 0 {
            matches += bestsize;
            if alo < besti && blo < bestj {
                queue.push((alo, besti, blo, bestj));
            }
            if besti + bestsize < ahi && bestj + bestsize < bhi {
                queue.push((besti + bestsize, ahi, bestj + bestsize, bhi));
            }
        }
    }
    2.0 * matches as f64 / (la + lb) as f64
}

/// difflib's `find_longest_match` with `isjunk=None`: the j2len walk finds
/// the longest block *anchored* on non-popular elements, then one plain
/// extension pass grows it over anything equal (popular included; the junk
/// pass is a structural no-op with an empty bjunk).
fn find_longest_match(
    a: &[char],
    b: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, ch) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(indices) = b2j.get(ch) {
            for &j in indices {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j > 0 {
                    j2len.get(&(j - 1)).copied().unwrap_or(0)
                } else {
                    0
                } + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    // The plain extension pass (difflib's "non-junk" pass; with isjunk=None
    // the subsequent junk pass can never fire).
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generated from CPython 3.12 difflib — regenerate with:
    /// `python3 -c "import difflib,json; pairs=[...]; print(json.dumps(
    ///  [[a,b,difflib.SequenceMatcher(None,a,b).ratio()] for a,b in pairs]))"`
    /// (the pair list mirrors the cases below). Tolerance is exact-f64: the
    /// arithmetic is integer counts and one division on both sides.
    #[test]
    fn ratio_matches_cpython_difflib() {
        let fox200 = "the quick brown fox ".repeat(10);
        let cases: Vec<(String, String, f64)> = vec![
            ("".into(), "".into(), 1.0),
            ("".into(), "a".into(), 0.0),
            ("abc".into(), "".into(), 0.0),
            ("abc".into(), "abc".into(), 1.0),
            (
                "the capital of france".into(),
                "the capital of spain".into(),
                0.829_268_292_682_926_8,
            ),
            (
                "What is the powerhouse of the cell".into(),
                "the powerhouse of the cell is what".into(),
                0.764_705_882_352_941_1,
            ),
            ("kitten".into(), "sitting".into(), 0.615_384_615_384_615_4),
            ("abcd".into(), "dcba".into(), 0.25),
            (
                "héllo wörld".into(),
                "hello world".into(),
                0.818_181_818_181_818_2,
            ),
            ("日本語のテキスト".into(), "日本語のテクスト".into(), 0.875),
            (
                "emoji 🦤 test".into(),
                "emoji 🦆 test".into(),
                0.916_666_666_666_666_6,
            ),
            ("a b".into(), "ab".into(), 0.8),
            // The autojunk boundary: at len(b)=199 spaces anchor (no junking);
            // at len(b)=200 popular elements lose anchoring.
            (
                fox200.clone(),
                fox200[..199].into(),
                0.997_493_734_335_839_5,
            ),
            (fox200.clone(), fox200.clone(), 1.0),
            (
                format!("{} end", "x".repeat(250)),
                format!("{} fin", "x".repeat(250)),
                0.992_125_984_251_968_5,
            ),
            (
                format!("{}{}", "a".repeat(100), "b".repeat(100)),
                format!("{}{}", "a".repeat(100), "c".repeat(100)),
                0.5,
            ),
            (
                "word ".repeat(50),
                format!("{}{}", "word ".repeat(39), "diff ".repeat(11)),
                0.78,
            ),
        ];
        for (a, b, expected) in cases {
            let got = sequence_ratio(&a, &b);
            assert!(
                (got - expected).abs() < 1e-12,
                "ratio({:?}…, {:?}…) = {got}, expected {expected}",
                &a.chars().take(24).collect::<String>(),
                &b.chars().take(24).collect::<String>(),
            );
        }
    }

    #[test]
    fn autojunk_pairs_agree_with_cpython_in_both_directions() {
        // b2j (and so autojunk) is built over `b` only, so direction could
        // matter; CPython yields the same ratio both ways for these pairs
        // (anchoring lost to junking is recovered by the extension pass) —
        // pin both orientations so the port can't diverge directionally.
        let a = "x".repeat(250);
        let b = format!("{a} end");
        let expected = 0.992_063_492_063_492_1; // CPython, both directions
        assert!((sequence_ratio(&a, &b) - expected).abs() < 1e-12);
        assert!((sequence_ratio(&b, &a) - expected).abs() < 1e-12);
        let c = "ab".repeat(100); // popular pair-chars when on the b side
        let d = "ab".repeat(99);
        let expected = 0.994_974_874_371_859_3;
        assert!((sequence_ratio(&c, &d) - expected).abs() < 1e-12);
        assert!((sequence_ratio(&d, &c) - expected).abs() < 1e-12);
    }
}
