"""Typo-query generator with gold-by-construction for the fuzzy-recall eval.

The fuzzy rare-trigram cap trades posting-read cost against recall on typo'd
queries, and nothing measured that recall before this lane. The hand-curated
``search_quality`` corpus (~46 cards) is far too small to measure a trigram-count
trade — at that scale a query's rarest 6 vs rarest 9 trigrams retrieve the same
handful of notes. So gold is generated **by construction** over the realistic perf
corpus (5k/50k notes): a content phrase is drawn from a sampled note, a typo is
injected into it, and the gold target is **every note whose clean indexed text
contains the un-perturbed phrase** (the exact-substring path, for honest recall).
This yields thousands of labelled fuzzy queries at real scale, deterministically.

Two perturbation models, both seeded:

- **Synthetic edits** — substitution (adjacent-keyboard and random), transposition,
  deletion, insertion (added and doubled), case error, and a simple phonetic swap.
  Multiple edits per query, stratified into 1/2/3-typo buckets, so the eval shows
  how each cap policy handles increasingly degraded queries. Each query records the
  edit kinds applied, for a per-edit-type breakdown.
- **Real misspellings** — when a sampled phrase contains a word the curated
  ``misspelling -> correction`` map knows (keyed on the CORRECT word), the
  misspelling is substituted in. This covers genuine human error the synthetic
  model cannot reproduce.

The phrase LENGTH is the independent variable: a 1-word phrase produces ~4 trigrams
(the cap stays at its floor), a 6-word phrase ~40 (the curve acts), so varying it
populates the full ``n``-trigram range the cap policies span. Everything is keyed
off the note id and a generator version, so a query set is byte-reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

# Bump when the generation logic changes in a way that alters the produced queries;
# it folds into the query-set version so a stale set is regenerated, not reused.
GENERATOR_VERSION = 1


class EditKind(StrEnum):
    """The perturbation applied to a phrase. ``REAL_MISSPELLING`` is the curated
    human-error substitution; the rest are the synthetic edit model."""

    SUBSTITUTE_ADJACENT = "substitute_adjacent"  # a neighbouring-key letter
    SUBSTITUTE_RANDOM = "substitute_random"  # any random letter
    TRANSPOSE = "transpose"  # swap two adjacent letters (protien)
    DELETE = "delete"  # drop a letter (mitochndria)
    INSERT = "insert"  # add a random letter
    DOUBLE = "double"  # double an existing letter
    CASE = "case"  # flip a letter's case
    PHONETIC = "phonetic"  # a simple sound-alike swap (ph<->f, ie<->ei)
    REAL_MISSPELLING = "real_misspelling"  # a curated misspelling->correction pair


# QWERTY adjacency (lowercase). A substitution drawn from a key's neighbours models
# a real fat-finger error far better than a uniform random letter — the trigram it
# breaks is the same, but the resulting word is a plausible human mistype.
_ADJACENT: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

# Simple phonetic / digraph swaps applied as a substring rewrite (the first one
# whose left side is present fires). Models the "sounds the same" error class.
_PHONETIC_SWAPS: tuple[tuple[str, str], ...] = (
    ("ph", "f"),
    ("ie", "ei"),
    ("ei", "ie"),
    ("tion", "sion"),
    ("ck", "k"),
    ("c", "k"),
    ("s", "z"),
)

# The synthetic edit kinds the generator draws from (REAL_MISSPELLING is applied on
# a separate, content-keyed path, not drawn here).
_SYNTHETIC_KINDS: tuple[EditKind, ...] = (
    EditKind.SUBSTITUTE_ADJACENT,
    EditKind.SUBSTITUTE_RANDOM,
    EditKind.TRANSPOSE,
    EditKind.DELETE,
    EditKind.INSERT,
    EditKind.DOUBLE,
    EditKind.CASE,
    EditKind.PHONETIC,
)

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


class GoldResolver(Protocol):
    """Maps a clean phrase to the gold note id set — every note whose indexed text
    contains it. The driver supplies one backed by the derived store's substring
    path; typing it as a protocol keeps this generator independent of the native
    engine."""

    def __call__(self, phrase: str) -> Sequence[int]: ...


@dataclass(frozen=True)
class TypoQuery:
    """One generated fuzzy query: the perturbed text, the gold note set, and the
    provenance the eval aggregates over (which note it was drawn from, the clean
    phrase, its trigram count and word length, and the edit kinds applied)."""

    query: str
    gold_ids: frozenset[int]
    source_note_id: int
    clean_phrase: str
    n_trigrams: int
    n_words: int
    edits: tuple[EditKind, ...]

    @property
    def typo_count(self) -> int:
        return len(self.edits)


def _query_rng(seed: int, note_id: int) -> random.Random:
    """A per-(seed, note) generator, independent yet deterministic across runs."""
    return random.Random((seed << 27) ^ (note_id * 0x9E3779B1) ^ GENERATOR_VERSION)


def _apply_one_synthetic(word: str, kind: EditKind, rng: random.Random) -> str:
    """Apply one synthetic edit of ``kind`` to ``word``, returning the perturbed
    word. A word too short for the edit is returned unchanged (the caller records
    the attempted kind regardless, so the per-type aggregate counts the intent)."""
    chars = list(word)
    n = len(chars)
    if n == 0:
        return word
    if kind is EditKind.SUBSTITUTE_ADJACENT:
        i = rng.randrange(n)
        neighbours = _ADJACENT.get(chars[i].lower())
        if neighbours:
            repl = rng.choice(neighbours)
            chars[i] = repl.upper() if chars[i].isupper() else repl
        return "".join(chars)
    if kind is EditKind.SUBSTITUTE_RANDOM:
        i = rng.randrange(n)
        repl = rng.choice(_LETTERS)
        chars[i] = repl.upper() if chars[i].isupper() else repl
        return "".join(chars)
    if kind is EditKind.TRANSPOSE:
        if n < 2:
            return word
        i = rng.randrange(n - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if kind is EditKind.DELETE:
        if n < 2:  # never delete to empty
            return word
        del chars[rng.randrange(n)]
        return "".join(chars)
    if kind is EditKind.INSERT:
        i = rng.randrange(n + 1)
        chars.insert(i, rng.choice(_LETTERS))
        return "".join(chars)
    if kind is EditKind.DOUBLE:
        i = rng.randrange(n)
        chars.insert(i, chars[i])
        return "".join(chars)
    if kind is EditKind.CASE:
        i = rng.randrange(n)
        chars[i] = chars[i].upper() if chars[i].islower() else chars[i].lower()
        return "".join(chars)
    if kind is EditKind.PHONETIC:
        lower = word.lower()
        for left, right in _PHONETIC_SWAPS:
            pos = lower.find(left)
            if pos >= 0:
                return word[:pos] + right + word[pos + len(left) :]
        return word  # no digraph present — unchanged
    return word


def _inject_synthetic(
    phrase: str, typo_count: int, rng: random.Random
) -> tuple[str, list[EditKind]]:
    """Apply ``typo_count`` synthetic edits to ``phrase``, each to a re-chosen word
    (the longest words are preferred so the edit lands on content, not a function
    word). Returns the perturbed phrase and the applied edit kinds in order."""
    words = phrase.split()
    if not words:
        return phrase, []
    applied: list[EditKind] = []
    for _ in range(typo_count):
        # Prefer a longer word (>= 4 chars) so the edit hits a content word and the
        # trigram damage is meaningful; fall back to any word if none qualify.
        candidates = [i for i, w in enumerate(words) if len(w) >= 4] or list(range(len(words)))
        wi = rng.choice(candidates)
        kind = rng.choice(_SYNTHETIC_KINDS)
        words[wi] = _apply_one_synthetic(words[wi], kind, rng)
        applied.append(kind)
    return " ".join(words), applied


def _inject_real_misspelling(
    phrase: str, correction_index: Mapping[str, str], rng: random.Random
) -> tuple[str, list[EditKind]]:
    """If ``phrase`` contains a word ``correction_index`` knows (keyed on the
    correctly-spelled word), replace ONE such occurrence with its misspelling.
    Returns the perturbed phrase and ``[REAL_MISSPELLING]`` when one fired, else the
    phrase unchanged and ``[]`` (the caller falls back to a synthetic edit)."""
    words = phrase.split()
    hits = [i for i, w in enumerate(words) if w.lower() in correction_index]
    if not hits:
        return phrase, []
    wi = rng.choice(hits)
    wrong = correction_index[words[wi].lower()]
    # Preserve a leading capital (sentence-start words are capitalized in the corpus).
    words[wi] = wrong.capitalize() if words[wi][:1].isupper() else wrong
    return " ".join(words), [EditKind.REAL_MISSPELLING]


def _invert_misspellings(misspellings: Mapping[str, str]) -> dict[str, str]:
    """``correction -> misspelling``, inverted from the source ``misspelling ->
    correction`` map so a correctly-spelled corpus word can be perturbed into its
    known error. On a correction with several misspellings the first in iteration
    order wins (deterministic for an ordered map)."""
    index: dict[str, str] = {}
    for wrong, right in misspellings.items():
        index.setdefault(right, wrong)
    return index


def _trigram_count(text: str) -> int:
    """The number of character trigrams in ``text`` lowercased — the ``n`` the cap
    policy keys on (mirrors the native ``trigrams`` window count: chars - 2 per run,
    counting spaces, floored at 0)."""
    lowered = text.lower()
    return max(0, len(lowered) - 2)


def length_bucket(n_trigrams: int) -> str:
    """The query-length bucket (by trigram count) the eval aggregates within. The
    cap curve only acts on n > the floor, so a global mean would hide its effect —
    the buckets straddle the floor (6) and the ceiling (12)."""
    if n_trigrams <= 6:
        return "n<=6"
    if n_trigrams <= 11:
        return "n7-11"
    if n_trigrams <= 17:
        return "n12-17"
    return "n18+"


#: The phrase word-length strata, drawn round-robin so each query set spans the
#: full trigram range the cap policies act on (a 1-word phrase pins the cap at its
#: floor; a 6-word phrase drives the curve toward the ceiling).
_PHRASE_WORD_LENGTHS = (1, 2, 3, 4, 6)

#: The typo-count strata (1 / 2 / 3 edits), drawn round-robin so each cap policy is
#: measured against increasingly degraded queries.
_TYPO_COUNTS = (1, 2, 3)


def _clean_phrase(text: str, n_words: int, rng: random.Random) -> str | None:
    """A contiguous ``n_words``-word phrase drawn from ``text`` (a note's clean
    indexed field). ``None`` when the text has too few words. Trailing punctuation
    is stripped so the phrase is a literal substring of the indexed text (the gold
    substring probe is exact)."""
    words = [w.strip(".,;:!?()[]\"'") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n_words:
        return None
    start = rng.randrange(len(words) - n_words + 1)
    return " ".join(words[start : start + n_words])


def generate_queries(
    note_texts: Mapping[int, str],
    gold_for_phrase: GoldResolver,
    *,
    seed: int = 0,
    sample_size: int = 500,
    misspellings: Mapping[str, str] | None = None,
) -> list[TypoQuery]:
    """Generate a deterministic typo-query set over ``note_texts`` (note id -> its
    clean indexed first-field text).

    For each sampled note: draw a phrase (cycling the word-length strata), pick a
    typo count (cycling the strata), inject a real misspelling when the phrase
    carries a known word else ``typo_count`` synthetic edits, and resolve gold via
    ``gold_for_phrase`` (all notes whose clean text contains the un-perturbed
    phrase). A note whose text yields no usable phrase, or whose perturbation left
    the phrase unchanged, is skipped (so every returned query is a genuine typo with
    a non-empty gold set).

    ``gold_for_phrase`` is injected (not computed here) so the generator stays pure
    of the store — the driver passes a resolver backed by ``search_substring``.
    """
    correction_index = _invert_misspellings(misspellings or {})
    note_ids = sorted(note_texts)
    rng_sample = random.Random((seed << 17) ^ GENERATOR_VERSION)
    sampled = (
        rng_sample.sample(note_ids, sample_size) if sample_size < len(note_ids) else list(note_ids)
    )
    out: list[TypoQuery] = []
    for idx, note_id in enumerate(sampled):
        text = note_texts[note_id]
        rng = _query_rng(seed, note_id)
        n_words = _PHRASE_WORD_LENGTHS[idx % len(_PHRASE_WORD_LENGTHS)]
        phrase = _clean_phrase(text, n_words, rng)
        if phrase is None:
            continue
        gold = gold_for_phrase(phrase)
        if not gold:
            continue  # phrase not found in the index (e.g. markup) — drop it
        typo_count = _TYPO_COUNTS[idx % len(_TYPO_COUNTS)]
        # Try a real misspelling first (one edit); top up to typo_count with
        # synthetic edits so the typo-count strata are honoured either way.
        perturbed, edits = _inject_real_misspelling(phrase, correction_index, rng)
        if edits and typo_count > 1:
            perturbed, more = _inject_synthetic(perturbed, typo_count - 1, rng)
            edits = edits + more
        elif not edits:
            perturbed, edits = _inject_synthetic(phrase, typo_count, rng)
        if perturbed == phrase:
            continue  # no effective perturbation (e.g. all edits no-op'd) — drop
        out.append(
            TypoQuery(
                query=perturbed,
                gold_ids=frozenset(gold),
                source_note_id=note_id,
                clean_phrase=phrase,
                n_trigrams=_trigram_count(phrase),
                n_words=n_words,
                edits=tuple(edits),
            )
        )
    return out
