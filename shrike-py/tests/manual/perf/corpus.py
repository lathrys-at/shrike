"""Deterministic synthetic corpus generator for the performance harness.

Builds repeatable Anki collections at the sizes that matter — 500, 5k, and 50k
notes — in ``text`` and ``text+image`` variants, through the REAL native write path
(``upsert_notes``), so the fixture is shaped like production rather than planted
behind it. The content is synthetic but production-shaped: varied field lengths,
HTML and cloze markup, tags, and a spread across several decks and the default
notetypes. In the text+image variant ~1 note in 10 carries a procedurally-generated
image (random shapes or rendered words, varied per note), so the media path is
exercised without every note being media.

Everything is seeded — a given :class:`CorpusSpec` yields a byte-identical
collection every run. The build is disposable and gitignored: it lands under
``.cache/perf/corpora/<key>/`` keyed by a content hash of the spec and is reused
if already present. No binary fixture is committed.

Run it directly to (re)build one::

    python shrike-py/tests/manual/perf/corpus.py --notes 10000 --variant text+image
    python shrike-py/tests/manual/perf/corpus.py --notes 50000 --variant text --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from a bare checkout without an editable install. _ROOT
# (shrike-py) carries the `tests.manual.perf.*` package; _SRC carries `shrike.*`.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shrike.harness.collection import CollectionWrapper  # noqa: E402
from tests.manual.perf import wordlist  # noqa: E402

# Bump when the generation logic changes in a way that alters the produced bytes;
# it folds into the cache key so a stale corpus is rebuilt rather than reused.
# (The active vocabulary's fingerprint also folds into the key — see CorpusSpec.key.)
GENERATOR_VERSION = 6

VARIANTS = ("text", "text+image")

# The canonical corpus sizes the harness benchmarks at. Small enough to keep the
# dev loop tractable, large enough to surface the O(collection) failure modes a
# perf audit hunts (per-op scans, N+1 hydration); 50k is the heaviest standard
# run, with 500/5k as the fast-feedback rungs.
STANDARD_SIZES = (500, 5_000, 50_000)

# The default cache root (repo-root .cache, gitignored — never ~/.cache).
DEFAULT_CACHE_ROOT = _ROOT.parent / ".cache" / "perf" / "corpora"

# The OFFLINE FALLBACK vocabulary. The real corpus draws from a large public-domain
# wordlist (`wordlist.ensure_wordlist` / `vocab()`); this small domain-flavoured set
# is used only when that cache is absent (offline, or a manual test that hasn't
# fetched it), so generation still works — with reduced trigram realism. It folds
# through `vocab()` into the cache-key fingerprint, so a fallback-built corpus never
# aliases a full-built one.
_FALLBACK_VOCAB = [
    "mitochondria",
    "enzyme",
    "synthesis",
    "gradient",
    "membrane",
    "catalyst",
    "photon",
    "entropy",
    "vector",
    "matrix",
    "tensor",
    "manifold",
    "topology",
    "lemma",
    "theorem",
    "proof",
    "axiom",
    "neuron",
    "synapse",
    "cortex",
    "dopamine",
    "receptor",
    "ligand",
    "peptide",
    "ribosome",
    "codon",
    "allele",
    "orbital",
    "valence",
    "isotope",
    "covalent",
    "reaction",
    "reagent",
    "solvent",
    "titration",
    "molarity",
    "epoch",
    "latency",
    "throughput",
    "cache",
    "pipeline",
    "kernel",
    "buffer",
    "mutex",
    "semaphore",
    "thread",
    "harvest",
    "sediment",
    "basalt",
    "tectonic",
    "erosion",
    "glacier",
    "estuary",
    "biome",
    "canopy",
    "nitrogen",
    "sonnet",
    "stanza",
    "metaphor",
    "cadence",
    "motif",
    "fugue",
    "counterpoint",
    "timbre",
    "register",
    "tonic",
    "dynasty",
    "treaty",
    "armistice",
    "suffrage",
    "republic",
    "doctrine",
    "schism",
    "reformation",
    "envoy",
    "quantum",
    "boson",
    "lepton",
    "hadron",
    "neutrino",
    "fermion",
    "plasma",
    "quark",
]

_VOCAB: list[str] | None = None


def vocab() -> list[str]:
    """The active vocabulary, shared by generation and the search workload: the
    large cached wordlist when present, else :data:`_FALLBACK_VOCAB`. Memoized so
    a single process sees one stable list (determinism). Never downloads — a perf
    run calls :func:`wordlist.ensure_wordlist` first."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = wordlist.load_wordlist(_FALLBACK_VOCAB)
    return _VOCAB


# Words are drawn Zipfian and PER TOPIC, not from one global distribution. A real
# collection is a mixture of decks, each about its own subject with its own
# vocabulary, and within a deck a small head of domain terms recurs heavily with a
# long rare tail. A single global distribution (or a uniform draw over the whole
# wordlist) is unrepresentative of FTS5 behaviour — it spreads every term across
# the entire collection. So the corpus is partitioned into ~`notes // _DECK_SIZE`
# topics (≈ one deck of _DECK_SIZE cards each), and each topic draws Zipfian from
# its OWN sub-vocabulary sampled from the wordlist: a term then lives in ~one deck,
# so a query for it matches ~one deck's worth of notes, not a thin global spread.
# The wordlist carries no frequency data, so per-topic sampling + Zipf weighting
# synthesizes only the SHAPE; the words (hence trigrams) stay real-English.
_DECK_SIZE = 500  # cards per deck/topic — a realistic deck size sets the topic count
_TOPIC_VOCAB_WORDS = 4000  # distinct terms a single deck's domain draws from
_ZIPF_EXPONENT = 1.07  # ~natural-language word-frequency exponent
_TOPIC_SEED = 0x70D1C
_TOPIC_CACHE: dict[int, tuple[list[str], list[float]]] = {}

# Fraction of words drawn from the shared common set rather than the deck's domain.
# Real prose is ~half function words; flashcards are terser and content-denser, so
# domain terms dominate here — but common words still carry the common English
# trigrams that recur across every deck. An explicit share, rather than letting
# Zipf head-position decide it (105 common words at the head swamped the domain at
# ~66% of tokens), keeps domain the majority while preserving cross-deck overlap.
_COMMON_SHARE = 0.4

# Common English function words (≥3 chars, so the trigram tokenizer indexes them),
# shared by EVERY deck. They are the most frequent words in any real text, and
# their trigrams ("the", "and", "tha", "ing", "ion", "hat", …) are the common
# English trigrams — so a fuzzy query carrying them matches broadly, while domain
# terms stay deck-local. Roughly frequency-ordered (Zipf-weighted among themselves).
_COMMON_WORDS = [
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "any",
    "can",
    "has",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "who",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
    "way",
    "that",
    "this",
    "with",
    "have",
    "from",
    "they",
    "what",
    "when",
    "your",
    "will",
    "said",
    "each",
    "which",
    "their",
    "would",
    "there",
    "about",
    "other",
    "were",
    "been",
    "than",
    "them",
    "then",
    "some",
    "into",
    "only",
    "over",
    "also",
    "back",
    "after",
    "first",
    "well",
    "year",
    "work",
    "such",
    "make",
    "even",
    "most",
    "give",
    "very",
    "just",
    "much",
    "like",
    "through",
    "between",
    "before",
    "because",
    "those",
    "these",
    "where",
    "while",
    "being",
    "under",
    "never",
    "again",
    "still",
    "every",
    "great",
    "might",
    "against",
    "during",
    "without",
    "another",
    "however",
    "people",
    "should",
    "could",
    "around",
]
_COMMON_SET = frozenset(_COMMON_WORDS)
# Cumulative Zipf weights over the common words themselves ("the" ≫ "around"),
# computed once; the list above is roughly frequency-ordered.
_COMMON_CUM = list(
    itertools.accumulate(1.0 / (r**_ZIPF_EXPONENT) for r in range(1, len(_COMMON_WORDS) + 1))
)


def n_topics(corpus_size: int) -> int:
    """The number of distinct topic vocabularies a corpus of ``corpus_size`` notes
    is built from — one per ~`_DECK_SIZE` notes (a realistic deck), at least one."""
    return max(1, corpus_size // _DECK_SIZE)


def _topic_domain(topic: int) -> tuple[list[str], list[float]]:
    """A deck's domain sub-vocabulary (sampled from the wordlist, seeded by
    ``topic``) and its cumulative Zipf weights, memoized. Common words are excluded
    so the two distributions don't overlap; different topics get near-disjoint
    samples (4k of ~359k), so decks barely share domain terms."""
    state = _TOPIC_CACHE.get(topic)
    if state is None:
        words = vocab()
        sample = random.Random(_TOPIC_SEED ^ topic).sample(
            words, min(_TOPIC_VOCAB_WORDS, len(words))
        )
        domain = [w for w in sample if w not in _COMMON_SET]
        cum = list(
            itertools.accumulate(1.0 / (r**_ZIPF_EXPONENT) for r in range(1, len(domain) + 1))
        )
        state = (domain, cum)
        _TOPIC_CACHE[topic] = state
    return state


def choose(rng: random.Random, k: int, topic: int = 0, *, ensure_domain: bool = False) -> list[str]:
    """``k`` words for ``topic``: each is, with probability :data:`_COMMON_SHARE`, a
    shared common word (Zipfian — its trigrams recur across every deck), else one
    of the deck's domain terms (Zipfian, deck-local). So a query carrying a common
    trigram matches broadly while a domain term matches ~one deck.

    ``ensure_domain`` (generation only) guarantees at least one domain word, so no
    note ends up all function words — every note carries searchable deck content.
    Queries leave it off (a real query may be all-common)."""
    domain, dom_cum = _topic_domain(topic)
    out = [
        rng.choices(_COMMON_WORDS, cum_weights=_COMMON_CUM, k=1)[0]
        if rng.random() < _COMMON_SHARE
        else rng.choices(domain, cum_weights=dom_cum, k=1)[0]
        for _ in range(k)
    ]
    if ensure_domain and k > 0 and domain and all(w in _COMMON_SET for w in out):
        out[rng.randrange(k)] = rng.choices(domain, cum_weights=dom_cum, k=1)[0]
    return out


def _word_count(rng: random.Random) -> int:
    """A field/sentence word count: a normal draw centred at 15, clamped to
    [5, 30] — middle-weighted and longer than a flat short range, so the text
    reads like real study notes and carries enough trigrams to be representative."""
    return min(30, max(5, round(rng.gauss(15, 5))))


# A few HTML wrappers sprinkled into back fields so the stripper does real work
# (the strip-skip fast path is byte-identity only when there is no `<`).
_HTML_WRAPS = ("<b>{}</b>", "<i>{}</i>", "<u>{}</u>", "{}<br>", "<span>{}</span>")


@dataclass(frozen=True)
class CorpusSpec:
    """What to generate: how many notes, which modality variant, and the seed
    that makes it reproducible."""

    notes: int
    variant: str = "text"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.notes < 1:
            raise ValueError(f"notes must be >= 1 (got {self.notes})")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS} (got {self.variant!r})")

    @property
    def key(self) -> str:
        """A short content hash of the spec — the cache directory discriminator.
        Includes the active vocabulary's fingerprint, so a corpus built from the
        small fallback never aliases (or is reused as) one built from the full
        wordlist, and a wordlist change invalidates cached corpora."""
        fp = wordlist.fingerprint(vocab())
        raw = f"v{GENERATOR_VERSION}:{self.notes}:{self.variant}:{self.seed}:{fp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def dirname(self) -> str:
        return f"{self.variant.replace('+', '_')}-{self.notes}-s{self.seed}-{self.key}"


@dataclass(frozen=True)
class BuiltCorpus:
    """A materialized corpus: where it lives and what it contains."""

    spec: CorpusSpec
    anki2_path: Path
    media_dir: Path
    note_count: int


def _note_rng(spec: CorpusSpec, index: int) -> random.Random:
    """A per-note generator, independent yet deterministic across runs."""
    return random.Random((spec.seed << 21) ^ (index * 0x9E3779B1) ^ GENERATOR_VERSION)


def _sentence(rng: random.Random, n_words: int, topic: int) -> str:
    words = choose(rng, n_words, topic, ensure_domain=True)
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _back_text(rng: random.Random, topic: int) -> str:
    parts = [_sentence(rng, _word_count(rng), topic) for _ in range(rng.randint(1, 4))]
    # Wrap a fraction of sentences in HTML so the field exercises the stripper.
    out = []
    for p in parts:
        out.append(rng.choice(_HTML_WRAPS).format(p) if rng.random() < 0.5 else p)
    return " ".join(out)


def _tags(rng: random.Random) -> list[str]:
    return rng.sample(vocab(), k=rng.randint(0, 3))


def _make_image(rng: random.Random, topic: int, size: int = 96) -> bytes:
    """A small procedurally-generated PNG with per-note diversity. Over a random
    background, an image is EITHER a handful of random shapes OR a single rendered
    word — real cards carry both diagrams and text, so mixing the two keeps the
    image set from being uniformly abstract and exercises the text-on-image path
    (image embedding now, OCR/recognition later). The word is drawn from the same
    vocabulary the search workload queries with (``vocab()``), in a colour that contrasts
    the background so it stays legible — so under a real CLIP backend a text query
    for the word lands near this image in the shared space. Deterministic per the
    note's seeded ``rng``."""
    from PIL import Image, ImageDraw, ImageFont

    def color() -> tuple[int, int, int]:
        return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

    bg = color()
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    if rng.random() < 0.4:
        # A single search-query term (a domain word), high-contrast on the background.
        word = choose(rng, 1, topic, ensure_domain=True)[0]
        ink = (0, 0, 0) if sum(bg) > 384 else (255, 255, 255)
        font = ImageFont.load_default(size=rng.randint(14, 24))
        draw.text((rng.randint(2, size // 3), rng.randint(2, size // 2)), word, fill=ink, font=font)
    else:
        for _ in range(rng.randint(3, 9)):
            x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
            x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
            box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            shape = rng.choice(("rectangle", "ellipse", "line"))
            if shape == "line":
                draw.line([x0, y0, x1, y1], fill=color(), width=rng.randint(1, 5))
            else:
                getattr(draw, shape)(box, fill=color(), outline=color())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generate_notes(spec: CorpusSpec, media_dir: Path) -> list[dict]:
    """Build the note dicts (and, for the image variant, write each note's image
    into ``media_dir`` and reference it in a field)."""
    decks = n_topics(spec.notes)
    notes: list[dict] = []
    for i in range(spec.notes):
        rng = _note_rng(spec, i)
        # The note's deck IS its topic: each deck draws from its own sub-vocabulary
        # (a domain), so terms cluster by deck like a real collection.
        topic = i % decks
        deck = f"Perf::Deck {topic:03d}"
        tags = _tags(rng)
        # Every fifth note is a cloze; the rest are Basic — two notetypes so the
        # write path resolves more than one field layout.
        if i % 5 == 4:
            word = choose(rng, 1, topic, ensure_domain=True)[0]
            text = f"{_sentence(rng, _word_count(rng), topic)} The key term is {{{{c1::{word}}}}}."
            fields = {"Text": text, "Back Extra": _back_text(rng, topic)}
            note_type, image_field = "Cloze", "Back Extra"
        else:
            fields = {
                "Front": _sentence(rng, _word_count(rng), topic),
                "Back": _back_text(rng, topic),
            }
            note_type, image_field = "Basic", "Back"
        # Only ~1 note in 10 carries an image — real collections aren't all-media.
        if spec.variant == "text+image" and i % 10 == 0:
            name = f"perf_{spec.seed}_{i}.png"
            (media_dir / name).write_bytes(_make_image(rng, topic))
            fields[image_field] += f'<img src="{name}">'
        notes.append({"deck": deck, "note_type": note_type, "tags": tags, "fields": fields})
    return notes


def build_corpus(spec: CorpusSpec, dest: Path) -> BuiltCorpus:
    """Build the corpus into ``dest`` (overwriting any prior build there)."""
    dest.mkdir(parents=True, exist_ok=True)
    anki2 = dest / "collection.anki2"
    # A fresh build: anki would otherwise reopen (and append to) an existing file.
    for suffix in ("", "-wal", "-shm"):
        leftover = anki2.with_name(anki2.name + suffix)
        if leftover.exists():
            leftover.unlink()

    wrapper = CollectionWrapper(str(anki2))
    media_dir = Path(wrapper.media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    try:
        notes = _generate_notes(spec, media_dir)
        # `allow`, not `error`: the synthetic first fields are short random
        # sentences, so at 50k+ notes two of them collide (birthday paradox) and
        # Anki's first-field-duplicate rule would abort the whole build. A handful
        # of duplicate fronts is harmless — even realistic — for a perf fixture, so
        # the corpus is always created at the requested size.
        results = wrapper.run_sync(
            lambda core: json.loads(core.upsert_notes(json.dumps(notes), "allow", False))
        )
    finally:
        wrapper.close()

    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        first = errors[0]
        raise RuntimeError(
            f"corpus build hit {len(errors)} upsert error(s); first: "
            f"note[{first.get('index')}] {first.get('error')}"
        )
    ok = sum(1 for r in results if r.get("status") in ("created", "updated"))
    return BuiltCorpus(spec=spec, anki2_path=anki2, media_dir=media_dir, note_count=ok)


def ensure_corpus(spec: CorpusSpec, cache_root: Path | None = None) -> BuiltCorpus:
    """Return a cached corpus for ``spec``, building it once if absent. The build
    is keyed by the spec's content hash; a completed build drops a ``.complete``
    marker so a half-written one (interrupted) is rebuilt rather than trusted."""
    root = cache_root or DEFAULT_CACHE_ROOT
    dest = root / spec.dirname
    marker = dest / ".complete"
    anki2 = dest / "collection.anki2"
    if marker.is_file() and anki2.is_file():
        # media_dir is a purely lexical derivation (<stem>.media, matching
        # CollectionWrapper.media_dir) — no need to open the collection on reuse.
        return BuiltCorpus(
            spec=spec,
            anki2_path=anki2,
            media_dir=anki2.with_suffix(".media"),
            note_count=spec.notes,
        )
    built = build_corpus(spec, dest)
    marker.write_text(f"{spec.dirname}\n{built.note_count} notes\n")
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, required=True, help="Number of notes to generate.")
    parser.add_argument("--variant", choices=VARIANTS, default="text", help="Modality variant.")
    parser.add_argument("--seed", type=int, default=0, help="Determinism seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Build directory (default: the cached .cache/perf/corpora/<key>/).",
    )
    args = parser.parse_args()
    # Fetch the large wordlist (once, cached) so a direct build uses the full
    # vocabulary rather than the small fallback.
    wordlist.ensure_wordlist()
    spec = CorpusSpec(notes=args.notes, variant=args.variant, seed=args.seed)
    built = build_corpus(spec, args.out) if args.out else ensure_corpus(spec)
    print(
        f"Built {built.note_count} notes ({spec.variant}) -> {built.anki2_path}"
        + (f"  (+images in {built.media_dir})" if spec.variant == "text+image" else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
