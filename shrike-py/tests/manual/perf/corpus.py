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
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from a bare checkout without an editable install.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shrike.harness.collection import CollectionWrapper  # noqa: E402

# Bump when the generation logic changes in a way that alters the produced bytes;
# it folds into the cache key so a stale corpus is rebuilt rather than reused.
GENERATOR_VERSION = 3

VARIANTS = ("text", "text+image")

# The canonical corpus sizes the harness benchmarks at. Small enough to keep the
# dev loop tractable, large enough to surface the O(collection) failure modes a
# perf audit hunts (per-op scans, N+1 hydration); 50k is the heaviest standard
# run, with 500/5k as the fast-feedback rungs.
STANDARD_SIZES = (500, 5_000, 50_000)

# The default cache root (repo-root .cache, gitignored — never ~/.cache).
DEFAULT_CACHE_ROOT = _ROOT.parent / ".cache" / "perf" / "corpora"

# A small fixed vocabulary (shared with the workloads): deterministic, varied text
# with no external data and no licensing surface. Domain-flavoured so fields read
# like real study notes.
VOCAB = [
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
        """A short content hash of the spec — the cache directory discriminator."""
        raw = f"v{GENERATOR_VERSION}:{self.notes}:{self.variant}:{self.seed}"
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


def _sentence(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _back_text(rng: random.Random) -> str:
    parts = [_sentence(rng, rng.randint(6, 18)) for _ in range(rng.randint(1, 4))]
    # Wrap a fraction of sentences in HTML so the field exercises the stripper.
    out = []
    for p in parts:
        out.append(rng.choice(_HTML_WRAPS).format(p) if rng.random() < 0.5 else p)
    return " ".join(out)


def _tags(rng: random.Random) -> list[str]:
    return rng.sample(VOCAB, k=rng.randint(0, 3))


def _make_image(rng: random.Random, size: int = 96) -> bytes:
    """A small procedurally-generated PNG with per-note diversity. Over a random
    background, an image is EITHER a handful of random shapes OR a single rendered
    word — real cards carry both diagrams and text, so mixing the two keeps the
    image set from being uniformly abstract and exercises the text-on-image path
    (image embedding now, OCR/recognition later). The word is drawn from VOCAB, the
    SAME vocabulary the search workload queries with, in a colour that contrasts
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
        # A single search-query term, high-contrast on the background.
        word = rng.choice(VOCAB)
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
    n_decks = min(50, max(1, spec.notes // 1000))
    notes: list[dict] = []
    for i in range(spec.notes):
        rng = _note_rng(spec, i)
        deck = f"Perf::Deck {i % n_decks:02d}"
        tags = _tags(rng)
        # Every fifth note is a cloze; the rest are Basic — two notetypes so the
        # write path resolves more than one field layout.
        if i % 5 == 4:
            word = rng.choice(VOCAB)
            text = f"{_sentence(rng, rng.randint(6, 14))} The key term is {{{{c1::{word}}}}}."
            fields = {"Text": text, "Back Extra": _back_text(rng)}
            note_type, image_field = "Cloze", "Back Extra"
        else:
            fields = {"Front": _sentence(rng, rng.randint(4, 10)), "Back": _back_text(rng)}
            note_type, image_field = "Basic", "Back"
        # Only ~1 note in 10 carries an image — real collections aren't all-media.
        if spec.variant == "text+image" and i % 10 == 0:
            name = f"perf_{spec.seed}_{i}.png"
            (media_dir / name).write_bytes(_make_image(rng))
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
        results = wrapper.run_sync(
            lambda core: json.loads(core.upsert_notes(json.dumps(notes), "error", False))
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
    spec = CorpusSpec(notes=args.notes, variant=args.variant, seed=args.seed)
    built = build_corpus(spec, args.out) if args.out else ensure_corpus(spec)
    print(
        f"Built {built.note_count} notes ({spec.variant}) -> {built.anki2_path}"
        + (f"  (+images in {built.media_dir})" if spec.variant == "text+image" else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
