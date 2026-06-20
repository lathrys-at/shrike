"""The perf corpus generator: deterministic, production-shaped collections
built through the real native write path.

Manual lane (off the per-PR critical path); it builds only tiny collections, so
it runs in seconds. The native upsert path is in every build, so no feature gate.
"""

from __future__ import annotations

import pytest

import tests.manual.perf.corpus as corpus_mod
from tests.manual.perf.corpus import (
    STANDARD_SIZES,
    CorpusSpec,
    build_corpus,
    ensure_corpus,
)


def test_text_corpus_builds_through_the_real_write_path(tmp_path):
    built = build_corpus(CorpusSpec(notes=20, variant="text"), tmp_path / "t")
    assert built.note_count == 20
    assert built.anki2_path.is_file()


def test_build_tolerates_first_field_duplicates(tmp_path, monkeypatch):
    # At 50k+ notes the short random first fields collide (birthday paradox), and
    # Anki's first-field-duplicate rule would abort the whole build. The generator
    # writes with on_duplicate="allow", so a collision creates the note rather than
    # failing. Force the collision deterministically with an all-identical front.
    colliding = [
        {
            "deck": "Perf::Deck 00",
            "note_type": "Basic",
            "tags": [],
            "fields": {"Front": "identical front", "Back": f"body {i}"},
        }
        for i in range(5)
    ]
    monkeypatch.setattr(corpus_mod, "_generate_notes", lambda spec, media_dir: colliding)
    built = build_corpus(CorpusSpec(notes=5, variant="text"), tmp_path / "dup")
    assert built.note_count == 5  # all written despite identical first fields


def test_image_corpus_attaches_images_to_a_fraction_of_notes(tmp_path):
    built = build_corpus(CorpusSpec(notes=30, variant="text+image"), tmp_path / "ti")
    assert built.note_count == 30
    pngs = list(built.media_dir.glob("*.png"))
    # ~1 note in 10 carries an image (i % 10 == 0): 30 notes -> notes 0, 10, 20.
    assert len(pngs) == 3


def test_generation_is_deterministic(tmp_path):
    a = build_corpus(CorpusSpec(notes=8, variant="text+image", seed=3), tmp_path / "a")
    b = build_corpus(CorpusSpec(notes=8, variant="text+image", seed=3), tmp_path / "b")
    for png in a.media_dir.glob("*.png"):
        assert png.read_bytes() == (b.media_dir / png.name).read_bytes()


def test_ensure_corpus_caches_and_reuses(tmp_path):
    spec = CorpusSpec(notes=6, variant="text")
    first = ensure_corpus(spec, cache_root=tmp_path)
    assert (first.anki2_path.parent / ".complete").is_file()
    second = ensure_corpus(spec, cache_root=tmp_path)
    assert second.anki2_path == first.anki2_path


def test_standard_sizes_are_pinned():
    assert STANDARD_SIZES == (500, 5_000, 50_000)


def test_spec_rejects_bad_input():
    with pytest.raises(ValueError, match="variant"):
        CorpusSpec(notes=1, variant="bogus")
    with pytest.raises(ValueError, match="notes"):
        CorpusSpec(notes=0)
