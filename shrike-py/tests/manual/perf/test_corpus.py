"""The perf corpus generator (#865): deterministic, production-shaped collections
built through the real native write path.

Manual lane (off the per-PR critical path); it builds only tiny collections, so
it runs in seconds. The native upsert path is in every build, so no feature gate.
"""

from __future__ import annotations

import pytest

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


def test_image_corpus_writes_one_png_per_note(tmp_path):
    built = build_corpus(CorpusSpec(notes=12, variant="text+image"), tmp_path / "ti")
    assert built.note_count == 12
    assert len(list(built.media_dir.glob("*.png"))) == 12


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
