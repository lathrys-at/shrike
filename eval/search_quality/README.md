# Search-quality adversarial corpus (#559)

This directory holds the **manual, never-in-CI** search-quality suite's corpus —
the data that drives `tests/integration/test_search_quality.py` (run only with
`SHRIKE_SEARCH_QUALITY=1`).

## What's here

- **`manifest.json`** — the graded adversarial corpus: cards (text-only,
  image-only-meaning, both-modality, distractors) + queries tagged by
  adversarial class (modality gap, portrait-vs-token-share, gate-no-inject,
  cross-lingual, over-return, semantic). Each `source: commons` image is keyed
  by a logical `handle` (substituted into a card's `$IMG:handle` at build time).
- **`resolved_urls.json`** — the **committed** pin: `handle` → Wikimedia Commons
  URL. Makes a replay reproducible without re-resolving.
- **`ASSETS.md`** — per-image attribution (Commons page / license / author),
  generated from the live API. Also backfills attribution for the reused
  `eval/multimodal/` images.
- **`cache/`** — **gitignored**: the downloaded image bytes. Never committed.

## Asset licensing

The image **bytes are not redistributed in this repository**. The manual suite
resolves each image via the Wikimedia Commons API and downloads it **on demand**
into the gitignored `cache/` at run time — only the pinned URLs and the
`ASSETS.md` attribution table are committed.

Licensing preference is **public domain / PD-art / CC0**; a few CC-BY-SA images
are used where no public-domain file fits a corpus need, and are attributed in
`ASSETS.md`. Each row there links its Commons page, where the full license terms
and authorship live.

## Regenerating

```bash
# resolve any new images, re-pin, regenerate ASSETS.md (incl. the multimodal backfill)
python scripts/eval_search_quality_corpus.py --backfill-multimodal
# re-resolve every term from scratch (a Commons file can be renamed/deleted)
python scripts/eval_search_quality_corpus.py --refresh --backfill-multimodal
```

## Running the suite

```bash
SHRIKE_SEARCH_QUALITY=1 pytest tests/integration/test_search_quality.py -m search_quality
```

It is excluded from CI three ways (Bazel `manual` + glob-exclude, the
`search_quality` marker + the `SHRIKE_SEARCH_QUALITY` env skip, and the coverage
`-m` exclusion).
