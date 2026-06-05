# Changelog

All notable changes to Shrike are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/). While in `0.x`, the public surface
(MCP tool schemas, CLI, config) may change between minor versions.

## [Unreleased]

### Added
- Decks can now be referenced by **ID** anywhere a deck name is taken — a bare
  numeric ID or a `#`-prefixed ID — across the CLI (`note list/create/update
  --deck`, `note search --deck`, `deck rename`, `deck delete`) and the MCP tools
  (`list_notes`, `search_notes`, `upsert_notes`, `delete_decks`). `#id` is always
  an ID; a bare number is tried as an ID first, falling back to a literal name if
  no deck has that ID (#88).
- Deck lifecycle (#74): `upsert_decks` and `delete_decks` MCP tools plus a
  `shrike deck create|rename|delete` CLI group.
  - `deck create NAME` makes an empty deck (nested `Parent::Child` ok);
    `deck rename OLD NEW` renames/reparents. `upsert_decks` mirrors `upsert_notes`
    (an `id` renames the existing deck; absent creates). Decks do not merge —
    renaming onto an existing name is an error.
  - `deck delete` / `delete_decks` is **empty-only**: a deck with cards in it or a
    subdeck is refused (reported `not_empty`). Move its notes out first, then
    delete — so deleting a deck can never delete a note.
- Tag curation beyond full-replace (#73): new MCP tools and matching CLI.
  - `update_note_tags` / `shrike note tag --set|--add|--remove`: edit tags on a
    set of notes. `--set` replaces (and `--set ""` clears); `--add`/`--remove`
    edit additively and combine in one call. Exactly one mode per call — there
    is no default, and `--set` cannot mix with `--add`/`--remove`.
  - `rename_tag` / `shrike tag rename OLD NEW [--note ID]`: rename a tag
    collection-wide, or exactly on specific notes (`jp` never touches `jp-verbs`).
  - Tag changes advance the vector index's stored `col_mod` without re-embedding
    (tags aren't part of embedding text), so a tag-only edit no longer forces a
    spurious full index rebuild on the next startup.
- `shrike server start --save-config` writes the resolved collection, server,
  embedding, cache, and index-tuning settings to the config file (#56).

### Changed
- `shrike server start` no longer writes `config.yml` on its own. It previously
  saved the flags only on the very first run (when no config existed) and then
  silently ignored later flags, so the on-disk config could diverge from how the
  daemon was actually running. The config file is now user-managed; pass the new
  `--save-config` flag to persist the resolved flags explicitly (#56).
- `shrike note tag` now requires choosing a mode (`--set`, `--add`, or
  `--remove`) and its output reports `notes_modified`/`not_found` instead of
  per-note upsert results (#73).

## [0.3.4] — 2026-06-01

### Fixed
- `shrike --version` crashed after the PyPI rename: Click looked up distribution
  metadata for `shrike`, but the published distribution is `shrike-mcp`. It now
  reads the version directly from `__version__` (#61).

## [0.3.3] — 2026-06-01

### Added
- Published to PyPI as `shrike-mcp` (the import package and `shrike` command are
  unchanged), with releases publishing via GitHub trusted publishing (#58).

## [0.3.2] — 2026-06-01

### Added
- Tag-triggered release workflow: pushing a `v*` tag runs the full cross-platform
  integration suite, builds the release artifacts (Python sdist + wheel, the
  `anki-cards.skill` bundle, and `SHA256SUMS`), and cuts a GitHub Release with them
  attached (notes from the matching `CHANGELOG.md` section) (#43).
- `readme = "README.md"` in package metadata, so the sdist/wheel carry a
  long description (#43).
- Reference `anki-cards` skill plugin for LLM-driven card creation, with a QA eval
  harness (`tests/qa/`) and a skill-packaging script (#21, #24, #25, #27, #29–#31).
- `shrike note tag <ids> --set …` for bulk tag replacement (#28).
- Configurable transport-security allow-list and a disable knob (#22).
- Embedding pooling-type setting and generic `llama-server` arg passthrough (#20).
- `shrike index save` and debounced index persistence (#19).
- `search_notes` query strings logged at DEBUG (#26).

### Changed
- Version is now derived from the git tag via hatch-vcs instead of a hand-edited
  `__version__` constant, ending tag/version drift (#42, #44).
- Note text is normalized (rendered, stable) before embedding (#32).
- Security hardening of the HTTP transport; concurrency serialization of
  collection access (#13, #17).

### Fixed
- Orphaned `llama-server` processes are reaped on startup (#15).

## [0.3.1] — 2026-05-29
### Changed
- Typed Pydantic schemas end-to-end as the single source of truth for every tool
  request/response (#12).

## [0.3.0] — 2026-05-29
### Added
- Standalone `shrike.client` library extracted from the CLI (#11).
- Embedding-service lifecycle (`shrike embedding start/stop`) with model-aware
  index invalidation (#10).
### Fixed
- Serialize collection access; upsert neighbor-retry hint (#9).

## [0.2.1] — 2026-05-27
### Changed
- Status-display polish, `llama-server` setup, and a CLI flag (#8).

## [0.2.0] — 2026-05-27
### Added
- Semantic search: `llama-server` embeddings, a USearch HNSW vector index, the
  `search_notes` tool, incremental index updates, startup drift detection, and the
  index build state machine (#7).

## [0.1.0] — 2026-05-25
### Added
- Initial release: the `shrike` CLI and the MCP server over streamable HTTP, with
  collection-info, note, and note-type tools; daemon lifecycle; tab completion (#6).

[Unreleased]: https://github.com/lathrys-at/shrike/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/lathrys-at/shrike/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/lathrys-at/shrike/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/lathrys-at/shrike/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/lathrys-at/shrike/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lathrys-at/shrike/releases/tag/v0.1.0
