# Changelog

All notable changes to Shrike are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/). While in `0.x`, the public surface
(MCP tool schemas, CLI, config) may change between minor versions.

## [Unreleased]

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
