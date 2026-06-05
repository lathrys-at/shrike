# Changelog

All notable changes to Shrike are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/). While in `0.x`, the public surface
(MCP tool schemas, CLI, config) may change between minor versions.

## [Unreleased]

### Added
- Cooperative collection locking (opt-in): `shrike server start --cooperative-lock`
  releases Anki's exclusive collection lock when the daemon is idle and re-opens
  it on demand, so an idle daemon no longer blocks launching Anki desktop against
  the same collection. The hold window is `--lock-hold-seconds` (config
  `server.lock_hold_seconds`, default 5s). On each re-acquire the index re-checks
  drift and rebuilds if the collection changed while released. The default
  permanent-hold behaviour is unchanged. `server status` (and `/status`) now
  report the locking mode and whether the collection is currently held (#64).
- `shrike collection reload` (and a `POST /reload` control endpoint) — close and
  re-open the collection without restarting the daemon, picking up changes made to
  the collection file on disk (a restored backup, a file-level sync or swap) and
  re-checking the search index for drift (rebuilding in the background if it
  moved). Groundwork for cooperative locking (#79, #64).
- `migrate_note_type` MCP tool and `shrike note migrate-type` — change a set of
  notes from one note type to another with an explicit field (and optional
  template) map, the way Anki's "Change Note Type" does: note IDs and, for mapped
  templates, review scheduling are preserved. It's the history-safe way to convert
  Basic↔Cloze, consolidate redundant note types, or adopt a richer template. The
  map is required and explicit — a source field you don't map is dropped (and
  reported), unknown names or ambiguous (two→one) maps error rather than guess —
  and it previews by `dry_run`/`--dry-run` before applying. `upsert_notes` still
  refuses a type change (#75).
- `collection_query` MCP tool and `shrike collection query "<expression>"` — find
  notes with a **raw Anki search expression**. The string is passed straight to
  Anki's search engine, so the full language works (`is:due`, `prop:ivl>=30`,
  `added:`, `rated:`, `flag:`, `nid:`/`cid:`, `OR`/`-`/parentheses). It's the
  power-user escape hatch, distinct from `search_notes` (meaning/text) and
  `list_notes` (structured filters), and returns the same note shape as
  `list_notes`. This restores the raw query removed in #86, now as an explicit
  tool rather than a `list_notes` param (#97).
- Collection cleanup: `collection_prune` MCP tool and `shrike collection prune`
  (a new `collection` CLI group) — clear unused tags, remove empty notes, and
  remove empty cards. Select cleanups with flags (`--unused-tags`/`--empty-notes`/
  `--empty-cards`); with none selected, all run. It **previews by default**
  (`dry_run` defaults true; the CLI applies only with `--apply`, confirming first
  unless `--yes`) because it deletes notes and cards collection-wide. An empty
  note is one whose every field is blank — no text *and* no media — so an image-
  or audio-only note is kept. Folds in the old `clear_unused_tags`/`tag clean`
  (removed in #90) and supersedes a standalone remove-empty-notes (#89, #78).
- `find_replace_note_types` MCP tool (#76, anki-connect's `findAndReplaceInModels`):
  literal-or-regex find/replace inside one note type's card-template HTML and
  shared CSS, scoped by `front`/`back`/`css` selectors, returning a replacement
  count and the templates/CSS that changed. It rewrites template text only — no
  note field values are touched and no note data is migrated — so it bumps the
  index `col_mod` without re-embedding. `match_case` defaults to true (template
  and CSS text is code). Typed `ShrikeClient.find_replace_note_types` too.
- Bulk find-and-replace across note fields: `find_replace_notes` MCP tool and
  `shrike note replace SEARCH REPLACE`. Scoped by `--deck`/`--tags`/`--type`/
  `--ids` (a scope is required), optional `--regex` (Anki's engine; `$1` capture
  refs), `--match-case`, and `--field` to limit to one field. The CLI previews the
  changes and asks before applying (`--dry-run` to preview only, `--yes` to skip
  the prompt); the MCP tool applies by default with a `dry_run` option. Changed
  notes are re-embedded so semantic search stays correct, and the edit is undoable
  in Anki (#85).
- `search_notes` / `shrike note search` now match each query **both** by semantic
  similarity and as an exact, case-insensitive substring of note fields, folded
  into one result list. Each hit carries a `score` when semantically ranked and a
  `substring` annotation (matched fields + snippet) when the text occurs
  literally — both when both apply. Exact matches are returned even when the
  embedding index is unavailable (the response notes semantic ranking was
  skipped), so text search works without embeddings. This is the single-query /
  annotated-evidence contract future retrieval backends plug into (n-gram, #98).
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

### Removed
- `shrike note list --query` and the `query` param of `list_notes` (the raw Anki
  search escape hatch). Text search now lives in `search_notes`; structured
  filters cover deck/tag/type. Raw Anki query expressions (`is:due`, `prop:`,
  `added:`, …) will return as a dedicated `shrike collection query` tool (#97,
  #86).

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
