# Changelog

All notable changes to Shrike are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/). While in `0.x`, the public surface
(MCP tool schemas, CLI, config) may change between minor versions.

## [Unreleased]

### Added
- The config file understands the new capability sections (#498): an
  `embedders:` list (each entry declaring its `modalities` and `runtime:
  onnx | remote`), a `recognizers:` map, and a `managed:` section for the
  components Shrike runs for you. These replace the
  `embedding:`/`recognition:` sections and the `--embedding-*` flags, which
  still work for one more release and print a deprecation pointer. A config
  that declares something this build or this release can't serve fails with
  an error naming what's missing — never a silent no-op. Settings declared
  in the new sections can't be overridden by the old flags or environment
  variables (the config file is their only home).
- Embeddings from a server Shrike doesn't run (#498): an `embedders:` entry
  with an `endpoint` uses any OpenAI-compatible embeddings service — a cloud
  provider (the API key is referenced by env-var name via `api_key_env`,
  never written in the config), a server on your tailnet, or a llama-server
  another process owns via `managed.llama_server.manage: attach` (Shrike
  verifies it answers and uses it, but never starts, restarts, or stops it).
  `manage: auto` keeps today's spawn-and-own behavior.
- `/status` (and `shrike server status`) report what semantic search can
  reach (#498): the embedding block names the modalities the running space
  embeds, and a new `coverage` map says, per modality (text/image/audio),
  whether a live embedding space serves it — so "can I search images by
  content right now" has a direct answer instead of being inferred from the
  backend name.

### Removed
- In-process Apple Vision OCR is no longer compiled into the server (#496):
  on-device platform models belong to the mobile apps, and the server-side
  replacement is recognition over a configured endpoint (#502). Selecting
  `--ocr-backend apple` now reports recognition as unavailable instead of
  scanning images; text recognized by earlier versions stays searchable.

### Fixed
- The published wheel can now actually run (#497): `shrike-mcp` ships as a
  platform-tagged wheel (cp312-abi3; macOS arm64/x86_64, Linux x86_64/aarch64)
  with the `shrike_native` extension inside. The previous pure-Python wheel
  imported a native module that was published nowhere, so a pip/pipx install
  could not start the server.

### Changed
- The CLI command tree is reorganized so each command sits under its natural
  parent (#683). **Breaking:** several top-level commands moved and the old
  spellings are gone (they now error).
  - `shrike note search` is now `shrike search` (the new top-level retrieval
    group). `shrike search <query>` searches; `shrike collection query` moved
    to `shrike search query`; the cross-modal coverage matrix has its own home,
    `shrike search coverage` (it no longer appears in `shrike server status`).
  - `shrike info`, `shrike export`, `shrike import`, `shrike tag rename`, and
    `shrike media …` moved under `shrike collection` (`shrike collection info`,
    `shrike collection export`, `shrike collection import`, `shrike collection
    tag rename`, `shrike collection media …`).
  - `shrike embedding …` and `shrike index …` moved under `shrike server`
    (`shrike server embedding …`, `shrike server index …`).
  - `shrike profile` stays at the top level.
  - Every group now lists its subcommands in a fixed, task-ordered sequence in
    `--help` and shell completions instead of alphabetically.
- `onnxruntime` is now a regular dependency — it ships with every install as
  the runtime the in-process embedding engines load. **Breaking for install
  scripts:** the `onnx` and `clip` extras are gone (no longer needed — plain
  `pip install shrike-mcp` covers both backends), and the `onnx-gpu` extra is
  renamed to `gpu` (`pip install 'shrike-mcp[gpu]'`; the CUDA/TensorRT
  onnxruntime build still must replace the base one, which pip cannot do for
  you — uninstall `onnxruntime` after installing).

## [0.4.0] — 2026-06-06

_Media release: author cards with images and audio end to end. Get assets into
the collection, read and list them back, clean up unused files, and check media
integrity, over both MCP and the CLI._

### Added
- Media support (#70): new MCP tools and a `shrike media` command group for the
  asset side of card authoring, which until now had no way into the collection
  (you could write `<img src="x.png">` into a field, but not get the file there).
  - `store_media` writes files into the collection's media folder in bulk (1 to
    10 per call), from base64 bytes or an `http(s)` URL the server fetches. Each
    item reports its own result so one bad item does not sink the batch, and Anki
    resolves name collisions (identical content keeps the name, different content
    gets a hashed suffix), so reference the returned filename.
  - `fetch_media` locates files and returns, per file, a `url` (the server's new
    `GET /media/<name>` route) and a server-side path, rather than base64 bytes,
    so a model can pull an asset with its own download tool without bloating its
    context. The standalone client adds `ShrikeClient.read_media(name)` for
    programmatic byte access.
  - `list_media` lists filenames (with the media directory and an optional glob),
    and `delete_media` moves files to Anki's recoverable trash.
  - `collection_check` reports media integrity: files on disk that no note
    references, references to files that are missing, and the note IDs that hold
    them. `collection_prune` gains an `unused_media` cleanup to trash files no
    note references.
  - CLI: `shrike media store|fetch|list|delete`, `shrike collection check`, and
    `shrike collection prune --unused-media`.
- `--public-url` (config `server.public_url`, env `SHRIKE_PUBLIC_URL`) sets the
  externally visible base URL used to build the media links in `fetch_media` /
  `list_media`, for running behind a reverse proxy where the bind host is not
  reachable (#70).
- Optional `socks` install extra (`shrike-mcp[socks]`) for routing the server's
  media URL fetch through a SOCKS proxy (#70).

### Security
- `store_media`'s server-side URL fetch is guarded against SSRF: `http`/`https`
  only; the host is resolved and refused unless every resolved address is globally
  routable (an allowlist that also rejects carrier-grade NAT, the cloud-metadata
  address, and other non-public ranges a denylist misses); redirects are followed
  manually with the guard re-run on each hop; and the connection is pinned to the
  vetted IP, with TLS still validated against the hostname, so the resolved
  address cannot change between the check and the connect. Trusted internal hosts
  can be allowed with `--allow-private-media-fetch` (config
  `server.media_allow_private_fetch`, env `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH`)
  (#70, #165).
- The `GET /media/<name>` route and the fetch/delete tools confine filenames to
  the collection's media directory (#70).
- `store_media` can read a file from the server's own filesystem with a `path`
  source, but this is **off by default**. It is honored only when the operator
  configures one or more `--media-path-root` directories (repeatable; config
  `server.media_path_roots`, env `SHRIKE_MEDIA_PATH_ROOTS`), the daemon is in its
  default loopback-only configuration, and the file resolves inside one of those
  roots. This keeps a remote or proxied caller from making the server read its
  disk, and bounds a permitted local caller to the named directories. The CLI
  exposes it as `shrike media store --server-path` (#164, #170).

### Changed
- A server started with DNS-rebinding protection on but no usable Host allow-list
  (a non-loopback bind given only `--allowed-origin`, which would otherwise reject
  every request) now logs a warning at startup instead of failing silently (#166).

## [0.3.5] — 2026-06-05

_Collection-management release: cooperative locking, note-type editing, deck and
tag tools, find/replace, raw queries, and collection cleanup — plus incremental
index reconcile on drift._

### Added
- `update_note_type_field_metadata` MCP tool — set a note type's per-field editor metadata:
  the `font` and `size` used when editing a field in Anki, and the field
  `description` (hint text). `collection_info` note-type details now include each
  field's font/size/description too, so the values are readable. It's cosmetics
  only (no effect on note content, cards, or search), so it bumps `col.mod`
  without re-embedding. Completes the #76 epic (#119).
- Busy-acquire error surface for cooperative locking: when the daemon can't
  re-acquire the collection because another process holds it (typically Anki
  desktop is open), tool calls now fail with a distinct, typed "collection busy"
  error instead of a leaked exception string. `ShrikeClient` raises a
  `CollectionBusyError` (a sibling of `ServerError`) callers can catch and retry,
  and the CLI prints an actionable "the collection is in use by another process
  (is Anki open?)" message. It returns busy immediately (no retry) (#65).
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
- `update_note_type_fields` and `update_note_type_templates` MCP tools (#76):
  edit a note type's fields or card templates **by name** with a sequence of
  `add`/`remove`/`rename`/`reposition` ops — the data-safe, identity-based
  counterpart to the positional `fields`/`templates` replace in
  `upsert_note_types`. They delegate to Anki's own primitives so note data and
  cards migrate by identity (a reposition is a true move; a non-trailing remove
  drops only that field's data / that template's cards). Each call is atomic — the
  op sequence is validated before any change runs (#101, #102).
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
- Index drift now **reconciles incrementally** instead of re-embedding the whole
  collection. When the collection changes outside Shrike (Anki GUI, sync, import),
  a per-note embedding-text hash sidecar lets the index re-embed only the notes
  whose text changed, add new notes, and drop deleted ones — the end state is
  identical to a full rebuild, but a drift touching a handful of notes now costs a
  handful of embeddings, not the whole collection. Explicit `index rebuild` stays
  full (#38, #144).
- `upsert_notes` now runs Anki's own add-note validation on each new note and
  takes an `on_duplicate` policy — `error` (default; reported, not written),
  `skip`, or `allow`. Structurally invalid notes (empty first field, broken cloze)
  are always reported and never written, regardless of policy; `dry_run` runs the
  same validation but writes nothing. This replaces the idea of a separate
  duplicate pre-check tool (which would be racy) (#77).
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

### Fixed
- An empty-at-boot server now indexes notes added later in the same session. A
  daemon started against an empty collection never materialized its vector index,
  so the incremental upsert path was skipped and notes stayed semantically
  unsearchable (with a misleading "embedding service not running" message) until a
  restart. Boot now materializes an empty, ready index so later upserts are
  indexed incrementally (#148).
- Note-type field/template updates no longer lose data. The positional
  `fields`/`templates` replace in `upsert_note_types` rebuilt the note type from
  scratch, blanking note data and deleting cards on any edit; it now replaces by
  position (data preserved) and rejects reorders/inserts/non-trailing removes that
  would silently re-label note data — pointing at the by-identity tools instead
  (#99).

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

[Unreleased]: https://github.com/lathrys-at/shrike/compare/v0.3.5...HEAD
[0.3.5]: https://github.com/lathrys-at/shrike/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/lathrys-at/shrike/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/lathrys-at/shrike/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/lathrys-at/shrike/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/lathrys-at/shrike/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/lathrys-at/shrike/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/lathrys-at/shrike/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/lathrys-at/shrike/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lathrys-at/shrike/releases/tag/v0.1.0
