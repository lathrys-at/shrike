# The tool surface

Shrike exposes 26 MCP tools. This is an orientation to *where they live* and the
behaviours a developer must preserve when touching them. For the wire shapes
(parameters and responses) see [`../mcp-tools.md`](../mcp-tools.md); for *why* each
behaves as it does, see [`decisions.md`](decisions.md).

Each tool is a Pydantic request/response model in `schemas.py` (the binding;
`shrike-schemas` in Rust is canonical), an action in `api/actions.py`, and a
function in `api/tools.py` that returns the response model so FastMCP emits an
`outputSchema`. The standalone `ShrikeClient` exposes a typed method per tool.

## The tools

| Tool | Purpose |
|------|---------|
| `collection_info` | Collection structure: note types, decks, tags, stats. |
| `list_notes` | Filter/retrieve notes by deck, tags, type, IDs, date. |
| `search_notes` | Per-modality semantic similarity + exact substring + fuzzy, RRF-fused. |
| `collection_query` | Raw Anki search expression → notes (read-only escape hatch). |
| `upsert_notes` | Create or update notes in bulk (1–100); `on_duplicate` + `dry_run`; returns neighbors. |
| `upsert_note_types` | Create or update note type definitions (1–10). |
| `update_note_type_fields` | Edit a note type's fields by name: add/remove/rename/reposition. |
| `update_note_type_templates` | Edit a note type's card templates by name. |
| `find_replace_note_types` | Find/replace text in one note type's template HTML + CSS. |
| `update_note_type_field_metadata` | Set per-field editor metadata (font/size/description). |
| `update_note_tags` | Edit tags on a note set (1–1000): `set` XOR `add`/`remove`. |
| `rename_tag` | Rename a tag collection-wide or on a note set (exact match). |
| `find_replace_notes` | Bulk find/replace across fields in a scoped set; literal or regex. |
| `migrate_note_type` | Change notes' note type via field/template name map. |
| `upsert_decks` | Create or rename/reparent decks in bulk (1–100). |
| `delete_decks` | Delete decks by name, only if empty. |
| `delete_notes` | Permanently delete notes by ID. |
| `delete_note_types` | Delete note types by ID (only if unused). |
| `collection_prune` | Cleanup: unused tags + empty notes + empty cards + unused media. |
| `collection_check` | Read-only media diagnostics. |
| `store_media` | Store media (1–10) from base64, URL (SSRF-guarded), or server-local path. |
| `fetch_media` | Locate media (1–10); returns a `url`, never bytes. |
| `list_media` | List media filenames (+ `media_dir`), optional glob. |
| `delete_media` | Delete media → Anki's recoverable trash. |
| `list_profiles` | Enumerate the collection/profile registry; read-only. |
| `export_package` | Export to `.apkg`/`.colpkg`; returns a download `url` or a gated server-local `path`. |

Most write-path logic lives in `CollectionWrapper` (`harness/collection.py`);
note-type edits are native in `shrike-collection/src/note_types.rs`.

## Notes: duplicates and validation

Duplicate handling lives **inside `upsert_notes`**, not in a separate pre-check.
Before each new note is written, Anki's own add-note validation runs
(`note.fields_check()` via `CollectionWrapper._check_new_note`). A first-field
duplicate (same first field as an existing note of that type — Anki's rule,
collection-wide and deck-independent) is governed by `on_duplicate`:

- `error` (**default**) — reported, not written;
- `skip` — `status: skipped`;
- `allow` — written anyway.

Structurally invalid notes (empty first field, broken cloze) are *always* reported
as errors regardless of policy, and never written. `dry_run: true` runs the same
validation but writes nothing. Validation applies to creates; updates are checked
for existence and fields but not duplicate/empty. A standalone `canAddNotes`-style
tool was rejected as racy (a check-then-write TOCTOU); `dry_run` covers the only
capability it would have added.

## Note-type edits come in two flavours, both data-safe

Anki migrates a note's field *values* (and a template's *cards*) by **ordinal**.
Two tool families exploit this:

- **By position** — `upsert_note_types` replaces the whole `fields`/`templates`
  list. The entry at each position keeps its note data even when renamed; only
  shortening drops the tail.
- **By identity** — `update_note_type_fields` / `update_note_type_templates` are a
  sequence of `add`/`remove`/`rename`/`reposition` ops addressed by *name*, so they
  can express a true move, an insert, or a non-trailing remove that position-replace
  can't. They delegate to Anki's data-safe primitives (`rename_field`,
  `reposition_field`, etc.).

Each call is **atomic**: the op sequence is validated against a simulated name list
before any primitive runs, then persisted with one `update_dict`. The two are
**reconciled** so they can't overlap dangerously — the positional replace *rejects*
any update where an existing name moves to a different position (a reorder, insert,
or non-trailing remove, which would silently re-label note data), erroring with a
pointer to the identity tool. None of these do inline index maintenance; the
`col.mod` bump drives a drift rebuild on next startup (correct for a removed field,
conservative for the rest).

`find_replace_note_types` is separate again: a literal-or-regex rewrite over one
model's `qfmt`/`afmt`/shared `css`. It touches *template text*, not fields, so it
migrates nothing and is a metadata-only index op. `match_case` defaults **true**
(template/CSS is code).

## Decks and tags

- **Decks never merge.** `upsert_decks` mirrors `upsert_notes` (id = rename, absent
  = create); renaming onto an existing name is an error. `delete_decks` is
  **empty-only** — move notes out first — so deck deletion can never delete a note.
- **Setting tags is a full replace.** `update_note_tags` `set` (and the `{id, tags}`
  form of `upsert_notes`) replaces the whole set; `add`/`remove` (mutually exclusive
  with `set`) is the additive/subtractive path.

Tags, deck names, and the metadata tools are all metadata-only index ops (see
[`indexing-and-search.md`](indexing-and-search.md)).

## Collection maintenance

`collection_prune` runs any of four cleanups — clear unused tags, remove empty
notes, remove empty cards, trash unused media (selecting none runs all). It
**applies by default** (`dry_run` defaults false; the CLI previews, confirms, then
applies unless `--dry-run`). An *empty* note is one whose every field is blank by
`embed_text.field_is_blank` — no text *and* no media — so an image- or audio-only
note is never deleted. Its index handling is **mixed**: empty-note/empty-card
removal drops vectors; clearing tags and trashing media leave vectors unchanged.

## Media

Media tools wrap `col.media` and never touch `col.mod` or embedding text, so they
need no index bump. `store_media` is a bulk (1–10) per-item batch where each item
is base64 `data`, a `url` the server fetches, or a server-local `path`.

### URL fetch is SSRF-guarded

`_fetch_media_url`: http/https only; the host is resolved and refused unless every
address is **globally routable** (`ipaddress.is_global` — an allowlist, so it also
rejects CGN, benchmarking ranges, loopback/RFC1918/link-local/metadata/reserved/
multicast), unless `--allow-private-media-fetch`. Redirects are followed **manually,
re-running the guard on every hop**, because httpx's `follow_redirects` would jump
to an attacker-chosen private address unchecked. The connection is **pinned to the
vetted IP**: the request connects to one validated address and never re-resolves the
name (closing the DNS-rebinding TOCTOU), with the `Host` header and TLS SNI carrying
the original name. Capped at `MEDIA_MAX_BYTES`.

### The server-local `path` source is gated

A `path` source stores a file zero-copy via `col.media.add_file`, **off by default**,
honored only when all three hold:

1. the operator set one or more `--media-path-root DIR` (config
   `server.media_path_roots`);
2. the server is **purely-local** (`_server_is_purely_local`: loopback bind, no
   `--allow-remote`, guard on, no added allowed-hosts/origins);
3. the path is **contained** in one of those roots after `..`/symlink resolution
   (`os.path.commonpath` on realpath'd sides, not `startswith`).

The two gates compose: purely-local stops a remote/proxied caller reaching it, the
roots bound what a permitted caller may read. `main()` validates each root once at
startup. Within a root, `path` is an arbitrary read of those files at the server
user's privileges — a deliberate, documented part of the unauthenticated-loopback
trust model. Narrow roots keep the blast radius small.

### fetch_media never returns bytes

base64 in a tool response is useless to a model and wrecks context. Each present
file returns a `url` (the server's `GET /media/<name>`) plus a server-side path; a
missing file is `MediaMissing`. The only way to bytes is the url. `list_media`
defaults to 100 files. fetch/delete and the media route sanitize filenames to a
basename inside the media dir.
