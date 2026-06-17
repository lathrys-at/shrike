# Shrike CLI Reference

The Shrike CLI manages your Anki collection through the MCP server. If the server isn't running when you run a command, the CLI starts the daemon automatically, as long as a collection is configured (in the config file or `SHRIKE_COLLECTION`). On a fresh setup, start it yourself first with `shrike server start --collection`.

All commands accept `--json` for machine-readable output and `--pretty/--no-pretty` for controlling Rich formatting. These flags work on both the root command and any subcommand (`shrike --json collection info` and `shrike collection info --json` are equivalent).

The top-level groups are `collection`, `search`, `server`, `note`, `deck`, `type`, `profile`, and `completion`. Collection-scoped commands (info, import/export, media, tags, maintenance) live under `collection`; retrieval (semantic/substring search, raw queries, coverage) under `search`; daemon control (the embedding service and vector index) under `server`.

Note IDs accept an optional `#` prefix, so `shrike note show #1779749914797` and `shrike note show 1779749914797` are the same. Type commands accept either a name or numeric ID wherever an identifier is expected. **Decks** are likewise referenceable by name, numeric ID, or `#id` anywhere a deck is taken (`--deck`, `deck rename`, `deck delete`): `#id` is always an ID, a bare number is tried as an ID then falls back to a name.

## Global Options

```
shrike [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|---|---|
| `-c, --config PATH` | Path to config file. Defaults to the platform config directory. |
| `--url TEXT` | Server URL (overrides config). Env: `SHRIKE_URL`. |
| `--json` | Output raw JSON instead of formatted text. |
| `--pretty/--no-pretty` | Enable or disable Rich styling (default: `--pretty`). |
| `--version` | Show the version and exit. |

---

## `shrike collection`

Collection-wide operations: summary info, integrity checks, maintenance, import/export, tags, and media.

### `shrike collection info`

Show collection information. Without flags, prints a compact summary (note count, deck count, note type count). Use flags to include additional detail.

| Option | Description |
|---|---|
| `--types` | List note types with their fields. |
| `--decks` | List decks with note counts. |
| `--tags` | List all tags. |
| `--stats` | Show scheduling statistics (due cards, new cards, per-deck breakdown). |
| `--type-details NAME` | Show full templates and CSS for a specific note type. |

```bash
shrike collection info                         # compact summary
shrike collection info --types --decks --stats # everything
shrike collection info --type-details Basic    # full definition for Basic
```

### `shrike collection check`

Report media-integrity issues without changing anything: files on disk no note references (prune candidates), references to files that are missing, the notes holding those broken references, and whether Anki's media trash holds anything.

```bash
shrike collection check
```

### `shrike collection export [DEST]`

Export the collection (or a deck/selection) to an Anki package: a `.apkg` (shareable, scopable) or `.colpkg` (whole-collection backup). By default the server writes a temporary package and the CLI downloads it to `DEST`. With `--server-path` the server writes directly to a path on its own disk (no download), for a co-located operator.

| Option | Description |
|---|---|
| `--deck TEXT` | Export only this deck (name, id, or `#id`). Mutually exclusive with `--note-id`. |
| `--note-id INTEGER` | Export only these notes. Repeatable. Mutually exclusive with `--deck`. |
| `--format` | Package format (`apkg`/`colpkg`). Default inferred from `DEST`'s extension, else `apkg`. A `.colpkg` is a whole-collection backup (no `--deck`/`--note-id`). |
| `--scheduling/--no-scheduling` | Include review/scheduling data and deck options (default: off). |
| `--media/--no-media` | Bundle referenced media into the package (default: on). |
| `--server-path PATH` | Write the package to this path on the **server's** disk (zero-copy; requires a purely-local daemon with a matching `--export-path-root`). |

```bash
shrike collection export backup.colpkg
shrike collection export spanish.apkg --deck Spanish
shrike collection export deck.apkg --deck Spanish --scheduling
shrike collection export --server-path /srv/exports/backup.colpkg
```

### `shrike collection import <PATH>`

Import an Anki package (`.apkg`/`.colpkg`) into the collection. This **merges** the package's notes into your collection (added or updated) — it is not a destructive restore, and your collection is never replaced, even for a `.colpkg`. `PATH` is read by the **server** from its own filesystem (the operator must have enabled it with `--import-path-root` on a purely-local daemon); a relative `PATH` is absolutized against your current directory first.

| Option | Description |
|---|---|
| `--update-notes` | How to handle an imported note whose GUID matches an existing one: `if_newer` (default), `always`, or `never`. New notes always add. |
| `--update-notetypes` | The same condition, applied to note types. |
| `--with-scheduling` | Import the package's review scheduling (due dates, intervals). Off by default. |
| `--merge-notetypes` | Merge imported note types into existing ones by name rather than adding new ones. |

```bash
shrike collection import ~/Downloads/shared-deck.apkg
shrike collection import /srv/anki/backup.colpkg --with-scheduling
shrike collection import deck.apkg --update-notes always
```

### `shrike collection prune`

Tidy up the collection: remove unused tags, empty notes, empty cards, and unused media. Select cleanups with the flags below; **with none selected, all four run.** By default this only **previews** what would be removed — pass `--apply` to actually remove (it previews, asks for confirmation, then applies). An empty note has every field blank, where a field is blank only if it has no text **and** no media, so an image- or audio-only note is kept.

| Option | Description |
|---|---|
| `--unused-tags` | Remove tag-registry names no note uses. |
| `--empty-notes` | Delete notes whose every field is blank (text- and media-free). |
| `--empty-cards` | Remove cards that render empty; a note that loses its last card is deleted. |
| `--unused-media` | Move media files no note references to Anki's recoverable trash. |
| `--apply` | Apply the changes. Without it, the command only previews. |
| `-y, --yes` | Skip the confirmation prompt (with `--apply`). |

```bash
shrike collection prune                          # preview every cleanup
shrike collection prune --unused-tags            # preview just unused tags
shrike collection prune --apply                  # preview, confirm, then prune all
shrike collection prune --empty-notes --apply -y # remove empty notes, no prompt
```

`--apply` is destructive (deleted notes and cards cannot be recovered; trashed media can, from Anki's media trash); preview first.

### `shrike collection reload`

Close and re-open the collection without restarting the daemon. Picks up changes made to the collection file on disk underneath a running daemon (a restored backup, a file-level sync or swap) and re-checks the search index for drift, rebuilding it in the background if the collection changed. Reports the new `col_mod` and whether a rebuild was triggered.

```bash
shrike collection reload
```

Note: in the default locking mode the daemon holds the collection's lock for its whole life, so external tools (Anki desktop, sync) generally cannot edit the collection underneath you — reload is mainly for file-level replacement. With `--cooperative-lock`, the daemon already re-checks for external edits each time it re-acquires the collection, so a manual reload is rarely needed.

### `shrike collection tag`

Collection-level tag operations. Per-note tag editing (set/add/remove on specific notes) lives under `shrike note tag`; this acts on the collection's tag taxonomy.

#### `shrike collection tag rename <OLD> <NEW>`

Rename a tag. With no `--note`, the tag is renamed everywhere it appears, children included (renaming `history` also moves `history::ww2`). With `--note`, only those notes are affected and the tag is matched exactly, so renaming `jp` never touches `jp-verbs`.

| Option | Description |
|---|---|
| `--note NOTE_ID` | Restrict the rename to this note ID. Repeatable. Omit to rename across the whole collection. |

```bash
shrike collection tag rename history::ww2 history::wwii
shrike collection tag rename jp japanese --note 1779749914797 --note 1779749914798
```

### `shrike collection media`

Manage the collection's media folder — the images and audio that note fields reference with `<img src="...">` and `[sound:...]`. Anki resolves name collisions (identical content keeps the name; different content under the same name gets a hashed suffix), so always reference the filename the store command *returns*.

#### `shrike collection media store [PATHS]...`

Store local files and/or URLs into the media folder. Local paths are read here and uploaded as bytes, so they work against a remote daemon. URLs are fetched by the server by default (http/https only; private and internal addresses are refused unless the server runs with `--allow-private-media-fetch`).

| Option | Description |
|---|---|
| `--name TEXT` | Override the stored filename (single item only). |
| `--url URL` | URL to fetch and store. Repeatable. |
| `--client-fetch` | Download `--url` files locally and upload the bytes — for when this machine has the network path or proxy, not the server. |
| `--server-path PATH` | Store a file already on the *server's* disk without sending bytes (repeatable). Off by default: the server must be started with one or more `--media-path-root DIR` (config `server.media_path_roots`, env `SHRIKE_MEDIA_PATH_ROOTS`) on a purely-local configuration, and the file must be under one of those roots. |

```bash
shrike collection media store diagram.png
shrike collection media store a.png b.jpg c.ogg
shrike collection media store --url https://example.com/cell.png
shrike collection media store --server-path /data/big-lecture.mp4
```

#### `shrike collection media fetch <NAMES>...`

Write media files out to local disk.

| Option | Description |
|---|---|
| `-o, --output PATH` | Write a single file to this path. |
| `--out-dir PATH` | Directory to write files into (default: current directory). |

```bash
shrike collection media fetch diagram.png -o /tmp/out.png
shrike collection media fetch a.png b.jpg --out-dir ./assets
```

#### `shrike collection media list [PATTERN]`

List media filenames, with size and type, optionally filtered by a glob pattern.

| Option | Description |
|---|---|
| `--limit INTEGER` | Maximum files to show. |

```bash
shrike collection media list
shrike collection media list '*.png' --limit 20
```

#### `shrike collection media delete <NAMES>...`

Move media files to Anki's recoverable trash. This does **not** check whether a note still references the file — run `shrike collection check` first to find unused media.

| Option | Description |
|---|---|
| `-y, --yes` | Skip confirmation. |

```bash
shrike collection media delete orphan.png --yes
```

---

## `shrike search`

Search the collection and inspect retrieval. `shrike search <query>` runs the default search (semantic + exact-substring); `shrike search query` is the raw-Anki-expression escape hatch; `shrike search coverage` shows the cross-modal coverage matrix.

### `shrike search [QUERIES]...`

Search the collection by meaning **and** by text. Each query is matched by semantic similarity (needs the embedding service and a built index), as an exact, case-insensitive substring of note fields, and fuzzily (trigram matching, so a typo like `protien` still finds protein cards). Results are folded together: each shows a similarity score when ranked and the matched field + a snippet when the text matched. Text matches work even with no embedding service (you'll see a note that semantic ranking was skipped). Accepts query strings, note IDs to find similar notes, or both.

| Option | Description |
|---|---|
| `--similar-to ID` | Find notes similar to this note ID (semantic only). Repeatable. |
| `--top-k INTEGER` | Results per mechanism per query (default: 10). |
| `--threshold FLOAT` | Minimum *semantic* similarity, 0–1 (default: 0.5). Exact matches ignore it. |
| `--deck TEXT` | Restrict search to this deck (name, numeric ID, or `#id`). |
| `--tags TEXT` | Restrict search to notes with these tags. Repeatable and comma-separated. |
| `--brief` | Show only IDs, badges, and snippet, not full note content. |

```bash
shrike search "electron transport chain"   # semantic + exact
shrike search --similar-to 1779749914797
shrike search "mitochondria" --deck Biochemistry
```

### `shrike search query <EXPRESSION>`

Find notes with a raw [Anki search expression](https://docs.ankiweb.net/searching.html) — the power-user escape hatch. EXPRESSION is passed straight to Anki's search engine, so the full language works (`is:due`, `prop:ivl>=30`, `added:`, `rated:`, `flag:`, `OR`, `-`, parentheses). For meaning/text search use `shrike search`; for plain deck/tag/type filters use `shrike note list`.

| Option | Description |
|---|---|
| `--brief` | Show only IDs and metadata, not field content. |
| `--limit INTEGER` | Max notes to return (default 50, max 200). |

```bash
shrike search query "is:due prop:ivl>=30"
shrike search query "added:7 -tag:done" --brief
shrike search query "deck:Japanese (tag:verb OR tag:adj)" --limit 100
```

### `shrike search coverage`

Show the cross-modal coverage matrix: for each (query → target) modality pair, how the target is reachable — **native** (one embedding space covers both modalities), **via text** (a recognizer derives text from the target into the text space), or **unavailable**. Reflects the server's configured embedders and recognizers, so e.g. `text → audio` reads "via text" only when ASR reaches it.

```bash
shrike search coverage
```

---

## `shrike server`

Manage the Shrike MCP server daemon. The embedding service and the vector index are controlled under this group too (`shrike server embedding …`, `shrike server index …`).

### `shrike server start`

Start the server as a background daemon. The collection path can come from `--collection`, the config file, or the `SHRIKE_COLLECTION` environment variable.

| Option | Description |
|---|---|
| `--collection PATH` | Path to the Anki collection file (`collection.anki2`). |
| `--port INTEGER` | Port to listen on (default: 8372). |
| `--host TEXT` | Host to bind to (default: `127.0.0.1`). |
| `--allow-remote` | Permit binding to a non-loopback host. Every endpoint is unauthenticated, so this exposes the full collection API to the network. Use it only behind your own auth or network controls. |
| `--allowed-host TEXT` | Additional `Host` header to trust beyond loopback (repeatable). For a reverse-proxy or VPN hostname; a proxy forwards `name:port`, so use the `name:*` port-wildcard form. |
| `--allowed-origin TEXT` | Additional `Origin` header to trust beyond loopback (repeatable). Native MCP clients usually send none (always allowed); add one only if a browser client is rejected with 403. |
| `--no-dns-rebinding-protection` | Disable `Host`/`Origin` validation entirely, on any bind, for deployments where the network is the trust boundary (behind a reverse proxy, on a VPN/tailnet, firewalled). Endpoints stay unauthenticated. |
| `--foreground` | Run in the foreground instead of daemonizing. |
| `--log-dir PATH` | Directory for log files (default: platform-specific). |
| `--log-level` | `debug`, `info`, `warning`, or `error` (default: `info`). |
| `--cache-dir PATH` | Directory for the vector index and other caches (default: platform-specific). |
| `--index-save-delay FLOAT` | Seconds of idle after the last index change before flushing to disk (default: 60). |
| `--index-save-threshold INTEGER` | Unsaved index changes that force an immediate flush (default: 100). |
| `--cooperative-lock` | Release the collection lock when idle and re-open on demand, so an idle daemon doesn't block launching Anki (opt-in; default holds the lock for the daemon's lifetime). |
| `--lock-hold-seconds FLOAT` | In cooperative mode, seconds to hold the collection after the last operation before releasing it (default: 5). |
| `--no-embedding` | Start without the embedding service even if one is configured. |
| `--save-config` | Persist the resolved operational flags to the config file. Without this, `server start` never writes config; it stays under your control and start always reflects the flags you pass. |

Embedding models, recognition, and managed processes are declared in the config file, not on the command line: an `embedders:` entry plus, where it applies, a `managed:` section. The [README's semantic search section](../README.md#semantic-search) walks through each shape. The older embedding flags (`--embedding-*`, `--llama-server`, `--ocr-backend`) still exist on `server start` and `server embedding start` for one more release, and a legacy `embedding:` config section still works and prints a pointer to the new shape; both are rejected when the config file declares the new sections, and both are removed in [#523](https://github.com/lathrys-at/shrike/issues/523).

`server start` never edits your config file on its own. Pass `--save-config` once to write the operational flags you started with (collection, ports, cache and index tuning) so you can drop them from later commands; without it, the file is yours alone to edit.

### `shrike server stop`

Stop the running daemon.

### `shrike server status`

Show whether the server is running, and if so, its URL, PID, collection path, and uptime. Below that it reports the index, derived-text store, recognition, and embedding blocks (in that order), each with its status on its own line. The index block breaks down per modality sub-index; the embedding block lists one entry per configured space, keyed by its modalities. In cooperative-locking mode it also shows whether the collection is currently held or released (idle).

### `shrike server logs`

View server log output with optional Rich styling.

| Option | Description |
|---|---|
| `-f, --follow` | Stream new log output as it arrives (like `tail -f`). |
| `-n, --lines INTEGER` | Number of lines to show (default: 50). |
| `-p, --process` | Which log to view: `shrike` (default) or `llama`. |
| `--stdin` | Read log lines from stdin instead of the log file. |

```bash
shrike server logs              # last 50 lines, styled
shrike server logs -f           # follow
shrike server logs -n 100       # last 100 lines
shrike --json server logs       # raw JSON log entries
cat shrike.log | shrike server logs --stdin
```

### `shrike server embedding`

Manage the embedding service that powers semantic search, whichever backend it runs on. It can be cycled independently of the Shrike server (model swaps, freeing GPU/RAM).

#### `shrike server embedding start`

Start the embedding service on a running server, using whatever the config file declares (the daemon resolves its `embedders:`/`managed:` sections). With a legacy flag-driven setup it accepts the same deprecated embedding options as `shrike server start`, and unspecified ones fall back to what the server booted with. Re-attaches the index and rebuilds it if the model changed or the index drifted.

#### `shrike server embedding stop`

Save the index, then stop the embedding service and mark the index `unavailable`. The server and collection stay up.

#### `shrike server embedding status`

Show each configured embedding space, one entry per space keyed by its modalities (`Embedding [text]`, `Embedding [text, image]`), with that space's status, URL, PID, model, provider, and batching. A multi-space profile reports every space, not just the primary.

### `shrike server index`

Build and inspect the semantic-search vector index. The index is a derived cache over note content; it can lag the collection and is rebuilt from it.

#### `shrike server index rebuild`

Drop the existing index and re-embed every note from scratch. Runs in the background on the server; requires the embedding service to be running. Use after the embedding model changes or if the index is suspected stale.

#### `shrike server index save`

Force an immediate flush of the in-memory index to disk (off the event loop). Normally the index auto-saves via a debounced flush, so this is rarely needed.

#### `shrike server index status`

Show index state (`ready` / `building` / `unavailable` / `error`) and on-disk path, with a per-modality breakdown of each sub-index's vector count and dimensions (a text+image collection reports its `text` and `image` sub-indexes separately, since they differ in dimensionality).

---

## `shrike note`

Create, list, update, and delete notes. For text or semantic search, use `shrike search`.

### `shrike note create`

Create one or more notes. For inline creation, `--deck`, `--type`, and at least one `-f` field are required. For bulk creation, pipe a JSON array to stdin with `--json-input`.

| Option | Description |
|---|---|
| `--deck TEXT` | Target deck. |
| `--type TEXT` | Note type (e.g., `Basic`, `Cloze`). |
| `-f, --field KEY=VALUE` | Field value. Repeatable. |
| `--tags TEXT` | Tags for the note. Repeatable and comma-separated. |
| `--json-input` | Read a JSON array of note objects from stdin. |

```bash
# Inline
shrike note create --deck "Japanese::Vocabulary" --type Basic \
  -f Front="読む (よむ)" -f Back="to read"

# Bulk from JSON
echo '[
  {"deck": "Biochemistry", "note_type": "Basic",
   "fields": {"Front": "What is the role of ATP synthase?",
              "Back": "Catalyzes ATP synthesis from ADP and Pi."},
   "tags": ["metabolism"]}
]' | shrike note create --json-input
```

#### JSON input schema (for `--json-input`)

Each object in the array:

| Field | Type | Required | Description |
|---|---|---|---|
| `deck` | `string` | yes | Target deck. |
| `note_type` | `string` | yes | Note type name. |
| `fields` | `object` | yes | Field key-value pairs. |
| `tags` | `string[]` | no | Tags for the note. |

### `shrike note update <NOTE_ID>`

Update an existing note. Only the specified fields are changed; everything else is left as-is. Tags are fully replaced if `--tags` is provided.

| Option | Description |
|---|---|
| `-f, --field KEY=VALUE` | Field to update. Repeatable. |
| `--tags TEXT` | Replace all tags. Repeatable and comma-separated. |
| `--deck TEXT` | Move note to this deck. |

```bash
shrike note update 1779749914797 -f Back="to read; to interpret"
shrike note update 1779749914797 --tags newtag,kept-tag
shrike note update 1779749914797 --deck "Other::Deck"
```

### `shrike note delete <NOTE_IDS>...`

Permanently delete notes and their cards. Prompts for confirmation unless `--yes` is passed.

| Option | Description |
|---|---|
| `-y, --yes` | Skip confirmation. |

```bash
shrike note delete 1779749914797
shrike note delete 1779749914797 1779749914798 --yes
```

### `shrike note list`

List notes matching structured filters. At least one filter is required.

| Option | Description |
|---|---|
| `--deck TEXT` | Filter by deck name (includes child decks). |
| `--tags TEXT` | Filter by tag. Repeatable and comma-separated, ANDed together. |
| `--type TEXT` | Filter by note type. |
| `--ids ID` | Fetch specific note IDs. Repeatable. |
| `--since TEXT` | Notes modified after this date (ISO 8601). |
| `--brief` | Show only IDs and metadata (type, deck, tags, modified), not field content. |
| `--limit INTEGER` | Max notes to return (default: 50). |

For text or semantic search, use `shrike search`.

```bash
shrike note list --deck "Japanese::Vocabulary"
shrike note list --tags verb,chapter-3
shrike note list --type Cloze --brief --limit 20
shrike note list --since 2026-05-01
```

### `shrike note show <NOTE_ID>`

Show a single note by ID. Shorthand for `note list --ids <ID>` with an error if the note doesn't exist.

```bash
shrike note show 1779749914797
shrike note show '#1779749914797'
```

### `shrike note tag <NOTE_IDS>...`

Edit the tags on one or more notes. Pick exactly one mode — there is no default. `--set` replaces wholesale; `--add`/`--remove` edit additively and combine in a single call. `--set` cannot be combined with `--add`/`--remove`. Fields and decks are untouched.

| Option | Description |
|---|---|
| `--set TEXT` | Replace all tags with this set. Repeatable and comma-separated. Pass `--set ""` to clear all tags. Mutually exclusive with `--add`/`--remove`. |
| `--add TEXT` | Add these tags, leaving other tags intact. Repeatable and comma-separated. |
| `--remove TEXT` | Remove these tags, leaving other tags intact. Repeatable and comma-separated. |

```bash
shrike note tag 1779749914797 --set world-war-2,history
shrike note tag 1779749914797 1779749914798 --add needs-review
shrike note tag 1779749914797 --add jp --add verbs --remove jp-verbs
shrike note tag 1779749914797 --set ""        # clear all tags
```

### `shrike note replace <SEARCH> <REPLACE>`

Find and replace text across note fields, scoped to a selection. A scope is required. `SEARCH` is literal unless `--regex`. By default it previews the changes, asks for confirmation, then applies; changed notes are re-embedded, and the edit is undoable in Anki.

| Option | Description |
|---|---|
| `--deck TEXT` | Scope to this deck (name, numeric ID, or `#id`). |
| `--tags TEXT` | Scope to notes with these tags. Repeatable and comma-separated. |
| `--type TEXT` | Scope to this note type. |
| `--ids ID` | Scope to these note IDs. Repeatable. |
| `--field TEXT` | Restrict to a single field (default: all fields). |
| `--regex` | Treat `SEARCH` as a regular expression (capture refs in `REPLACE` use `$1`). |
| `--match-case` | Case-sensitive match. |
| `--dry-run` | Preview the changes without applying them. |
| `-y, --yes` | Skip the confirmation prompt. |

```bash
shrike note replace "teh" "the" --deck "Biology" --dry-run
shrike note replace "colou?r" "color" --regex --tags spelling -y
```

### `shrike note migrate-type <NOTE_IDS>...`

Change one or more notes to a different note type, moving field content by an explicit map. The notes must all currently share one note type. This is Anki's "Change Note Type": note IDs and (for mapped templates) review scheduling are preserved. **Source fields you don't `--map` are dropped and their content is lost** — by default the command previews (showing the drops), asks to confirm, then applies.

| Option | Description |
|---|---|
| `--to TEXT` | Target note type (required). |
| `--map OLD=NEW` | Field mapping, source=target. Repeatable; at least one required. |
| `--template-map OLD=NEW` | Optional card-template mapping, source=target. Repeatable. |
| `--dry-run` | Preview the migration without applying it. |
| `-y, --yes` | Skip the confirmation prompt. |

```bash
shrike note migrate-type 1700000000123 --to Cloze --map Front=Text --map "Back=Back Extra"
shrike note migrate-type 170...1 170...2 --to Basic --map Text=Front --dry-run
```

---

## `shrike deck`

Create, rename, and delete decks. Deletion requires the deck to be empty — move its notes elsewhere first (e.g. `shrike note update <id> --deck …`), then delete. `rename` and `delete` accept a deck name, numeric ID, or `#id`.

### `shrike deck create <NAME>`

Create an empty deck. Use `::` for nesting. No-op if it already exists.

```bash
shrike deck create "Japanese::Vocabulary"
```

### `shrike deck rename <OLD> <NEW>`

Rename or reparent a deck. **Decks do not merge** — renaming onto a name another deck already uses is an error. To consolidate, move the notes instead.

```bash
shrike deck rename "Japanese::Vocabulary" "Japanese::Vocab"
shrike deck rename "Misc::French" "French"      # reparent to top level
```

### `shrike deck delete <NAMES>...`

Delete one or more decks, each of which must already be empty (no cards in it or any subdeck). Prompts for confirmation unless `--yes`. Exits non-zero if any named deck was not empty or not found.

| Option | Description |
|---|---|
| `-y, --yes` | Skip confirmation. |

```bash
shrike deck delete "Old Deck"
shrike deck delete "A" "B" --yes
```

---

## `shrike type`

Create, list, show, update, and delete note type definitions. All subcommands accept either a note type name or numeric ID as the identifier.

### `shrike type create`

Create a new note type definition. For inline creation, `--name`, `--field`, and `--template` are required. For complex types, pipe a JSON definition to stdin with `--json-input`.

| Option | Description |
|---|---|
| `--name TEXT` | Name for the note type. |
| `--field TEXT` | Field name. Repeatable, order matters. |
| `--template NAME:FRONT:BACK` | Card template. Repeatable. The delimiter is `:`, split on the first two occurrences (template HTML may contain `:`). |
| `--css TEXT` | CSS styling for all cards. |
| `--cloze` | Create a cloze deletion note type. |
| `--json-input` | Read a JSON note type definition from stdin. |

```bash
# Inline
shrike type create --name "Vocab" --field Word --field Meaning \
  --template 'Card 1:{{Word}}:{{FrontSide}}<hr>{{Meaning}}' \
  --css ".card { font-size: 20px; }"

# Cloze type
shrike type create --name "My Cloze" --field Text --field Extra --cloze \
  --template 'Cloze:{{cloze:Text}}:{{cloze:Text}}<br>{{Extra}}'

# From JSON
echo '{
  "name": "Japanese Vocabulary",
  "fields": ["Word", "Reading", "Meaning"],
  "templates": [
    {"name": "Recognition", "front": "{{Word}}", "back": "{{FrontSide}}<hr>{{Meaning}}"},
    {"name": "Recall", "front": "{{Meaning}}", "back": "{{FrontSide}}<hr>{{Word}}"}
  ],
  "css": ".card { font-family: sans-serif; }"
}' | shrike type create --json-input
```

#### JSON input schema (for `--json-input`)

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | yes | Note type name. |
| `fields` | `string[]` | yes | Ordered list of field names. |
| `templates` | `object[]` | yes | Card templates (see below). |
| `css` | `string` | yes | CSS styling. |
| `is_cloze` | `boolean` | no | `true` for cloze deletion types. |

Each template object:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | yes | Template name (e.g., `"Card 1"`). |
| `front` | `string` | yes | Front side HTML. Use `{{FieldName}}` for field substitution. |
| `back` | `string` | yes | Back side HTML. `{{FrontSide}}` renders the front side. |

### `shrike type update <IDENTIFIER>`

Update an existing note type. For simple changes, use `--name` or `--css`. For structural changes (fields, templates), pipe a JSON definition with `--json-input`; the identifier is resolved and the ID is injected automatically.

| Option | Description |
|---|---|
| `--name TEXT` | New name for the note type. |
| `--css TEXT` | New CSS styling. |
| `--json-input` | Read a full JSON definition from stdin (merged with the resolved ID). |

```bash
shrike type update Basic --css ".card { color: red; }"
shrike type update 1779649378945 --name "Renamed"
echo '{"fields": ["A", "B", "C"], "templates": [...]}' | \
  shrike type update Basic --json-input
```

### `shrike type delete <IDENTIFIERS>...`

Delete note types by name or ID. A note type can only be deleted if no notes use it. Prompts for confirmation unless `--yes` is passed.

| Option | Description |
|---|---|
| `-y, --yes` | Skip confirmation. |

```bash
shrike type delete "Old Type"
shrike type delete 1779649378945 1779649378946 --yes
```

### `shrike type list [IDENTIFIER]`

Without an identifier, lists all note types in a table (ID, name, kind, fields). With an identifier, shows the full definition including templates and CSS.

```bash
shrike type list                 # table of all note types
shrike type list Basic           # full detail for Basic
shrike type list 1779649378945   # full detail by ID
```

### `shrike type show <IDENTIFIER>`

Shorthand for `type list <IDENTIFIER>`.

---

## `shrike profile`

Register collections by a friendly name, set an active default, and list them. The registry lives in the config file and is managed entirely client-side — these commands never talk to the server. (Verb names are unchanged in this release.)

```bash
shrike profile add work ~/Anki2/Work/collection.anki2 --default
shrike profile list
shrike profile default personal
shrike profile remove old
```

---

## `shrike completion`

Generate shell completion scripts.

```bash
eval "$(shrike completion zsh)"     # zsh (add to ~/.zshrc)
eval "$(shrike completion bash)"    # bash (add to ~/.bashrc)
shrike completion fish > ~/.config/fish/completions/shrike.fish
```
