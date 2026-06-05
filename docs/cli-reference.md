# Shrike CLI Reference

The Shrike CLI manages your Anki collection through the MCP server. If the server isn't running when you run a command, the CLI starts the daemon automatically, as long as a collection is configured (in the config file or `SHRIKE_COLLECTION`). On a fresh setup, start it yourself first with `shrike server start --collection`.

All commands accept `--json` for machine-readable output and `--pretty/--no-pretty` for controlling Rich formatting. These flags work on both the root command and any subcommand (`shrike --json info` and `shrike info --json` are equivalent).

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

## `shrike server`

Manage the Shrike MCP server daemon.

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
| `--embedding-model PATH` | Path to a GGUF embedding model. Enables semantic search. |
| `--embedding-port INTEGER` | Port for the embedding server (default: 8373). |
| `--embedding-context-size INTEGER` | Context size for the embedding model. |
| `--embedding-threads INTEGER` | CPU threads for embedding inference. |
| `--embedding-gpu-layers INTEGER` | Layers to offload to the GPU. |
| `--embedding-pooling` | llama-server pooling type: `mean`, `last`, `cls`, or `none`. Defaults to the model's GGUF setting. Set `last` for last-token models (Jina v5, Qwen3-Embedding), whose metadata omits it; changing it forces an index rebuild. |
| `--embedding-arg TOKENS` | Extra `llama-server` flag passed through verbatim, repeatable and `shlex`-split (e.g. `--embedding-arg='--flash-attn'`). For runtime-only flags; Shrike-owned flags (`--model`/`--host`/`--port`/`--embeddings`) are rejected, and any change forces an index rebuild. Vector-affecting flags belong in typed settings like `--embedding-pooling`. |
| `--llama-server PATH` | Path to the `llama-server` binary (default: `LLAMA_SERVER_PATH` or `PATH`). |
| `--no-embedding` | Start without the embedding service even if a model is configured. |
| `--save-config` | Persist the resolved flags to the config file. Without this, `server start` never writes config — it stays under your control and start always reflects the flags you pass. |

The embedding flags above are also accepted by `shrike embedding start`, which cycles the service on an already-running server.

`server start` never edits your config file on its own. Pass `--save-config` once to write the flags you started with (collection, ports, embedding model, cache and index tuning) so you can drop them from later commands; without it, the file is yours alone to edit.

### `shrike server stop`

Stop the running daemon.

### `shrike server status`

Show whether the server is running, and if so, its URL, PID, collection path, and uptime.

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

---

## `shrike info`

Show collection information. Without flags, prints a compact summary (note count, deck count, note type count). Use flags to include additional detail.

| Option | Description |
|---|---|
| `--types` | List note types with their fields. |
| `--decks` | List decks with note counts. |
| `--tags` | List all tags. |
| `--stats` | Show scheduling statistics (due cards, new cards, per-deck breakdown). |
| `--type-details NAME` | Show full templates and CSS for a specific note type. |

```bash
shrike info                         # compact summary
shrike info --types --decks --stats # everything
shrike info --type-details Basic    # full definition for Basic
```

---

## `shrike note`

Create, list, update, search, and delete notes.

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

For text or semantic search, use `shrike note search`.

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

### `shrike note delete <NOTE_IDS>...`

Permanently delete notes and their cards. Prompts for confirmation unless `--yes` is passed.

| Option | Description |
|---|---|
| `-y, --yes` | Skip confirmation. |

```bash
shrike note delete 1779749914797
shrike note delete 1779749914797 1779749914798 --yes
```

### `shrike note search [QUERIES]...`

Search the collection by meaning **and** by exact text. Each query is matched both by semantic similarity (needs the embedding service and a built index) and as an exact, case-insensitive substring of note fields. Results are folded together: each shows a similarity score when ranked and the matched field + a snippet when the text occurs literally. Exact matches work even with no embedding service (you'll see a note that semantic ranking was skipped). Accepts query strings, note IDs to find similar notes, or both.

| Option | Description |
|---|---|
| `--similar-to ID` | Find notes similar to this note ID (semantic only). Repeatable. |
| `--top-k INTEGER` | Results per mechanism per query (default: 10). |
| `--threshold FLOAT` | Minimum *semantic* similarity, 0–1 (default: 0.5). Exact matches ignore it. |
| `--deck TEXT` | Restrict search to this deck (name, numeric ID, or `#id`). |
| `--tags TEXT` | Restrict search to notes with these tags. Repeatable and comma-separated. |
| `--brief` | Show only IDs, badges, and snippet, not full note content. |

```bash
shrike note search "electron transport chain"   # semantic + exact
shrike note search --similar-to 1779749914797
shrike note search "mitochondria" --deck Biochemistry
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

## `shrike tag`

Collection-level tag operations. Per-note tag editing (set/add/remove on specific notes) lives under `shrike note tag`; this acts on the collection's tag taxonomy.

### `shrike tag rename <OLD> <NEW>`

Rename a tag. With no `--note`, the tag is renamed everywhere it appears, children included (renaming `history` also moves `history::ww2`). With `--note`, only those notes are affected and the tag is matched exactly, so renaming `jp` never touches `jp-verbs`.

| Option | Description |
|---|---|
| `--note NOTE_ID` | Restrict the rename to this note ID. Repeatable. Omit to rename across the whole collection. |

```bash
shrike tag rename history::ww2 history::wwii
shrike tag rename jp japanese --note 1779749914797 --note 1779749914798
```

---

## `shrike type`

Create, list, show, update, and delete note type definitions. All subcommands accept either a note type name or numeric ID as the identifier.

### `shrike type list [IDENTIFIER]`

Without an identifier, lists all note types in a table (ID, name, kind, fields). With an identifier, shows the full definition including templates and CSS.

```bash
shrike type list                 # table of all note types
shrike type list Basic           # full detail for Basic
shrike type list 1779649378945   # full detail by ID
```

### `shrike type show <IDENTIFIER>`

Shorthand for `type list <IDENTIFIER>`.

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

---

## `shrike index`

Build and inspect the semantic-search vector index. The index is a derived cache
over note content; it can lag the collection and is rebuilt from it.

### `shrike index status`

Show index state (`ready` / `building` / `unavailable` / `error`), vector count,
dimensions, and on-disk path.

### `shrike index rebuild`

Drop the existing index and re-embed every note from scratch. Runs in the
background on the server; requires the embedding service to be running. Use after
the embedding model changes or if the index is suspected stale.

### `shrike index save`

Force an immediate flush of the in-memory index to disk (off the event loop).
Normally the index auto-saves via a debounced flush, so this is rarely needed.

---

## `shrike embedding`

Manage the `llama-server` embedding service that powers semantic search. It can
be cycled independently of the Shrike server (model swaps, freeing GPU/RAM).

### `shrike embedding status`

Show whether the embedding service is running, its URL, PID, and model.

### `shrike embedding start`

Start the embedding service on a running server. Accepts the same embedding
options as `shrike server start` (`--embedding-model`, `--embedding-port`,
`--embedding-pooling`, `--embedding-arg`, `--llama-server`, …); unspecified ones
fall back to the config/env the server booted with. Re-attaches the index and
rebuilds it if the model changed or the index drifted.

### `shrike embedding stop`

Save the index, then stop the embedding service and mark the index
`unavailable`. The server and collection stay up.

---

## `shrike completion`

Generate shell completion scripts.

```bash
eval "$(shrike completion zsh)"     # zsh (add to ~/.zshrc)
eval "$(shrike completion bash)"    # bash (add to ~/.bashrc)
shrike completion fish > ~/.config/fish/completions/shrike.fish
```
