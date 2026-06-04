# Shrike

Sometimes you want to work on your Anki collection without opening Anki. That's what Shrike is for. It's a command-line client and an MCP server for your collection, so you can manage cards from a terminal or hand them to an LLM agent.

Shrike works directly on the collection's SQLite database, without using the Anki desktop app or relying on the AnkiConnect bridge. You get:

- a `shrike` CLI to browse, create, edit, and delete notes and note types
- an MCP server exposing the same operations to agents
- semantic search over your notes, with embeddings computed locally

## Requirements

- Python 3.12 or newer
- An Anki collection (a `collection.anki2` file)

## Getting started

Install Shrike from PyPI:

```bash
pipx install shrike-mcp   # or: pip install shrike-mcp
```

The package is `shrike-mcp`; the command it installs is `shrike`.

Point Shrike at your collection and start the daemon:

```bash
shrike server start --collection ~/path/to/collection.anki2
```

That starts a background daemon with your collection open. The other `shrike` commands talk to it, so they don't repeat `--collection`:

```bash
shrike info
shrike note list --deck Default
```

And when you're done:

```bash
shrike server stop
```

## Semantic search

`shrike note search` finds notes by meaning instead of keywords. It needs two things you supply yourself: a `llama-server` binary (from llama.cpp) to compute embeddings, and a GGUF embedding model for it to run.

Get llama.cpp by building it or installing it from a package manager; you want the `llama-server` binary it provides. For the model, any GGUF embedding model works. A small one like [all-MiniLM-L6-v2](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) is a good default: it runs on CPU and is plenty for finding related cards.

Start Shrike with both:

```bash
shrike server start --collection ~/path/to/collection.anki2 \
  --llama-server ~/llama.cpp/build/bin/llama-server \
  --embedding-model ~/models/all-MiniLM-L6-v2-Q4_K_M.gguf
```

If `--llama-server` or `--embedding-model` isn't given on the command line or in your config file, Shrike falls back to `LLAMA_SERVER_PATH` and a `llama-server` on your `PATH` for the binary, and to `SHRIKE_EMBEDDING_MODEL` for the model.

Shrike builds an index of your notes in the background. A large collection takes a little while the first time; search will tell you if it's still indexing. Once it's ready:

```bash
shrike note search "electron transport chain"
shrike note search --similar-to 1700000000123
```

## CLI

`shrike info` summarizes the collection: note types, decks, tags, and scheduling stats.

```bash
shrike info
shrike info --types --decks --stats
```

List and read notes:

```bash
shrike note list --deck "Organic Chemistry" --limit 20
shrike note show 1779749914797
```

Create a note by passing fields with `-f`:

```bash
shrike note create --deck "Japanese::Vocabulary" --type Basic \
  -f Front="読む (よむ)" -f Back="to read"
```

Cloze notes work the same way; the deletions go in the field:

```bash
shrike note create --deck "Systems Design" --type Cloze \
  -f Text="The {{c1::CAP theorem}} says a distributed system can guarantee at most two of {{c2::consistency}}, {{c3::availability}}, and {{c4::partition tolerance}}."
```

To add notes in bulk, pipe a JSON array to `--json-input`:

```bash
echo '[
  {"deck": "Biochemistry", "note_type": "Basic",
   "fields": {"Front": "What does ATP synthase do?",
              "Back": "Builds ATP from ADP and phosphate, driven by the proton gradient across the inner mitochondrial membrane."},
   "tags": ["metabolism", "mitochondria"]}
]' | shrike note create --json-input
```

Update fields, edit tags, or delete:

```bash
shrike note update 1779749914797 -f Back="to read; to interpret"
shrike note tag 1779749914797 --set verb,jlpt-n5
shrike note delete 1779749914797 --yes
```

`shrike note tag` edits a note's tags in one of three modes, and you pick one explicitly:

```bash
shrike note tag 1779749914797 --set verb,jlpt-n5         # replace all tags
shrike note tag 1779749914797 --add needs-review         # add, leaving others
shrike note tag 1779749914797 --add jp --remove jp-verb  # add and remove together
shrike note tag 1779749914797 --set ""                   # clear all tags
```

For tags across the whole collection, `shrike tag` renames and cleans up:

```bash
shrike tag rename history::ww2 history::wwii   # rename everywhere it appears
shrike tag rename jp japanese --note 1779749914797   # only on these notes
shrike tag clean                               # drop tags no note uses anymore
```

Note types have their own commands:

```bash
shrike type list
shrike type show Basic
shrike type create --name Vocab --field Word --field Meaning \
  --template 'Card 1:{{Word}}:{{FrontSide}}<hr>{{Meaning}}'
```

Every command takes `--json` for scriptable output. See [the CLI reference](docs/cli-reference.md) for the full list of commands and flags.

## Connect an MCP client

Start the server first (see Getting started). Clients connect to the running daemon at `http://127.0.0.1:8372/mcp`.

**Claude Code** connects over streamable HTTP directly:

```bash
claude mcp add --transport http shrike http://127.0.0.1:8372/mcp
```

**Claude Desktop and claude.ai** native URL connectors require OAuth, which Shrike doesn't implement yet, so connect through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge instead. Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "shrike": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:8372/mcp", "--allow-http", "--transport", "http-only"]
    }
  }
}
```

The endpoint is unauthenticated and bound to loopback by default. Before exposing it past `127.0.0.1`, read the transport-security and remote-access options in [the CLI reference](docs/cli-reference.md).

## MCP Tools

Shrike exposes these MCP tools:

- **collection_info**: collection structure, note types, decks, tags, and stats
- **list_notes**: filter and retrieve notes by deck, tags, type, IDs, or date
- **search_notes**: semantic search over note embeddings
- **upsert_notes**: create or update notes in bulk
- **upsert_note_types**: create or update note type definitions
- **update_note_tags**: set, add, or remove tags on a set of notes
- **rename_tag**: rename a tag collection-wide or on specific notes
- **clear_unused_tags**: remove tags no longer used by any note
- **delete_notes**: permanently delete notes by ID
- **delete_note_types**: delete note types by ID, if no notes use them

The machine schema is whatever the running server advertises via `tools/list`. [`docs/mcp-tools.md`](docs/mcp-tools.md) is the human-readable companion.

## Configuration

Shrike reads settings from a YAML file in the platform config directory: `~/.config/shrike/config.yml` on Linux, `~/Library/Application Support/shrike/config.yml` on macOS. The file is yours to manage; `shrike server start` never writes it on its own. To save the flags you started with so you don't have to repeat them, pass `--save-config` and Shrike writes the resolved settings to the file:

```bash
shrike server start --collection ~/path/to/collection.anki2 \
  --embedding-model ~/models/all-MiniLM-L6-v2-Q4_K_M.gguf --save-config
```

Most settings also take an environment variable (`SHRIKE_URL`, `SHRIKE_COLLECTION`, `SHRIKE_EMBEDDING_MODEL`, and more) or a command-line flag. Command-line flags take precedence, then environment variables, then the file. [The CLI reference](docs/cli-reference.md) has the full list.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/unit -v                         # unit tests (no server)
pytest tests/integration -v -m integration   # integration tests (real server subprocess)
ruff check src/shrike/
mypy src/shrike/
```

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers branching, versioning, releases, and how defects get tracked.

## License

AGPL-3.0-or-later
