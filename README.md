# Shrike

An alternative Anki command-line client and MCP server. Shrike manages your collection without running Anki's desktop app and provides programmatic access through a CLI and an MCP server for LLM agents.

## Install

Requires Python 3.12+.

```bash
pip install -e .
```

## Usage

### Start the server

```bash
# As a daemon (recommended):
shrike server start --collection /path/to/collection.anki2

# Or in the foreground:
python -m shrike.server --collection /path/to/collection.anki2
```

The server listens on `http://127.0.0.1:8372/mcp` (JSON-RPC 2.0). Any MCP client can connect to it. The CLI auto-starts the daemon if it isn't already running.

### Connect an MCP client

Start the server (above) first — it holds an exclusive lock on the collection, so it must be the one process that has it open.

**Claude Code** speaks streamable HTTP directly:

```bash
claude mcp add --transport http shrike http://127.0.0.1:8372/mcp
```

**Claude Desktop / claude.ai** native *URL connectors* require OAuth, which Shrike does not yet implement, so connect through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge instead. Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "shrike": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://127.0.0.1:8372/mcp",
        "--allow-http",
        "--transport",
        "http-only"
      ]
    }
  }
}
```

The endpoint is unauthenticated and bound to loopback by default — see the transport-security and remote-access options in [`docs/cli-reference.md`](docs/cli-reference.md) before exposing it beyond `127.0.0.1`.

### CLI

```bash
# See what's in the collection
shrike info
shrike info --types --decks --stats

# Browse notes
shrike note list --deck "Organic Chemistry" --limit 20
shrike note show 1779749914797

# Create a card
shrike note create --deck "Japanese::Vocabulary" --type Basic \
  -f Front="読む (よむ)" -f Back="to read"

# Cloze deletion
shrike note create --deck "Systems Design" --type Cloze \
  -f Text="The {{c1::CAP theorem}} states that a distributed system can provide at most two of three guarantees: {{c2::consistency}}, {{c3::availability}}, and {{c4::partition tolerance}}."

# Bulk create from JSON
echo '[
  {"deck": "Biochemistry", "note_type": "Basic",
   "fields": {"Front": "What is the role of ATP synthase?",
              "Back": "Catalyzes the synthesis of ATP from ADP and inorganic phosphate, driven by a proton gradient across the inner mitochondrial membrane."},
   "tags": ["metabolism", "mitochondria"]}
]' | shrike note create --json-input

# Update and delete
shrike note update 1779749914797 -f Back="to read; to interpret"
shrike note delete 1779749914797 --yes

# Manage note types
shrike type list
shrike type show Basic
shrike type create --name "Vocab" --field Word --field Meaning \
  --template 'Card 1:{{Word}}:{{FrontSide}}<hr>{{Meaning}}'
```

Pass `--json` for machine-readable output or `--no-pretty` to disable Rich formatting. See [`docs/cli-reference.md`](docs/cli-reference.md) for the full command reference.

### Configuration

YAML config at the platform config directory (`~/.config/shrike/config.yml` on Linux, `~/Library/Application Support/shrike/config.yml` on macOS). Environment variables `SHRIKE_URL` and `SHRIKE_COLLECTION` are also supported. CLI flags take precedence over both.

## MCP Tools

Shrike exposes the following MCP tools:

- **collection_info** — collection structure, note types, decks, tags, and stats
- **list_notes** — filter and retrieve notes by deck, tags, type, IDs, or date
- **search_notes** — semantic similarity search over note embeddings
- **upsert_notes** — create or update notes in bulk
- **upsert_note_types** — create or update note type definitions
- **delete_notes** — permanently delete notes by ID
- **delete_note_types** — delete note types by ID (only if no notes use them)

The authoritative machine schema is whatever the running server advertises via
`tools/list`; [`docs/mcp-tools.md`](docs/mcp-tools.md) is the human-readable companion.

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

Conventions for branching, versioning, releases, and how defects get tracked are
in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Roadmap

Shipped through **v0.3.x**: CLI + MCP server, semantic search (local embeddings +
USearch), and the reference skill plugin for LLM-driven card creation. Next up is
sync (v0.4). Tracked work lives in
[GitHub issues and milestones](https://github.com/lathrys-at/shrike/milestones),
not here — see `CONTRIBUTING.md` for how the roadmap is organised.

## License

AGPL-3.0-or-later
