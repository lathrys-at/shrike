# Shrike

Manage Anki flashcard collections without running Anki. Shrike opens the collection directly via Anki's headless Python library and exposes operations through an MCP server and CLI.

## Install

Requires Python 3.12+.

```bash
pip install -e .
```

## Usage

### MCP server

```bash
# Start the daemon (backgrounded, manages its own lifecycle):
shrike server start --collection /path/to/collection.anki2

# Or run in the foreground:
python -m shrike.server --collection /path/to/collection.anki2
```

The server listens on `http://127.0.0.1:8372/mcp` and speaks JSON-RPC 2.0. Any MCP client (Claude Desktop, Claude Code, etc.) can connect directly.

### CLI

```bash
shrike info                          # Collection overview
shrike note list --deck "My Deck"    # List notes
shrike note create --deck "My Deck" --type Basic --fields '{"Front": "Q", "Back": "A"}'
shrike type list                     # List note types
```

The CLI talks to the MCP server over HTTP. If the server isn't running, it auto-starts the daemon.

### Configuration

YAML config at `~/.config/shrike/config.yml`. Environment variables `SHRIKE_URL` and `SHRIKE_COLLECTION` also work. CLI flags override everything.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `collection_info` | Collection structure, note types, decks, tags, stats |
| `list_notes` | Filter notes by deck, tags, type, IDs, date |
| `search_notes` | Semantic similarity search (not yet implemented) |
| `upsert_notes` | Create or update notes in bulk |
| `upsert_note_types` | Create or update note type definitions |
| `delete_notes` | Delete notes by ID |

Full schemas: [`docs/mcp-schema.json`](docs/mcp-schema.json)

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/unit -v
pytest tests/integration -v -m integration
ruff check src/shrike/
mypy src/shrike/
```

## Roadmap

- **v0.1.0** — CLI and MCP server, fully tested and polished
- **v0.2.0** — Semantic search via local embeddings (llama-server + USearch)
- **v0.3.0** — AnkiWeb and self-hosted sync support
- **v0.4.0** — Desktop application (Tauri)

## License

AGPL-3.0-or-later
