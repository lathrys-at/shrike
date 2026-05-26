# Shrike

An alternative Anki client that runs headless — no GUI, no Qt, no desktop app required. Shrike opens your collection directly through Anki's Python library and gives you programmatic access via a CLI and an MCP server that LLM agents can talk to.

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

The server listens on `http://127.0.0.1:8372/mcp` and speaks JSON-RPC 2.0. Any MCP client (Claude Desktop, Claude Code, etc.) can connect to it directly.

You don't need to start the server manually — if you run a CLI command and the server isn't up, it auto-starts the daemon for you.

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

All output supports `--json` for machine-readable output and `--pretty/--no-pretty` for controlling Rich formatting.

### Configuration

YAML config at the platform config directory (`~/.config/shrike/config.yml` on Linux, `~/Library/Application Support/shrike/config.yml` on macOS). Environment variables `SHRIKE_URL` and `SHRIKE_COLLECTION` work too. CLI flags override everything.

## MCP Tools

Shrike exposes seven tools over MCP:

- **collection_info** — collection structure, note types, decks, tags, and stats
- **list_notes** — filter and retrieve notes by deck, tags, type, IDs, or date
- **search_notes** — semantic similarity search (not yet implemented)
- **upsert_notes** — create or update notes in bulk
- **upsert_note_types** — create or update note type definitions
- **delete_notes** — permanently delete notes by ID
- **delete_note_types** — delete note types by ID (only if no notes use them)

Full input/output schemas are in [`docs/mcp-schema.json`](docs/mcp-schema.json).

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/unit -v                         # 93 unit tests
pytest tests/integration -v -m integration   # 103 integration tests
ruff check src/shrike/
mypy src/shrike/
```

## Roadmap

- **v0.1.0** — CLI and MCP server *(done)*
- **v0.2.0** — Semantic search via local embeddings (llama-server + USearch)
- **v0.3.0** — AnkiWeb and self-hosted sync support
- **v0.4.0** — Desktop application (Tauri)

## License

AGPL-3.0-or-later
