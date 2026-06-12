# Shrike

[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lathrys-at/shrike/badges/coverage.json)](https://github.com/lathrys-at/shrike/actions/workflows/coverage.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

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

`shrike note search` finds notes by meaning instead of keywords. You supply an embedding model to turn your notes into vectors, and pick a backend to run it. Shrike has three backends, and which one suits you depends mostly on what's on your cards.

### Choosing a backend

For plain-text cards, which is the common case, the ONNX backend is the simpler choice. It runs inside Shrike, so there's no separate binary to install or process to keep alive, single-card lookups are quick, and a small quantized model keeps memory use low. Reach for this one if you just want related-card search to work.

Pick the llama-server backend if you already use llama.cpp or want to run a GGUF or MLX model. Pick the CLIP backend if your cards carry images you want to search by content — it embeds text and images into one shared space, so a text query can surface a card whose meaning lives in its picture. All three can use a GPU; see the ONNX section below for the NVIDIA and Apple paths.

Whichever you pick, search works the same, text-only models are fully supported, and search quality comes from the model you choose rather than the backend.

### ONNX backend

Install the extra, then put an ONNX embedding model in a directory: a `model.onnx` file and its `tokenizer.json`. Good small models live in the [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) repository, including quantized exports a few times smaller than the default; download the `.onnx` file you want, save it as `model.onnx`, and put `tokenizer.json` beside it.

```bash
pip install 'shrike-mcp[onnx]'

shrike server start --collection ~/path/to/collection.anki2 \
  --embedding-backend onnx \
  --embedding-model ~/models/all-MiniLM-L6-v2-onnx
```

If you have an NVIDIA GPU, install `shrike-mcp[onnx-gpu]` instead of `[onnx]` (the two onnxruntime builds can't be installed together) and add `--embedding-onnx-provider CUDAExecutionProvider`. On a Mac the plain install already runs on the Apple GPU. Either way, `shrike server status` shows the provider that actually loaded, so you can confirm the accelerator is in use rather than a quiet fall back to the CPU. A floating-point model (rather than a small quantized one) is what makes a GPU worthwhile, since it can embed cards in larger batches.

### llama-server backend

You supply a `llama-server` binary (from llama.cpp) and a GGUF embedding model. Get llama.cpp by building it or installing it from a package manager. Any GGUF embedding model works; a small one like [all-MiniLM-L6-v2](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) runs on CPU and is plenty for finding related cards.

```bash
shrike server start --collection ~/path/to/collection.anki2 \
  --llama-server ~/llama.cpp/build/bin/llama-server \
  --embedding-model ~/models/all-MiniLM-L6-v2-Q4_K_M.gguf
```

If `--llama-server` or `--embedding-model` isn't given on the command line or in your config file, Shrike falls back to `LLAMA_SERVER_PATH` and a `llama-server` on your `PATH` for the binary, and to `SHRIKE_EMBEDDING_MODEL` for the model.

### CLIP backend (search images by content)

Install the `clip` extra and point `--embedding-model` at a CLIP ONNX export — a directory holding `text_model.onnx`, `vision_model.onnx`, `tokenizer.json`, and `preprocessor_config.json` (the layout of [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32); a larger model like jina-clip-v2 searches noticeably better):

```bash
pip install 'shrike-mcp[clip]'

shrike server start --collection ~/path/to/collection.anki2 \
  --embedding-backend clip \
  --embedding-model ~/models/clip-vit-base-patch32
```

With this backend, a query like "diagram of the Krebs cycle" can find a card whose answer is a picture, even when the card's text never says so. Text-only search still works exactly as on the other backends.

### Reading text inside images (OCR)

On macOS, `--ocr-backend apple` runs Apple's Vision OCR over the images on your cards in the background (nothing extra to install — Vision ships with the OS). Recognized text becomes searchable like field text: exact and typo-tolerant matches work with any backend, and with an embedding backend the text is indexed for semantic search too. Results tell you which image matched.

### The index

Shrike builds an index of your notes in the background. A large collection takes a little while the first time; search will tell you if it's still indexing. Once it's ready:

```bash
shrike note search "electron transport chain"
shrike note search --similar-to 1700000000123
```

`shrike note search` also matches your query as exact text and as a near-miss (so `protien` still finds protein cards); each result shows you which applied. These text matches work even without the embedding service running.

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

Fix text across many notes at once with `shrike note replace`. It needs a scope, previews the changes, and asks before applying:

```bash
shrike note replace "teh" "the" --deck "Biology"          # preview, confirm, apply
shrike note replace "colou?r" "color" --regex --tags spelling --dry-run
```

For tags across the whole collection, `shrike tag` renames:

```bash
shrike tag rename history::ww2 history::wwii   # rename everywhere it appears
shrike tag rename jp japanese --note 1779749914797   # only on these notes
```

Manage decks with `shrike deck`:

```bash
shrike deck create "Japanese::Vocabulary"      # nested decks use ::
shrike deck rename "Misc::French" "French"      # rename or reparent
shrike deck delete "Old Deck" --yes            # delete (must be empty first)
```

Anywhere a command takes a deck (including `--deck` on note commands), you can pass the deck's name, its numeric ID, or `#id` instead.

Note types have their own commands:

```bash
shrike type list
shrike type show Basic
shrike type create --name Vocab --field Word --field Meaning \
  --template 'Card 1:{{Word}}:{{FrontSide}}<hr>{{Meaning}}'
```

`shrike media` gets images and audio into the collection (reference the stored name from a note field with `<img src="...">` or `[sound:...]`):

```bash
shrike media store diagram.png                  # upload a local file
shrike media store --url https://example.com/cell.png
shrike media list '*.png'
shrike media fetch diagram.png -o /tmp/out.png  # download it back
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
- **search_notes**: semantic similarity and exact-substring search over notes
- **collection_query**: find notes with a raw Anki search expression (`is:due`, `prop:`, …)
- **find_replace_notes**: bulk find and replace across note fields in a scoped set
- **migrate_note_type**: change notes' note type with a field/template map (preserves history)
- **upsert_notes**: create or update notes in bulk
- **upsert_note_types**: create or update note type definitions
- **update_note_type_fields**: add, remove, rename, or reposition a note type's fields (data-safe)
- **update_note_type_templates**: add, remove, rename, or reposition a note type's card templates (data-safe)
- **find_replace_note_types**: find and replace text in a note type's template HTML and CSS
- **update_note_type_field_metadata**: set a note type's per-field editor metadata (font, size, description)
- **update_note_tags**: set, add, or remove tags on a set of notes
- **rename_tag**: rename a tag collection-wide or on specific notes
- **upsert_decks**: create or rename/reparent decks in bulk
- **delete_decks**: delete decks by name, if empty
- **delete_notes**: permanently delete notes by ID
- **delete_note_types**: delete note types by ID, if no notes use them
- **store_media**: store media files from base64 data, a URL, or a server-local path
- **fetch_media**: locate media files; returns a download URL per file, never bytes
- **list_media**: list media filenames, optionally filtered by a glob
- **delete_media**: move media files to Anki's recoverable trash
- **collection_check**: read-only media diagnostics — unused files, missing files, broken references
- **collection_prune**: clean up unused tags, empty notes, empty cards, and unused media (previews by default)

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
