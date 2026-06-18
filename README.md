# Shrike

[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lathrys-at/shrike/badges/coverage.json)](https://github.com/lathrys-at/shrike/actions/workflows/coverage.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Sometimes you want to work on your Anki collection without opening Anki. That's what Shrike is for. It's a command-line client and an MCP server for your collection, so you can manage cards from a terminal or hand them to an LLM agent.

Shrike works directly on the collection's SQLite database, without using the Anki desktop app or relying on the AnkiConnect bridge. You get:

- a `shrike` CLI to browse, create, edit, and delete notes and note types
- an MCP server exposing the same operations to agents
- semantic search over your notes, with embeddings computed locally

This README covers running Shrike yourself: install the package, point it at your collection, run the daemon. Desktop, mobile, and web apps built on the same core are planned; [docs/distribution.md](docs/distribution.md) is the map of where each is headed.

## Requirements

- Python 3.12 or newer
- An Anki collection (a `collection.anki2` file)

## Getting started

Install Shrike from PyPI:

```bash
pipx install shrike-py   # or: pip install shrike-py
```

The package is `shrike-py`; the command it installs is `shrike`. (It was published as `shrike-mcp` before; that name now resolves to a thin shim that depends on `shrike-py` and warns on import, so an existing `pip install shrike-mcp` keeps working while it migrates.)

Point Shrike at your collection and start the daemon:

```bash
shrike server start --collection ~/path/to/collection.anki2
```

That starts a background daemon with your collection open. The other `shrike` commands talk to it, so they don't repeat `--collection`:

```bash
shrike collection info
shrike note list --deck Default
```

And when you're done:

```bash
shrike server stop
```

## Semantic search

`shrike search` finds notes by meaning instead of keywords. To turn it on, declare an embedder in your config file (see [Configuration](#configuration) for where the file lives), then start the server as usual. The entry says what kinds of content you want searchable and where the model runs.

### Text search with a local model

The simplest setup runs a small ONNX model inside the Shrike process. There is no separate binary to install or keep alive, single-card updates are quick, and a quantized model keeps memory low enough for small machines; this is the setup for a Raspberry Pi. Put a `model.onnx` and its `tokenizer.json` in a directory ([all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) has good small exports, including quantized ones a few times smaller than the default), then:

```yaml
embedders:
  - modalities: [text]
    runtime: onnx
    model: ~/models/all-MiniLM-L6-v2-onnx
```

### Searching images too

If your cards carry images worth searching by content, use a CLIP export instead: one model that embeds text and images into the same space, so a query like "diagram of the Krebs cycle" can find a card whose answer is a picture, even when the card's text never says so.

```yaml
embedders:
  - modalities: [text, image]
    runtime: onnx
    model: ~/models/clip-vit-base-patch32
```

The model directory holds `text_model.onnx`, `vision_model.onnx`, `tokenizer.json`, and `preprocessor_config.json`; that's the layout of [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32), and a larger model like jina-clip-v2 searches noticeably better.

### A GGUF model through llama-server

If you already use llama.cpp, or the model you want ships as GGUF, declare a remote entry with no endpoint and Shrike runs llama-server for you: spawned at start, stopped with the daemon.

```yaml
embedders:
  - modalities: [text]
    runtime: remote
    model: ~/models/all-MiniLM-L6-v2-Q4_K_M.gguf
managed:
  llama_server:
    manage: auto
    binary: ~/llama.cpp/build/bin/llama-server   # omit if llama-server is on PATH
```

Last-token models (Jina v5, Qwen3-Embedding) need `pooling: last` on the entry; their GGUF metadata doesn't carry it, and without it the vectors come out wrong. BERT-family models like MiniLM and bge need nothing.

### A llama-server you already run

If another process owns a llama-server, point Shrike at it instead of spawning a second one. Shrike checks it answers and uses it; it never starts, restarts, or stops it.

```yaml
embedders:
  - modalities: [text]
    runtime: remote
managed:
  llama_server:
    manage: attach
    port: 8373
```

### A model served somewhere else

Any OpenAI-compatible embeddings endpoint works: a cloud provider, a box on your tailnet. Name the environment variable holding the API key; the key itself never goes in the config file. Keep in mind your note text goes to that endpoint to be embedded, so think before pointing Shrike at a third party.

```yaml
embedders:
  - modalities: [text]
    runtime: remote
    model: text-embedding-3-small
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
```

### Using a GPU

For the in-process entries on an NVIDIA GPU, swap the bundled CPU runtime for the GPU build (`pip uninstall onnxruntime`, then `pip install onnxruntime-gpu`; the two can't be installed together) and add `providers: [CUDAExecutionProvider]` to the entry. On a Mac the plain install already runs on the Apple GPU. Either way, `shrike server status` shows the provider that actually loaded, so you can confirm the accelerator is in use rather than a quiet fall back to the CPU. For a managed llama-server, set `gpu_layers:` under `managed.llama_server`. A floating-point model rather than a small quantized one is what makes a GPU worthwhile, since it can embed cards in larger batches.

### Picking a model

Search quality comes from the model, not from where it runs. Text-only models are fully supported everywhere.

| You want | In process (`runtime: onnx`) | Via llama-server (`runtime: remote`) |
|---|---|---|
| Text search | [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) ONNX export | [all-MiniLM-L6-v2](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) GGUF |
| Text and image search | [clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32); jina-clip-v2 for better results | not yet; omni models over llama-server are [#501](https://github.com/lathrys-at/shrike/issues/501) |

One embedder entry at a time for now; running a dedicated text model alongside a CLIP model as separate entries is tracked in [#229](https://github.com/lathrys-at/shrike/issues/229), and the config error will tell you exactly that if you try.

### Reading text inside images (OCR)

OCR support is being reworked. The server no longer bundles Apple's Vision OCR (on-device platform models belong to the mobile apps; the server's replacement is recognition over a configured endpoint, tracked in [#502](https://github.com/lathrys-at/shrike/issues/502)), so `--ocr-backend apple` currently reports recognition as unavailable rather than scanning your images. Text already recognized by an earlier version stays searchable. When recognition is active, recognized text becomes searchable like field text: exact and typo-tolerant matches work with any backend, and with an embedding backend the text is indexed for semantic search too.

### The index

Shrike builds an index of your notes in the background. A large collection takes a little while the first time; search will tell you if it's still indexing. Once it's ready:

```bash
shrike search "electron transport chain"
shrike search --similar-to 1700000000123
```

`shrike search` also matches your query as exact text and as a near-miss (so `protien` still finds protein cards); each result shows you which applied. These text matches work even without the embedding service running.

## CLI

`shrike collection info` summarizes the collection: note types, decks, tags, and scheduling stats.

```bash
shrike collection info
shrike collection info --types --decks --stats
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

For tags across the whole collection, `shrike collection tag` renames:

```bash
shrike collection tag rename history::ww2 history::wwii   # rename everywhere it appears
shrike collection tag rename jp japanese --note 1779749914797   # only on these notes
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

`shrike collection media` gets images and audio into the collection (reference the stored name from a note field with `<img src="...">` or `[sound:...]`):

```bash
shrike collection media store diagram.png                  # upload a local file
shrike collection media store --url https://example.com/cell.png
shrike collection media list '*.png'
shrike collection media fetch diagram.png -o /tmp/out.png  # download it back
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
- **collection_check**: read-only media diagnostics: unused files, missing files, broken references
- **collection_prune**: clean up unused tags, empty notes, empty cards, and unused media (previews by default)

The machine schema is whatever the running server advertises via `tools/list`. [`docs/mcp-tools.md`](docs/mcp-tools.md) is the human-readable companion.

## Configuration

Shrike reads a YAML file from the platform config directory: `~/.config/shrike/config.yml` on Linux, `~/Library/Application Support/shrike/config.yml` on macOS. The file is yours to manage; `shrike server start` never writes it on its own (pass `--save-config` to persist the operational flags you started with).

Models and managed processes are declared only in the file. The `embedders:` entries above, a `recognizers:` map (OCR and speech recognition; being reworked, see [#502](https://github.com/lathrys-at/shrike/issues/502)), and `managed:` for processes Shrike runs on your behalf. There is no flag or environment-variable spelling for these; a structured setting wants one home. If the file declares something this build or this release can't serve, the server names what's missing and refuses to start, rather than quietly skipping it.

A complete file looks like this:

```yaml
collection: ~/Anki/User 1/collection.anki2

embedders:
  - modalities: [text]
    runtime: onnx
    model: ~/models/all-MiniLM-L6-v2-onnx

server:
  port: 8372
```

Operational settings (the collection path, host and port, log level, cache directory) also take command-line flags and environment variables (`SHRIKE_URL`, `SHRIKE_COLLECTION`); flags win, then the environment, then the file. [The CLI reference](docs/cli-reference.md) has the full list.

If you configured embedding before this config shape existed: a legacy `embedding:` section keeps working for one more release and prints a pointer to the new shape, and the old embedding flags keep working alongside it ([#523](https://github.com/lathrys-at/shrike/issues/523) is the removal). Once your file declares `embedders:`, the old flags are rejected rather than silently ignored.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "shrike-py/[dev]"

pytest shrike-py/tests/unit -v                         # unit tests (no server)
pytest shrike-py/tests/integration -v -m integration   # integration tests (real server subprocess)
ruff check shrike-py/src/shrike/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
```

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers branching, versioning, releases, and how defects get tracked.

## License

AGPL-3.0-or-later
