# QA harness

A manual testing setup for driving a real Shrike server against a known,
disposable collection — used for hand-testing the card-authoring skill and any
other end-to-end behaviour that's awkward to cover in the automated suites.

It is **not** part of `pytest`. Nothing here runs in CI.

## What's here

| Path | Tracked? | What it is |
|---|---|---|
| `collection.json` | yes | The fixture corpus — decks, note types, notes, tags. The source of truth. |
| `build_collection.py` | yes | Generates a fresh `collection.anki2` from the corpus, via Shrike's own write path. |
| `config.yml` | yes | Portable server knobs (ports, fast index flush, debug logging). No paths. |
| `scenarios/` | yes | Hand-run skill test scenarios — prompt + expected outcome, one `.md` each. See `scenarios/README.md`. |
| `run/` | **no** (gitignored) | Everything mutable: the built `working.anki2`, the vector index cache, and logs. Recreated on each launch. |

There is no binary collection in git: the JSON corpus is the fixture, rebuilt
into `run/working.anki2` every launch. Grow the corpus by editing
`collection.json` (keep it valid JSON); the `_`-prefixed keys are documentation
and are ignored by the generator.

## Prerequisites

- The repo installed into `.venv` (`pip install -e ".[dev]"`). The launch script
  prefers `.venv/bin/shrike`, so a stale global `shrike` won't interfere.
- For semantic search and upsert neighbors (the interesting half), the
  `llama-server` binary and an embedding model. `scripts/fetch-llama-server.sh`
  fetches a pinned `llama-server` (and prints the export lines for it).

- **Model choice matters for QA.** That fetch script's default —
  `all-MiniLM-L6-v2` (384-dim) — is what CI uses: small and fast, but weak at
  telling a near-duplicate from a merely-related card, which is exactly the
  signal this skill leans on. For realistic QA, use a stronger model. The
  recommended default is **bge-m3** (BERT-family, multilingual so it also covers
  the Spanish deck, a clean drop-in through the same launch path):

  ```bash
  mkdir -p .cache/models
  curl -L "https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf?download=true" \
    -o .cache/models/bge-m3-Q8_0.gguf

  export LLAMA_SERVER_PATH="$PWD/.cache/llama-server/llama-server"   # from fetch-llama-server.sh
  export SHRIKE_EMBEDDING_MODEL="$PWD/.cache/models/bge-m3-Q8_0.gguf"
  ```

  Any BERT-family embedding GGUF works the same way (bge-large-en-v1.5,
  mxbai-embed-large, nomic-embed-text-v1.5, …); the model just needs to be
  consistent across a session, since changing it invalidates the index and
  forces a rebuild.

  Last-token-pooling models (Jina v5, Qwen3-Embedding) need their pooling type
  set explicitly — their GGUF metadata omits it, so without it llama-server
  defaults to mean and produces wrong embeddings. Point the harness at one with
  `SHRIKE_EMBEDDING_MODEL` and set `SHRIKE_EMBEDDING_POOLING=last` (or pass
  `--embedding-pooling last` to `shrike server start` / `shrike embedding
  start`). Note that some of these architectures (e.g. EuroBERT, which
  jina-v5-nano is built on) may need a newer `llama-server` than the pinned
  build — see `scripts/llama-server.lock`.

## Usage

```bash
# Clean rebuild + start as a daemon (default port 8372):
scripts/launch-qa-server.sh

# Foreground (watch logs live, Ctrl+C to stop):
scripts/launch-qa-server.sh -f

# Reuse the existing run without wiping/rebuilding (e.g. to inspect a persisted index):
scripts/launch-qa-server.sh --keep

# No embeddings (tests the non-semantic paths only):
scripts/launch-qa-server.sh --no-embedding
```

The server runs on the default port, so the plain CLI talks to it directly:

```bash
shrike server status
shrike info --decks --types --tags
shrike note search "mitochondria" --json
shrike server logs -f
shrike server stop          # when you're done
```

## Notes & limitations

- **Shared state dir.** `server.lock` / `server.pid` / `server.json` /
  `embedding.pid` live in the platform state dir (not configurable), so the QA
  server uses the *same* lock as a normal Shrike server — only one can run at a
  time. The launch script stops any running server first. Don't expect the QA
  run to be fully self-contained on disk; only the collection, index, and logs
  are redirected into `run/`.
- **Dedup targets.** Some corpus facts (mitochondria→ATP, capital of France,
  WWII end year, …) exist specifically so a test prompt that re-covers them
  gives the skill's pre-create search and neighbor net something real to catch.
  See `_dedup_targets` in `collection.json`.
- **Index flush is tuned tight** (2s idle / 5 changes) so persistence is quick
  to observe; production defaults are 60s / 100.
