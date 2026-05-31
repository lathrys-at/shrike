#!/usr/bin/env bash
#
# Launch a clean Shrike server for manual QA / skill testing.
#
# Each launch starts from a known state: it stops any running server, wipes the
# previous run, regenerates the synthetic fixture collection from
# tests/qa/collection.json, and starts the server pointed at run-local
# collection / cache / log directories. Everything mutable lives under
# tests/qa/run/ (gitignored); the tracked corpus and config never change.
#
# Semantic search & upsert neighbors need an embedding model. Set:
#   export SHRIKE_EMBEDDING_MODEL=/path/to/embedding-model.gguf
#   export LLAMA_SERVER_PATH=/path/to/llama-server   # or have it on PATH
# Or run with --no-embedding to test the non-semantic paths only.
#
# Usage:
#   scripts/launch-qa-server.sh                 # clean rebuild, daemon
#   scripts/launch-qa-server.sh -f              # foreground (Ctrl+C to stop)
#   scripts/launch-qa-server.sh --keep          # reuse existing run/ (no wipe/rebuild)
#   scripts/launch-qa-server.sh --no-embedding  # start without llama-server
#
# Stop it with:  shrike server stop

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QA="$ROOT/tests/qa"
RUN="$QA/run"

# Prefer the repo venv so we never run against a stale global install.
if [[ -x "$ROOT/.venv/bin/shrike" ]]; then
  SHRIKE="$ROOT/.venv/bin/shrike"
  PY="$ROOT/.venv/bin/python"
elif command -v shrike >/dev/null 2>&1; then
  SHRIKE="$(command -v shrike)"
  PY="$(command -v python3)"
else
  echo "!! 'shrike' not found. Activate the venv or 'pip install -e .' first." >&2
  exit 1
fi

KEEP=0
FOREGROUND=0
NO_EMBEDDING=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    -f | --foreground) FOREGROUND=1 ;;
    --no-embedding) NO_EMBEDDING=1 ;;
    -h | --help)
      sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

echo "==> Stopping any running shrike server (the server.lock is shared state)…"
"$SHRIKE" server stop >/dev/null 2>&1 || true

if [[ "$KEEP" -eq 0 ]]; then
  echo "==> Wiping previous run: $RUN"
  rm -rf "$RUN"
  mkdir -p "$RUN/cache" "$RUN/logs"
  echo "==> Generating fixture collection → $RUN/working.anki2"
  "$PY" "$QA/build_collection.py" --out "$RUN/working.anki2"
else
  echo "==> --keep: reusing existing $RUN (no wipe, no rebuild)"
  mkdir -p "$RUN/cache" "$RUN/logs"
  if [[ ! -f "$RUN/working.anki2" ]]; then
    echo "!! --keep set but $RUN/working.anki2 is missing; generating it." >&2
    "$PY" "$QA/build_collection.py" --out "$RUN/working.anki2"
  fi
fi

START=(server start
  --collection "$RUN/working.anki2"
  --cache-dir "$RUN/cache"
  --log-dir "$RUN/logs")

if [[ "$NO_EMBEDDING" -eq 1 ]]; then
  START+=(--no-embedding)
elif [[ -z "${SHRIKE_EMBEDDING_MODEL:-}" ]]; then
  echo "!! SHRIKE_EMBEDDING_MODEL is not set — semantic search and upsert" >&2
  echo "   neighbors will be UNAVAILABLE. Set it, or pass --no-embedding to" >&2
  echo "   acknowledge and silence this." >&2
fi

[[ "$FOREGROUND" -eq 1 ]] && START+=(--foreground)

echo "==> $SHRIKE --config $QA/config.yml ${START[*]}"
exec "$SHRIKE" --config "$QA/config.yml" "${START[@]}"
