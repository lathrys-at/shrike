# Shrike CLI — the commands this skill uses

Orientation for the `shrike` CLI, focused on the operations card authoring needs:
`info` and `note`. Read this once instead of probing `--help` repeatedly. (The
full CLI — `server`, `type`, `index`, `embedding` — is in the repo's
`docs/cli-reference.md`; a card-authoring agent doesn't touch those.)

> This is a hand-maintained **subset** of `docs/cli-reference.md`. If the two
> disagree, the live `shrike … --help` is the truth; keep this in sync when the
> CLI changes.

Global: every command takes `--json` (raw JSON instead of styled text). **Pass
`--json` whenever you need the structured payload** — the similarity *scores* on
search and the *neighbors* returned on create only appear in JSON.

## `shrike info` — orient in the collection

```
shrike info [--types] [--decks] [--tags] [--stats] [--type-details NAME]
```
No flags → a compact summary. Use it **once** to learn what exists before
drafting:

- `--types` — note types with their field names (use the real names; you can
  only create notes of a type that already exists).
- `--decks` — decks with note counts.
- `--tags` — the established tag vocabulary (adopt these; don't coin synonyms).
- `--stats` — scheduling stats. `--type-details NAME` — templates + CSS.

```bash
shrike info --decks --types --tags --json
```

## `shrike note search` — check for existing coverage

```
shrike note search [QUERIES]... [--similar-to ID] [--top-k N]
                   [--threshold FLOAT] [--deck TEXT] [--tags TEXT] [--brief]
```
Semantic search. Give natural-language query strings (the actual claim of a card
you drafted — not a bare keyword), and/or `--similar-to <id>` to find notes like
an existing one. `--threshold` is the min cosine (default 0.5); `--top-k`
default 10. Restrict with `--deck`/`--tags`.

```bash
shrike note search "mitochondria produce ATP by aerobic respiration" --json
shrike note search --similar-to 1700000000123 --json
```
`--json` returns `results[].matches[]` with `id`, `deck`, `tags`, `content`, and
`score`. **Read the content; don't judge from the score alone.**

## `shrike note list` / `shrike note show` — exact lookups

```
shrike note list (--deck TEXT | --tags TEXT | --type TEXT | --ids ID |
                  --since ISO8601 | --query "anki query") [--brief] [--limit N]
shrike note show NOTE_ID
```
At least one filter is required. `--brief` returns only IDs + metadata (no field
content) — handy for triage. `note show` is shorthand for `note list --ids ID`.

```bash
shrike note list --deck "Biology" --json
shrike note show 1700000000123 --json
```

## `shrike note create` — create notes

```
shrike note create --deck TEXT --type TEXT -f KEY=VALUE [-f …] [--tags a,b] 
shrike note create --json-input        # bulk: JSON array on stdin
```
Inline needs `--deck`, `--type`, and at least one `-f/--field`. For more than a
couple of cards, pipe a JSON array to `--json-input` (one upsert, 1–100 notes):

```bash
echo '[
  {"deck":"Biology","note_type":"Cloze",
   "fields":{"Text":"The citric acid cycle runs in the {{c1::mitochondrial matrix}}."},
   "tags":["biology","metabolism"]},
  {"deck":"Biology","note_type":"Basic",
   "fields":{"Front":"Where is ATP synthase located?","Back":"The inner mitochondrial membrane."},
   "tags":["biology","metabolism"]}
]' | shrike note create --json-input --json
```
With `--json`, the response carries per-note `neighbors` (the most similar
existing notes, each with `id`, `score`, `tags`) — your post-write dedup/tag
check. JSON note object: `deck`, `note_type`, `fields` (required), `tags`
(optional). Use the note type's real field names (`Front`/`Back` for Basic,
`Text`/`Back Extra` for Cloze).

## `shrike note update` — refine an existing note

```
shrike note update NOTE_ID [-f KEY=VALUE …] [--tags a,b] [--deck TEXT]
```
Only the fields you pass change; `--tags` **replaces** all tags (include the ones
you want to keep). Use this to improve a note that already covers a fact, rather
than creating a parallel one.

## `shrike note tag` — re-tag notes in bulk

```
shrike note tag NOTE_IDS... --set a,b
```
Replaces the tags on every listed note with the same set (`--set ""` clears).
Like `note update --tags` but across several notes at once, and it touches
nothing but tags. Handy when you've created a batch and want to align them all to
the neighborhood's vocabulary in one call.

## `shrike note delete` — remove notes

```
shrike note delete NOTE_IDS... [--yes]
```
Permanent. Per this skill's boundaries, only ever use it to clean up a duplicate
**you created in this same session** and confirmed against the original — never
a pre-existing note without the user's say-so.
