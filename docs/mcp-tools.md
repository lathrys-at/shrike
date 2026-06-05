# Shrike MCP Tools

These tools manage an Anki flashcard collection. The server maintains a local vector index over all note content, enabling semantic search and contextual neighbor suggestions without external API calls.

This document is the human-readable reference. The machine-readable schema each tool advertises (input and output) is generated at runtime by the server from the Pydantic models in `shrike/schemas.py`, which is the single source of truth.

Notes in Anki have a **note type** that defines their fields (e.g., a "Basic" note type has "Front" and "Back" fields; a "Cloze" note type has "Text" and "Extra"). A note type also defines **card templates** (HTML templates that control how cards are rendered) and **CSS styling** shared across all its cards. A single note produces one or more cards depending on its note type. Notes belong to a **deck** and can have **tags**.

---

## `collection_info`

Return the structure of the Anki collection: available note types with their fields, deck names, tags, and summary statistics. Use this to orient yourself before creating or searching for notes, especially to discover which note types, fields, and decks exist.

Called with no arguments, returns a compact summary (counts, dates, and collection path). Use the `include` parameter to request specific sections, or `"all"` for everything.

Note type summaries include field names and type (standard/cloze) but not full template HTML or CSS. To inspect or modify templates, request full details with `note_type_details`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `include` | `string[]` | no | Sections to return. Any combination of `"summary"`, `"note_types"`, `"decks"`, `"tags"`, `"stats"`, or `"all"`. Defaults to `["summary"]`. |
| `note_type_details` | `string[]` | no | List of note type names to return full definitions for, including card templates (HTML) and CSS. Omit to return only summaries. |

### Response

The default (no `include`) returns only the `summary` section:

```jsonc
{
  "summary": {
    "path": "/path/to/collection.anki2",
    "created": "2024-01-15",
    "modified": "2026-05-20T14:32:00Z",
    "notes": 3847,
    "cards": 4521,
    "decks": 12,
    "note_types": 5,
    "tags": 38,
    "due_today": 74
  }
}
```

Requesting `include: ["all"]` (or specific sections) adds them:

```jsonc
{
  "note_types": [
    {
      "name": "Basic",
      "id": 1234567890,
      "fields": ["Front", "Back"],
      "type": "standard",          // "standard" or "cloze"
      // present only when requested via note_type_details; null otherwise.
      // templates and css always travel together, so they're grouped here:
      "detail": {
        "templates": [
          {
            "name": "Card 1",
            "front": "{{Front}}",
            "back": "{{FrontSide}}<hr id=answer>{{Back}}"
          }
        ],
        "css": ".card { font-family: arial; font-size: 20px; text-align: center; }"
      }
    }
  ],
  "decks": [
    {
      "name": "Japanese::Vocabulary",  // "::" denotes nesting
      "id": 9876543210,
      "note_count": 482
    }
  ],
  "tags": ["verb", "chapter-3", "leech", "marked"],
  "stats": {
    "total_notes": 3847,
    "total_cards": 4521,
    "cards_due_today": 74,
    "new_cards": 312,
    "decks_summary": {
      "Japanese::Vocabulary": { "notes": 482, "due": 23 },
      "Pharmacology": { "notes": 1205, "due": 51 }
    }
  }
}
```

---

## `list_notes`

Retrieve notes by structured filters: deck, tags, note type, note IDs, or modification date. Use this for precise lookups: fetching specific notes by ID, listing everything in a deck, or finding notes matching exact criteria. For conceptual or fuzzy queries ("cards about mitochondrial membrane potential"), use `search_notes` instead.

Returns note metadata and content. Use `fields: "meta"` to return only metadata, which is useful when listing large result sets for triage before reading individual notes.

Results are capped by `limit`. The response includes `total` (the full count of matching notes) so you can tell whether your query matched more than was returned. If so, narrow your filters rather than attempting to retrieve everything; large result sets aren't useful in conversation.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `ids` | `integer[]` | no | Specific note IDs to retrieve. |
| `deck` | `string` | no | Filter to notes in this deck. Use `"::"` for nested decks (e.g., `"Japanese::Vocabulary"`). Includes child decks. Accepts a deck name, numeric ID, or `#id`. |
| `tags` | `string[]` | no | Filter to notes having **all** of these tags. Prefix a tag with `"-"` to exclude (e.g., `["-leech", "verb"]`). |
| `note_type` | `string` | no | Filter to notes using this note type (e.g., `"Basic"`, `"Cloze"`). |
| `modified_since` | `string` | no | ISO 8601 date or datetime. Only notes modified after this time. |
| `fields` | `string` | no | `"full"` (default) returns all field content. `"meta"` returns only note ID, note type, deck, tags, and modification time. |
| `limit` | `integer` | no | Maximum notes to return. Default `50`, max `200`. |

At least one filter (`ids`, `deck`, `tags`, `note_type`, or `modified_since`) must be provided. For text search, use `search_notes`.

### Response

```jsonc
{
  "notes": [
    {
      "id": 1700000000123,
      "note_type": "Basic",
      "deck": "Japanese::Vocabulary",   // first card's deck (see note below)
      "tags": ["verb", "chapter-3"],
      "modified": "2026-05-20T14:32:00Z",
      // included when fields: "full" (default)
      "content": {
        "Front": "食べる",
        "Back": "to eat (taberu); ichidan verb"
      }
    }
  ],
  "total": 482,     // total matching notes (before limit)
  "limit": 50
}
```

Anki permits per-card decks, so a single note's cards can live in different decks. `deck` reports the **first card's deck**. Shrike treats notes as belonging to one deck, so split-deck notes are represented by that first card only.

---

## `search_notes`

Search the collection by **meaning and by exact text in one call**. Each query string is matched two ways — semantic similarity (the vector index) and exact, case-insensitive substring over note fields — and the results are folded together. Every match carries a `score` when it was semantically ranked and a `substring` annotation (which fields matched + a snippet) when the query text occurs literally; both when both apply.

Use it for conceptual queries keyword search can't handle ("cards about electron transport chain regulation") and for finding exact wording. Note IDs in `ids` are semantic anchors only (no literal text to match).

Exact matches are returned even when the embedding index is unavailable — the response carries a `message` noting semantic ranking was skipped — and are **not** subject to `threshold` (a literal hit is always relevant). Within a group, literal hits are listed first, then by descending score.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `queries` | `string[]` | no | Search strings. Each is matched independently by semantic similarity **and** as an exact substring of note fields. Max 50 per call. |
| `ids` | `integer[]` | no | Note IDs to use as semantic anchors, finding notes similar to these. Source notes are excluded from results. Max 50 per call. |
| `top_k` | `integer` | no | Maximum results per mechanism per query/anchor. Default `10`, max `50`. |
| `threshold` | `number` | no | Minimum cosine similarity (0–1) for a *semantic* match. Default `0.5`. Does not apply to exact substring matches. |
| `deck` | `string` | no | Restrict to notes in this deck (includes child decks). Accepts a deck name, numeric ID, or `#id`. |
| `tags` | `string[]` | no | Restrict to notes matching all of these tags. |
| `exclude_ids` | `integer[]` | no | Additional note IDs to exclude from results. |

At least one of `queries` or `ids` must be provided.

`deck`/`tags` are applied after the vector search over a widened candidate window; if in-scope notes rank very deep a filtered semantic search may still return fewer than `top_k` (exact matches are filtered precisely). 

### Response

Each match is a note annotated with the evidence that produced it: `score` (semantic, `null`/omitted when only an exact hit) and `substring` (`{matched_fields, snippet}`, absent when there was no literal hit).

```jsonc
{
  "results": [
    {
      // one entry per query string or source ID
      "source": "electron transport chain regulation",  // or source note ID
      "matches": [
        {
          "id": 1700000000456,
          "note_type": "Basic",
          "deck": "Biochemistry",
          "tags": ["metabolism", "chapter-18"],
          "content": { "Front": "…", "Back": "…" },
          "score": 0.87,                 // semantic similarity; null/omitted if exact-only
          "substring": {                 // present only when the text matched literally
            "matched_fields": ["Front"],
            "snippet": "…electron transport chain…"
          }
        }
      ]
    }
  ],
  "message": null   // e.g. "Semantic ranking unavailable …" when the index is down
}
```

---

## `upsert_notes`

Create or update notes in bulk. If a note object includes an `id`, the existing note is updated. If `id` is absent, a new note is created.

When creating notes, `deck`, `note_type`, and `fields` are required. When updating, only `id` and the properties being changed need to be provided; omitted properties are left unchanged.

**Duplicate and validity checking.** Each new note is validated against Anki's own add-note rule before it is written. A first-field duplicate — a new note whose first field matches an existing note of the same type (Anki's rule, applied collection-wide and independent of deck) — is governed by `on_duplicate`: `error` (the default; the item is reported and not written), `skip` (`status: "skipped"`), or `allow` (created anyway). Notes that are malformed regardless of policy — an empty first field, or broken cloze structure — are always reported as errors and never written. As with any batch op, one bad note doesn't block the rest. This exact first-field check is distinct from the *semantic* `neighbors` below: a high neighbor score is a softer "this looks similar" hint, while `on_duplicate` enforces Anki's precise rule.

**Dry run.** Set `dry_run: true` to validate every note and write nothing — a pre-flight sanity check. Each result is `ok` (with `action: "create" | "update"`), `skipped`, or `error`, and the response echoes `dry_run: true`. (Because nothing is written, two identical new notes in the *same* dry-run call both validate clean; a real run catches the second.)

When a vector index is available (and not a dry run), each created or updated result includes `neighbors`: the most similar existing notes ranked by cosine similarity. Use these for tag consistency (adopt tags from nearby notes), spotting near-duplicates by meaning, or understanding where a new note sits in the collection.

If the index update fails transiently (for example, the embedding service is briefly unavailable), the notes are still saved but `neighbors` is omitted. Each affected result is flagged with `neighbors_unavailable: true`, and the response carries a top-level `message` naming the IDs to retry. The exact same neighbor data is reproducible afterward with `search_notes` keyed on the note ID (`search_notes(ids=[<note id>])`). It embeds the same note text against the same index, so the result is identical to what would have been attached here.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `notes` | `object[]` | **yes** | Array of note objects (1–100). See note schema below. |
| `on_duplicate` | `string` | no | Policy for a first-field duplicate on **create**: `"error"` (default), `"skip"`, or `"allow"`. Updates are unaffected. |
| `dry_run` | `boolean` | no | If `true`, validate everything and write nothing. Default `false`. |
| `top_k_neighbors` | `integer` | no | Maximum neighbors per result. Default `5`. Set to `0` to disable. |
| `neighbor_threshold` | `number` | no | Minimum cosine similarity for a neighbor. Default `0.5`. Higher values return only very similar notes. |

#### Note object schema

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `integer` | no | Note ID. Present = update, absent = create. |
| `deck` | `string` | create | Target deck. Required for new notes. On update, moves the note to this deck. Accepts a deck name, numeric ID, or `#id` (an unknown ID is an error; a new name is created). |
| `note_type` | `string` | create | Note type (e.g., `"Basic"`, `"Cloze"`). Required for new notes. Cannot be changed on update. |
| `fields` | `object` | create | Field key-value pairs (e.g., `{"Front": "...", "Back": "..."}`). On update, only specified fields are modified. |
| `tags` | `string[]` | no | Tags to set. On create, these are the note's tags. On update, **replaces** all existing tags, so include existing tags you want to keep. |

### Response

```jsonc
{
  "results": [
    {
      "status": "created",
      "id": 1700000000789,
      "neighbors": [               // present when vector index is available
        {
          "id": 1700000000456,
          "score": 0.82,
          "tags": ["metabolism", "chapter-18"]
        }
      ]
    },
    {
      "status": "updated",
      "id": 1700000000123,
      "neighbors": [{ "id": 1700000000001, "score": 0.71, "tags": ["verb"] }]
    },
    {
      "status": "skipped",          // on_duplicate: "skip" met a duplicate
      "index": 2,
      "reason": "duplicate"
    },
    {
      "status": "error",
      "index": 3,                  // position in the input array
      "error": "The first field duplicates an existing note of this type.",
      "reason": "duplicate"        // duplicate | empty | missing_cloze |
                                   // notetype_not_cloze | field_not_cloze |
                                   // unknown_note_type | unknown_field
    }
  ],
  "dry_run": false
}
```

A dry run writes nothing; would-succeed notes report `ok` with the action they would have taken:

```jsonc
{
  "results": [
    { "status": "ok", "index": 0, "action": "create" },
    { "status": "error", "index": 1, "error": "The first field is empty.", "reason": "empty" }
  ],
  "dry_run": true
}
```

When the index update fails transiently, saved notes carry `neighbors_unavailable` instead of `neighbors`, and the response adds a top-level `message`:

```jsonc
{
  "results": [
    {
      "status": "created",
      "id": 1700000000789,
      "neighbors_unavailable": true   // index hiccup; neighbors not computed
    }
  ],
  "message": "Notes were saved, but the vector index update failed, so neighbors could not be computed. Retry with search_notes(ids=[1700000000789]) to fetch the same neighbor data."
}
```

---

## `upsert_note_types`

Create or update note type definitions. A note type defines the schema for a group of notes: its fields, card templates (HTML for front and back of each card), and shared CSS styling.

If a note type object includes an `id`, the existing note type is updated. If `id` is absent, a new note type is created.

When creating, `name`, `fields`, `templates`, and `css` are required. When updating, only `id` and the properties being changed are needed.

Card templates use Anki's `{{FieldName}}` replacement syntax. The special `{{FrontSide}}` tag on the back template inserts the rendered front side. Cloze note types use `{{cloze:FieldName}}` in templates.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_types` | `object[]` | **yes** | Array of note type objects (1–10). See schema below. |

#### Note type object schema

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `integer` | no | Note type ID. Present = update, absent = create. |
| `name` | `string` | create | Name for the note type (e.g., `"Japanese Vocabulary"`). |
| `fields` | `string[]` | create | Ordered list of field names (e.g., `["Word", "Reading", "Meaning"]`). On update, replaces the field list **by position**: the field at each position keeps its note data even when renamed. Only shortening the list discards the trailing fields' data; lengthening it appends empty fields. |
| `templates` | `object[]` | create | Card templates. Each produces one card per note (except cloze types). On update, replaced **by position** like `fields`: existing cards (and their scheduling history) are preserved; only removing a trailing template deletes its cards. See template schema below. |
| `css` | `string` | create | CSS styling shared across all cards of this note type. |
| `is_cloze` | `boolean` | no | If `true`, this is a cloze deletion note type. Default `false`. Cannot be changed on update. |

#### Template object schema

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | yes | Template name (e.g., `"Recognition"`, `"Recall"`). |
| `front` | `string` | yes | Front side HTML. Use `{{FieldName}}` to insert fields. |
| `back` | `string` | yes | Back side HTML. Use `{{FieldName}}` and `{{FrontSide}}` (renders the front side). |

### Response

```jsonc
{
  "results": [
    {
      "status": "created",
      "id": 1234567890,
      "name": "Japanese Vocabulary"
    },
    {
      "status": "updated",
      "id": 9876543210,
      "name": "Basic"
    }
  ]
}
```

### Example: Japanese vocabulary note type

```jsonc
{
  "note_types": [
    {
      "name": "Japanese Vocabulary",
      "fields": ["Word", "Reading", "Meaning", "Pitch", "Example"],
      "is_cloze": false,
      "css": ".card { font-family: 'Noto Sans JP', sans-serif; font-size: 24px; text-align: center; color: #333; }\n.reading { font-size: 16px; color: #888; }\n.example { font-size: 18px; margin-top: 1em; font-style: italic; }",
      "templates": [
        {
          "name": "Recognition",
          "front": "<div class=\"word\">{{Word}}</div>",
          "back": "{{FrontSide}}<hr id=answer><div class=\"reading\">{{Reading}} — {{Pitch}}</div><div>{{Meaning}}</div><div class=\"example\">{{Example}}</div>"
        },
        {
          "name": "Recall",
          "front": "<div>{{Meaning}}</div>",
          "back": "{{FrontSide}}<hr id=answer><div class=\"word\">{{Word}}</div><div class=\"reading\">{{Reading}} — {{Pitch}}</div><div class=\"example\">{{Example}}</div>"
        }
      ]
    }
  ]
}
```

---

## `update_note_tags`

Edit tags on a set of notes (1–1000) without rewriting the whole note. Choose exactly one mode — there is no default:

- `set`: full replace. The notes end up with exactly the tags you pass; an empty list clears all tags.
- `add` and/or `remove`: additive/subtractive. Add tags without disturbing existing ones, remove specific tags, or both in one call (e.g. add `["jp","verbs"]` + remove `["jp-verbs"]` swaps one tag for two).

`set` cannot be combined with `add`/`remove`. To replace a note's tags as part of a broader edit (fields, deck), use `upsert_notes` instead.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_ids` | `integer[]` | **yes** | Note IDs whose tags to edit (1–1000). |
| `set` | `string[]` | no | Full replace (empty clears). Mutually exclusive with `add`/`remove`. |
| `add` | `string[]` | no | Tags to add, leaving other tags intact. |
| `remove` | `string[]` | no | Tags to remove, leaving other tags intact. |

### Response

```jsonc
{
  "notes_modified": 2,
  "not_found": [9999999999999]   // requested IDs that didn't match any note
}
```

---

## `rename_tag`

Rename a tag, collection-wide or on a set of notes. With no `note_ids`, the tag is renamed everywhere it appears, children included (renaming `history` also moves `history::ww2`). With `note_ids`, only those notes are affected and the tag is matched **exactly** — renaming `jp` never touches `jp-verbs`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `old` | `string` | **yes** | The tag to rename. |
| `new` | `string` | **yes** | The new tag name (must differ from `old`). |
| `note_ids` | `integer[]` | no | Restrict the rename to these notes. Omit to rename across the whole collection. |

### Response

```jsonc
{
  "notes_modified": 2
}
```

---

## `upsert_decks`

Create or rename decks in bulk (1–100), the same shape as `upsert_notes`. Each item's `name` is the desired deck name (nested with `::`, e.g. `Japanese::Vocabulary`). An optional `id` selects an existing deck to rename/reparent to `name`.

- item `{name}` (no id) → ensure a deck named `name` exists: `created` if newly made, `updated` if it already existed.
- item `{id, name}` → rename/reparent deck `id` to `name` (`updated`). **Decks do not merge**: renaming onto a name another deck already uses returns an `error` for that item. An unknown `id` is also an `error`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `decks` | `object[]` | **yes** | 1–100 deck objects. Each: `name` (string, required), `id` (integer, optional — present = rename that deck to `name`). |

### Response

```jsonc
{
  "results": [
    { "status": "created", "id": 1700000000111, "name": "Japanese::Vocabulary" },
    { "status": "updated", "id": 1700000000222, "name": "French" },
    { "status": "error", "index": 2, "name": "Dup", "error": "A deck named 'Dup' already exists" }
  ]
}
```

---

## `delete_decks`

Delete decks by name — **only if empty**. A deck is deletable only when neither it nor any of its subdecks contains cards. To remove a non-empty deck, move its notes elsewhere first (`upsert_notes` with a new `deck`), then delete the now-empty deck. This keeps deletion from ever destroying a note.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `decks` | `string[]` | **yes** | 1–100 decks to delete, each a name, numeric ID, or `#id`. |

### Response

```jsonc
{
  "deleted": ["Old Deck"],
  "not_found": ["Typo Deck"],     // no deck by that name
  "not_empty": ["Active Deck"]    // skipped: it (or a subdeck) still has cards
}
```

---

## `delete_notes`

Permanently delete notes and all their associated cards. This cannot be undone.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `ids` | `integer[]` | **yes** | Note IDs to delete (1–100). |

### Response

```jsonc
{
  "deleted": [1700000000123, 1700000000456],
  "not_found": [9999999999999]   // IDs that didn't match any note
}
```

---

## `delete_note_types`

Permanently delete note type definitions. A note type can only be deleted if no notes currently use it; attempting to delete a note type that has notes returns an error for that item.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `ids` | `integer[]` | **yes** | Note type IDs to delete (1–10). |

### Response

```jsonc
{
  "results": [
    {
      "status": "deleted",
      "id": 1234567890,
      "name": "Old Type"
    },
    {
      "status": "error",
      "id": 9876543210,
      "name": "Basic",
      "error": "Cannot delete: 482 note(s) use this type"
    },
    {
      "status": "not_found",
      "id": 9999999999999
    }
  ]
}
```
