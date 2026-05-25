# Anki MCP Server — Tool Interface

Six tools for managing an Anki flashcard collection. The server maintains a local vector index over all note content, enabling semantic search without external API calls. Duplicate detection is handled by the application and surfaced in its own UI — not by the LLM.

Notes in Anki have a **note type** that defines their fields (e.g., a "Basic" note type has "Front" and "Back" fields; a "Cloze" note type has "Text" and "Extra"). A note type also defines **card templates** — HTML templates that control how cards are rendered — and **CSS styling** shared across all its cards. A single note produces one or more cards depending on its note type. Notes belong to a **deck** and can have **tags**.

---

## `collection_info`

Return the structure of the Anki collection: available note types with their fields, deck names, tags, and summary statistics. Use this to orient yourself before creating or searching for notes — especially to discover which note types, fields, and decks exist.

Called with no arguments, returns everything (note type summaries, decks, tags, and stats). Use the `include` parameter to request only a subset.

Note type summaries include field names and type (standard/cloze) but not full template HTML or CSS. To inspect or modify templates, request full details with `note_type_details`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `include` | `string[]` | no | Subset of info to return. Any combination of `"note_types"`, `"decks"`, `"tags"`, `"stats"`. Defaults to all four. |
| `note_type_details` | `string[]` | no | List of note type names to return full definitions for, including card templates (HTML) and CSS. Omit to return only summaries. |

### Response

```jsonc
{
  "note_types": [
    {
      "name": "Basic",
      "id": 1234567890,
      "fields": ["Front", "Back"],
      "type": "standard",          // "standard" or "cloze"
      // included only when requested via note_type_details:
      "templates": [
        {
          "name": "Card 1",
          "front": "{{Front}}",
          "back": "{{FrontSide}}<hr id=answer>{{Back}}"
        }
      ],
      "css": ".card { font-family: arial; font-size: 20px; text-align: center; }"
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

Retrieve notes by structured filters — deck, tags, note type, note IDs, or modification date. Use this for precise lookups: fetching specific notes by ID, listing everything in a deck, or finding notes matching exact criteria. For conceptual or fuzzy queries ("cards about mitochondrial membrane potential"), use `search_notes` instead.

Returns note metadata and content. Use `fields: "meta"` to return only metadata, which is useful when listing large result sets for triage before reading individual notes.

Results are capped by `limit`. The response includes `total` (the full count of matching notes) so you can tell whether your query matched more than was returned. If so, narrow your filters rather than attempting to retrieve everything — large result sets aren't useful in conversation.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `ids` | `integer[]` | no | Specific note IDs to retrieve. |
| `deck` | `string` | no | Filter to notes in this deck. Use `"::"` for nested decks (e.g., `"Japanese::Vocabulary"`). Includes child decks. |
| `tags` | `string[]` | no | Filter to notes having **all** of these tags. Prefix a tag with `"-"` to exclude (e.g., `["-leech", "verb"]`). |
| `note_type` | `string` | no | Filter to notes using this note type (e.g., `"Basic"`, `"Cloze"`). |
| `modified_since` | `string` | no | ISO 8601 date or datetime. Only notes modified after this time. |
| `query` | `string` | no | Raw Anki search query for advanced filtering (e.g., `"is:due prop:ivl>=30"`). Combined with other filters via AND. See [Anki search docs](https://docs.ankiweb.net/searching.html). |
| `fields` | `string` | no | `"full"` (default) returns all field content. `"meta"` returns only note ID, note type, deck, tags, and modification time. |
| `limit` | `integer` | no | Maximum notes to return. Default `50`, max `200`. |

At least one filter (`ids`, `deck`, `tags`, `note_type`, `modified_since`, or `query`) must be provided.

### Response

```jsonc
{
  "notes": [
    {
      "id": 1700000000123,
      "note_type": "Basic",
      "deck": "Japanese::Vocabulary",
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

---

## `search_notes`

Semantic similarity search over the collection. Accepts a list of natural-language query strings, a list of note IDs (to find notes similar to existing ones), or both. Returns the top matches ranked by similarity score.

Use this for conceptual queries that keyword search can't handle: "cards about electron transport chain regulation", "anything related to this note about Japanese honorifics", or pre-creation checks ("do I already have a card covering this concept?"). Read the results and reason about overlap from the content — don't rely on the numeric scores for decision-making.

Results can be filtered by deck or tags to narrow the search space.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `queries` | `string[]` | no | Natural-language search strings. Each is embedded and matched against the collection independently. |
| `ids` | `integer[]` | no | Note IDs to use as search anchors — finds notes semantically similar to these. Source notes are automatically excluded from results. |
| `top_k` | `integer` | no | Maximum results per query or source ID. Default `10`, max `50`. |
| `deck` | `string` | no | Restrict search to notes in this deck (includes child decks). |
| `tags` | `string[]` | no | Restrict search to notes matching all of these tags. |
| `exclude_ids` | `integer[]` | no | Additional note IDs to exclude from results. |

At least one of `queries` or `ids` must be provided.

### Response

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
          "content": {
            "Front": "What are the three regulatory points of the ETC?",
            "Back": "Complex I (NADH dehydrogenase), Complex III (cytochrome bc1), and Complex IV (cytochrome c oxidase)"
          },
          "score": 0.87
        }
      ]
    }
  ]
}
```

---

## `upsert_notes`

Create or update notes in bulk. If a note object includes an `id`, the existing note is updated. If `id` is absent, a new note is created.

When creating notes, `deck`, `note_type`, and `fields` are required. When updating, only `id` and the properties being changed need to be provided — omitted properties are left unchanged.

The application independently checks new notes for semantic similarity against the collection and surfaces duplicate warnings in its own UI. This tool does not perform or control duplicate detection.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `notes` | `object[]` | **yes** | Array of note objects (1–100). See note schema below. |

#### Note object schema

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `integer` | no | Note ID. Present = update, absent = create. |
| `deck` | `string` | create | Target deck. Required for new notes. On update, moves the note to this deck. |
| `note_type` | `string` | create | Note type (e.g., `"Basic"`, `"Cloze"`). Required for new notes. Cannot be changed on update. |
| `fields` | `object` | create | Field key-value pairs (e.g., `{"Front": "...", "Back": "..."}`). On update, only specified fields are modified. |
| `tags` | `string[]` | no | Tags to set. On create, these are the note's tags. On update, **replaces** all existing tags — include existing tags you want to keep. |

### Response

```jsonc
{
  "results": [
    {
      "status": "created",
      "id": 1700000000789
    },
    {
      "status": "updated",
      "id": 1700000000123
    },
    {
      "status": "error",
      "index": 2,                  // position in the input array
      "error": "Note type 'Basicc' not found"
    }
  ]
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
| `fields` | `string[]` | create | Ordered list of field names (e.g., `["Word", "Reading", "Meaning"]`). On update, replaces the full field list. Removing fields deletes that field's data from all existing notes of this type. |
| `templates` | `object[]` | create | Card templates. Each produces one card per note (except cloze types). See template schema below. |
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
