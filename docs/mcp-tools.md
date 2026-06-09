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
        "css": ".card { font-family: arial; font-size: 20px; text-align: center; }",
        // per-field editor metadata (font/size used when editing, + hint text)
        "fields": [
          { "name": "Front", "font": "Arial", "size": 20, "description": "" },
          { "name": "Back", "font": "Arial", "size": 20, "description": "" }
        ]
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

## `collection_query`

Find notes with a **raw [Anki search expression](https://docs.ankiweb.net/searching.html)** — the power-user escape hatch. The `query` string is passed straight to Anki's search engine, so the full language is available: `is:due`, `prop:ivl>=30`, `added:`, `rated:`, `flag:`, `nid:`/`cid:`, and boolean `OR` / `-` / parentheses.

Reach for this when you need predicates the structured tools don't expose. For conceptual or exact-text search use [`search_notes`](#search_notes); for plain deck/tag/type filters use [`list_notes`](#list_notes). It returns the **same note shape** as `list_notes`. An invalid expression is reported as an input error.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | **yes** | A raw Anki search expression (e.g. `"deck:Japanese (tag:verb OR tag:adj) -is:suspended"`). |
| `fields` | `string` | no | `"full"` (default) returns all field content; `"meta"` returns only metadata. |
| `limit` | `integer` | no | Maximum notes to return. Default `50`, max `200`. |

### Response

Same shape as `list_notes` — `notes` (capped by `limit`), `total` (full match count), and `limit`:

```jsonc
{
  "notes": [
    { "id": 1700000000123, "note_type": "Basic", "deck": "Japanese::Vocabulary",
      "tags": ["verb"], "modified": "2026-05-20T14:32:00Z",
      "content": { "Front": "食べる", "Back": "to eat" } }
  ],
  "total": 27,
  "limit": 50
}
```

---

## `search_notes`

Search the collection by **meaning and by exact text in one call**. Each query string is matched two ways — semantic similarity (the vector index) and exact, case-insensitive substring over note fields — and the results are folded together. Every match carries a `score` when it was semantically ranked and a `substring` annotation (which fields matched + a snippet) when the query text occurs literally; both when both apply.

Use it for conceptual queries keyword search can't handle ("cards about electron transport chain regulation") and for finding exact wording. Note IDs in `ids` are semantic anchors only (no literal text to match).

When the server runs a multimodal (CLIP) embedding backend, semantic matching also covers a note's **image content** — a text query like "diagram of the Krebs cycle" can surface a card whose meaning lives in its image, even if the text doesn't say so. The query is ranked against text and image separately and the rankings are fused, so an image match isn't drowned out by text matches. On a text-only backend this is simply inert (no image vectors), and the request shape is identical either way. Image matches carry their (lower) cross-modal `score`. An activation gate keeps the image modality from contributing when none of its matches are good enough for a given query, so an off-topic query won't pull in loosely-related image cards.

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

Each match is a note annotated with the evidence that produced it: `score` (semantic, `null`/omitted when only an exact hit), `substring` (`{matched_fields, snippet}`, absent when there was no literal hit), and `provenance` — the list of signals that surfaced this result, best-ranked signal first.

Each `provenance` entry is `{signal, rank}`. `signal` names the retrieval signal: `text` and `image` are the per-modality semantic rankers (so the name doubles as the matched-modality facet — `image` means the query matched the note's *image* content, on a multimodal backend), and `exact` is a literal substring hit (`fuzzy` and `tag` signals will appear here as they land). `rank` is the note's 1-based position in that signal's own ranking. `provenance` is always present and non-empty for a returned match; `score`/`substring` remain the per-signal detail (the cosine magnitude, the matched fields/snippet) and stay consistent with it (`exact` in `provenance` ⟺ `substring` is set; a semantic signal present ⟺ `score` is non-null).

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
          },
          "provenance": [                // why it surfaced, best-ranked signal first
            { "signal": "exact", "rank": 1 },
            { "signal": "text",  "rank": 2 }
          ]
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
| `fields` | `string[]` | create | Ordered list of field names (e.g., `["Word", "Reading", "Meaning"]`). On update, replaces the field list **by position**: the field at each position keeps its note data even when renamed. Only shortening the list discards the trailing fields' data; lengthening it appends empty fields. May only rename in place, append, or drop trailing fields — a move, insert, or non-trailing remove is **rejected** (it would mislabel note data); use [`update_note_type_fields`](#update_note_type_fields). |
| `templates` | `object[]` | create | Card templates. Each produces one card per note (except cloze types). On update, replaced **by position** like `fields`: existing cards (and their scheduling history) are preserved; only removing a trailing template deletes its cards. May only rename/edit in place, append, or drop trailing templates — a move, insert, or non-trailing remove is **rejected** (it would re-label cards); use [`update_note_type_templates`](#update_note_type_templates). See template schema below. |
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

## `update_note_type_fields`

Edit an existing note type's fields **by name**, preserving note data. Where `upsert_note_types` replaces the whole field list by position (so it can only rename in place, append, or drop the trailing field), this tool applies a sequence of identity-addressed operations and can truly move a field, insert one at a position, or remove a non-trailing field — all migrating note data by field identity.

Operations apply in order, so a `rename` followed by an op naming the new name is valid. The whole call is **atomic**: if any operation is invalid (unknown field, name clash, out-of-range position, or removing the last remaining field), nothing is changed.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_type` | `string` | **yes** | Name of the note type to edit. |
| `operations` | `object[]` | **yes** | Field operations to apply in order (1–50). Each is one of the variants below, discriminated on `op`. |

#### Operation variants

| `op` | Fields | Effect |
|---|---|---|
| `add` | `name`, `position?` | Add a new (empty) field. Inserted at `position` (0-based) if given, else appended. |
| `remove` | `name` | Remove the field. **Drops that field's data from every note** of this type. Can't remove the last remaining field. |
| `rename` | `name`, `new_name` | Rename the field, preserving its data. |
| `reposition` | `name`, `position` | Move the field to `position` (0-based); its data moves with it. |

### Response

```jsonc
{
  "id": 1234567890,
  "name": "Japanese Vocabulary",
  "fields": ["Reading", "Word", "Meaning", "Notes"]   // the resulting order
}
```

---

## `update_note_type_templates`

Edit an existing note type's **card templates by name**, preserving cards. The template counterpart of [`update_note_type_fields`](#update_note_type_fields): where `upsert_note_types` replaces the whole template list by position (rename/edit in place, append, or drop the trailing template only), this tool can truly move a template, insert one, or remove a non-trailing one — migrating cards (and their scheduling) by template identity.

Operations apply in order; the whole call is **atomic** (an invalid op — unknown template, name clash, out-of-range position, or removing the last remaining template — changes nothing). To change a template's front/back **HTML in place**, use `upsert_note_types` (its positional replace is data-safe for in-place edits).

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_type` | `string` | **yes** | Name of the note type to edit. |
| `operations` | `object[]` | **yes** | Template operations to apply in order (1–50). Each is one of the variants below, discriminated on `op`. |

#### Operation variants

| `op` | Fields | Effect |
|---|---|---|
| `add` | `name`, `front`, `back`, `position?` | Add a new template (generates a card per note). Inserted at `position` (0-based) if given, else appended. |
| `remove` | `name` | Remove the template. **Deletes that template's cards** (and their scheduling) from every note. Can't remove the last remaining template. |
| `rename` | `name`, `new_name` | Rename the template — a label change only; cards are untouched. |
| `reposition` | `name`, `position` | Move the template to `position` (0-based); its cards move with it. |

### Response

```jsonc
{
  "id": 1234567890,
  "name": "Japanese Vocabulary",
  "templates": ["Recall", "Recognition"]   // the resulting order
}
```

---

## `find_replace_note_types`

Find and replace text inside a **single note type's card templates and shared CSS** — the note type *definition*, not note field values. No note is touched. Use `front`/`back`/`css` to pick where to search (all on by default). Typical uses: fix a `{{OldField}}` reference across a model's templates after a field rename, swap a CSS class or colour, or correct a typo in template markup for all of a note type's cards at once.

`search` is literal text unless `regex` is set, in which case it is a Python regular expression and `replace` may use `$1`/`\1` capture references. `match_case` defaults to **true** because template and CSS text is code (field names, class names) where case is significant. The model is saved only if at least one replacement is made.

To rename a *field* itself (and migrate note data), use [`update_note_type_fields`](#update_note_type_fields); this tool only rewrites the template text that references fields.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_type` | `string` | **yes** | Name of the note type to edit. |
| `search` | `string` | **yes** | Text (or regex, if `regex`) to find. |
| `replace` | `string` | **yes** | Replacement text. Literal by default; with `regex`, `$1`/`\1` refer to capture groups. May be empty (deletes matches). |
| `front` | `boolean` | no | Search each card template's front (question) HTML. Default `true`. |
| `back` | `boolean` | no | Search each card template's back (answer) HTML. Default `true`. |
| `css` | `boolean` | no | Search the note type's shared CSS. Default `true`. |
| `regex` | `boolean` | no | Treat `search` as a Python regular expression. Default `false`. |
| `match_case` | `boolean` | no | Case-sensitive match. Default `true`. |

At least one of `front`/`back`/`css` must be enabled.

### Response

```jsonc
{
  "id": 1234567890,
  "name": "Japanese Vocabulary",
  "replacements": 7,                 // total substitutions made
  "templates_changed": ["Recall"],   // templates whose front/back changed
  "css_changed": true
}
```

---

## `update_note_type_field_metadata`

Set a note type's **per-field editor metadata**: the `font` and `size` used when editing a field in Anki, and the field `description` (hint text shown in the editor). These are cosmetics — they have no effect on note content, card rendering, or search. Read the current values from [`collection_info`](#collection_info)'s note type details (each field carries `font`/`size`/`description`).

Each update is addressed by field `name` and sets only the attributes you provide; the rest are left unchanged. At least one attribute per update. The call is atomic — an unknown field name changes nothing. To add/remove/rename/reorder fields, use [`update_note_type_fields`](#update_note_type_fields).

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_type` | `string` | **yes** | Name of the note type to edit. |
| `fields` | `object[]` | **yes** | Per-field updates (1–100), each `{ name, font?, size?, description? }` — at least one of `font`/`size`/`description` set. |

### Response

```jsonc
{
  "id": 1234567890,
  "name": "Basic",
  "fields_updated": ["Front"]
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

## `collection_prune`

Tidy up the collection with one or more cleanups: remove **unused tags** (tag-registry names no note uses any more), **empty notes** (notes whose every field is blank), **empty cards** (cards that render empty, e.g. a cloze card with no matching deletion), and **unused media** (media files no note references). Enable the cleanups you want; **if you set none of them, all run.**

This is destructive and cannot be undone through this tool, so `dry_run` defaults to **true** — by default it only **previews**, reporting what would be removed without changing anything. Pass `dry_run: false` to apply. Notes and cards are deleted outright; unused media goes to Anki's recoverable trash. To inspect media issues without pruning, use [`collection_check`](#collection_check).

An **empty note** has every field blank, where a field is blank only if it has no text **and** no media — so a card that is just an image or audio clip is never removed. On apply, empty notes are removed first, then empty cards, then unused tags, then unused media (so tags and media freed by the deletions are cleared in the same call). Because the dry-run previews each cleanup independently, an apply may clear a few more tags/media than the preview showed.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `unused_tags` | `boolean` | no | Remove tag-registry names no note uses. Default `false`. |
| `empty_notes` | `boolean` | no | Delete notes whose every field is blank (text- and media-free). Default `false`. |
| `empty_cards` | `boolean` | no | Remove cards that render empty; a note that loses its last card is deleted. Default `false`. |
| `unused_media` | `boolean` | no | Move media files no note references to Anki's trash. Default `false`. |
| `dry_run` | `boolean` | no | Preview only — report without mutating. Default `true`. |

If none of `unused_tags`/`empty_notes`/`empty_cards`/`unused_media` is set, all run.

### Response

Each cleanup reports its own section; a section is `null` (absent) when that cleanup was not requested.

```jsonc
{
  "dry_run": true,
  "unused_tags": { "removed": 3, "tags": ["old-deck", "typo-tag", "temp"] },
  "empty_notes": { "removed": [1700000000123, 1700000000456] },
  "empty_cards": { "cards_removed": 2, "notes_deleted": [1700000000789] },
  "unused_media": { "removed": 1, "files": ["orphan.png"] }
}
```

---

## `collection_check`

Report collection media-integrity issues **read-only** — the sibling of [`collection_prune`](#collection_prune). Runs Anki's media check and reports what it finds without changing anything. Use it to preview unused media before pruning, or to discover broken references.

### Parameters

None.

### Response

```jsonc
{
  "media_dir": "/path/to/collection.media",
  "unused": ["orphan.png"],            // on disk, referenced by no note (prune candidates)
  "missing": ["ghost.jpg"],            // referenced by a note, absent from the media folder
  "missing_media_notes": [1700000000123],  // note IDs with a missing reference
  "have_trash": false                  // whether Anki's media trash holds anything
}
```

---

## `store_media`

Store media files in the collection's media folder (1–10 per call) — the write path for authoring cards with images or audio. Each item provides exactly one source: base64 `data` (which **requires** a `filename` with an extension, since the bytes alone don't say what the file is), a `url` the server fetches (filename derived from the URL or its `Content-Type` if you omit it), or a server-local `path` (see below). After storing, reference the returned `filename` in a note field (`<img src="NAME">` or `[sound:NAME]`).

URL fetches are restricted to `http`/`https` and **refuse any non-globally-routable address by default** (an SSRF guard that allowlists public IPs and re-checks each redirect hop; override with the server's `--allow-private-media-fetch` flag or `SHRIKE_MEDIA_ALLOW_PRIVATE_FETCH=1`).

A **`path`** reads a file on the **server's** filesystem and stores it zero-copy (no base64). It is **off by default** and honored only when **all three** hold: the operator set one or more `--media-path-root DIR` on the server (repeatable; config `server.media_path_roots`, env `SHRIKE_MEDIA_PATH_ROOTS`, `os.pathsep`-separated); the daemon is purely-local (loopback bind, no `--allow-remote`, DNS-rebinding guard on, no added `--allowed-host`/`--allowed-origin`); and the path is contained in **one of** those roots after resolving `..`/symlinks. Otherwise the `path` item is a clean per-item error. The stored name comes from the path's basename. To store a local file against *any* server, the CLI `shrike media store PATH` reads it and sends the bytes instead.

> **Security:** within a configured root, `path` is an **arbitrary read of those files at the server user's privileges** (stored, then readable via `fetch_media`/`GET /media/<name>`). It's off unless the operator opts in with one or more narrow `--media-path-root` dirs, and gated to a purely-local daemon, so a remote/proxied caller can't reach it and the blast radius is bounded to the named subtrees — intended for single-user/local use, consistent with Shrike's unauthenticated-loopback trust model.

Anki resolves name collisions: identical content keeps the name (reported `deduped: true`), different content under the same name gets a hashed suffix — so the stored `filename` may differ from what you asked for. Per-item errors (bad base64, unfetchable/blocked URL, disallowed/missing path, oversize) are reported per item and don't sink the batch.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `items` | `object[]` | **yes** | 1–10 media items. Each: exactly one of `data` (base64 string), `url` (string), or `path` (server-local file path), plus `filename` (string; required with `data`, optional/derived otherwise). |

### Response

```jsonc
{
  "results": [
    { "status": "stored", "index": 0, "filename": "cell.png", "mime": "image/png", "size_bytes": 20481, "deduped": false },
    { "status": "error", "index": 1, "filename": "bad.png", "error": "Only base64 data is allowed" }
  ]
}
```

---

## `fetch_media`

Locate media files in the collection (1–10 per call). **It never returns the bytes** — base64 in a tool response is useless to a model (it can't render or display it) and wrecks context. Each present file comes back as `found` with a `url` (the server's `GET /media/<name>` endpoint) and a server-side `path`; a non-existent file is `missing`. **To get the actual bytes, GET the `url`** with your download/fetch tool, or read `path` if you share the server's disk. Every `found` file reports `url`, `path`, `mime`, and `size_bytes`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `filenames` | `string[]` | **yes** | 1–10 media filenames to look up. |

### Response

```jsonc
{
  "results": [
    { "status": "found", "filename": "cell.png", "url": "http://127.0.0.1:8372/media/cell.png", "path": "/…/collection.media/cell.png", "mime": "image/png", "size_bytes": 20481 },
    { "status": "missing", "filename": "nope.png" }
  ]
}
```

The `url` is the server's media endpoint — `GET /media/{filename}` streams the raw bytes with the right `Content-Type`. It's read-only and behind the same Host/Origin guard as the other custom routes. `url` is `null` only when the server didn't advertise a base URL (e.g. the library is used without a running HTTP server). The standalone client offers `ShrikeClient.read_media(name) -> bytes` for programmatic byte access (it GETs this endpoint).

---

## `list_media`

List filenames in the collection's media folder (with the folder path), optionally filtered by a glob `pattern`. Each file carries a `url` (`GET /media/<name>`) so you can fetch its bytes directly. Covers anki-connect's `getMediaFilesNames` and `getMediaDirPath`.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `pattern` | `string` | no | Glob to filter filenames (e.g. `"*.png"`, `"cell-*"`). |
| `limit` | `integer` | no | Maximum filenames to return (`count` still reflects the full total). Default `100`. |

### Response

```jsonc
{
  "media_dir": "/path/to/collection.media",
  "count": 2,
  "files": [
    { "filename": "a.png", "url": "http://127.0.0.1:8372/media/a.png", "mime": "image/png", "size_bytes": 20481 },
    { "filename": "b.ogg", "url": "http://127.0.0.1:8372/media/b.ogg", "mime": "audio/ogg", "size_bytes": 9123 }
  ]
}
```

---

## `delete_media`

Delete media files by name, moving them to Anki's media **trash** (recoverable, sync-aware). It does **not** check whether a note still references the file, so removing a referenced asset leaves a broken `<img>`/`[sound:]` — use [`collection_check`](#collection_check) to find unused media first.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `filenames` | `string[]` | **yes** | Media filenames to delete (1–1000). |

### Response

```jsonc
{ "deleted": ["old.png"], "not_found": ["never.png"] }
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

## `find_replace_notes`

Find and replace text across the fields of a scoped set of notes. A scope is **required** — at least one of `deck`, `tags`, `note_type`, or `ids` (the same filters as `list_notes`). `search` is literal unless `regex` is set (Anki's regex engine; capture references in `replace` use `$1`). `field` limits the edit to a single field; otherwise all fields are searched.

By default the tool **applies** the change; pass `dry_run: true` to preview without modifying. Either way the response reports `notes_changed` and a sample of before/after edits. Changed notes are re-embedded so semantic search stays correct, and the edit is undoable in Anki. For literal searches the dry-run preview matches the apply exactly; for regex the preview is a best-effort sample and the apply is authoritative.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `search` | `string` | **yes** | Text (or regex) to find. |
| `replace` | `string` | **yes** | Replacement text. In regex mode, capture refs use Anki's `$1`. |
| `regex` | `boolean` | no | Treat `search` as a regular expression. Default `false`. |
| `match_case` | `boolean` | no | Case-sensitive match. Default `false`. |
| `field` | `string` | no | Restrict to this single field; omit for all fields. |
| `deck` | `string` | no | Scope: a deck (name, numeric ID, or `#id`; includes child decks). |
| `tags` | `string[]` | no | Scope: notes having all of these tags. |
| `note_type` | `string` | no | Scope: notes using this note type. |
| `ids` | `integer[]` | no | Scope: these note IDs. |
| `dry_run` | `boolean` | no | Preview only — change nothing. Default `false`. |

### Response

```jsonc
{
  "notes_changed": 3,
  "dry_run": false,
  "samples": [   // capped illustrative before/after, per changed field
    { "id": 1700000000123, "field": "Front", "before": "teh cell", "after": "the cell" }
  ]
}
```

---

## `migrate_note_type`

Change a set of notes from one note type to another, moving field content per an explicit map. This is Anki's "Change Note Type": **note IDs and — for mapped card templates — review scheduling are preserved**, so it's the history-safe way to convert Basic↔Cloze, consolidate redundant note types, or adopt a richer template. To create or edit notes *without* changing type, use [`upsert_notes`](#upsert_notes) (which refuses a type change).

All `note_ids` must currently share **one** note type (a single map can't apply to mixed types). The migration is **data-affecting**: a source field not named in `field_map` is dropped and its content lost (reported in `dropped_fields`); target fields nothing maps into start empty (`new_empty_fields`). The mapping is explicit — unknown field names, or two source fields mapping to one target, are errors, not guesses. Use `dry_run` to preview the drops first.

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `note_ids` | `integer[]` | **yes** | Notes to migrate (1–1000). Must all currently share one note type. |
| `new_note_type` | `string` | **yes** | Name of the note type to migrate to (must differ from the current one). |
| `field_map` | `object` | **yes** | Map of source field name → target field name (at least one). Unmapped source fields are dropped; two sources may not map to one target. |
| `template_map` | `object` | no | Optional source template name → target template name. Omit to let Anki map templates by position. |
| `dry_run` | `boolean` | no | Preview only — report drops without changing anything. Default `false` (applies). |

### Response

```jsonc
{
  "changed": [1700000000123, 1700000000456],
  "from_note_type": "Basic",
  "to_note_type": "Cloze",
  "dropped_fields": ["Hint"],        // source fields with no mapping (content lost)
  "new_empty_fields": ["Back Extra"],// target fields nothing mapped into
  "dry_run": false
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
