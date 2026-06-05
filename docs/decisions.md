# Design decisions

A log of non-obvious design choices for *shipped* work — the "why", kept out of
both the code (which shows the "what") and the issue tracker (which holds *future*
work). New entries go on top of their section. For the mechanics of how a feature
works, see `CLAUDE.md`'s "Key technical details"; this file is the reasoning that
isn't reconstructable from the code.

## Semantic search & the vector index

### The index is a derived cache, never a co-equal store

The Anki collection (SQLite) is always the source of truth; the USearch index is
a rebuildable projection of it. The index may lag the collection (stale search
results) but the collection never lags the index (data ops are always correct).
This is what lets drift detection be a single `col.mod` comparison with a
background rebuild, rather than per-note reconciliation. The full rationale,
drift-detection scheme, and model-fingerprint design live under "Vector index and
consistency" in `CLAUDE.md` — not duplicated here.

### Contextual upsert returns neighbours; it makes no suggestions

`upsert_notes` returns, for each created/updated note, the *k* most similar
existing notes (`{id, score, tags}`, defaults `top_k_neighbors=5`,
`neighbor_threshold=0.5`) — the same similarity operation `search_notes` runs,
queried by the upserted note's own content, with the batch's own notes excluded
from each other's results.

It returns **raw neighbour data and stops there.** The server suggests no tags,
flags no duplicates, makes no decisions. The reasoning: the server can't know the
caller's intent, and baking in a policy (auto-tagging, dup-rejection thresholds)
would be a guess that's wrong half the time and impossible to override cleanly.
Handing back the neighbours lets an LLM-driven caller ground new cards in the
collection's existing taxonomy (reuse tags that already exist) and spot
near-duplicates — while the *policy* for what to do with that lives in the skill,
not the server.

### Semantic duplicate detection is not a separate feature

There is no dedicated *semantic* duplicate-detection endpoint, and there won't be
one. A high similarity score in `search_notes` results or in upsert neighbours *is*
the soft duplicate signal; the caller applies its own threshold. Adding a second
code path that re-implements the same cosine-similarity lookup with a built-in
cutoff would be redundant surface area with a worse interface (a hard-coded
threshold the caller can't see or tune).

### Anki's exact duplicate rule lives inside `upsert_notes`, not a `canAddNotes` tool (#77)

Distinct from the semantic signal above, Anki has a *precise* duplicate rule:
a note duplicates another if it shares the first field with an existing note of the
same type (collection-wide, deck-independent). #77 asked for a pre-flight check for
this, mirroring anki-connect's `canAddNotes`. We folded it into `upsert_notes` (an
`on_duplicate` policy defaulting to `error`, plus a `dry_run` flag) rather than
shipping a standalone checker, because a separate check is the wrong shape for this
codebase:

- **It would be racy.** A check-then-write pair has a TOCTOU gap; the collection
  can change between the two calls, so the check can lie. Folding the rule into the
  write makes creation a single atomic, race-free operation.
- **Its result is only actionable by another call.** A checker that says "this would
  be a duplicate" just sends you to `upsert_notes` anyway — extra surface area and a
  round-trip for no committed outcome.
- **It overlaps the existing per-item result union.** `error`/`skipped`/`ok`
  variants already fit `UpsertNoteResult`; a second tool would re-model the same
  thing.

The one capability a standalone checker has that an inline policy doesn't — a pure
zero-write preview — is covered by `dry_run`, which runs the identical validation
path and writes nothing. So `dry_run` with the default `on_duplicate="error"` is a
full `fields_check`-based sanity pass over a batch, without a second tool or a
second code path. Structurally invalid notes (empty first field, broken cloze) are
always errors regardless of `on_duplicate` — they're malformed, not merely
duplicated. The default is `error` (not the old silent-create behaviour) because
silently writing a duplicate or an empty note is almost always a mistake; callers
who genuinely want duplicates opt in with `allow`.

### One query, many retrieval mechanisms, annotated results (#86)

`search_notes` takes a single `queries` input and runs it through *every*
retrieval mechanism, folding the evidence into one result list rather than
exposing a separate param/tool per mechanism. Today that's semantic similarity
(the vector index) and exact case-insensitive substring matching; each match
carries a `score` when semantically ranked and a `substring` annotation (matched
fields + snippet) when the text occurs literally — both when both apply, and a
given annotation is simply absent otherwise (a returned match always has at
least one).

This was deliberately chosen over a `--substring`/`substring` parameter: a
separate flag reads as a filter and forces a union-vs-intersection decision,
whereas "the query is matched every way we can, and results tell you how" needs
no mode. It also degrades gracefully — with the embedding index down, a query
still returns its exact matches (no `score`) plus an advisory `message`. The
optional match-evidence fields are the extension point: a future n-gram/fuzzy
backend (#98) adds another annotation, never a new param or tool. Exact matches
are **not** subject to the semantic `threshold` (a literal hit is always
relevant), and within a group literal hits are listed first, then by score.

### Raw Anki query is removed, to return as its own tool

`note list --query` / `list_notes.query` (the raw Anki search escape hatch) is
gone. Text search lives in `search_notes`; deck/tag/type are structured filters.
The remaining raw-Anki power (`is:due`, `prop:ivl>=30`, `added:`, `flag:`, …) is
review/scheduling-oriented — outside Shrike's non-review scope — and will return
as an explicit `shrike collection query` tool (#97) rather than a leaky param.

### Find-and-replace edits via Anki's engine; a scope is required (#85)

`find_replace_notes` / `shrike note replace` runs the actual edit through Anki's
`col.find_and_replace` (Rust regex — linear-time, no catastrophic backtracking —
and undo-able), rather than re-implementing replacement. A **scope is required**
(`deck`/`tags`/`note_type`/`ids`) so a collection-wide edit is always an explicit
choice. The MCP tool applies by default with a `dry_run` option; the CLI previews
then confirms (`--dry-run`/`--yes`).

Two subtleties worth recording:
- **Re-embedding.** Unlike tag/deck ops, this changes field bodies, which *are*
  embedding text — so changed notes are re-embedded (the `upsert_notes` index
  path), not just `col_mod`-bumped. The changed set is found by diffing
  `notes.flds` before/after the apply; `note.mod` is only second-resolution, so a
  same-second edit wouldn't show a bump.
- **Preview faithfulness.** The dry-run preview is rendered in Python
  (`apply_replacement`): exact for literal edits, illustrative for regex (Anki's
  `$1` capture syntax vs Python's `\1`), with the apply authoritative. Acceptable
  because literal is the common case and the apply is what mutates.

## Tags

### Setting tags is a full replace; add/remove is a separate operation (#73)

When you *set* tags — `upsert_notes` partial updates (`{id, tags}`),
`shrike note update --tags`, and `update_note_tags`/`shrike note tag --set`
(`--set ""` clears) — the note ends up with *exactly* the set you sent. Replace
never silently merges.

We rejected an additive/subtractive `mode` parameter that changes the meaning of
the `tags` field on `upsert_notes` — that's the "bag of optionals / hidden state"
the `schemas.py` house style warns against, where one field's meaning depends on
another. Create-time tagging stays a full-set decision ("this card's tags are X,
Y, Z").

Additive/subtractive editing is instead its **own** tool, `update_note_tags`,
with explicit, non-overlapping fields: `set` (full replace) is mutually exclusive
with `add`/`remove`, and `add`/`remove` combine freely (e.g. add `["jp","verbs"]`
+ remove `["jp-verbs"]` swaps one tag for two). There is no default mode — the
caller picks one. So replace and merge are distinct, named operations rather than
a single overloaded field.

### Retroactive collection-wide tag curation is built (#73)

Earlier this was deliberately unbuilt "until a concrete need appears" — #73 was
that need. Bulk add/remove over existing notes (`update_note_tags`, backed by
Anki's `bulk_add`/`bulk_remove`) and collection-wide and note-scoped tag rename
(`rename_tag` / `shrike tag rename`) are available. Note-scoped rename matches the
tag *exactly* (find notes carrying it, then swap) rather than a substring
find/replace, so renaming `jp` never touches `jp-verbs`. Unused-tag cleanup
shipped briefly here too, but was pulled back out: it belongs with other
collection-maintenance chores (remove-empty-notes, etc.) under a single
`collection_prune` op rather than a one-off `tag clean` (#89).

Because tags are not part of a note's embedding text, these ops bump `col.mod`
but leave every vector valid; each advances the stored index `col_mod` without
re-embedding, so a tag-only change doesn't trigger a full rebuild on next
startup.

## Decks

### Deck deletion is empty-only; decks never merge (#74)

`delete_decks` / `shrike deck delete` refuses unless the deck *and every subdeck*
is empty. There is deliberately no "delete the cards too" or "move cards to
Default" mode. Emptying a deck is a separate, composable step — move its notes
elsewhere (`upsert_notes` with a new `deck`, `shrike note update --deck`) — and
then the deck is deletable. The payoff: **deck deletion can never delete a note**,
so it has no bearing on the collection's note set or the vector index (a deck name
isn't embedding text). A destructive cards-and-notes wipe, if ever wanted, is just
`delete_notes` on the deck's notes — no need to overload deck delete with it.

Renaming a deck onto a name another deck already uses is an **error**, not a
merge. Anki's backend would silently disambiguate (`B` → `B+`), which is
surprising and litters the tree; we reject the collision instead and tell the
caller to move notes if they meant to consolidate. So `upsert_decks` mirrors
`upsert_notes` (id present = rename the existing deck; absent = create) without
ever introducing a hidden merge.

Like tag ops, deck create/rename/delete bump `col.mod` but change no vector, so
they advance the stored index `col_mod` (shared `_bump_col_mod_after_metadata_change`
helper) without re-embedding.

### Deck references accept name, numeric ID, or `#id` (#88)

Anywhere a deck is *referenced* (not created) — `list_notes`/`note list --deck`,
`search_notes`, `upsert_notes`' `deck`, `delete_decks`, `deck rename`'s target —
the value may be a deck name, a bare numeric ID, or a `#`-prefixed ID. One rule,
applied server-side in `CollectionWrapper._resolve_deck_ref` and mirrored by the
CLI's `_match_deck` for `deck rename`:

- `#<id>` is **always** an ID; it resolves to that deck's name, or is "not found"
  if no deck has it (never silently treated as a name).
- a **bare integer** is tried as an ID first, then falls back to a literal name —
  so a deck genuinely named `123` is still reachable, while `123` meaning deck-id
  123 is the common case.
- anything else is a name.

This mirrors note IDs' `#`-prefix handling (`NoteIDType`). Resolution lives on the
server because that's where the collection is; the CLI passes references through
untouched (except `deck rename`, which must resolve to an ID client-side for the
`upsert_decks` call). On note **create**, a name that doesn't exist is still
auto-created as before; only an explicit unknown `#id` is an error.

## Note types

### Field and template updates are applied by position, preserving note data (#76)

`upsert_note_types` lets you replace a note type's whole `fields`/`templates`
list on update. Anki migrates a note's field *values* (and a template's *cards*)
by **ordinal** — the field/template at position *N* keeps its data as long as
position *N* survives. The original implementation rebuilt `flds`/`tmpls` from
fresh `new_field`/`new_template` objects, which carry no ordinal, so Anki saw
every existing field/template as removed and every incoming one as new. The
effect was catastrophic and silent: **any** update carrying a `fields` key blanked
every note's content for that type, and **any** `templates` key deleted every
card (losing all scheduling/review history) — even re-sending the *identical*
list did it.

The fix (`_set_fields` / `_set_templates` in `note_types.py`) reuses the existing
field/template dicts in place — renaming/retitling the ones whose position
survives, appending only for added positions, and dropping the tail for removed
ones. So a whole-list replace is now data-safe and matches Anki's
"keyed by position" rule: rename-in-place and edit-in-place preserve data;
lengthening appends empty fields / new cards; shortening discards only the
trailing entries (the standard meaning of removing a field/template). A genuine
*reorder* (moving a field/template to a new position while keeping its identity)
is necessarily a separate, explicit operation — a positional name swap reads as
two renames, which is non-destructive but not a move.

### Identity-based field/template ops are separate tools, not modes of `upsert_note_types` (#76)

The genuine move/insert/non-trailing-remove that position-replace can't express
lives in two tools: `update_note_type_fields` and `update_note_type_templates` —
sequences of `add`/`remove`/`rename`/`reposition` operations addressed by **name**.
They exist separately from `upsert_note_types` rather than as another shape of its
`fields`/`templates` params because the two are fundamentally different contracts —
a *declarative* "the fields/templates are now exactly this list" (position-keyed)
versus an *imperative* "move X to 0, rename Y" (identity-keyed). Conflating them in
one param would make "is `["B","A"]` a reorder or two renames?" ambiguous; keeping
them apart makes each unambiguous.

They delegate to Anki's own data-safe primitives (`rename_field` /
`reposition_field` / `add_field` / `remove_field` and the `*_template`
equivalents), which migrate note data / cards by identity — so a reposition is a
true move (the data/cards follow) and a non-trailing remove drops only that
field's data / that template's cards. (A template `rename` is a pure label change:
cards key by ordinal, not name, so no primitive and no card migration.) Each call
is **atomic**: the op sequence is validated against a simulated name list first, so
an invalid op (unknown name, clash, out-of-range position, removing the last entry)
changes nothing; only once every op is known-good are the primitives applied to one
in-memory notetype and persisted with a single `update_dict`. Like
`upsert_note_types`, they do no inline index maintenance — the `col.mod` bump
drives a drift-rebuild on next startup (correct, since a removed field changes a
note's embedding text). The two tools share `_simulate_struct_op` (the op shape is
identical) and raise `NoteTypeOpError`, which the tool layer turns into a
`ToolInputError`.

With correct movers in hand, the position vs identity tools are **reconciled** so
they don't overlap dangerously: `upsert_note_types`' positional `fields`/`templates`
replace now *refuses* any update where an existing name lands at a different
position — a reorder, an insert before another entry, or a non-trailing remove
(which shifts the names after it). Positionally those silently re-label note data /
cards (the value/cards stay in their slot while the slot's name changes), the
footgun the position-keyed contract can't avoid. The check is one shared rule
(`_reject_unsound_positional_replace`: an existing name may not change index) whose
error points at the matching identity tool. So `upsert_note_types` keeps only the
*unambiguous* positional edits — rename/edit-in-place, append, trailing-remove —
and every move/insert/non-trailing-remove goes through the identity tools. The
overlap that remains (those simple edits doable both ways) is the benign
PUT-vs-PATCH kind; the dangerous overlap is gone.

### `find_replace_note_types` rewrites template text, not fields (#76)

anki-connect's `findAndReplaceInModels` is the third note-type edit tool, and it
is deliberately a *different* operation from the find-and-replace over note
*content* (#85, `find_replace_notes`): this one searches a single note type's
card-template HTML (`qfmt`/`afmt`) and shared CSS — the model definition — and
touches no note field values. So the two find-and-replaces don't share code or a
selector vocabulary; they operate on different layers (one model's templates vs a
scoped set of notes' fields), and shipping the model one separately from the
unmerged #85 was the clean call.

It scopes to **one model per call** with `front`/`back`/`css` booleans (mirroring
anki-connect's shape) and returns a replacement count plus which templates/CSS
changed. Three decisions worth recording:

- **No data migration, so no re-embed — but a `col_mod` bump.** Templates and CSS
  aren't part of a note's embedding text, so every vector stays valid. Like the
  tag/deck metadata ops, it advances the stored `col_mod` without re-embedding
  (via `_bump_col_mod_after_metadata_change`) so the `update_dict` mod-bump
  doesn't trigger a spurious rebuild. This is the opposite of the *structural*
  field/template ops, whose removes change embedding text and correctly let a
  drift-rebuild happen.
- **`match_case` defaults to true**, unlike a note-content find-and-replace where
  prose makes case-insensitive the friendlier default. Template and CSS text is
  *code* — field names (`{{Front}}`), CSS class names, colours — where case is
  significant, so a case-sensitive default is the safe one.
- **Literal mode inserts the replacement verbatim.** Literal `search` is
  `re.escape`d and the replacement is applied through a constant function, so a
  replacement containing `\1` or `$2` is inserted as those characters, not
  interpreted as a backreference; capture refs are available only under `regex`.
  We reused Python `re` (not Anki's `find_and_replace`, which is note-scoped)
  because the substitution target is model strings we already hold in memory.

It does **not** rename a field — Anki's `rename_field` already rewrites template
references when a field is renamed via `update_note_type_fields`, so this tool is
for the cases that primitive doesn't cover (CSS edits, literal markup typos,
collapsing two field refs into one). Producing a template that references a
missing field still fails Anki's own save validation, as it should.

Remaining for #76: the field font/description metadata getters/setters.

## Collection maintenance

### One `collection_prune` tool, not scattered cleanups (#89)

Small "tidy up" chores — clear unused tags, remove empty notes, remove empty
cards — live behind one `collection_prune` tool / `shrike collection prune`
rather than a tool each. `clear_unused_tags` actually shipped standalone first
(`shrike tag clean`, #73) and was deliberately **removed** (#90) to fold in here,
and this supersedes a standalone remove-empty-notes (#78, closed wontfix). The
reasoning mirrors the "one query, many mechanisms" call for search: these are all
"maintenance passes over the whole collection," so one entry point with opt-in
flags (none selected → run all) beats N one-off verbs cluttering the surface. The
`collection` CLI group it introduces is also where `collection query` (#97) will
land.

Three decisions worth recording:

- **Preview by default — the opposite of `find_replace_notes`.** `dry_run`
  defaults **true**; the CLI previews unless `--apply`. The note find/replace
  applies by default (it's scoped to an explicit selector, and the edit is
  undoable in Anki). Prune is collection-wide *and* deletes notes and cards, with
  no per-call scope to contain a mistake — so it errs the safe way and makes you
  ask for the mutation. Same two primitives (`dry_run` / a confirm flow), opposite
  default, chosen per blast radius.

- **"Empty" is media-safe.** A note is empty only if *every* field is blank by
  `embed_text.field_is_blank` — no text **and** no media reference. This is
  stricter than the embedding normalizer (which drops media to `""`): an
  image-only or audio-only card has real content and must never be pruned. We did
  *not* use Anki's "generates no cards" definition, which would delete a note
  whose only content sits in a field no template renders — silent data loss.

- **Apply ordering: notes → cards → tags.** Empty notes are removed first, then
  empty cards, then unused tags, so a tag orphaned by the deletions is cleared in
  the same call. The dry-run previews each cleanup independently against the
  current state (and subtracts empty-note ids from the empty-cards list to avoid
  double-listing), so an apply can legitimately clear a few more tags than the
  preview showed — preview is advisory, apply is authoritative.

Index handling is **mixed**, which is the whole reason prune isn't a plain
metadata-bump op: empty-note/empty-card removal deletes notes, so their vectors
leave the index via `index.remove` exactly like `delete_notes`, while unused-tag
clearing leaves every vector valid. The tool does both in one pass and advances
`col_mod` once when anything changed.
