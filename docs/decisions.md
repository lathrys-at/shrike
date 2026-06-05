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

### Identity-based field ops are a second tool, not a mode of `upsert_note_types` (#76)

The genuine move/insert/non-trailing-remove that position-replace can't express
is the `update_note_type_fields` tool: a sequence of `add`/`remove`/`rename`/
`reposition` operations addressed by field **name**. It exists separately from
`upsert_note_types` rather than as another shape of its `fields` param because the
two are fundamentally different contracts — a *declarative* "the fields are now
exactly this list" (position-keyed) versus an *imperative* "move X to 0, rename Y"
(identity-keyed). Conflating them in one param would make "is `["B","A"]` a reorder
or two renames?" ambiguous; keeping them apart makes each unambiguous.

It delegates to Anki's own data-safe primitives (`rename_field`,
`reposition_field`, `add_field`, `remove_field`), which migrate note data by
field identity, so a reposition is a true move (data follows the field) and a
non-trailing remove drops only that field. The call is **atomic**: the whole op
sequence is validated against a simulated field-name list first, so an invalid op
(unknown field, name clash, out-of-range position, removing the last field)
changes nothing; only once every op is known-good are the primitives applied to
one in-memory notetype and persisted with a single `update_dict`. Like
`upsert_note_types`, it does no inline index maintenance — the `col.mod` bump
drives a drift-rebuild on next startup (correct, since a removed field changes a
note's embedding text).

With a correct mover in hand, the two tools are **reconciled** so they don't
overlap dangerously: `upsert_note_types`' positional `fields` replace now
*refuses* any update where an existing field name lands at a different position —
a reorder, an insert before another field, or a non-trailing remove (which shifts
the names after it). Positionally those silently re-label note data (the value
stays in its slot while the slot's name changes), and that's exactly the footgun
the position-keyed contract can't avoid. The check is one rule
(`_reject_unsound_field_replace`: an existing name may not change index) and its
error points at `update_note_type_fields`. So `upsert_note_types` keeps only the
*unambiguous* positional edits — rename-in-place, append, trailing-remove — and
every move/insert/middle-remove goes through the identity tool. The overlap that
remains (rename/append/trailing-remove doable both ways) is the benign PUT-vs-PATCH
kind; the dangerous overlap is gone. Remaining for #76: `findAndReplaceInModels`
(overlaps the separate find-and-replace issue #85) and the field font/description
metadata getters/setters.

The same positional footgun still exists for **templates** (a reorder re-labels
cards), but there's no template-ops tool to redirect to yet, so it's left
unguarded for now — a follow-up when identity-based template ops land.
