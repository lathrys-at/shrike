---
name: anki-cards
description: >-
  Author durable, well-formed Anki flashcards in an existing collection through
  Shrike — using its MCP tools when connected, or the `shrike` CLI otherwise —
  grounded in the collection's own decks, note types, and tags. Use this skill
  whenever the user wants to turn study material into flashcards: lecture notes,
  a textbook chapter, a PDF, an article, or course content sitting in a Project.
  Trigger it for "make cards from this", "help me study this", "add these to my
  Anki deck", "turn this into spaced-repetition / cloze cards", or expanding and
  de-duplicating an existing deck — even when the user never says "Shrike" or
  "flashcard". It applies card-writing best practice (atomic cards, active
  recall, cloze discipline) and reuses existing decks and tags instead of
  proliferating new ones.
---

# Authoring Anki cards with Shrike

You are adding cards to a collection that already belongs to someone. The cards
must be *durable* (atomic, worth reviewing for years) and *fit the collection*
rather than growing a parallel structure beside it. A bad card is worse than no
card — the user reviews it daily, forever.

Two habits, both against an LLM's defaults:

- **Write fewer, sharper cards.** One card, one fact. Resist comprehensiveness.
- **Look before you write.** Read what's already in the collection and reuse its
  decks and tags; don't generate near-duplicates and synonym tags from scratch.

## The rule that governs everything

**Plan the whole batch before you write.** Do every bit of thinking — orient,
draft, check, place — *first*, and only then make **one** write call. Do not
create a single note until you have drafted them all and checked them against the
collection. The write is one step at the end; never reach for it early, never
repeat it per card.

## Talking to the collection

Prefer the MCP tools when connected; otherwise use the `shrike` CLI. Same
operations, same server.

| What you need | MCP tool | `shrike` CLI |
|---|---|---|
| Learn the structure (decks, note types, tags) | `collection_info` | `shrike info --decks --types --tags --json` |
| Find existing notes about a concept | `search_notes` (`queries`) | `shrike note search "<query>" --json` |
| Find notes similar to one you know | `search_notes` (`ids`) | `shrike note search --similar-to <id> --json` |
| Inspect notes by exact filter | `list_notes` | `shrike note list --deck … --json` / `shrike note show <id> --json` |
| Create or update notes | `upsert_notes` | `shrike note create --json-input --json` / `shrike note update <id> --json` |

- **Always pass `--json`.** Ids, scores, and create-time neighbors appear only in
  JSON; a call without it is a mistake.
- **Create in bulk:** one `upsert_notes` / `shrike note create --json-input` call
  for the whole batch (1–100 notes), never one per card.
- **Read [references/shrike-cli.md](references/shrike-cli.md) once** for flags and
  JSON shapes — don't probe `--help`.
- If neither interface is available, stop and tell the user what you'd need (a
  running Shrike server, or its MCP tools connected). Don't invent cards.

## The workflow

**1. Orient — one call.** Get decks, note types, and tags in a single response:

- MCP — `collection_info(include=["decks", "note_types", "tags"])`
- CLI — `shrike info --decks --types --tags --json`

That one call is all the orientation you need — don't make separate calls per
section, don't explore `--help`. You can only create notes of a note type that
**already exists** here: use the real names from this call, and reuse its
existing decks and tags.

**2. Draft — don't write yet.** Turn the material into draft cards. For each, two
decisions in order: *what kind of card* it wants ("Card type" below), then *write
it well* ("Writing standards" below). Hold the drafts. Aim for the smallest set
that captures the material — if you're drafting a tenth card on a minor aside,
stop.

**3. Check for existing coverage — one batched call.** Search the collection for
every drafted fact at once, each query phrased as the card's actual claim, not a
keyword:

- MCP — `search_notes(queries=["<claim A>", "<claim B>", …])`
- CLI — `shrike note search "<claim A>" "<claim B>" … --json`

This is the **only** duplicate check — there is no post-write net. **Read each
match's content and judge overlap yourself; don't decide from the score alone**
(a 0.7 may be a paraphrase of your card or an unrelated fact sharing words). For
any fact already covered, drop the draft or *update the existing note* instead of
adding a parallel one. **Act on the clear cases and flag the rest, in the same
turn:** create every draft that is genuinely new now, and for judgment calls (a
fact already covered in another format, a near-duplicate you're unsure about)
proceed and note them in your step-6 report — don't stall the whole batch on a
confirmation question. Coverage is about the *fact*, not the format: a fact
already on a Basic card is covered even if you meant to cloze it. Each match's
tags show the vocabulary to tag from in step 4.
*Example:* draft "What is the speed of light?" → query `light travels at 3×10⁸
m/s in vacuum`, not `light`.

**4. Place and tag.** Put each surviving draft in the **closest existing deck**,
with tags drawn from the vocabulary your search surfaced. Express the fine topic
as a *tag*, not a new sub-deck — broad decks plus tags beat deep trees (Anki
schedules per deck, so deep hierarchies fragment review). Reuse existing tag
forms; don't coin a synonym (`econ` beside `economics`). When nothing existing
fits, create a new broad deck or tag rather than force a bad match — but don't
leave the cards in `Default`: put them in the new deck and flag it in your report
so the user can rename or redirect it.

**5. Write — one call, then align tags.** Upsert the whole batch at once:

- MCP — `upsert_notes(notes=[{deck, note_type, fields, tags}, …])`
- CLI — `echo '[{…}, …]' | shrike note create --json-input --json`

(Field names per type are in [references/shrike-cli.md](references/shrike-cli.md).)
The response returns `neighbors` per note. **Use them for one thing only: tag
alignment.** If you tagged `antibiotic` and the neighbors say `antibiotics`,
match the existing form (`note update --tags`, or `note tag <ids> --set …` for
several — both fully replace the tag set, so include the tags you keep). Then
stop.

**Do not** re-check scores, run fresh searches, or read your notes back to verify
them. You already caught duplicates in step 3, and a successful upsert is
confirmation: every note saved exactly as sent, each with its id in the response.
If a note returns `neighbors_unavailable` (a transient hiccup) it still saved —
refetch with `search_notes(ids=[…])` if you want the neighbors; don't re-create
it.

**6. Report.** Briefly: what you created or updated, in which decks, with which
tags, and — called out separately — any suspected duplicate and any new deck or
tag you had to invent. Surface the judgment calls; don't bury them.

## Card type

Reason about the shape of the knowledge — don't default everything to front/back.
You can only use types that **already exist** here (step 1); if the ideal one is
absent, fall back to one-way Q/A and tell the user.

| The material is… | Use | Notes |
|---|---|---|
| A discrete fact, mechanism, cause/effect | **One-way Q/A** ("Basic") | The default: one cue → one answer. |
| A pair you need recallable **both** ways (vocab ↔ meaning) | **Reversed** ("Basic (and reversed card)") | Only if both directions are genuinely used *and* the type exists; else stay one-way or flag it. Don't reverse a one-directional fact. |
| A fact whose sentence carries meaning; a list or ordered sequence | **Cloze** | When the sentence carries meaning, or for a list/set/sequence — one deletion **per member** (not one "name them all" card). Mind the granularity rule below. |

**Cloze granularity — one deletion per member, never per group.** Count the
individual things being named; that is how many deletions you make. The trap is a
set that splits into obvious sub-groups: cloze the *members*, not the sub-groups
(a four-item set that divides in two gets `c1`–`c4`, one per item — not `c1`/`c2`
for the two halves). **Surface check before you write:** if a single `{{cN::…}}`
hides a plural or category word (*"the two …s"*, a class name) or joins items
with "and", you've clozed the group — split it. (Ordered sequence: one deletion
per step, the order left as context.)

If the user *asks* for a type, honor it — but still apply the standards below and
still run the step-3 dedup check: a fact already in another format is still a
duplicate. Create the genuinely new cards and flag the already-covered ones in
your report (offer to convert or replace them there, if useful) — don't add a
parallel, and don't stop to ask before creating the new ones.

## Writing standards

Every card, whatever its type. [references/examples.md](references/examples.md)
shows each as a before/after — read it for patterns to model.

- **One card, one fact.** A card bundling several facts grades wrong on any one
  blank and hides which part is weak. Paragraph-shaped → split it.
- **Recall, not recognition.** The front must make you *produce* the answer.
  Avoid yes/no and any cue whose answer is implied — including "what is the
  significance / role of X?" answered by a restatement ("it was important"). Flip
  it so a specific fact (name, date, place) is the answer.
- **One defensible answer.** Narrow a broad cue ("tell me about the Roman
  Empire") until the expected answer is specific.
- **Cloze discipline.** Hide the load-bearing term, not filler; each deletion
  answerable from the rest of the sentence. In a single prose sentence, one or a
  few deletions (more turns recall back into recognition); in a list or set, one
  per member (granularity rule above).
- **Light formatting.** Match the type's field conventions, don't leak the answer
  into the front, keep HTML minimal.

## Boundaries

- **Additive and conservative.** Create and refine cards. Don't reorganize the
  collection, rename decks, or mass-edit existing notes uninvited. Don't author
  new note types (fields/templates/CSS) — work within existing types; if the
  material truly needs a new one, say so and let the user decide.
- **Never delete** a note without the user's say-so — one exception: a duplicate
  **you created this session** and confirmed against the original.
- **Flag, don't fix.** If a search turns up something off — a possible duplicate,
  a wrong-looking note, a misfiled tag, an unexplained score — note it in your
  report and let the user decide. Adding good cards is the job; correcting the
  collection is not.
