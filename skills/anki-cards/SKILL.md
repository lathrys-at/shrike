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

You are adding cards to a collection that already belongs to someone. The whole
value of this skill is that the cards you write are *durable* — atomic, worth
reviewing for years — and that they *fit the collection they land in* rather
than growing a parallel structure beside it. A pile of mediocre, redundant cards
is worse than none: the user pays for every bad card in review time, daily,
indefinitely.

Two habits make the difference, and both run against an LLM's defaults:

- **Write fewer, sharper cards.** The instinct to be comprehensive produces
  bloated cards that bundle several facts. Resist it. One card, one fact.
- **Look before you write.** The instinct to generate from scratch produces
  decks full of near-duplicates and synonym tags. Read what's already there
  first, and reuse it.

## Talking to the collection

Shrike exposes the collection two ways. Prefer the native MCP tools when they're
connected to the session; fall back to the `shrike` CLI (via the shell)
otherwise. They are the same operations over the same server.

| What you need | MCP tool | `shrike` CLI |
|---|---|---|
| Learn the structure (decks, note types, tags) | `collection_info` | `shrike info --decks --types --tags --json` |
| Find existing notes about a concept | `search_notes` (`queries`) | `shrike note search "<query>" --json` |
| Find notes similar to one you know | `search_notes` (`ids`) | `shrike note search --similar-to <id> --json` |
| Inspect notes by exact filter | `list_notes` | `shrike note list --deck … --json / shrike note show <id> --json` |
| Create or update notes | `upsert_notes` | `shrike note create --json-input --json / shrike note update <id> --json` |

**For exact flags, options, and the `--json` response shapes of these commands,
read [references/shrike-cli.md](references/shrike-cli.md).** That's your CLI
orientation — read it once; you don't need to rediscover the interface through
`--help`.

**Every CLI call takes `--json`.** You work from the structured payload — ids,
similarity scores, and the neighbors returned on create — which the styled text
output drops; a call without `--json` is a mistake. **Create in bulk:** pipe a
JSON array to `shrike note create --json-input` — the upsert takes 1–100 notes,
so it's one call for the whole batch, never one per card.

If neither interface is available, stop and tell the user — say what you'd need
(a running Shrike server, or its MCP tools connected) rather than inventing
cards you can't place.

## The workflow

Follow this spine. Each step exists to serve one of the two habits above.

### 1. Orient yourself first

Start with **exactly one** call that returns the whole landscape — decks, note
types, and the established tag vocabulary — in a single response. Run exactly:

- MCP — `collection_info(include=["decks", "note_types", "tags"])`
- CLI — `shrike info --decks --types --tags --json`

That one call is all the orientation you need. **Do not make several `info` /
`collection_info` calls** (one for decks, then another for types, then tags…) —
the single call above already returns every section. And don't go spelunking
through `--help` to "get oriented": the operations you need are the handful in
the table above. Run the one command, then move on to drafting.

Why it matters: you can only create a note of a note type that **already
exists** in this collection — don't assume "Basic" or "Cloze" are present; use
the real names from that call. And knowing the existing decks and tags is what
stops you from adding a "Chemistry" deck next to the user's "Chemical Sciences",
or tagging `econ` when every neighbor says `economics`.

### 2. Draft the cards (don't write yet)

Turn the source material into draft cards. Two decisions, in order: for each
piece of material, first reason about **what kind of card it wants** (see
"Choosing the card type" below) — this is a genuine design choice, not a
default — then **write that card well** (the standards under "Writing the card
well"). Hold the drafts; you'll check them against the collection before
committing anything. Aim for the smallest set of cards that genuinely captures
the material; if you're writing a tenth card on a minor aside, stop.

### 3. Check for existing coverage

Before writing, search the collection for every fact you drafted — **in one
batched call**, each query phrased as the card's actual claim (its question or
the real fact it states), never a bare keyword. Run exactly:

- MCP — `search_notes(queries=["<claim of card A>", "<claim of card B>", …])`
- CLI — `shrike note search "<claim A>" "<claim B>" … --json`

This search is the **one** place a pre-existing version gets caught — there's no
post-write net — so phrase each query as the *exact intention of the card you
intend to create*, not its topic; if a duplicate ever slips through, the query
didn't carry the card's intent, and that's what to sharpen. One query per
distinct concept (or per cluster of related drafts) is enough — not one per card
when several share a topic — but fire them **together in that one call**, never
one at a time. A single word pulls back noise and misses paraphrases; the real
claim finds the note that already states it. *Example:* drafted card *"What is
the speed of light in a vacuum?"* → query `light travels at 3×10⁸ m/s in
vacuum`, **not** `light`.

**Read the returned content and judge overlap yourself.** Do not decide from the
score alone: a 0.7 can be a close paraphrase of your card *or* an unrelated fact
that merely shares vocabulary. The number narrows the candidates; your reading
decides. Each match also carries its **tags** — note the vocabulary the
neighborhood uses, because that's what you'll tag from in step 4.

If the collection already covers a fact well, drop that draft, or plan to
*improve the existing note* (update it) instead of adding a parallel one.
**Coverage is about the fact, not the card format:** a fact already carried by a
Basic card is covered even if you were about to make a cloze of it (and vice
versa). A second card on the same fact in a different format isn't new material —
it's a duplicate the user now reviews twice. The goal is coverage of the
material, not a count of new cards.

### 4. Choose a home and tags

Put each surviving draft in the **closest existing deck**, and tag it with terms
drawn from the vocabulary your searches surfaced. Express the fine-grained topic
as a *tag*, not a new sub-deck — broad decks organized by tags beat deep trees:
Anki schedules review per deck, so a thicket like
`Chemistry::Elements::Alkali-Metals::Sodium` fragments your reviews into tiny
sessions. And reuse the collection's existing tag forms rather than coining a
parallel — tags rot into synonym sets (`econ`, `economics`, `economy`) that
splinter a collection so no single tag finds everything; the neighbor and search
data already show you the established term. Only propose a new deck or tag when
nothing existing fits — and when you do, flag it explicitly in your report so the
user can veto it.

### 5. Write, then align tags with the neighbors

Upsert the surviving drafts in **one call** — the whole batch, never one per
card. Run exactly:

- MCP — `upsert_notes(notes=[{deck, note_type, fields, tags}, …])`
- CLI — `echo '[{…}, …]' | shrike note create --json-input --json`

(The note object is `deck`, `note_type`, `fields`, `tags`; for the exact field
names per note type, see [references/shrike-cli.md](references/shrike-cli.md).)

Every created/updated note comes back with `neighbors` **in the upsert
response** — the existing notes most similar to it, each with its tags. Their
job here is **tag consistency**, not a second duplicate hunt: these are the
closest notes in the collection, so their tags are the best evidence for
whether you tagged yours to match the established vocabulary. If the neighborhood
is tagged `pharmacology`, `antibiotics` and you used `antibiotic`, align to the
existing form (adjust with a quick `note update --tags` — or `note tag <ids>
--set …` to re-tag several at once — if you drifted; both fully replace the tag
set, so include the tags you want to keep).

You already guarded against duplicates in the step-3 pre-check, so this step is
**only about tags**: read the neighbors' tags, decide whether to align or add
tags to the notes you just created, and stop there. **Don't re-audit the neighbor
scores, don't run fresh searches, and don't go looking at or editing other
notes** — that score-fixated re-checking is the behavior to avoid. **And a
successful write is confirmation, not a maybe:** when the upsert returns success,
every note is in the collection exactly as you sent it, each with its id in the
response — don't `list` or `show` them back to check they saved; the success
already says so. The neighbors are evidence for one decision only: did you tag
your new notes to match the established vocabulary?

If a result says `neighbors_unavailable` (a transient index hiccup), the notes
*were* saved; the response tells you how to refetch the same data with
`search_notes(ids=[…])`. Don't re-create the notes.

### 6. Report what you did

Close with a concise account: what you created or updated, in which decks, with
which tags, and — called out separately — anything you suspect is a duplicate,
and anywhere you had to invent a new deck or tag. The collection is the user's;
surface the judgment calls instead of burying them in a wall of confirmations.

## Choosing the card type

The first decision in drafting is what *kind* of card the material wants — and it
is a decision, made by reasoning about the shape of the knowledge, not a reflex
that turns everything into front/back. Picking the wrong form is a major reason
decks become tedious: facts forced into the wrong card are awkward to recall and
grade inconsistently. Work through the material and match it:

- **A directed fact → one-way question/answer (a "Basic"-style type).** A
  discrete fact, a mechanism, a cause and effect, "what does X do" — anything
  with a single clear question and a single clear answer in one direction. This
  is the workhorse; default to it when one cue should produce one response.

- **A bidirectional association → a two-way / reversed card.** When you truly
  need the pair recallable *both* ways — a vocabulary word ↔ its meaning, a term
  ↔ a definition you must both recognize and produce — a reversed card (which
  generates front→back *and* back→front) earns the extra review. Only when both
  directions are actually useful: if you only ever need "function, given the drug
  name" and never the reverse, a two-way card just doubles your daily load for
  nothing. Reversed cards require a note type with a reverse template (often
  named "Basic (and reversed card)"). If the collection has none, stay one-way or
  flag it — don't silently drop the second direction the user needs.

- **A fact that lives in context, or an enumeration → cloze.** When the
  surrounding sentence carries meaning worth learning or genuinely cues the
  answer — a definition, idiomatic phrasing, an ordered sequence of steps, or a
  set/list — cloze preserves that context where a stripped-down Q/A would throw
  it away. For a list or sequence, give *each* item its own deletion
  (`{{c1::…}}`, `{{c2::…}}`, …) so every item is scheduled on its own, rather
  than one "name all of them" card that grades all-or-nothing. **An "item" is the
  smallest thing you want to recall by name — not the category it sits in.** When
  a set has obvious sub-groups, don't cloze the *groups*: hiding the two or three
  categories a list divides into tests those categories and never the individual
  members you set out to learn. One deletion per member you'd want quizzed by
  name. Then hold it to the cloze discipline below.

Reconcile the choice with reality. You can only use note types that already
exist in this collection — you saw them in step 1. If the ideal type isn't
present, adapt (a one-way Q/A is almost always available as a fallback) and tell
the user if the material would genuinely be better served by a type they don't
have. And if the user *asked* for a specific kind of card — "make cloze cards
for the cranial nerves" — honor it, but still apply every standard below; their
request fixes the type, not the right to be sloppy. **It doesn't override the
step-3 dedup check either:** "make cloze cards" is no licence to re-make a fact
that already exists in another format — a different format is still a duplicate
(see step 3), so flag it and offer to convert or replace the existing note rather
than adding a parallel. If the requested type truly fights the material, build
the good version and explain why.

## Writing the card well

Whatever type you chose, the card has to meet these standards. They're old,
well-worn spaced-repetition wisdom; the reasons matter more than the rules, so
they're given.

**One card, one fact (the minimum-information principle).** A card should test a
single thing you can retrieve in one go. "The four inner planets are Mercury,
Venus, Earth, and Mars" is four facts wearing one card; learned as a unit it
grades wrong if you blank on any part, and you never find out *which* part is
weak. Split it, or use a cloze with separate deletions, so each fact rides its
own scheduling curve. When a card feels like a paragraph, it's several cards.

**Demand recall, not recognition.** The front should force you to *produce* the
answer from memory. Yes/no and true/false cards test almost nothing — you'll be
right half the time by luck. "Is the Great Barrier Reef off the coast of
Australia? (yes/no)" teaches you to nod; "Off which country's coast is the Great
Barrier Reef?" makes you retrieve. Avoid cards whose answer is implied by the
question — including the quiet version of this failure: a front that asks for the
*significance*, *role*, or *importance* of something, answered by a back that
merely restates the front ("What was the significance of X?" → "It was
important"). Nothing is recalled. When the material gives you a thing and why it
mattered, don't make the why a definition prompt — flip the card so the
significance becomes the cue and the specific fact you must produce (a name, a
date, a place) is the answer.

**Make the cue unambiguous.** A good front has essentially one defensible answer
in context. "Tell me about the Roman Empire" has a hundred, so it trains
hesitation and you grade it inconsistently. If a cue could pull up many
different answers, it's too broad — narrow it until the expected answer is
specific.

**Cloze discipline (once you've chosen cloze).** Hide the load-bearing term, not
filler, and keep it to one or a few deletions per note — hiding five things in
one sentence turns recall back into recognition. Each deletion should be
answerable on its own from the rest of the sentence; don't cloze a word that
grammar alone gives away.

**Keep formatting light and consistent.** Match the field conventions of the note
type you're filling. Don't leak the answer into the front (a giveaway in the
phrasing, a tell in the formatting). Keep HTML minimal — content over styling.

For worked examples organized by card type — when each type fits, and before/
after fixes (bloated cards split into atomic ones, an asymmetric pair that
shouldn't be reversed, an enumeration done as separate cloze deletions, an
over-clozed sentence repaired, ambiguous cues sharpened) — read
[references/examples.md](references/examples.md) when you want concrete patterns
to model.

## Boundaries

This skill is **additive and conservative**. It creates and refines cards; it
does not reorganize the user's collection, rename decks, or mass-edit existing
notes uninvited. Authoring new *note types* (custom fields, templates, CSS) is
out of scope here — work within the types the collection already has, and if the
material truly needs a new type, say so and let the user decide. Never delete a
note without the user's say-so, with one narrow exception: a duplicate **you
created moments ago in this same session** and have confirmed against the
original — clean up your own mess, but nothing pre-existing.

**When something looks off, raise it — don't act on it silently, and don't bury
it.** Searching the collection will sometimes turn up more than you went looking
for: a possible duplicate you're unsure about, an existing note that looks wrong
or contradicts your source material, a near-identical pair, a tag or deck that
seems misfiled, an oddly low or high similarity score you can't explain. None of
that is yours to quietly fix or quietly ignore. Note it and surface it to the
user — briefly, in your report — and let them decide what to do. Your job is to
add good cards; flagging the discrepancies you happen to notice along the way is
part of that, chasing them down and "correcting" the collection is not.
