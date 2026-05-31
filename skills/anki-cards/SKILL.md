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
| Learn the structure (decks, note types, tags) | `collection_info` | `shrike info --decks --types --tags` |
| Find existing notes about a concept | `search_notes` (`queries`) | `shrike note search "<query>"` |
| Find notes similar to one you know | `search_notes` (`ids`) | `shrike note search --similar-to <id>` |
| Inspect notes by exact filter | `list_notes` | `shrike note list --deck … / shrike note show <id>` |
| Create or update notes | `upsert_notes` | `shrike note create … / shrike note update <id>` |

**For exact flags, options, and the `--json` response shapes of these commands,
read [references/shrike-cli.md](references/shrike-cli.md).** That's your CLI
orientation — read it once; you don't need to rediscover the interface through
`--help`.

CLI tips: pass `--json` so you get the structured payload — you need the
similarity **scores** and the **neighbors** that come back on create, and the
styled output drops them. Bulk-create by piping a JSON array to
`shrike note create --json-input`. Upsert takes 1–100 notes at a time; batch
them rather than firing one call per card.

If neither interface is available, stop and tell the user — say what you'd need
(a running Shrike server, or its MCP tools connected) rather than inventing
cards you can't place.

## The workflow

Follow this spine. Each step exists to serve one of the two habits above.

### 1. Orient yourself first

Start with **exactly one** call that returns the whole landscape — decks, note
types, and the established tag vocabulary — in a single response:

- MCP: `collection_info(include=["decks", "note_types", "tags"])`
- CLI: `shrike info --decks --types --tags`

That one call is all the orientation you need. **Do not make several `info` /
`collection_info` calls** (one for decks, then another for types, then tags…) —
the single call above already returns every section. And don't go spelunking
through `--help` to "get oriented": the operations you need are the handful in
the table above. Run the one command, then move on to drafting.

Why it matters: you can only create a note of a note type that **already
exists** in this collection — don't assume "Basic" or "Cloze" are present; use
the real names from that call. And knowing the existing decks and tags is what
stops you from adding a "Biology" deck next to the user's "Biological Sciences",
or tagging `cardio` when every neighbor says `cardiology`.

### 2. Draft the cards (don't write yet)

Turn the source material into draft cards. Two decisions, in order: for each
piece of material, first reason about **what kind of card it wants** (see
"Choosing the card type" below) — this is a genuine design choice, not a
default — then **write that card well** (the standards under "Writing the card
well"). Hold the drafts; you'll check them against the collection before
committing anything. Aim for the smallest set of cards that genuinely captures
the material; if you're writing a tenth card on a minor aside, stop.

### 3. Check for existing coverage

Now — before writing — search the collection for what you're about to add,
using the cards you just drafted as the source of your queries
(`search_notes`, or `shrike note search`). **Plan the queries from the drafts
first:** each query is a *specific, content-bearing phrase* — the card's own
question or the actual claim it makes — not a bare keyword. A single word pulls
back noise and misses paraphrases; the real claim finds the note that already
states it.

**Example.** Drafted card: *"What is the primary function of the
mitochondria?"* → search `function of the mitochondria` or `mitochondria
produce ATP by aerobic respiration` — **not** `mitochondria`. One good query per
distinct concept (or per cluster of related drafts) is enough; you don't need
one per card when several share a topic.

**Send the queries as one call, not one per fact.** `search_notes` takes a
`queries` array, and `shrike note search` takes several query strings as
positional arguments —
`shrike note search "claim of card A" "claim of card B" --json` returns a
`results[]` group per query in a single round-trip. Fire them together; serial
one-query-at-a-time calls are just slower, not more thorough.

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
as a *tag*, not as a new sub-deck (see "Prefer existing structure" below). Only
propose a new deck or a new tag when nothing existing fits — and when you do,
flag it explicitly in your report so the user can veto it.

### 5. Write, then align tags with the neighbors

Upsert the batch. Every created/updated note comes back with `neighbors` **in
the upsert response** — the existing notes most similar to it, each with its
tags. Their job here is **tag consistency**, not a second duplicate hunt: these
are the closest notes in the collection, so their tags are the best evidence for
whether you tagged yours to match the established vocabulary. If the neighborhood
is tagged `pharmacology`, `antibiotics` and you used `antibiotic`, align to the
existing form (adjust with a quick `note update` on the tags if you drifted).

You already guarded against duplicates in the step-3 pre-check, so **don't
re-audit the neighbor scores** — that's the redundant, score-fixated behavior to
avoid, and you don't run fresh searches afterward either. **And a successful
write is a guarantee, not a maybe** — when the upsert returns success, every note
is already in the collection exactly as you sent it: same fields, same deck, same
tags, each with its id handed back in the response. There is nothing to verify.
Don't `list` or `show` the notes back to confirm they saved or that the fields
took — the success *is* the confirmation, and re-fetching what you just wrote
tells you nothing it didn't already promise. (The neighbors are keyed on the
note's full content, so on the rare chance a near-identical one surfaces that
your draft-phrased query missed, resolve it — delete the one you just made,
improve the original — but that's a backstop, not the point of this step.)

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
  than one "name all of them" card that grades all-or-nothing. Then hold it to
  the cloze discipline below.

Reconcile the choice with reality. You can only use note types that already
exist in this collection — you saw them in step 1. If the ideal type isn't
present, adapt (a one-way Q/A is almost always available as a fallback) and tell
the user if the material would genuinely be better served by a type they don't
have. And if the user *asked* for a specific kind of card — "make cloze cards
for the cranial nerves" — honor it for the cards you do create, but still apply
every standard below; their request fixes the type, it doesn't license sloppy
cards. **It also doesn't override the step-3 dedup check:** "make cloze cards" is
not licence to re-make a fact that already exists as a Basic card (or any other
form) — a different format is still a duplicate (see step 3). When a
requested-format card lands on a fact the collection already covers, treat it as
covered: flag it, and offer to convert or replace the existing note rather than
adding a format-parallel beside it. If the requested type truly fights the
material, build the good version and explain why.

## Writing the card well

Whatever type you chose, the card has to meet these standards. They're old,
well-worn spaced-repetition wisdom; the reasons matter more than the rules, so
they're given.

**One card, one fact (the minimum-information principle).** A card should test a
single thing you can retrieve in one go. "The heart has four chambers — left and
right atria and ventricles" is four facts wearing one card; learned as a unit it
grades wrong if you blank on any part, and you never find out *which* part is
weak. Split it, or use a cloze with separate deletions, so each fact rides its
own scheduling curve. When a card feels like a paragraph, it's several cards.

**Demand recall, not recognition.** The front should force you to *produce* the
answer from memory. Yes/no and true/false cards test almost nothing — you'll be
right half the time by luck. "Is ATP synthase a membrane protein?" teaches you
to nod; "Where in the cell is ATP synthase located, and what does it do there?"
makes you retrieve. Avoid cards whose answer is implied by the question.

**Make the cue unambiguous.** A good front has essentially one defensible answer
in context. "Tell me about the mitochondria" has a hundred, so it trains
hesitation and you grade it inconsistently. If a cue could pull up many
different answers, it's too broad — narrow it until the expected answer is
specific.

**Cloze discipline (once you've chosen cloze).** Hide the load-bearing term, not
filler, and keep it to one or a few deletions per note — hiding five things in
one sentence turns recall back into recognition. Each deletion should be
answerable on its own from the rest of the sentence; don't cloze a word that
grammar alone gives away.

**Prefer existing structure over inventing new.** Before adding a deck or a tag,
use what's there. Broad decks organized by tags beat deep deck trees: Anki
schedules and mixes review per deck, so a thicket like
`Biology::Cell::Organelles::Mitochondria` fragments your reviews into tiny
sessions and is tedious to maintain. Put the card in the nearest existing deck
and let a *tag* carry the finer topic. Create a new deck only for a genuinely new
subject area with no home in the collection.

**Reuse the collection's vocabulary.** Tags rot into synonym sets — `cardio`,
`cardiology`, `heart` — that splinter a collection so no single tag finds
everything. The neighbor and search data show you what similar notes are already
tagged; adopt those terms instead of coining a parallel one.

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
