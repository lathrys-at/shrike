# Worked examples

Concrete patterns for the standards in `SKILL.md`, organized by card type. Use
the first section to confirm you picked the right *type*; use the rest to model
the *writing*. Aim for the "after" column.

## Quick map: which type, when

| The material is… | Use | Because |
|---|---|---|
| A discrete fact, mechanism, cause/effect, "what does X do" | One-way Q/A | One cue, one answer, one direction |
| A pair you must recall **both** ways (vocab, term↔definition) | Two-way / reversed | Both directions are genuinely used |
| A fact whose sentence carries meaning, or an ordered list/set | Cloze | Context aids recall; list items schedule separately |

When in doubt, a one-way Q/A is the safe default. Reach for the others only when
the material clearly calls for them — and only if the collection has a note type
that supports them.

---

## One-way Q/A (Basic)

The workhorse. A single cue producing a single answer.

### Splitting a bloated card (minimum information)

A card carrying several facts is the most common failure. Split it.

**Before** — four facts welded into one card:

```
Front: Describe the structure of an atom.
Back:  The nucleus holds protons and neutrons; the atomic number is the number
       of protons; the mass number is protons plus neutrons; and electrons,
       which carry negative charge, occupy shells around the nucleus.
```

You grade it wrong whenever you miss any one piece, and never learn *which*
piece is weak. Each fact also wants its own review schedule.

**After** — four atomic cards (or a cloze; see below):

```
Front: What does an atom's atomic number count?
Back:  Its number of protons.

Front: How is an atom's mass number calculated?
Back:  Protons + neutrons.

Front: Which subatomic particle carries a negative charge?
Back:  The electron.

Front: In a neutral atom, how do the proton and electron counts compare?
Back:  They are equal.
```

### Recognition → recall

**Before** — answerable by luck:

```
Front: Is the resting membrane potential of a neuron about −70 mV? (yes/no)
Back:  Yes.
```

**After** — forces retrieval of the value:

```
Front: What is the approximate resting membrane potential of a typical neuron?
Back:  About −70 mV.
```

### The cue that answers itself (restatement)

A subtler version of the same failure: the front asks for "the significance" or
"the importance" of something, and the back just rephrases the question. Nothing
is recalled — the cue already contains the shape of the answer.

**Before** — the back restates the front:

```
Front: What was the significance of the Rosetta Stone?
Back:  It was important for understanding ancient Egypt.
```

"What was the significance?" → "it was important" tests nothing: you can't get it
wrong, so you learn nothing. Flip it so the answer is a specific thing you must
produce:

**After** — a lean cue whose answer must be retrieved:

```
Front: Which artifact gave scholars the key to deciphering Egyptian hieroglyphs?
Back:  The Rosetta Stone.
```

Mind *both* ways to get this wrong. Don't let the back merely restate the front —
but don't over-correct by stuffing every identifying detail into the cue ("Which
2nd-century-BC granodiorite stele bearing the same decree in Greek, Demotic, and
hieroglyphs…"). A question that lists half the answer is just recognition wearing
a question mark; keep the cue lean and let incidental specifics live in the back.
And when the material is genuinely thin, don't manufacture detail to pad it —
test the one association it actually contains.

### Sharpening an ambiguous cue

**Before** — the front maps to dozens of answers, so you can't grade it
consistently:

```
Front: Insulin.
Back:  A peptide hormone from pancreatic beta cells that lowers blood glucose by
       promoting cellular uptake; also drives glycogen, lipid, and protein
       synthesis.
```

**After** — specific cues, each with one defensible answer:

```
Front: Which cells secrete insulin?
Back:  Pancreatic beta cells (islets of Langerhans).

Front: What is insulin's primary effect on blood glucose, and by what
       mechanism?
Back:  Lowers it — by promoting glucose uptake into cells (notably muscle and
       fat via GLUT4).
```

---

## Two-way / reversed cards

Use only when you genuinely need the association in **both** directions, and the
collection has a reversed note type (often "Basic (and reversed card)").

### When it fits

A symbol-and-name pair you must read *and* produce — you meet `Na` in a formula
and must recall "sodium", and you reach for `Na` when writing one:

```
Front (→): Na
Back  (→): Sodium

Back  (←): Sodium
Front (←): Na
```

Both directions get used, so the second template earns its daily review.

### When it doesn't (asymmetric knowledge)

**Tempting but wrong** — making this reversible:

```
Front: What enzyme does aspirin irreversibly inhibit?
Back:  Cyclooxygenase (COX-1 and COX-2).
```

The reverse ("what drug irreversibly inhibits COX?") has several valid answers
and you rarely need to retrieve in that direction anyway. Keep it **one-way**.
Doubling the cards here just adds daily review for a direction you don't use.

---

## Cloze

Use when the surrounding sentence carries meaning or cues the answer, or for an
enumeration where each item should schedule independently. Then keep to the
cloze discipline: hide the load-bearing term, not filler; one or a few deletions
per note; each answerable from the rest of the sentence.

### A fact in meaningful context

```
Text: At sea-level pressure, water boils at {{c1::100}} °C.
```

The qualifier ("at sea-level pressure") is doing real work — it's the context
that makes the boiling point a fact worth stating — so the sentence is worth
keeping whole rather than stripped to "Boiling point of water? → 100 °C".

### An enumeration as separate deletions

**Before** — one all-or-nothing card:

```
Front: Name the four nitrogenous bases of DNA.
Back:  Adenine, thymine, guanine, cytosine.
```

**After** — one cloze note, each item independently scheduled:

```
Text: The four nitrogenous bases of DNA are {{c1::adenine}}, {{c2::thymine}},
      {{c3::guanine}}, and {{c4::cytosine}}.
```

**Also wrong** — clozing the sub-*groups* instead of the members. The bases split
into two classes (purines, pyrimidines), and that grouping is a trap:

```
Text: DNA's bases are the purines {{c1::adenine and guanine}} and the
      pyrimidines {{c2::thymine and cytosine}}.
```

Only two deletions, so you're tested on "name the purines" / "name the
pyrimidines" — never the four bases individually. Notice each deletion joins two
names with "and": that's the tell you've clozed the group. One deletion per base,
as in the "after" above.

### An ordered sequence (where order is the point)

Like an enumeration, but the *sequence* itself is part of the knowledge — so keep
it in one cloze note with a deletion per step, which tests each link while the
surrounding order cues it. Don't flatten an ordered process into a single Q/A
("list the stages of X") that grades all-or-nothing and throws the ordering away.

**Before** — the whole sequence as one answer:

```
Front: What are the four stages of the water cycle, in order?
Back:  Evaporation, condensation, precipitation, collection.
```

**After** — each step scheduled on its own, the order preserved as context:

```
Text: The water cycle runs {{c1::evaporation}} → {{c2::condensation}} →
      {{c3::precipitation}} → {{c4::collection}}.
```

### Over-cloze, repaired

**Before** — so many deletions the sentence is fill-in-the-blank recognition:

```
Text: The {{c1::Sun}} is a {{c2::star}} at the centre of the {{c3::Solar System}},
      composed mainly of {{c4::hydrogen}} and {{c5::helium}}.
```

**After** — one load-bearing deletion per card, context left to cue it:

```
Text: The Sun is composed mainly of {{c1::hydrogen}}, with helium second.

Text: The star at the centre of the Solar System is the {{c1::Sun}}.
```

---

## Placement: deck vs. tag

Independent of card type. The collection already has a `Chemistry` deck and notes
tagged `organic`, `reactions`, `mechanisms`.

**Before** — a new deck tree for one chapter of substitution-reaction cards:

```
deck: Chemistry::Organic::Reactions::Nucleophilic-Substitution
tags: []
```

This fragments review into a tiny isolated deck and buries the cards three
levels down.

**After** — land in the existing deck; let tags (reusing the collection's
vocabulary) carry the topic:

```
deck: Chemistry
tags: ["organic", "reactions", "nucleophilic-substitution"]
```

---

## Using the pre-write search to catch a duplicate

You drafted:

```
Front: What is the mechanism of action of aspirin?
Back:  Irreversibly inhibits cyclooxygenase (COX-1 and COX-2), blocking
       prostaglandin and thromboxane synthesis.
```

Before writing, search the claim (step 3): `search_notes(queries=["aspirin
irreversibly inhibits cyclooxygenase, blocking prostaglandin synthesis"])`. A
match comes back at score 0.91 — note `1700000000123`:

```
Front: How does aspirin work?
Back:  Irreversibly acetylates COX-1/COX-2, reducing prostaglandin and
       thromboxane production.
```

Same fact. Resolve it before writing: drop the draft and, if the original could
be sharper, update it instead. Note the merge in your report. Also notice the
match's tag is `nsaids` — if your draft used `nsaid`, align to the existing form
before the upsert.
