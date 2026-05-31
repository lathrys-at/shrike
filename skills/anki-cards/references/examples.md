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
Front: Describe the citric acid cycle.
Back:  It occurs in the mitochondrial matrix, starts when acetyl-CoA combines
       with oxaloacetate to form citrate, produces 3 NADH, 1 FADH2, and 1 GTP
       per turn, and is regulated by isocitrate dehydrogenase.
```

You grade it wrong whenever you miss any one piece, and never learn *which*
piece is weak. Each fact also wants its own review schedule.

**After** — four atomic cards (or a cloze; see below):

```
Front: Where in the cell does the citric acid cycle take place?
Back:  The mitochondrial matrix.

Front: What two molecules combine to start the citric acid cycle, and what do
       they form?
Back:  Acetyl-CoA + oxaloacetate → citrate.

Front: What reduced electron carriers and nucleotide are produced per turn of
       the citric acid cycle?
Back:  3 NADH, 1 FADH2, 1 GTP.

Front: Which enzyme is the main regulatory control point of the citric acid
       cycle?
Back:  Isocitrate dehydrogenase.
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
Front: What was the significance of the Battle of Midway?
Back:  It was a turning point in the Pacific theatre.
```

"What was the significance?" → "it was significant" tests nothing: you can't get
it wrong, so you learn nothing. The fix is **not** to pad the back with facts the
source never gave (carrier losses, dates you'd be inventing) — it's to turn the
real fact the material *does* contain into a cue whose answer you must produce:

**After** — the answer is a specific thing to retrieve, not a paraphrase:

```
Front: Which 1942 naval battle was the turning point of the Pacific theatre?
Back:  The Battle of Midway.
```

Same single fact from the same source sentence, flipped so the cue is a real
question. When the material is genuinely thin ("X was a turning point" and no
more), the recall-worthy content is that event↔significance pairing — test
*that* directly; don't manufacture detail to fill a back.

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

Vocabulary is the classic case — you must both read the word and produce it:

```
Front (→): 食べる (taberu)
Back  (→): to eat

Back  (←): to eat
Front (←): 食べる (taberu)
```

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
Text: Oxidative phosphorylation occurs across the mitochondrion's
      {{c1::inner membrane}}.
```

### An enumeration as separate deletions

**Before** — one all-or-nothing card:

```
Front: Name the four chambers of the heart.
Back:  Left atrium, right atrium, left ventricle, right ventricle.
```

**After** — one cloze note, each item independently scheduled:

```
Text: The four chambers of the heart are the {{c1::left atrium}},
      {{c2::right atrium}}, {{c3::left ventricle}}, and {{c4::right ventricle}}.
```

### Over-cloze, repaired

**Before** — so many deletions the sentence is fill-in-the-blank recognition, and
`{{c1}}` ("the powerhouse") is a cliché that gives itself away:

```
Text: The {{c1::mitochondrion}} is {{c2::the powerhouse}} of the cell, producing
      {{c3::ATP}} through {{c4::oxidative phosphorylation}} across the
      {{c5::inner membrane}}.
```

**After** — one load-bearing deletion per card, context left to cue it:

```
Text: The mitochondrion produces ATP through {{c1::oxidative phosphorylation}}.

Text: Oxidative phosphorylation occurs across the mitochondrion's
      {{c1::inner membrane}}.
```

---

## Placement: deck vs. tag

Independent of card type. The collection already has a `Medicine` deck and notes
tagged `pharmacology`, `cardiology`, `antibiotics`.

**Before** — a new deck tree for one chapter of beta-blocker cards:

```
deck: Medicine::Pharmacology::Cardiovascular::Beta-Blockers
tags: []
```

This fragments review into a tiny isolated deck and buries the cards three
levels down.

**After** — land in the existing deck; let tags (reusing the collection's
vocabulary) carry the topic:

```
deck: Medicine
tags: ["pharmacology", "cardiology", "beta-blockers"]
```

---

## Using neighbors to catch a duplicate

After upsert, you just created:

```
Front: What is the mechanism of action of aspirin?
Back:  Irreversibly inhibits cyclooxygenase (COX-1 and COX-2), blocking
       prostaglandin and thromboxane synthesis.
```

The response attaches a neighbor at score 0.91:

```
{ "id": 1700000000123, "score": 0.91, "tags": ["pharmacology", "nsaids"] }
```

Read note `1700000000123`:

```
Front: How does aspirin work?
Back:  Irreversibly acetylates COX-1/COX-2, reducing prostaglandin and
       thromboxane production.
```

Same fact. Resolve it: delete the note you just created and, if the original
could be sharper, update it instead. Note the merge in your report. Also notice
the original's tag is `nsaids` — if your new card used `nsaid`, the neighbor data
just told you to align.
