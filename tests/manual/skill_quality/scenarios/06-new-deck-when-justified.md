# 06 — Create a new deck only when justified (and flag it)

**Exercises:** the complement to deck-reuse. The skill prefers existing decks,
but when material is a genuinely new subject with **no** reasonable home, it
should create a new deck *and surface that decision* for the user to veto —
rather than forcing the cards into an ill-fitting existing deck.

**Why this material:** the fixture decks are `Biology`, `History`, `Spanish`,
`Geography`. Music theory fits none of them.

## Prompt

```
Make cards from this: a major scale has seven notes; the circle of fifths
arranges the twelve keys by ascending perfect fifths; a triad is a three-note
chord built in thirds.
```

## Expected outcome

- **Recognizes no existing deck fits** — it does not jam these into `Biology` or
  any current deck just to avoid a new one.
- **Creates a new deck** with a sensible broad name (e.g. `Music Theory`) — flat,
  not a deep hierarchy.
- **Explicitly flags the new deck in its report** so the user can rename or
  redirect it ("created a new `Music Theory` deck since nothing existing fit —
  let me know if you'd prefer elsewhere").
- Card types fit the facts (mostly one-way Q/A here); sensible new tags.
- Contrast with scenario 01: there, an existing deck fit and *no* new deck should
  appear. Here, a new deck is the *correct* move — the test is that it's done
  deliberately and surfaced, not silently.

## Fixture dependencies

None of the four fixture decks covers music; this is intentionally
home-less material.
