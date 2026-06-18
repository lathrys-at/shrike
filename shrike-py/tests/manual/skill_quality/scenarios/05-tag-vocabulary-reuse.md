# 05 — Tag-vocabulary reuse (no synonyms)

**Exercises:** the skill should adopt the collection's **existing** tag
vocabulary rather than coining parallel synonyms — the thing that otherwise rots
a collection into `cardio`/`cardiology`/`heart` splinters. It should look at what
nearby notes are tagged (search / upsert neighbors) and match.

**Why this material:** these WWII events aren't in the fixture (so nothing is a
duplicate), but the `History` deck already establishes the tag `world-war-2`.
The test is whether new WWII cards adopt `world-war-2` instead of inventing
`wwii` / `ww2` / `world-war-ii`.

## Prompt

```
Make cards from these WW2 facts: the Battle of Midway (June 1942) was a turning
point in the Pacific theatre; the Yalta Conference (February 1945) shaped the
postwar order in Europe.
```

## Expected outcome

- **Tags reuse the existing vocabulary:** new cards are tagged **`world-war-2`**
  (and `history`), matching the established tag — *not* a new synonym like
  `wwii`, `ww2`, or `world-war-ii`.
- Confirm with `shrike info --tags` afterward: the tag list should **not** have
  gained a WWII synonym alongside `world-war-2`.
- **Placement:** `History` deck; one-way Q/A is fine for these discrete facts.
- **Report** ideally notes that it aligned tags with existing ones.

## Fixture dependencies

`History` deck has notes tagged `world-war-2` (D-Day, 1939 invasion, 1945 end).
Midway and Yalta are not in the fixture.
