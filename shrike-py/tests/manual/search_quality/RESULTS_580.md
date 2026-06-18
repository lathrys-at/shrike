# #580 cross-space floor-admission (+ #582 per-note floor) — measured results

Real MiniLM (text) + CLIP (image) 2-space corpus (`SHRIKE_SEARCH_QUALITY=1`,
`tests/manual/search_quality/cross_space_floor_admit.py`). Derived secondary image floor =
**0.246** (the `clip` space's calibrated `mean + margin·std`, margin 1.0). Single
image space throughout (the production-valid config; >1 image space is a
`ProfileError`). `FloorAdmit` is the SHIPPED production default since #580; the
other rows are eval-only (`SHRIKE_CROSS_SPACE_FUSION_MODE`) for reproducing the
historical tables. The #582 per-NOTE floor is folded into the `FloorAdmit` rows.

## Axis 2 (modality_gap recall) + Axis 3 (over-return leak)

| mode | mg_R1 | mg_R5 | mg_Rk | mg_MRR | mg_nDCG | OR_clip |
| --- | --- | --- | --- | --- | --- | --- |
| RelativeFloor (pre-#580 gate, eval) | 0.737 | 0.947 | 1.000 | 0.822 | 0.865 | 0 |
| **FloorAdmit (PROD, #580+#582)** | **0.947** | **1.000** | 1.000 | **0.965** | **0.974** | **0** |
| FloorAdmitBudget B=1.0 (N=1, eval) | 0.947 | 1.000 | 1.000 | 0.965 | 0.974 | 0 |
| SoftFloorAdmit τ=0.05 (eval) | 0.947 | 0.947 | 1.000 | 0.956 | 0.966 | 0 |
| SoftFloorAdmit τ=0.10 (eval) | 0.947 | 0.947 | 1.000 | 0.956 | 0.966 | **2** |

The production `FloorAdmit` row carries the #582 per-note floor; vs the prior
per-space gate it LIFTED mg_R5 0.947→1.000, MRR 0.956→0.965, nDCG 0.966→0.974
(dropping below-floor tail cards tightened the rankings). `OR_n` is 10 for every
mode — `fuzzy`-only lexical noise on the ∅-gold query (`score=None`), unrelated to
cross-space fusion. `OR_clip` (the leak metric) stays 0 under production.

## Axis 1a — filename-collision CORROBORATION (the win)

The 7 cards whose `<img src>` filename lexically wins the text space. `C` =
`image#clip` reached RRF provenance; `r` = the card's overall rank.

| mode | heart | skeleton | plant_cell | animal_cell | citric_acid | periodic_table | great_wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RelativeFloor (pre-#580 gate) | ·r5 | ·r1 | ·r2 | ·r2 | ·r1 | ·r1 | ·r4 |
| **FloorAdmit (PROD)** | **C r1** | C r1 | **C r1** | **C r1** | C r1 | C r1 | **C r1** |

Floor-admission adds the `image#clip` corroborating vote to all 7 (baseline: 0)
and lifts every rank to 1 (baseline ranks 5/1/2/2/1/1/4). All 7 are above the
floor, so the #582 per-note filter keeps them.

## Axis 1b — spurious-filename PRECISION (the guard)

Homonym matched pairs (one filename word, two visual senses). The #582 per-note
floor keeps a card's `image#clip` only when its own image clears the floor — but
BOTH homonym senses are above the floor (CLIP genuinely matched both above
noise), so both are kept, and that is correct. The precision that matters is
**ordering**: the on-topic sense ranks first. Identical under baseline and every
floor-admission mode:

| query | on-topic | off-topic | cosine sep |
| --- | --- | --- | --- |
| jaguar (spotted big cat) | id60 r1 cos 0.316 | id61 r2 cos 0.259 | 0.057 |
| crane (wading bird) | id62 r1 cos 0.329 | id63 r2 cos 0.292 | 0.037 |
| bass (freshwater fish) | id64 r1 cos 0.321 | id65 r2 cos 0.280 | 0.041 |

CLIP genuinely separates the senses; the on-topic image wins rank-1 on every
pair under every mode.

## #582 — the per-note floor trims the below-floor tail

On a text-answered fact query (`"what was Napoleon's final defeat"`), the old
per-space gate admitted the WHOLE image ranking — 6 cards carried `image#clip`,
including a below-floor tail (cos 0.232/0.228/0.225/0.222, all below the 0.246
floor). The per-note floor keeps only the above-floor card (the Napoleon portrait
@ 0.307, a genuine weak CLIP match) and drops the tail's `image#clip`: 6→1 cards.
The correct TEXT answer still wins rank-1 (cos 0.77); the portrait sits at rank-2
and never out-ranks the canonical answer. The tail's tiny image votes are removed
uniformly — a strict tightening, never a loosening.

## binary vs soft — over-return leak vs τ (the #580 §5 separator)

τ sweep of SoftFloorAdmit on the ∅-gold query (floor 0.246; the two leaking
cards sit at cos 0.229 / 0.221, just below the floor):

| mode | OR_clip | mg_R1 |
| --- | --- | --- |
| binary | 0 | 0.947 |
| soft τ=0.02 | 0 | 0.947 |
| soft τ=0.05 | 0 | 0.947 |
| soft τ=0.08 | 1 | 0.947 |
| soft τ=0.10 | 2 | 0.947 |
| soft τ=0.15 | 3 | 0.947 |
| soft τ=0.20 | 4 | 0.947 |

Soft buys **zero** recall over binary (corroboration is identical at every τ) and
re-opens the leak monotonically with τ. **Binary dominates** — the shipped choice.
