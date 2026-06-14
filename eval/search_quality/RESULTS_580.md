# #580 cross-space floor-admission — measured results

Real MiniLM (text) + CLIP (image) 2-space corpus (`SHRIKE_SEARCH_QUALITY=1`,
`scripts/eval_cross_space_floor_admit.py`). Derived secondary image floor =
**0.246** (the `clip` space's calibrated `mean + 1·std`, #576). Single image
space throughout (the production-valid config; N≥2 image spaces is a config
error, see the memo).

## Axis 2 (modality_gap recall) + Axis 3 (over-return leak)

| mode | mg_R1 | mg_R5 | mg_Rk | mg_MRR | mg_nDCG | OR_clip |
| --- | --- | --- | --- | --- | --- | --- |
| RelativeFloor (PROD baseline) | 0.737 | 0.947 | 1.000 | 0.822 | 0.865 | 0 |
| **FloorAdmit (binary)** | **0.947** | 0.947 | 1.000 | **0.956** | **0.966** | **0** |
| FloorAdmitBudget B=1.0 (N=1) | 0.947 | 0.947 | 1.000 | 0.956 | 0.966 | 0 |
| SoftFloorAdmit (τ=0.05) | 0.947 | 0.947 | 1.000 | 0.956 | 0.966 | 0 |
| SoftFloorAdmit (τ=0.10) | 0.947 | 0.947 | 1.000 | 0.956 | 0.966 | **2** |

`OR_n` is 10 for every mode — those are `fuzzy`-only lexical noise on the ∅-gold
query (`score=None`), unrelated to cross-space fusion. `OR_clip` is the leak
metric (`image#clip` cards on the ∅-gold query); it is the discriminator.

## Axis 1a — filename-collision CORROBORATION (the win)

The 7 cards whose `<img src>` filename lexically wins the text space. `C` =
`image#clip` reached RRF provenance; `r` = the card's overall rank.

| mode | heart | skeleton | plant_cell | animal_cell | citric_acid | periodic_table | great_wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RelativeFloor (baseline) | ·r5 | ·r1 | ·r2 | ·r2 | ·r1 | ·r1 | ·r4 |
| **FloorAdmit** | **C r1** | C r1 | **C r1** | **C r1** | C r1 | C r1 | **C r1** |

Floor-admission adds the `image#clip` corroborating vote to all 7 (baseline: 0)
and lifts every rank to 1 (baseline ranks 5/1/2/2/1/1/4).

## Axis 1b — spurious-filename PRECISION (the guard)

Homonym matched pairs (one filename word, two visual senses). The image
activation signal is **space-level** (once the floor opens, the whole image
ranking enters as one `image#clip` signal), so both senses carry the provenance
whenever either does — this is the established #201b gate, **not** a #580 change.
The precision that matters is **ordering**: the on-topic sense must rank above
the off-topic one. Identical under baseline and every floor-admission mode:

| query | on-topic | off-topic | cosine sep |
| --- | --- | --- | --- |
| jaguar (spotted big cat) | id60 r1 cos 0.316 | id61 r2 cos 0.259 | 0.057 |
| crane (wading bird) | id62 r1 cos 0.329 | id63 r2 cos 0.292 | 0.037 |
| bass (freshwater fish) | id64 r1 cos 0.321 | id65 r2 cos 0.280 | 0.041 |

CLIP genuinely separates the senses; the on-topic image wins rank-1 on every
pair under every mode.

## binary vs soft — over-return leak vs τ (the separator)

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
re-opens the #576 leak monotonically with τ. The only τ where soft matches binary
is the τ→0 limit (where soft *is* binary). **Binary dominates.**
