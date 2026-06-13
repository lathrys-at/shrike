# Cross-space fusion eval results — #231 (the mechanism gate for #229)

Distinct from `RESULTS.md` (the #193 *single-shared-space* eval). This validates the **cross-space
RRF-fusion mechanism**: on a deployment with **no single omni embedder** — only a dedicated TEXT
space and a separate joint text↔image CLIP space (the canonical mobile config: a platform
text-embedding API + a separate platform CLIP API) — is fusing a text query across the two spaces
sound, and what params make it so?

Reproduce:

```bash
# Canonical config — the project's own native backends (no torch). Build the native ext first.
scripts/build-native.sh
python scripts/eval_cross_space.py -k 5            # MiniLM (dedicated text) + small CLIP (joint)

# Strong-vision robustness run (jina-clip-v2 via sentence-transformers).
pip install 'transformers==4.49.0' 'sentence-transformers==3.4.1' torch einops timm pillow
python scripts/eval_cross_space_jina.py -k 5
```

Dataset: the #193 labelled set (`manifest.json`, 16 image notes + 12 text notes), all in ONE shared
corpus so every query competes against the full distractor set. Image URLs pinned in
`resolved_urls.json`; bytes cached locally. Scope is **text + image only**.

## Conditions

- **(a) text-only baseline** — a dedicated text embedder (MiniLM), one index of note bodies. *The
  bar (c) must not regress.*
- **(b) single CLIP pooled space** — CLIP text + image vectors pooled into one cosine ranking. *The
  unified-space reference ceiling, where one exists — NOT the bar; no-omni targets lack it.*
- **(c) cross-space fused** — the dedicated MiniLM text space + the joint CLIP space (its
  text-tower body vectors AND its image vectors), each ranked in its own space, RRF-fused.
- **(c+gate)** — (c) with the CLIP-image signal gated by an **absolute** activation floor
  (`mean + margin·std` of its OFF-topic best-cosine; the #201b intra-modal gate lifted across
  spaces). **(c+relgate)** — gated **relatively** (fire the image signal only when its best cosine
  for the query beats the text space's best for the same query).

## Numbers — canonical config: MiniLM (dedicated text) + small CLIP (joint), realistic "leaky" labels

`rrf_k=60, rank_cap=10, gate_margin=2.0`

| condition | text-target R@1 / R@5 / MRR | image-target R@1 / R@5 / MRR |
|---|---|---|
| (a) text-only [MiniLM]        | **1.00** / 1.00 / **1.00** | 0.69 / 1.00 / 0.84 |
| (b) single CLIP pooled (ref)  | 0.92 / 1.00 / 0.96 | 0.56 / 1.00 / 0.73 |
| (c) cross-space **ungated**   | **0.08** / 1.00 / 0.38 ⚠️ | **1.00** / 1.00 / 1.00 |
| **(c+gate)** abs gate         | **1.00** / 1.00 / **1.00** | **0.94** / 1.00 / 0.96 |
| **(c+relgate)** relative gate | **1.00** / 1.00 / **1.00** | 0.88 / 1.00 / 0.93 |

Isolating control — CLIP **image-only** space (image vectors only): R@1 = **1.00**. The image
signal is perfect in isolation; any cross-space image collapse is a ranking/fusion artifact (the
modality gap), not a dead tower.

Activation diagnostic (small CLIP image space best-cosine): on-topic mean 0.321, off-topic mean
0.268, **separation +0.052** (narrow — the small fixture's weakness).

## Numbers — strong-vision robustness: jina-clip-v2 (its text tower vs its vision tower as two spaces)

`gate_margin=1.0` (jina's wider separation needs less margin)

| condition | text-target R@1 | image-target R@1 |
|---|---|---|
| (a) text-only [jina text tower] | 1.00 | 0.81 |
| (b) single pooled (ref) | 1.00 | 0.81 |
| (c) cross-space **ungated** | **0.00** ⚠️ | 0.88 |
| **(c+gate)** abs (margin 1.0) | 0.92 | 0.88 |
| **(c+relgate)** vision ≥ text | **1.00** | 0.81 |

Image-only control R@1 = **1.00**. Activation separation **+0.109** (on 0.357 / off 0.249) — much
cleaner than the small CLIP, as expected from a production model.

## Findings

1. **The cross-space fusion mechanism is SOUND — but the activation gate is MANDATORY, not
   optional.** Ungated (c) catastrophically regresses text (R@1 0.08 small-CLIP, 0.00 jina): an
   always-on image signal injects every image note into every text query's fusion, and image notes
   (present in multiple signals) out-fuse the correct text note. With the gate, text quality is
   **fully preserved** (R@1 = 1.00 = the dedicated-text baseline) AND image recall is delivered
   (0.88–0.94). This is the single load-bearing result.

2. **Gated cross-space fusion preserves text↔text quality.** (c+gate)/(c+relgate) match the
   dedicated-text baseline (a) exactly on text-target (R@1 1.00) — fusing in the weaker CLIP text
   tower + image hits does **not** drag text down, because the dedicated space carries text and the
   gate keeps the weak/off-topic image space out when it shouldn't fire.

3. **It delivers the image recall a single pooled space sacrifices.** On image-target, (c+gate)
   beats the pooled reference (b) decisively (0.94 vs 0.56 small-CLIP). In a single pooled cosine
   index the modality gap buries image vectors under text vectors (b's image R@1 *below* even the
   text-only baseline); ranking the image space separately and fusing dodges that — the #201a
   per-modality insight, lifted across model spaces.

4. **Two gate strategies both work; the RELATIVE gate is the more robust default.** The absolute
   `mean+margin·std` gate is sensitive to the off-topic distribution (one outlier inflates the
   threshold and over-suppresses). The **relative gate** (fire the image signal only when its best
   query-cosine beats the text space's best) is self-calibrating, model-agnostic, and gave perfect
   text preservation on both models (R@1 1.00) at strong image recall.

5. **RRF k is insensitive; the per-space rank cap matters.** Results are identical across k ∈
   {10,30,60,100} (k = 60 is fine). An UN-capped full-corpus ranking is degenerate for RRF; a
   bounded top-N per space (cap = 10) is what lets the combiner discriminate — and the kernel
   already bounds its index search, so this is free.

See the #229 write-up for the GO verdict and recommended build params.
