# Router training

How to train the router of a `DeltaMoE` / `HierDeltaMoE` built from fine-tunes, with the expert weights frozen.
These lessons come from Qwen3.5-4B MoEs built from full fine-tunes. The bracketed numbers are entries in the
project's research log. The training script is `qwen3-vl-moe/scripts/train_hier_router.py`, which lives outside
this library.

## Objective

Only `mlp.router.weight` trains (fp32); every other weight is frozen. The loss is

$$
L = L_{lm} + \lambda\, L_{route}, \qquad \lambda = \texttt{--route-weight}\ (\text{default } 1).
$$

**$L_{lm}$, language-model loss.**

- It is the next-token cross-entropy of the whole MoE: every token of a raw-text window, or the last assistant turn
  of an SFT example.
- The router gets its gradient only through the routing weights that mix the expert outputs: the group softmax $h$
  and the within-group probabilities $p$.
- With top-1 inside a group, the renormalised weight is always 1, so the within-group rows get no gradient from
  $L_{lm}$.

**$L_{route}$, routing loss.**

- Each data source is labelled with where it should go: `--data name@g=PATH` for group $g$, `name@g.e=PATH` for
  expert $e$ of group $g$.
- `@g.e` adds the same cross-entropy over group $g$'s expert rows against $e$.
- Because of the stop-gradient, $L_{route}$ trains only the routers it scores and never reaches earlier layers.

At every layer $\ell$ the router logits are scored against the label, and the result is averaged over the layers.
For the group level, with $x^{\ell}$ the router input of layer $\ell$ and $\mathrm{sg}$ the stop-gradient:

$$
L_{route} = \frac{1}{N_L}\sum_{\ell} \mathrm{CE}\big(R_0\,\mathrm{sg}(x^{\ell}),\ g\big)
$$

## Recommended setup

- **Loss:** $L_{lm} + L_{route}$, with labels at every level the router decides (group and expert).
- **Layout:** flat, with one expert per domain; same-domain seeds are averaged into that expert first.
- **Steps:** about 1,000, oversampling the sources that are hard to tell apart.
- **Inference:** top-1.
- **Initialisation:** any. SVD and class centroids end up the same.

## Why

| Finding | Evidence |
|---|---|
| $L_{lm}$ alone collapses routing onto one expert (usually the generation expert) | flat MoE 68.0 / 78.0 vs 72.2 / 93.1 with $L_{route}$ [7] [8]; hierarchical 2–17 points below flat [56]; NLI4CT −9 to −11 when retrained for top-2 / soft [39] |
| A within-group router needs expert labels | trained on $L_{lm}$ alone its choice ignores the input; `@g.e` gives NLI4CT +3.9 [58] [59] |
| Steps matter, initialisation does not | an untrained SVD router is near random [7]; BC5CDR −6.8 at 300 steps, all eight sets within 1.5 points of their expert at 1,000 [12] |
| The inference rule hardly matters | top-1, top-2 and top-p 0.7 / 0.9 tie; soft is about 1 point lower on two sets; unseen sets differ by under 1 point [38] [43] |
| Flat beats hierarchical | best 4B layout: engineering 73.6, history 75.8, law 67.3, NLI4CT 77.8; two-level layouts were worse [60] |

## What the router cannot fix

- **Overlapping domains.** About 15% of law tokens go to the LEXam expert, which costs about 1.5 points [60].
- **Shared non-MLP weights.** NLI4CT stays about 3 points below the medical expert even under forced routing. The
  cause is that the expert's attention and norm changes are averaged into the shared weights [60].
- **Neither is fixed by training the experts together with the router.** Joint training scored below router-only
  training [24] [25].

## Qwen3.8-27B CPT experts

The 27B experts are full-parameter CPT models: P1 / P3 / P4 literature, textbook + guidelines, and drugs.

**First router run [53].**

- Layout: three department groups, each holding its literature expert plus copies of textbook and drugs.
- Training: 20 steps, $L_{lm}$ only, top-2.
- Within each group, routing already followed the data: drugs text sent 74–76% of tokens to the drugs expert.
- Across groups it did not separate departments: about 70% of every department's literature went to the P1 group.

**Delta similarity (MLP cosine, 19 models).**

- Literature experts share a large common direction. Cross-department cosine is 0.3–0.55, and P1–P4 is 0.54 whether
  or not P4 has finished training.
- Textbook and drugs are nearly orthogonal to the rest (≤ 0.07).
- Expect literature to be the hard part of the routing, and textbook and drugs to separate easily. The router should
  be trained with source labels as above.
- Some models need cleaning before merging. Models trained with MIMIC (`*_d4`, `P3_lit_mimic`) carry one or two
  outlier `linear_attn.conv1d` tensors that hold 24–53% of their delta energy. These tensors need screening or repair.
