# Router training: what worked

These are lessons from building `DeltaMoE` and `HierDeltaMoE` out of full fine-tunes of Qwen3.5-4B. There were two
expert sets: in-domain classification and QA experts, and five domain experts (engineering, history, law, medical,
generation). A first run on Qwen3.8-27B continued-pretraining (CPT) experts is included as well. Experiment ids in
brackets refer to the research log (`tasks/inprogress/moe-indomain-acc/log.yaml` in the project repo).

The training script lives outside this library, in
`qwen3-vl-moe/scripts/train_hier_router.py`.

- It freezes every weight except `mlp.router.weight`, which it keeps in fp32.
- Each step takes one row per data source.
- `--data name@g=PATH` adds a group-level route loss.
- `--data name@g.e=PATH` also adds an expert-level route loss inside group `g`.

## Recipe

| Choice | Use | Why |
|---|---|---|
| Loss | LM loss + route cross-entropy, `--route-weight 1` | LM loss alone collapses routing (below) |
| Labels | the data source of each row, at the level the router decides; for `HierDeltaMoE` the expert inside the group too (`@g.e`) | a within-group router trained on LM loss alone ignores its input |
| Layout | flat: one group, one expert per domain; average same-domain seeds into one expert first | two-level grouping was worse |
| Inference `top_k` | 1 | top-2, top-p 0.7 / 0.9 and soft routing were all within 1 point; soft routing was about 1 point lower on two sets |
| Steps | about 1,000; oversample sources that are hard to separate | 300 steps left one set 6.8 points down |
| Init | anything (SVD or centroid) | it does not matter once the router trains |

The route loss is the cross-entropy of each layer's router logits against the source's label, averaged over layers.
The router inputs are detached. With top-1 inside a group, the renormalised weight is 1, so the expert rows get no
gradient from the LM loss at all. The route loss is then their only training signal.

## Evidence

**LM loss alone collapses routing.** Every time, the router sent most tokens to one expert, usually the
generation expert.

- Flat MoE: 68.0 / 78.0 on the first two sets, against 72.2 / 93.1 with the route loss [7] [8].
- Two-level MoE: the group router collapsed onto the generation group, 2-17 points below the flat `DeltaMoE` [56].
- Retraining for top-2 or soft routing: NLI4CT 9-11 points down [39].

With the route loss and enough steps, the MoE comes within about 1 point of each dense expert [9] [12] [56].

**The within-group router needs expert labels.** Trained on LM loss alone, its choice did not depend on the input.
With `@g.e` labels it separated the experts, NLI4CT +3.9 [58] [59].

**Initialisation is irrelevant, steps are not.**

- An SVD-initialised router without training routes close to random [7].
- 300 steps left BC5CDR at −6.8. At 1,000 steps all eight sets were within 1.5 points of their expert [12].
- ChemProt sampled ×5 for 1,000 steps routed 94-99% correctly on three seeds.

**The routing rule at inference hardly matters.**

- Top-1, top-2 and top-p tie [38] [43].
- On five unseen sets (MedMCQA, PubMedQA, LEXam, CS-Bench, FinTrain CFA) no routing rule differs by more than 1 point.
- The experts are close to one another. Cross-domain delta cosine is 0.44-0.51, the same as same-domain seeds
  (about 0.51) [44]. Any off-domain expert already gives most of the general gain.

**Flat beats grouped.** The best 4B layout [60] is one expert per domain in one flat group, with expert labels and
top-1. It scores engineering 73.6, history 75.8, law 67.3 and NLI4CT 77.8. The two-level layouts were worse. Two
losses remain that the router cannot fix:

- **Law misrouting** (about −1.5). About 15% of law tokens go to the LEXam (lex) expert.
- **Shared non-MLP weights** (NLI4CT about −3, also under forced routing). The medical expert's attention and norm
  changes are averaged away when the non-MLP weights are shared. This is not a routing error.

**Training the experts together with the router does not recover that loss.** Joint router + expert training was
below router-only training [24] [25].

## Qwen3.8-27B CPT experts (first run)

The CPT experts are P1 / P3 / P4 literature, textbook + guidelines, and drugs. They were built as three department
groups with nine experts, in which each group held its own literature expert plus a copy of textbook and drugs. The
router was trained for 20 steps on LM loss only, with top-2 inside each group [53].

- Both routers got gradients.
- Inside each group, routing already followed the data: drugs text sent 74-76% to the drugs expert, textbook text
  sent 64-76% to the textbook expert.
- The group router did not separate departments. About 70% of every department's literature went to the P1 group:
  the groups differ only in their literature expert, and P1 changed the most.

The 4B results above say to train this with route labels (the source of each CPT corpus), a flat layout and top-1.
Two caveats for 27B:

- **Literature experts are close.** Their whole-model delta cosine is 0.28-0.54 (the MLP-only cosine in [53] is up to 0.70), against 0.44-0.51 between the 4B domains.
  Expect literature routing to be the hard part.
- **Textbook and drugs are easy.** They are nearly orthogonal to the rest (cosine ≤ 0.07).
