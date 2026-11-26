# Similarity of fine-tunes from one base

Before several dense fine-tunes of one base are upscaled into a [DeltaMoE](deltamoe.md) or a
[HierDeltaMoE](hier.md), three questions decide the layout: which models belong in one group, how the shared
(non-MLP) parameters should be merged, and which experts carry nothing of their own and can be merged or dropped.
`similarity.py` answers them from the weights (a Gram matrix of the deltas, cheap, no data) and from the loss (dense
models built from delta combinations, evaluated on each domain's validation set).

## Usage

```bash
python -m shared_delta_moe.similarity gram --base Qwen3.5-4B --ft eng=ckpt/eng law=ckpt/law med=ckpt/med \
    --out gram.$k.pt --shard $k/4                      # one GPU per shard; a 4B model with 30 fine-tunes: minutes
python -m shared_delta_moe.similarity report gram.?.pt \
    --fit med=eng,law --set sa=eng,law,med --pair eng,law --json report.json
python -m shared_delta_moe.similarity loss --base Qwen3.5-4B --ft eng=ckpt/eng law=ckpt/law \
    --data eng=val/eng.jsonl law=val/law.jsonl \
    --spec base eng law avg=eng*0.5+law*0.5 "route_eng=mlp:eng;rest:eng*0.5+law*0.5" \
    --score last --out loss.json
```

`gram` reads every language-model tensor once (vision tower and MTP head skipped). `report` prints Markdown tables.
In a `loss` spec, `A*0.5+B*0.5` is a delta combination applied to every tensor, `mlp:...;rest:...` gives the MLP and
the other tensors their own; `base` is the base itself. `--score` picks what a chat row is scored on: `reply` (the
whole last turn, template tokens included), `content` (only the reply's text) or `last` (its last token: the option
letter of an answer-only multiple-choice reply). Raw-text rows are scored on every token.

## What is measured

For each tensor, with base weight $W$ and deltas $\Delta_i = \theta_i - W$, `gram` stores the Gram matrix of
$[W, \Delta_1, \dots, \Delta_n]$ (fp64), so every quantity below is a sum over tensors and needs no second pass:

- relative size $\|\Delta_i\| / \|W\|$, and $\cos(\Delta_i, W)$ (weight decay shows as a small negative value);
- cosine $\cos(\Delta_i, \Delta_j)$, per group (MLP, attention, linear attention, embeddings, norms), per tensor kind
  and per layer;
- fit: the least-squares $\beta$ of $\Delta_t \approx \sum_j \beta_j \Delta_{s_j}$ and the energy it leaves
  unexplained, $\|\Delta_t - \sum_j \beta_j \Delta_{s_j}\|^2 / \|\Delta_t\|^2$;
- shared part of a set: the top eigenvalue of its cosine matrix over its size, and each member's $\cos^2$ with the
  normalised mean direction;
- merge coefficients matching a set's sizes (below);
- per-channel overlap of the MLP: each intermediate channel's delta energy (its gate and up rows, its down column),
  as a correlation and as the overlap of the top 10 % channels (chance: 10 %);
- bf16 granularity: the fraction of entries a fine-tune left unchanged and of those that moved by exactly one bf16
  ulp.

The weights say how the deltas relate geometrically; only the loss says whether that matters. The two disagree often
enough (below) that neither is enough alone.

## Merging theory

Take the merged weights $\theta = W + \sum_j \alpha_j \Delta_j$ and, for expert $i$'s domain, a quadratic model of the
loss around the expert, $L_i(W + x) \approx L_i(\theta_i) + \tfrac12 (x - \Delta_i)^\top H_i (x - \Delta_i)$. If
the domain's curvature is felt mainly along the expert's own direction $u_i = \Delta_i / \|\Delta_i\|$,
$H_i \approx h_i u_i u_i^\top$, the loss of the merge is

$$
\sum_i L_i(\theta) - L_i(\theta_i) \approx \sum_i \frac{h_i}{2} \bigl(\langle \theta - W, u_i \rangle - \|\Delta_i\|\bigr)^2 ,
$$

which is zero when every expert's projection is kept at its own size,
$\langle \sum_j \alpha_j \Delta_j, \Delta_i \rangle = \|\Delta_i\|^2$, i.e.

$$
G \alpha = \operatorname{diag}(G), \qquad G_{ij} = \langle \Delta_i, \Delta_j \rangle .
$$

This is `matching_coefficients`. Orthogonal deltas give $\alpha = 1$ (task arithmetic); identical ones any $\alpha$
with $\sum_j \alpha_j = 1$ (averaging); $n$ deltas of equal size with pairwise cosine $\rho$ give

$$
\alpha = \frac{1}{1 + (n-1)\rho} .
$$

Averaging ($\alpha = 1/n$) therefore shrinks every expert whenever $\rho < 1$, and summing ($\alpha = 1$) overshoots
whenever $\rho > 0$. The assumption is optimistic: each domain also pays for the other experts' components it does
not want, $\tfrac12 \sum_{j \ne i} \alpha_j^2 \Delta_j^\top H_i \Delta_j$, which the Gram cannot see and the loss
matrix measures.

Forced routing, the dense equivalent of a (Hier)DeltaMoE whose router always picks expert $i$, takes the MLP from
$\theta_i$ and the shared tensors from the merge. Its loss on domain $i$ is the merge's error restricted to the
shared tensors, so it bounds what any router can reach with that shared-parameter merge, and the gap between
"MLP of $i$, rest of $i$" and "MLP of $i$, rest merged" is the price of sharing the rest.

## Findings

From Qwen3.5-4B SFT experts (engineering, history, law, CS, finance and law-exam MCQ, medical, generation; three
seeds each where available), Qwen3.5-4B and Qwen3.8-27B medical continued pre-training (lr $10^{-6}$), and
Qwen3.5-9B medical SFT experts.

Size and granularity:

- SFT moves the MLP by 0.5–4 % of its norm; low-lr CPT by 0.02–0.45 % (a few hundred steps). Of a CPT delta, 45–77 % of the entries are
  unchanged in bf16 and another 15–30 % moved by exactly one ulp. Merge in fp32 and keep the result as a delta: a
  bf16 average of such deltas rounds most of them back to zero. A delta that is mostly $\{0, \pm 1\ \text{ulp}\}$ is
  already close to ternary, which helps [compression](delta-compression.md).
- A checkpoint saved in a lower precision than the base (fp32 norms written back as bf16) shows a constant
  "delta" on those tensors; check dtypes before reading small deltas.

Seeds and domains (MLP, SFT):

- Changing only the seed gives cosine 0.47–0.54. Different domains give 0.1–0.5, and the closest pairs (4B
  engineering–medical 0.49, law–CS MCQ 0.47) are as close as two seeds. The weights do form clusters (4B: engineering,
  history and medical against law and the MCQ sets; 9B: clinical against drug, literature and multitask), but they
  do not follow the domain labels or the data format, and their loss meaning has to be checked.
- A seed's delta is about 70–75 % its own: a second seed of the same domain explains only 24–29 % of its energy,
  every other domain together 23–36 %. The shared part of three seeds is 66–69 % of the energy (top eigenvalue).
- So clustering experts by weight cosine treats seed noise as signal; group by the
  loss matrix instead.
- A multitask model is not a combination of the single-domain experts trained on its parts: they explain 19–25 % of
  its delta. Continual fine-tuning keeps the earlier delta: training on top of a fine-tune explains 66–69 % of the
  result by the first stage.

CPT:

- Low-lr CPT is orthogonal to SFT (cosine 0.00–0.02). Runs on overlapping data mixtures are close (0.8–0.97; a rerun
  0.97), short runs on a narrow source far (0.03–0.26).
- A four-department literature CPT gains about the same on every department's held-out literature (4B: −0.081 to
  −0.098 nats); the department-specific part is under 0.02 nats. Groups by department have little to separate.
- Two runs with MLP cosine 0.97 differed by 0.15 nats on held-out clinical notes: one had a few linear-attention
  conv1d channels blown up (taps from about 0 to $\pm 2$–6, impossible for Adam at that lr). Screen every tensor's
  $\max|\Delta|$ against $\max|W|$ before merging; a whole-model cosine hides single channels.

Loss of SFT experts (4B, answer-letter NLL on 100 validation items per domain):

- Scoring the whole reply measures the shared answer format: every expert gained about 0.28 nats on every MCQ set.
  Score the option letter (`--score last`) or the reply text (`--score content`).
- Letter NLL measures confidence, not the argmax, and on 100 items it misled: it showed no own-domain gain for the
  engineering and history experts, while top-1 accuracy (`acc` in the output) does. Every domain's seed-mean expert
  is best on its own set (history: 75 % against 62–73 % for the others), and merging engineering, history, CS and
  finance into one MLP expert costs 4 points on history, as the converted HierDeltaMoE under oracle routing did
  ($-4.6$). Use accuracy to compare layouts, loss only alongside it.
- Averaging seeds is as good as the best seed and steadier than any one.
- On 100 items the proxy ranks MLP-expert merges, not shared-parameter merges: medical's NLI4CT accuracy on
  validation was 9 points higher with the shared tensors left at the base, while on the test set the converted
  HierDeltaMoE's oracle was the same for the base and for 5-, 8- and 11-way averages (all about 3 below the medical
  expert, which is the cost of dropping its own non-MLP delta).
- A flat layout with one seed-averaged expert per domain beat grouping by Gram clusters on the test sets; the
  group router was too soft. Weight neighbours are not functional neighbours: law is closer to CS (MLP cosine 0.43)
  than to the law-exam MCQ expert (0.31), merging law with CS costs neither, and the history expert is as good as
  the law expert on law. Which misroutes matter follows from the accuracy matrix, not from the cosine.
- Forced routing with the shared tensors averaged over five experts keeps the experts' own-domain gains; the dense
  average of the same five loses 14 points on medical dialogue. The MLP delta alone keeps the full gain as well.
  Summing experts hurts almost everywhere.
- Weight cosine does not predict function: engineering and medical have MLP cosine 0.49, and both are specific.

Loss of 27B CPT models (literature by department, textbook, drugs):

- Department-specific gains exist but are small (0.006–0.015 nats, under 16 % of the gain), and the average of two
  department models is within 0.006 of each on its own department: one literature expert is enough.
- Textbook and drug CPT are orthogonal to the literature CPT (cosine 0.03–0.07) and each only helps its own data,
  yet a dense merge with them interferes: the plain average keeps 43 % of the drug gain, and the matching
  coefficients keep the drug gain but make clinical notes worse by 0.07 nats and halve the literature gain. The
  matching condition only sees each delta's own direction, not what the others cost on its domain. Forced routing
  keeps 86–95 % of each gain: separate MLP experts.

Loss along and across deltas (4B CPT):

- Along one delta the loss is lowest near $\alpha = 1$: $0.5\,\Delta$ keeps 76 % of the gain, $2\,\Delta$ is worse
  than $1\,\Delta$.
- The MLP delta alone keeps about 89 % of the gain, the other tensors alone 55 %; the two overlap.
- Averaging an unrelated model's shared-tensor delta into the forced-routing model ("MLP of $i$, rest of $i$ and an
  SFT model averaged") is worse than leaving the rest at the base: merge the shared tensors only across related
  experts.

## Guidance for upscaling

1. Screen each fine-tune per tensor (size, $\max|\Delta|$, dtype) and drop or repair outliers before anything else.
2. Compute the Gram for all candidates and read seeds as the noise floor: a pair of experts whose cosine is not above
   their seed-to-seed cosine has no weight evidence of being different (or alike).
3. Build the loss matrix (every expert on every domain's validation set, scored on the answer, `--score last` for
   MCQ; compare top-1 accuracy, not only NLL) and group experts whose rows are alike; a group's members should each beat the others on their own
   domain, otherwise one expert or a merge serves the group.
4. Merge the shared tensors (a plain average of the SFT experts was enough at 4B, and the choice did not change the
   test-set oracle; `matching_coefficients` when the
   deltas are near-orthogonal, as CPT on different sources) and check the forced-routing models on the loss matrix:
   each should match its expert on its domain.
5. Seeds of one domain are merged (their $\alpha$ is close to $1/(1 + 2\rho)$, i.e. about 0.5 for three seeds), not
   kept as separate experts: the router has nothing to tell them apart by.
6. Experts without a specific gain on the loss matrix are merged into one expert, for the same reason.
