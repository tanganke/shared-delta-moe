# Per-expert attention and `lm_head` deltas

A DeltaMoE replaces only the decoder MLPs. Fine-tunes that changed every weight, such as continued pretraining, also
need their attention, linear attention and `lm_head` per expert. `RoutedDeltaLinear` (`models/routed_linear.py`)
gives those layers a delta per expert, routed per token like the MLP. The Qwen3.5 / Qwen3.8 integration builds them
from the text config's `routed_deltas`, and the HierDeltaMoE converter writes them with `--routed-deltas`.

## Why not one shared attention

Three Qwen3.8-27B continued-pretraining experts were tested (medical literature, textbook and guideline text, drug
labels; [delta compression](delta-compression.md), ntu-projects moe-indomain-acc log [62]-[64]). In each test, every
weight outside the MLPs was replaced by a shared copy, with the expert's own MLP delta kept dense. The table gives
next-token KL(expert || model) on 16 x 1024 tokens of each expert's held-out text.

| weights outside the MLPs | literature | textbook | drugs |
|---|---:|---:|---:|
| the base | 0.0044 | 0.0232 | 0.077 |
| mean of the three experts | 0.0028 | 0.0118 | 0.038 |
| best merge (TIES, top 50% kept) | 0.0061 | 0.0052 | 0.0094 |
| each expert's own, 2-bit grid | 0.0005 | 0.0007 | 0.0014 |

Even the best merge is 7-12 times further from each expert than per-expert 2-bit deltas. For scale, dropping an
expert's MLP delta altogether costs 0.032 / 0.089 / 0.430.

## The layer

A `RoutedDeltaLinear` holds the base weight $W_b$ (its `weight`) and $N$ deltas $\Delta_e$. Token $t$, with routing
slots $(e_i, r_i)$ and $\sum_i r_i = 1$, gets

$$
y_t = W_b x_t + \sum_i r_i \, \Delta_{e_i} x_t = \Big(W_b + \sum_i r_i \Delta_{e_i}\Big) x_t .
$$

That is the routed experts' weights mixed per token. The layer is linear, so mixing outputs and mixing weights agree.
In the MLP they don't: there each slot keeps its own SwiGLU activation ([DeltaMoE](deltamoe.md)). A slot with
$e_i \ge N$, which is a zero expert or the base, adds nothing.

The deltas are stored in one of two ways:

- **Dense:** `deltas`, of shape (N, R, C).
- **Group grid:** as in [group grids](grid.md).
  - `codes`: (N, R, C b / 32), int32.
  - `steps`: (N, R, C / g).
  - `lows`: (N, R, C / g), asymmetric grids only.

  Each entry is $\hat\delta = \ell + s\,q$ with $q \in \{0, \dots, L-1\}$. At 2 bits with $g = 128$, that is 2.25 bits
  per entry.

The forward is $W_b x$ for all tokens (cuBLAS) plus the routed deltas. With 2-bit grids on CUDA in bf16 / fp16 and
no gradient needed anywhere, the deltas take one fused Triton launch: the MoE down kernel (mode 3 of
`triton_kernels/fused.py`) over a zero base, decoding $\ell + s q$ tile by tile, with no host sync. Otherwise, or past
$2^{28}$ (pair, output) entries (e.g. `lm_head` over a whole sequence), a loop over the routed experts: `index_add` of
$r \cdot \Delta_e x$ over that expert's rows, differentiable in the routing weights, the grid steps and lowest
levels, and dense deltas.

## Routing

The routing comes from outside the layer, in its `route` attribute: (idx, r), each of shape (T, K).

- **Attention blocks:** each attention block (`self_attn` or `linear_attn`) gets a `delta_router`, shaped like its
  layer's MoE router. A forward pre-hook on the block routes the block's input, `hidden_states` (after
  `input_layernorm`), with `layer.mlp.topk(x, delta_router)`. It then sets that route on every projection of the
  block.

  The attention therefore routes exactly as the layer's MLP does (groups, zero experts, top-k, `group_top_k`), but
  from its own router and its own input. Expert $e$ is the same fine-tune in the MLP, the attention and `lm_head`.
- **`lm_head`:** has its own `delta_router`, routed like the last layer's MoE, on the final hidden states.
  `logits_to_keep` works because the route is computed on whatever reaches `lm_head`.

The routers start at zero, as the MLP routers do. That makes routing uniform until they are trained.

## Sets: one delta for several experts

`--routed-sets a,b,...` stores one routed delta per set of consecutive experts (in expert order): the mean of the
set's deltas, $\bar\Delta_s$. Use it when a domain has several fine-tunes, for example seeds or epochs: $S$ sets
store $S$ deltas instead of $N$. Each slot maps its expert to the expert's set $s(e)$:

$$
y_t = W_b x_t + \sum_i r_i \, \bar\Delta_{s(e_i)} x_t .
$$

A token's weight on a set is the sum of its slots' weights on that set's experts. Zero experts and the base still
add nothing. The sets don't depend on the MoE's routing groups, so a flat MoE over all experts can use the same sets
as a two-level one.

## What stays shared

| weight | routed per expert | why |
|---|---|---|
| attention `q_proj`, `k_proj`, `v_proj`, `o_proj` | yes | not shareable (table above) |
| linear attention `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` | yes | same |
| `lm_head` | yes | sharing it multiplies the KL by 2-4 |
| embedding | no | sharing it costs nothing; it comes before any router |
| norms, linear attention `conv1d`, `A_log`, `dt_bias` | no | sharing costs little (see below) |
| vision tower, MTP | no | not on the text path |

The unrouted weights are merged with `--merge`. Use `--merge task_arithmetic --merge-arg scaling_factor=0` to take the
base itself, which is what was measured.

Sharing the norms, `conv1d`, `A_log`, `dt_bias` and the embedding from the base, with every other delta at 2 bits,
gives KL 0.0012 / 0.0009 / 0.0017 on the three experts, against 0.0005 / 0.0008 / 0.0015 with nothing shared. Almost
all of the literature expert's increase comes from `conv1d`. That expert was trained with another recipe, and the
cause hasn't been investigated.

## Converting

```bash
python -m shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert BASE DST --group ... \
    --routed-deltas [--routed-levels 4 --routed-group 128 [--routed-symmetric]] [--routed-sets 4,4,4,2] \
    [--levels 4 --grid-group 128 [--symmetric]] [--merge task_arithmetic --merge-arg scaling_factor=0]
```

- **`--routed-levels`:** rounds the routed deltas to a grid. It uses round to nearest and needs no data: at 27B it
  matches GPTQ, while clipping and rotations hurt or don't help ([delta compression](delta-compression.md)). Without
  it the routed deltas are stored dense.
- **`--routed-group`:** the grid group $g$. It must divide each projection's input width.
- **`--levels`:** stores the MLP deltas as grids too. The model then builds `Qwen3_5GridHierDeltaMoeBlock`.
- **GPU:** the rounding runs on CUDA when it is available. Run the converter on a compute node.

Keys per projection, next to its `weight` (= $W_b$):

- **Dense:** `deltas`.
- **Grid:** `codes`, `steps`, and `lows` for asymmetric grids.

Each block also gets `delta_router.weight`, and so does `lm_head` (`lm_head.delta_router.weight`). The text config
gets `routed_deltas: {levels, group, symmetric, sets}`, where `levels` is null for dense and `sets` is optional.
`lm_head` must be untied (`tie_word_embeddings` false).

## Loading and training

```python
import shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe  # patches the decoder layer and lm_head
model = Qwen3_5ForConditionalGeneration.from_pretrained(DST, dtype=torch.bfloat16)
```

Importing either Qwen3.5 integration patches the decoder layers, the attention blocks and `lm_head`; configs without
`routed_deltas` are unchanged. With `device_map` (accelerate) the hook moves the block input to the router's device.

**To train only the routers:** keep `requires_grad` on the parameters whose name ends in `router.weight`
(`mlp.router.weight` and every `delta_router.weight`) and freeze the rest. The routers get their gradients through
the routing weights.

**Known gaps:**

- `nn.DataParallel` replicas are not supported: the hooks hold the original modules. DDP and FSDP wrap the model in
  place, so they work.
- Gradient checkpointing and the KV cache work. A checkpointed block recomputes its forward in backward from
  inputs that need not be bitwise equal (atomic adds, non-deterministic kernels), and a near-tie could then flip a
  top-k pick. So inside a non-reentrant checkpoint (the transformers default) every router's picks are kept on the
  checkpoint and reused in the recompute (`ops.replay`); the routing weights are still recomputed, so the routers
  get their gradients. Reentrant checkpoints get no such protection.

## Size at 27B

Qwen3.8-27B has 8.6 B routed parameters: attention 1.8 B, linear attention 5.6 B, `lm_head` 1.3 B. At 2.25 bits that
is 2.4 GB per expert or set. With the MLP deltas at 2.25 bits (17.4 B parameters, 4.9 GB), an expert costs about
7.3 GB on top of the 54 GB base.

## Checks and status

`tests/test_routed_linear.py` checks:

- the layer against per-token weight mixing;
- a converted tiny Qwen3.5 routed to one expert against that fine-tune, with dense and grid deltas, grid MLPs and sets;
- that the routers get gradients.

On the three 27B experts (moe-indomain-acc [65]), the converted MoE was run with every router forced to one expert,
with dense MLP deltas and 2-bit routed deltas. Its KL from that expert:

| | literature | textbook | drugs |
|---|---:|---:|---:|
| own text | 0.0014 | 0.0012 | 0.0021 |
| general text | 0.0030 | 0.0008 | 0.0008 |

This is 0.0002-0.0004 above the offline estimate for the same storage. The cause is not found yet; computing
$W_b + \Delta_e$ first instead of the two products gives the same numbers.

Inference with 2-bit grids uses the fused kernels (MLP and routed deltas); the timing above is from the earlier
loop. Training goes through the differentiable loop. [vLLM](vllm.md) serves routed 2-bit deltas, sets included,
with tensor parallelism.
