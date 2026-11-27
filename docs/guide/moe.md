# Plain MoE

`MoE` and `HierMoE` (`models/moe.py`) are the MoEs that the delta forms are equivalent to: every expert is stored as
its own full weights, with no shared base. They route exactly as `DeltaMoE` and `HierDeltaMoE`, so a delta model and
its `to_moe()` compute the same function. They are the baseline that the tests and benchmarks compare against, and a
plain module to train or convert from when the delta layout is not wanted.

## MoE

A top-k SwiGLU MoE over $E$ experts. Routing is the same as in [DeltaMoE](deltamoe.md): a float32 softmax over the
router logits, top-$k$, renormalised over the $k$ picks to weights $r_i$:

$$
y = \sum_{i=1}^{k} r_i\,\mathrm{MLP}(x;\,W_{e_i})
$$

**Storage.** One layer's state dict:

| key | shape | holds |
|---|---|---|
| `gate`, `up` | $(E, I, H)$ | each expert's gate and up weights |
| `down` | $(E, H, I)$ | each expert's down weight |
| `router.weight` | $(E,\ H)$ | row $e$ routes to expert $e$ |

**Building one.** Either from full experts, or from a delta model:

```python
import torch
from shared_delta_moe import DeltaMoE, MoE, random_experts

H, I = 1024, 4096
full = random_experts(H, I, 4, torch.bfloat16, "cuda")        # (gate, up, down) each
router = torch.randn(4, H, dtype=torch.bfloat16, device="cuda") * H**-0.5
moe = MoE.from_experts(full, router, top_k=2)                # E = 4

delta = DeltaMoE.from_experts(full[0], full[1:], router, top_k=2, num_zero_experts=1)
plain = delta.to_moe()                                       # the same MoE, full experts again
x = torch.randn(32, H, dtype=torch.bfloat16, device="cuda")
y = plain(x)                                                 # any leading shape (..., H)
```

`DeltaMoE.to_moe()` writes out $[W_b+\Delta_1, \dots, W_b+\Delta_N]$ followed by one copy of $W_b$ for each zero
expert, and keeps the router and top-$k$ unchanged ($E = N + Z$). DARE, BitDelta and grid deltas are unpacked first.
To start from shapes instead, use `MoE(hidden, inter, n_experts, top_k)` (uninitialised, also on the meta device) and
load the state dict.

## HierMoE

The two-level MoE of [HierDeltaMoE](hier.md), with every expert stored in full. Group $i$ holds $E_i$ experts
$\theta_{ij}$:

$$
y = \sum_i h_i \sum_{j \in \text{top-}k_1} \tilde p_{ij}\,\mathrm{MLP}(x;\,\theta_{ij})
$$

Here $h$ is the softmax over the $G$ groups (all groups when `group_top_k=None`, otherwise top-$k_0$ renormalised),
and $\tilde p_{ij}$ is the top-$k_1$ of group $i$'s own softmax, renormalised. The router has
$G + \sum_i E_i$ rows: first $R_0$ ($G$ rows), then each group's expert rows, group by group. `gate`, `up` and `down`
hold the experts in the same order.

- `HierMoE.from_experts(groups, router, top_k, group_top_k)`, with `groups[i]` group $i$'s experts as
  `(gate, up, down)`; `top_k` is $k_1$, `group_top_k` is $k_0$.
- `HierDeltaMoE.to_moe()` gives group $i$ its $E_i$ experts $W_b+\Delta_{ij}$ and then $Z_i$ copies of $W_b$ (group
  sizes $E_i+Z_i$). The router layout, $k_1$ and $k_0$ stay the same.

The routing is flattened into $S\cdot k_1$ slots per token ($S = G$ when dense, otherwise $k_0$), so every `MoE`
forward runs unchanged. The routing code (`hier_topk`, `hier_tables`) is shared with `HierDeltaMoE`.

## Forwards

| method | computation |
|---|---|
| `_forward_grouped` | all (token, slot) pairs sorted by expert, one `F.grouped_mm` per projection, like transformers' `grouped_mm` path |
| `_forward_loop` | a loop over the routed experts, one MLP on the rows routed to each |
| `forward` | `_forward_grouped` when `F.grouped_mm` takes the dtype and device (bf16/fp16/fp32; CPU, or CUDA sm80+) and there is more than one pair, otherwise `_forward_loop` (the same rule as `forward_matrix`, see [Forward dispatch](../internals/dispatch.md)) |

Both are differentiable: gradients reach $x$, the router (through the routing weights) and every expert weight. No
Triton kernel is used: that kernel's gain is reading $W_b$ and $\Delta_e$ without writing their sum, which a plain MoE
does not need.

## How it compares with the delta forms

- **Same function.** `to_moe()` forms $W_b+\Delta_e$ in the model dtype, the same weights the materialised
  `forward_param` path forms, and routes with the same code, ties included. The fused Triton kernel rounds
  differently (it adds $W_b$ and $\Delta_e$ in registers), so in bf16 a delta model and its `to_moe()` agree to
  rounding, not bit for bit.
- **Size.** $E$ MLPs of weights plus the router. A `DeltaMoE` stores $1+N$ MLPs, so the two are the same size at
  $Z = 1$, and every extra zero expert is one more stored MLP here. At hidden 2560, inter 9216, one MLP in bf16 is
  141.6 MB.
- **Speed.** The plain-MoE columns in [Benchmarks](../internals/benchmarks.md) (`plain grouped`, `dense grouped
  (plain MoE)`) are `_forward_grouped`'s computation; they were measured before this module existed, through
  `DeltaMoE.forward_dense`, which ran the same operations.

## Limits

- There is no transformers or vLLM model class for a plain MoE; the [Qwen3.5 converters](qwen3_5.md) write the delta
  layouts only. For a whole stock MoE checkpoint, transformers' own MoE is the plain model.
- `tests/test_moe.py` checks the loop, grouped and module forwards and all gradients against the MoE written out token
  by token (for `HierMoE`, two-level routing computed from the router weight, not through `topk`), plus bf16, meta
  construction and state-dict loading. `tests/test_delta_moe.py`, `test_hier.py`, `test_dare.py`, `test_bitdelta.py`
  and `test_grid.py` check that `to_moe()` equals the delta model.
