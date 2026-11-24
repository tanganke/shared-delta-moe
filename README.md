# shared-delta-moe

A minimal implementation (one file, torch only) of a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, showing two forward forms that are exactly equivalent to the original MoE.

Each routed expert's weights are $W_e = W_b + \Delta_e$, applied to gate, up and down separately. Router index 0 is the **zero expert**: it has no parameters and stands for the base MLP itself. So base + n deltas has exactly as many parameters as the original MoE's n+1 experts.

Routing is softmax, top-k (`top_k`, default 1), renormalised over the k picks to weights $r_i$. The target is the original MoE:

$$y = \sum_{i=1}^{k} r_i\,\mathrm{MLP}(x;\,W_b + \Delta_{e_i})$$

| Form | Computation | Function |
|---|---|---|
| Parameter aggregation | Per expert, materialise $W_b + \Delta_e$ once and run one MLP on the (token, slot) pairs routed to it; multiply by $r_i$ and `index_add` back | `DeltaMoE.forward_param` |
| Per-matrix output aggregation | gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `DeltaMoE.forward_matrix` |
| Reference: plain MoE | Stored full expert weights, grouped the same way | `DeltaMoE.forward_dense` |

Note two things that would not be equivalent:

- **Adding whole-MLP outputs.** SwiGLU is nonlinear in its weights, so $\mathrm{MLP}(x;W_b) + \mathrm{MLP}(x;\Delta_e) \ne \mathrm{MLP}(x;W_b+\Delta_e)$. The addition has to happen on each weight matrix.
- **Merging into one activation when $k>1$.** Adding every slot's $r_i\Delta_{e_i}x$ into a single activation amounts to merging weights per token, $\mathrm{MLP}(x;\,W_b+\sum_i r_i\Delta_{e_i})$. That is a different function from the MoE above. So in per-matrix output aggregation each slot keeps its own activation; only $W_b x$ in gate/up is shared across slots. Down needs $k$ passes.

With $k=1$, $r=1$ and both forms reduce to one MLP. Under top-k the zero expert can be one of the k picks: its share $r_i$ goes to the base MLP.

Build one from a plain MoE with `DeltaMoE.from_experts(base, experts, router, top_k)`:

- `base`: the base expert's `(gate (I, H), up (I, H), down (H, I))`, i.e. expert 0 of the original MoE.
- `experts`: the other n experts' full weights $W_e$ (not deltas), each in the same `(gate, up, down)` form; $\Delta_e = W_e - W_b$ is computed inside.
- `router`: the original router weight `(n+1, H)`, with row 0 routing to the base.

transformers' fused layout maps over as `gate, up = gate_up_proj[e].chunk(2)` and `down = down_proj[e]`. In bf16, $W_b + (W_e - W_b)$ is not always exactly $W_e$ (about 0.6% of elements in our model).

**No custom kernels.** Everything is `F.linear` (cuBLAS) plus `index_add`.

```bash
python delta_moe.py                  # float64 self-check (top-1 and top-2, both forms == plain MoE, token by token), then a top-1 benchmark if CUDA is available
python delta_moe.py --top-k 2        # top-2 benchmark
python delta_moe.py --no-bench
```

## Speed (single MoE layer, H200, bf16, eager, ms)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts):

| tokens | top-1 plain MoE | top-1 param agg | top-1 matrix agg | top-2 plain MoE | top-2 param agg | top-2 matrix agg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.242 | 0.304 | 0.296 | 0.352 | 0.418 | 0.440 |
| 32 | 0.800 | 1.292 | 0.771 | 0.832 | 1.311 | 1.480 |
| 512 | 0.853 | 1.349 | 0.974 | 0.926 | 1.428 | 1.834 |
| 8192 | 2.708 | 3.197 | 5.633 | 4.600 | 5.218 | 10.258 |

Neither form beats a plain MoE implemented the same way:

- **Parameter aggregation** writes out a full weight matrix per expert. With few tokens that extra memory traffic dominates (top-1, 32 tokens: 1.6×). Under top-k each expert still materialises its weights only once, so going from top-1 to top-2 adds little.
- **Per-matrix output aggregation** has no weight sum, but does about twice the compute. Under top-1, with few tokens, it is on par with the plain MoE. Under top-k every slot has to run the full base down projection plus the deltas, so it is the slowest at top-2 (8192 tokens: 2.2×).

The benefits of these forms are elsewhere. The delta $\Delta_e$ is a standalone tensor that can be compressed, sparsified or trained on its own; the zero expert has no parameters; and in per-matrix output aggregation the routing weight can stay in the computation graph.

## Why it looked "faster than the original MoE" in transformers

Running the full model with transformers (Qwen3.5-4B MoE, 6 experts, H200), batch-32 decode measured 338 tok/s for the original MoE, while both forms here reached 711 / 860 tok/s. That is **not** because these forms are faster. It's because the original MoE was given a slow path by transformers:

1. Qwen3.5 MoE experts default to `grouped_mm` (`torch._grouped_mm`, one cutlass grouped GEMM for all experts). A single layer takes 0.45 ms at 32 tokens, faster than the plain MoE in this repo.
2. But `generate()` uses `_optimize_model_for_decode` to automatically **switch every `grouped_mm` to `batched_mm` during decode** (transformers 5.16, `generation/utils.py`).
3. `batched_mm` gathers **one full copy of the expert weights per token** (`self.gate_up_proj[expert_ids]`), then runs `bmm`. At batch 1 that's about the same as reading the weights directly. At batch 32, every layer copies 32 expert-sized matrices (about 4.5 GB). The profile shows 90% of the time in `aten::index` (gather) and `aten::bmm`.
4. The two forms here don't go through the experts module, so they're never switched.

Disable the switch and the original MoE becomes the fastest:

| Batch-32 decode (full model, ctx 512) | tok/s | ms/step |
|---|---:|---:|
| Original MoE (with transformers' switch to `batched_mm`) | 338 | 94.7 |
| Original MoE (stays on `grouped_mm`) | **867** | 36.9 |
| Parameter aggregation | 711 | 45.0 |
| Per-matrix output aggregation | 860 | 37.2 |

How to disable it (batch 1 gets slightly slower, 31 → 28 tok/s):

```python
import contextlib
from transformers.generation.utils import GenerationMixin
GenerationMixin._optimize_model_for_decode = contextlib.contextmanager(lambda self: (yield))
```

The takeaway: with large experts (9216 × 2560) and batch > 1, transformers' "switch to `batched_mm` during decode" heuristic makes things slower.
