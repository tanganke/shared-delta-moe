# shared-delta-moe

A minimal implementation (one file, torch only) of a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, showing two forward forms that are exactly equivalent to the original MoE.

Each routed expert's weights are $W_e = W_b + \Delta_e$, applied to gate, up and down separately. The router's first n rows are the deltas; its last `num_zero_experts` rows (Z, default 1) are **zero experts**: they have no parameters and each stands for the base MLP itself. Whatever Z is, base + n deltas has exactly as many parameters as the original MoE's n+1 distinct experts. With Z = 0 the base is only a shared component, never chosen on its own; with Z > 1 the original MoE is equivalent to one with Z copies of the base expert (which gives the router more room to send tokens to the base).

Routing is softmax, top-k (`top_k`, default 1), renormalised over the k picks to weights $r_i$. The target is the original MoE:

$$y = \sum_{i=1}^{k} r_i\,\mathrm{MLP}(x;\,W_b + \Delta_{e_i})$$

| Form | Computation | Function |
|---|---|---|
| Parameter aggregation | Per expert, materialise $W_b + \Delta_e$ once and run one MLP on the (token, slot) pairs routed to it; multiply by $r_i$ and `index_add` back | `DeltaMoE.forward_param` |
| Per-matrix output aggregation | gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `DeltaMoE.forward_matrix` |
| Reference: plain MoE | Stored full expert weights, grouped the same way | `DeltaMoE.forward_dense` |
| Per-matrix output aggregation, grouped GEMM | Same as above, but flattens all (token, slot) pairs, sorts them by expert, and computes the deltas for each projection in one `F.grouped_mm` call (no Python loop) | `DeltaMoE.forward_matrix_grouped` |
| Reference: plain MoE, grouped GEMM | One `F.grouped_mm` per projection, which is what transformers' default `grouped_mm` path does | `DeltaMoE.forward_dense_grouped` |

Note two things that would not be equivalent:

- **Adding whole-MLP outputs.** SwiGLU is nonlinear in its weights, so $\mathrm{MLP}(x;W_b) + \mathrm{MLP}(x;\Delta_e) \ne \mathrm{MLP}(x;W_b+\Delta_e)$. The addition has to happen on each weight matrix.
- **Merging into one activation when $k>1$.** Adding every slot's $r_i\Delta_{e_i}x$ into a single activation amounts to merging weights per token, $\mathrm{MLP}(x;\,W_b+\sum_i r_i\Delta_{e_i})$. That is a different function from the MoE above. So in per-matrix output aggregation each slot keeps its own activation; only $W_b x$ in gate/up is shared across slots. Down needs $k$ passes.

With $k=1$, $r=1$ and both forms reduce to one MLP. Under top-k the zero expert can be one of the k picks: its share $r_i$ goes to the base MLP.

Build one from a plain MoE with `DeltaMoE.from_experts(base, experts, router, top_k)`:

- `base`: the base expert's `(gate (I, H), up (I, H), down (H, I))`.
- `experts`: the other n experts' full weights $W_e$ (not deltas), each in the same `(gate, up, down)` form; $\Delta_e = W_e - W_b$ is computed inside.
- `router`: the original router weight `(n+Z, H)`, with rows n..n+Z-1 routing to the base (if the base is expert 0 in the original MoE, move its row to the end first).
- `num_zero_experts`: Z, default 1.

transformers' fused layout maps over as `gate, up = gate_up_proj[e].chunk(2)` and `down = down_proj[e]`. In bf16, $W_b + (W_e - W_b)$ is not always exactly $W_e$ (about 0.6% of elements in our model).

**No custom kernels.** Everything is `F.linear` (cuBLAS), `F.grouped_mm` (PyTorch's own grouped GEMM: CUDA sm80+ or CPU, bf16/fp16/fp32, no fp64) plus `index_add`.

```bash
python delta_moe.py                  # float32 self-check (Z = 0/1/2 x top-1/top-2, every form == plain MoE, token by token), then a top-1 benchmark if CUDA is available
python delta_moe.py --top-k 2        # top-2 benchmark
python delta_moe.py --no-bench
```

## Speed (single MoE layer, H200, bf16, eager, ms)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts). "transformers" is the stock `Qwen3_5MoeSparseMoeBlock` (`grouped_mm`, 6 experts) at the same size.

top-1:

| tokens | plain loop | plain grouped | transformers | param agg | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.265 | 0.367 | 0.360 | 0.244 | 0.231 | 0.335 |
| 32 | 0.828 | 0.578 | 0.472 | 1.310 | 0.782 | 0.611 |
| 512 | 0.888 | 0.571 | 0.494 | 1.369 | 0.985 | 0.736 |
| 8192 | 2.722 | 2.450 | 2.719 | 3.220 | 5.637 | 5.130 |

top-2:

| tokens | plain loop | plain grouped | transformers | param agg | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.374 | 0.392 | 0.405 | 0.428 | 0.471 | 0.445 |
| 32 | 0.843 | 0.597 | 0.498 | 1.335 | 1.482 | 0.632 |
| 512 | 0.947 | 0.662 | 0.591 | 1.426 | 1.833 | 0.917 |
| 8192 | 4.539 | 4.402 | 4.732 | 5.195 | 10.248 | 8.631 |

- **Grouped GEMM** replaces the per-expert Python loop with one kernel per projection. From 32 to 512 tokens it is 1.3–2.3× faster than the loop, and top-2 gains the most, because the loop runs every expert of every slot separately. At 1 token, the extra sort and count make it slightly slower.
- **Per-matrix output aggregation, grouped**: 1.3× the transformers MoE at 32 tokens (top-1 0.61 vs 0.47, top-2 0.63 vs 0.50), and about 1.9× at 8192 tokens (computing both $W_b$ and $\Delta_e$ roughly doubles the compute).
- **Parameter aggregation** writes out a full weight matrix per expert, so it is the slowest at mid-range token counts. It does not fit grouped GEMM well: you would first have to write out $W_b+\Delta_e$ for every expert, which costs even more memory traffic.
- The remaining gap between plain grouped and transformers (0.58 vs 0.47) is mainly the `nonzero` and `bincount` here, which synchronise the CPU and GPU; transformers uses `histc` and a fixed number of groups, which needs no sync.

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
