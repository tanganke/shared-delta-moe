# shared-delta-moe

A minimal implementation (one file, torch only) of a top-1 SwiGLU MoE written as **a shared base expert plus per-expert deltas**, showing two forward forms that are exactly equivalent to the original MoE.

Each routed expert's weights are $W_k = W_b + \Delta_k$, applied to gate, up and down separately. Router index 0 is the **zero expert**: it has no parameters, and the tokens routed to it compute only the base MLP. So base + n deltas has exactly as many parameters as the original MoE's n+1 experts.

| Form | Computation | Function |
|---|---|---|
| Parameter aggregation | Tokens are grouped by routed expert; each group materialises $W_b + \Delta_k$ once, then runs one MLP | `DeltaMoE.forward_param` |
| Per-matrix output aggregation | Each projection computes $W_b x$ (all tokens at once), plus $\Delta_k x$ for each group (`index_add`) | `DeltaMoE.forward_matrix` |
| Reference: plain MoE | Stored full expert weights, grouped the same way | `DeltaMoE.forward_dense` |

Note that you cannot add the outputs of a whole MLP instead: SwiGLU is nonlinear in its weights, so $\mathrm{MLP}(x;W_b) + \mathrm{MLP}(x;\Delta_k) \ne \mathrm{MLP}(x;W_b+\Delta_k)$. The addition has to happen on each weight matrix.

**No custom kernels.** Everything is `F.linear` (cuBLAS) plus `index_add`.

```bash
python delta_moe.py              # float64 self-check (both forms == plain MoE, token by token), then a benchmark if CUDA is available
python delta_moe.py --no-bench
```

## Speed (single MoE layer, H200, bf16, eager, ms)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts):

| tokens | plain MoE | parameter aggregation | per-matrix output aggregation |
|---:|---:|---:|---:|
| 1 | 0.177 | 0.264 | 0.224 |
| 32 | 0.595 | 1.129 | 0.678 |
| 512 | 0.651 | 1.179 | 0.869 |
| 8192 | 2.236 | 2.825 | 5.471 |

Neither form beats a plain MoE implemented the same way:

- **Parameter aggregation** writes out a full weight matrix per expert. With few tokens that extra memory traffic dominates (32 tokens: 1.9×).
- **Per-matrix output aggregation** has no weight sum, but does about twice the compute. It costs little with few tokens and is clearly slower with many (8192 tokens: 2.4×).

The benefits of these forms are elsewhere. The delta $\Delta_k$ is a standalone tensor that can be compressed, sparsified or trained on its own; the zero expert has no parameters; and in per-matrix output aggregation the routing weight can stay in the computation graph.

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
