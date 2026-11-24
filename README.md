# shared-delta-moe

A small Python library: a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, with several forward implementations that are exactly equivalent to the original MoE, including a Triton kernel that fuses $W_b+\Delta_e$ into the GEMM.

```
src/shared_delta_moe/delta_moe.py       DeltaMoE, from_experts, the forward implementations
src/shared_delta_moe/triton_kernels.py  delta_gmm: grouped GEMM over (W_b + Delta_g), the sum formed in registers
tests/test_delta_moe.py                 every forward == plain MoE, token by token (Triton part needs CUDA)
benchmarks/bench.py                     per-layer time of each implementation (CUDA)
```

```bash
uv sync                                  # or: pip install -e .   (Triton ships with torch on Linux + CUDA)
uv run pytest                            # or: python tests/test_delta_moe.py
uv run python benchmarks/bench.py --top-k 2 --tokens 1,32,512,8192
```

```python
from shared_delta_moe import DeltaMoE
m = DeltaMoE.from_experts(base, experts, router, top_k=1, num_zero_experts=1)
y = m.forward_param(x)   # or forward_matrix(x); forward_dense(x, m.dense_experts()) is the plain-MoE baseline
```

Each routed expert's weights are $W_e = W_b + \Delta_e$, applied to gate, up and down separately. The router's first n rows are the deltas; its last `num_zero_experts` rows (Z, default 1) are **zero experts**: they have no parameters and each stands for the base MLP itself. Whatever Z is, base + n deltas has exactly as many parameters as the original MoE's n+1 distinct experts. With Z = 0 the base is only a shared component, never chosen on its own; with Z > 1 the original MoE is equivalent to one with Z copies of the base expert (which gives the router more room to send tokens to the base).

Routing is softmax, top-k (`top_k`, default 1), renormalised over the k picks to weights $r_i$. The target is the original MoE:

$$y = \sum_{i=1}^{k} r_i\,\mathrm{MLP}(x;\,W_b + \Delta_{e_i})$$

| Public method | Computation | Implementations (private), picked automatically |
|---|---|---|
| `forward_param` | Parameter aggregation: each (token, slot) pair is multiplied by $W_b + \Delta_e$ of its expert, weighted by $r_i$ | `_forward_param_triton`: one Triton launch per projection for all pairs, adding $W_b$ and $\Delta_e$ tiles in registers (no weights written, 1× FLOPs); `_forward_param(x, min_rows)`: materialise $W_b + \Delta_e$ per expert and run cuBLAS (experts with fewer than `min_rows` rows compute $W_b x + \Delta_e x$ instead) |
| `forward_matrix` | Per-matrix output aggregation: gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `_forward_matrix_grouped`: flattens all (token, slot) pairs, sorts them by expert, one `F.grouped_mm` per projection for the deltas; `_forward_matrix`: loop over experts |
| `forward_dense(x, experts)` | Reference: plain MoE from stored full expert weights | `_forward_dense_grouped`: one `F.grouped_mm` per projection, like transformers' `grouped_mm` path; `_forward_dense`: loop over experts |

Selection rules (thresholds measured on one H200 at the shape below; retune for other GPUs or shapes):

- `forward_param`: Triton on CUDA bf16/fp16 for 2–4096 (token, slot) pairs. A single pair is cheaper through the loop; beyond about 4096 the work is compute-bound, and cuBLAS on materialised weights wins.
- `forward_matrix` / `forward_dense`: grouped GEMM when `F.grouped_mm` exists, the dtype is bf16/fp16/fp32, the device is CPU or a CUDA GPU with sm80+, and there is more than one pair; otherwise the loop.

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

Apart from the Triton kernel, everything is `F.linear` (cuBLAS), `F.grouped_mm` (PyTorch's own grouped GEMM: CUDA sm80+ or CPU, bf16/fp16/fp32) and `index_add`. bf16/fp16 are the targets; fp32 is supported (and used for the CPU tests), fp64 is not.

## Speed (single MoE layer, H200, bf16, eager, ms)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts). For comparison, transformers' stock `Qwen3_5MoeSparseMoeBlock` (`grouped_mm`) at the same size takes 0.36 / 0.47 / 0.49 / 2.72 ms at 1 / 32 / 512 / 8192 tokens (top-1).

top-1:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.249 | 0.349 | 0.229 | 0.221 | 0.336 | 0.229 | 0.315 |
| 8 | 0.572 | 0.481 | 0.935 | 0.748 | **0.354** | 0.609 | 0.568 |
| 32 | 0.794 | 0.586 | 1.263 | 1.018 | **0.359** | 0.758 | 0.597 |
| 512 | 0.850 | 0.562 | 1.321 | 1.115 | **0.557** | 0.963 | 0.733 |
| 2048 | 1.149 | 0.958 | 1.649 | 1.655 | 1.322 | 1.864 | 1.636 |
| 4096 | 1.677 | 1.319 | 2.205 | 2.178 | 1.934 | 3.122 | 2.730 |
| 8192 | 2.730 | 2.409 | 3.251 | 3.237 | 3.655 | 5.670 | 5.139 |

top-2:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.358 | 0.387 | 0.417 | 0.390 | **0.354** | 0.450 | 0.419 |
| 8 | 0.789 | 0.578 | 1.254 | 1.004 | **0.361** | 1.266 | 0.633 |
| 32 | 0.824 | 0.573 | 1.287 | 1.051 | **0.362** | 1.435 | 0.623 |
| 512 | 0.919 | 0.659 | 1.416 | 1.267 | 0.712 | 1.818 | 0.923 |
| 2048 | 1.654 | 1.319 | 2.180 | 2.144 | 1.783 | 3.438 | 2.444 |
| 4096 | 2.615 | 2.439 | 3.152 | 3.120 | 3.621 | 5.712 | 4.574 |
| 8192 | 4.717 | 4.601 | 5.456 | 5.258 | 7.172 | 10.293 | 8.668 |

- **Triton fused parameter aggregation** is the fastest of everything here from 8 to 512 tokens, the plain MoE and transformers included (top-1, 32 tokens: 0.36 vs 0.59 / 0.47). Decode is memory-bound. The kernel reads $W_b$ and $\Delta_e$ once each, writes no weights, and does a single GEMM's FLOPs; all pairs go through one launch per projection, and nothing synchronises the host.
- Three things mattered in the kernel:
  - **Masked delta load instead of a branch.** The zero expert's delta load is masked off rather than skipped with an `if`, so Triton can pipeline the loop. At 32 tokens: 0.57 → 0.48 ms.
  - **Summed weight as operand A.** The summed weight is the A operand, computing $y^T = (W_b+\Delta)\,x^T$; Hopper's wgmma can take A from registers. At 8192 tokens: 4.55 → 3.67 ms.
  - **No host sync for counts.** Group counts come from `scatter_add`, not `bincount`, and with few pairs the grid is sized from the pair count, not the largest group, so nothing synchronises the host. At 32 tokens: 0.48 → 0.36 ms.
- With many tokens the work is compute-bound, and this simple Triton GEMM loses to cuBLAS (8192 tokens: 3.66 vs 2.41 plain grouped). So `forward_param` switches to the materialised version past 4096 pairs; a better GEMM kernel would be needed to close that gap.
- **Per-matrix output aggregation** needs no custom kernel. It is about 1.3× the plain MoE at small token counts and about 2× at large ones, since it computes both $W_b$ and $\Delta_e$.
- **Grouped GEMM vs loop**: from 32 to 512 tokens, the grouped versions are 1.3–2.3× faster than the per-expert loops; at a single token the sort and count cost more than they save.

The delta form has uses beyond speed: $\Delta_e$ is a standalone tensor that can be compressed, sparsified or trained on its own; the zero expert has no parameters; and in per-matrix output aggregation the routing weight can stay in the computation graph.

## Why it looked "faster than the original MoE" in transformers

Running the full model with transformers (Qwen3.5-4B MoE, 6 experts, H200), batch-32 decode measured 338 tok/s for the original MoE, while both forms here reached 711 / 860 tok/s. That is **not** because these forms are faster. It's because the original MoE was given a slow path by transformers:

1. Qwen3.5 MoE experts default to `grouped_mm` (`torch._grouped_mm`, one cutlass grouped GEMM for all experts). A single layer takes 0.47 ms at 32 tokens.
2. But `generate()` uses `_optimize_model_for_decode` to automatically **switch every `grouped_mm` to `batched_mm` during decode** (transformers 5.16, `generation/utils.py`).
3. `batched_mm` gathers **one full copy of the expert weights per token** (`self.gate_up_proj[expert_ids]`), then runs `bmm`. At batch 1 that's about the same as reading the weights directly. At batch 32, every layer copies 32 expert-sized matrices (about 4.5 GB). The profile shows 90% of the time in `aten::index` (gather) and `aten::bmm`.
4. The forms here don't go through the experts module, so they're never switched.

Disable the switch and the original MoE becomes the fastest:

| Batch-32 decode (full model, ctx 512) | tok/s | ms/step |
|---|---:|---:|
| Original MoE (with transformers' switch to `batched_mm`) | 338 | 94.7 |
| Original MoE (stays on `grouped_mm`) | **867** | 36.9 |
| Parameter aggregation (materialised, pure PyTorch) | 711 | 45.0 |
| Per-matrix output aggregation (loop, pure PyTorch) | 860 | 37.2 |

How to disable it (batch 1 gets slightly slower, 31 → 28 tok/s):

```python
import contextlib
from transformers.generation.utils import GenerationMixin
GenerationMixin._optimize_model_for_decode = contextlib.contextmanager(lambda self: (yield))
```

A better fix than disabling it outright is to use `batched_mm` only when there are few tokens. Full-model decode (H200, context 512, ms per step):

| batch | transformers default (always `batched_mm` in decode) | always `grouped_mm` | auto, threshold 8 |
|---:|---:|---:|---:|
| 1 | 31.7 | 38.2 | 33.3 |
| 2 | 32.2 | 39.4 | 34.0 |
| 4 | 31.8 | 39.6 | 34.0 |
| 8 | 33.1 | 38.9 | 34.4 |
| 16 | 52.9 | 39.8 | 36.7 |
| 32 | 94.3 | 39.1 | 37.5 |

`grouped_mm` is flat at about 39 ms (bound by framework overhead, not memory traffic); `batched_mm` is faster up to batch 8, then grows with the number of weight copies. Registering an `auto` implementation (`batched_mm` when tokens × top_k ≤ 8, otherwise `grouped_mm`; prefill naturally takes `grouped_mm`) stays within about 2 ms of the best at every batch size. Between-run noise is about ±2 ms, so 4 and 8 are indistinguishable as thresholds.

```python
import contextlib
import transformers.integrations.moe as moe
from transformers.generation.utils import GenerationMixin

def auto_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    f = moe.batched_mm_experts_forward if top_k_index.numel() <= 8 else moe.grouped_mm_experts_forward
    return f(self, hidden_states, top_k_index, top_k_weights)

moe.ExpertsInterface.register("auto", auto_experts_forward)
GenerationMixin._optimize_model_for_decode = contextlib.contextmanager(lambda self: (yield))
model.set_experts_implementation("auto")
```

The takeaway: with large experts (9216 × 2560) and batch > 1, transformers' "switch to `batched_mm` during decode" heuristic makes things slower.
