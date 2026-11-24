# shared-delta-moe

A minimal implementation (one file, torch only) of a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, showing two forward forms that are exactly equivalent to the original MoE.

Each routed expert's weights are $W_e = W_b + \Delta_e$, applied to gate, up and down separately. The router's first n rows are the deltas; its last `num_zero_experts` rows (Z, default 1) are **zero experts**: they have no parameters and each stands for the base MLP itself. Whatever Z is, base + n deltas has exactly as many parameters as the original MoE's n+1 distinct experts. With Z = 0 the base is only a shared component, never chosen on its own; with Z > 1 the original MoE is equivalent to one with Z copies of the base expert (which gives the router more room to send tokens to the base).

Routing is softmax, top-k (`top_k`, default 1), renormalised over the k picks to weights $r_i$. The target is the original MoE:

$$y = \sum_{i=1}^{k} r_i\,\mathrm{MLP}(x;\,W_b + \Delta_{e_i})$$

| Public method | Computation | Implementations (private), picked automatically |
|---|---|---|
| `forward_param` | Parameter aggregation: per expert, materialise $W_b + \Delta_e$ once and run one MLP on the (token, slot) pairs routed to it; multiply by $r_i$ and `index_add` back. Experts with fewer than 256 routed rows skip materialisation and compute $W_b x + \Delta_e x$ per projection directly | `_forward_param(x, min_rows)` |
| `forward_matrix` | Per-matrix output aggregation: gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `_forward_matrix_grouped`: flattens all (token, slot) pairs, sorts them by expert, one `F.grouped_mm` per projection for the deltas; `_forward_matrix`: loop over experts |
| `forward_dense(x, experts)` | Reference: plain MoE from stored full expert weights | `_forward_dense_grouped`: one `F.grouped_mm` per projection, like transformers' `grouped_mm` path; `_forward_dense`: loop over experts |

Selection rule: use grouped GEMM when `F.grouped_mm` exists, the dtype is bf16/fp16/fp32, the device is CPU or a CUDA GPU with sm80+, and there is more than one (token, slot) pair (for a single pair, sorting and counting cost more than the loop saves). Otherwise use the loop. The threshold was measured on an H200 only.

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

`transformers` is the stock `Qwen3_5MoeSparseMoeBlock` (`grouped_mm`); the other columns are the public methods (automatic selection), with the private implementations in brackets.

top-1:

| tokens | transformers | `forward_dense` (loop / grouped) | `forward_param` (pure param agg) | `forward_matrix` (loop / grouped) |
|---:|---:|---:|---:|---:|
| 1 | 0.360 | 0.271 (0.273 / 0.363) | 0.240 (0.234) | 0.250 (0.231 / 0.326) |
| 32 | 0.472 | 0.577 (0.817 / 0.578) | 1.056 (1.285) | 0.623 (0.932 / 0.618) |
| 512 | 0.494 | 0.576 (0.962 / 0.578) | 1.207 (1.380) | 0.745 (1.007 / 0.744) |
| 8192 | 2.719 | 2.462 (2.720 / 2.441) | 3.241 (3.317) | 5.171 (5.651 / 5.106) |

top-2:

| tokens | transformers | `forward_dense` (loop / grouped) | `forward_param` (pure param agg) | `forward_matrix` (loop / grouped) |
|---:|---:|---:|---:|---:|
| 1 | 0.405 | 0.396 (0.377 / 0.392) | 0.405 (0.422) | 0.459 (0.456 / 0.437) |
| 32 | 0.498 | 0.601 (0.848 / 0.600) | 1.083 (1.309) | 0.635 (1.488 / 0.632) |
| 512 | 0.591 | 0.668 (0.949 / 0.664) | 1.281 (1.428) | 0.920 (1.835 / 0.920) |
| 8192 | 4.732 | 4.506 (4.569 / 4.431) | 5.223 (5.219) | 8.664 (10.239 / 8.773) |

- **Grouped GEMM** replaces the per-expert Python loop with one kernel per projection. From 32 to 512 tokens it is 1.3–2.3× faster than the loop, and top-2 gains the most, because the loop runs every expert of every slot separately. At 1 token, the extra sort and count make it slightly slower.
- **Per-matrix output aggregation, grouped**: 1.3× the transformers MoE at 32 tokens (top-1 0.61 vs 0.47, top-2 0.63 vs 0.50), and about 1.9× at 8192 tokens (computing both $W_b$ and $\Delta_e$ roughly doubles the compute).
- **Parameter aggregation** writes out a full weight matrix per expert, so it is the slowest at mid-range token counts. The hybrid that `forward_param` uses (small groups skip materialisation) only shaves this down (top-1 32 tokens 1.35 → 1.17 ms, top-2 1.33 → 1.10): each small group still reads $W_b$ once, so it can't beat per-matrix output aggregation, which shares $W_b x$ across all tokens (0.86 / 0.64). It does not fit grouped GEMM well: you would first have to write out $W_b+\Delta_e$ for every expert, which costs even more memory traffic.
- The remaining gap between plain grouped and transformers (0.58 vs 0.47) probably comes from the `nonzero` and `bincount` here, which synchronise the CPU and GPU (transformers uses `histc` and a fixed number of groups, which needs no sync). Not profiled.

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
