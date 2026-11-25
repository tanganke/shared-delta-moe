# shared-delta-moe

A small Python library: a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, with several forward implementations that are exactly equivalent to the original MoE, including a Triton kernel that fuses $W_b+\Delta_e$ into the GEMM.

```
src/shared_delta_moe/models/delta_moe.py  DeltaMoE: from_experts, routing, the forward implementations
src/shared_delta_moe/ops.py               stateless helpers: mlp, grouping routed rows, one grouped GEMM
src/shared_delta_moe/triton_kernels/      fused.py: the fused parameter-aggregation forward (sort, gate_up + SiLU,
                                          down + routing weight; dense or seeded-DARE deltas; Hopper TMA)
src/shared_delta_moe/models/dare.py       seeded DARE: Philox mask, its bitmap, pack, DareDeltaMoE
src/shared_delta_moe/models/qwen3_5/qwen3_5_delta_moe/
                                          transformers' Qwen3.5 MoE with DeltaMoE blocks (forward_param): modeling.py
                                          (patches the decoder layer for configs with delta_moe_zero_experts), convert.py
                                          (plain MoE checkpoint -> delta layout)
src/shared_delta_moe/constants.py         HAS_TRITON and the thresholds that pick an implementation
src/shared_delta_moe/type_hints.py        shape-annotated tensor aliases (T tokens, H hidden, ...)
src/shared_delta_moe/testing.py           random_experts for tests and benchmarks
tests/test_delta_moe.py                   every forward == plain MoE, token by token (Triton part needs CUDA)
tests/test_dare.py                        Philox known-answer, CPU == CUDA masks, bitmap == mask, DARE forwards and gradients
tests/test_qwen3_5_delta_moe.py           tiny Qwen3.5 MoE: converted == stock MoE at top-1, top-2, soft routing
tests/test_fused.py                       the fused kernels: sort, every autotune config, split-K, int64 offsets,
                                          CUDA graph, torch.compile
benchmarks/bench.py                       per-layer time of each implementation (CUDA)
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
| `forward_param` | Parameter aggregation: each (token, slot) pair is multiplied by $W_b + \Delta_e$ of its expert, weighted by $r_i$ | `_forward_param_triton`: three Triton launches for all pairs (sort; gate/up with SiLU·up; down with the routing weight), adding $W_b$ and $\Delta_e$ tiles in registers (no weights written, 1× FLOPs); `_forward_param(x, min_rows)`: materialise $W_b + \Delta_e$ per expert and run cuBLAS (experts with fewer than `min_rows` rows compute $W_b x + \Delta_e x$ instead) |
| `forward_matrix` | Per-matrix output aggregation: gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `_forward_matrix_grouped`: flattens all (token, slot) pairs, sorts them by expert, one `F.grouped_mm` per projection for the deltas; `_forward_matrix`: loop over experts |
| `forward_dense(x, experts)` | Reference: plain MoE from stored full expert weights | `_forward_dense_grouped`: one `F.grouped_mm` per projection, like transformers' `grouped_mm` path; `_forward_dense`: loop over experts |

Selection rules (thresholds measured on an H200 and an RTX 6000 Ada at the shape below; retune for other GPUs or shapes):

- `forward_param`: Triton on CUDA bf16/fp16 without gradients, up to 8192 (token, slot) pairs on Hopper (sm90+) or 16384 on Ada and older. Beyond the limit the work is compute-bound, and cuBLAS on materialised weights wins; on Hopper that happens earlier, because cuBLAS's wgmma kernels are better than Triton's. With gradients (training) it always takes the materialised path; see [Training](#training).
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

Apart from the Triton kernels, everything is `F.linear` (cuBLAS), `F.grouped_mm` (PyTorch's own grouped GEMM: CUDA sm80+ or CPU, bf16/fp16/fp32) and `index_add`. bf16/fp16 are the targets; fp32 is supported (and used for the CPU tests), fp64 is not.

## Qwen3.5 in transformers

```bash
python -m shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe.convert PLAIN_MOE OUT [--base INDEX] [--top-k K]
```

```python
import shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe  # patches Qwen3_5MoeDecoderLayer
model = AutoModelForImageTextToText.from_pretrained(OUT, num_experts_per_tok=2)  # or AutoModelForCausalLM
```

The plain MoE's expert BASE becomes $W_b$ and the zero expert; the others become $\Delta_e$. Its shared expert must
have no output. `num_experts_per_tok` is the top-k at load time (= `num_experts` for soft routing over all experts).

## Seeded DARE deltas

`DareDeltaMoE.from_delta_moe(m, keep=0.2, block=64, seed=0)` keeps a random `keep` of each $\Delta_e$ (whole
$1 \times$ `block` blocks along the input dimension; `block=1` is element-wise DARE), rescaled by $1/\text{keep}$, and
stores only the kept values plus one int64 row start per weight row. The mask is not stored: it is Philox4x32-10 of
(unit, 3e + projection) under the seed, pure 32-bit integer arithmetic, so it can be regenerated anywhere
(`tests/test_dare.py` checks a Random123 known-answer vector and CPU == CUDA). The fused kernel reads it as a bitmap,
1 bit per block (`mask_bits`), built once from the seed and kept as a non-persistent buffer (`m_gate` / `m_up` /
`m_down`, rebuilt after loading): 0.7 MB per layer at block 64, 44 MB element-wise, against 142 MB of kept values at
keep 0.2. A row's bitmap of up to 8 words (blocks of 64: 2 to 5 words) is loaded once per tile into registers, so
the K loop only does bit arithmetic before gathering the kept values; longer rows read their word per tile. Storage
per delta: `keep` of dense, e.g. 20%, plus the bitmap.

The bitmap replaced Philox in the kernel (one Philox per block per tile). ms per layer, top-1, 5 deltas + zero
expert, hidden 2560, inter 9216:

| tokens | dense $\Delta$ | block 64, Philox | **block 64, bitmap** | element-wise, Philox | **element-wise, bitmap** |
|---:|---:|---:|---:|---:|---:|
| RTX 6000 Ada, 8 | 0.69 | 0.50 | **0.44** | 4.79 | **0.61** |
| RTX 6000 Ada, 32 | 1.02 | 0.64 | **0.54** | 7.08 | **0.92** |
| RTX 6000 Ada, 512 | 1.15 | 1.35 | **1.28** | 6.05 | **1.67** |
| RTX 6000 Ada, 2048 | 2.96 | 4.03 | **3.55** | 18.9 | **5.0** |
| H200, 8 | 0.231 | 0.739 | **0.488** | | 0.693 |
| H200, 32 | 0.267 | 0.852 | **0.676** | | 0.932 |
| H200, 512 | 0.384 | 1.153 | **0.982** | | 1.379 |
| H200, 8192 | 3.43 | 8.85 | **7.64** | | 10.59 |

Element-wise DARE was compute-bound under Philox (one per element, every expert group recomputing its mask each
call); with the bitmap it is within 1.5x of blocks. Reading the bitmap per tile instead of from registers made block 64
on the Ada slower than Philox at decode (8 tokens 0.78 ms): the mask load and the gather that depends on it could not
be pipelined. On the H200 DARE stays about 2x slower than dense deltas: the gather of kept values, not the mask, is
what is left. Accuracy of block DARE on the real model:
ntu-projects `moe-indomain-acc` log [35] (blocks 4 to 256 at 20% match element-wise DARE).

## Fused Triton kernel

`triton_kernels/fused.py`, behind `forward_param` (inference):

- **Three launches, no host sync.** A one-program counting sort of the (token, slot) pairs by expert group (the routing itself is `DeltaMoE.topk` in torch, so every forward picks the same experts, ties included). Then gate/up with SiLU·up in the epilogue, writing $h$ in sorted order. Then down, scaled by the routing weight and written to the pair's slot. The kernels are persistent and derive the tile list from the group starts on the device, so a whole forward can be captured in a CUDA graph.
- **CUDA graph / `torch.compile`.** Eager calls launch directly. Under `torch.compile` the forward is one opaque custom op (`shared_delta_moe::fused_moe_dense` / `_dare`), so `fullgraph=True` and `mode="reduce-overhead"` work.
- **Tile order.** Grouped (GROUP_M) so programs running together share weight tiles in L2.
- **Split-K for decode.** When there are too few output tiles to occupy the SMs (one or two tokens), down splits K and adds float32 partials atomically. Dense deltas only.
- **Hopper.** From 256 pairs on sm90+ with dense deltas, weight tiles come in through TMA (host tensor descriptors). A 3-D descriptor over the deltas makes the zero expert's reads land out of bounds, so they read as zeros without a mask. Below 256 pairs, plain pointer loads are faster, because building descriptors costs more than TMA saves. DARE stays on pointer loads: with TMA for $W_b$ alone it was 2.3× slower (2048 tokens: 5.95 vs 2.63 ms). Two Triton 3.7.1 problems: `tl.range(warp_specialize=True)` fails to compile (NVGPUWarpSpecialization pass), and a 256-wide TMA box gives wrong results; configs with BN 256 are left out under TMA.
- **Rounding like the model.** The epilogues round $g$, $u$, SiLU($g$), $h$, $y$ and the routing weight to bf16 wherever the PyTorch MLP does (cuBLAS accumulation order aside, it matches `_forward_param` to about 3e-4 norm-relative per layer). An fp32 epilogue is closer to exact but drifts from the numerics the model was trained with: on a 32-layer Qwen3.5 its argmax agreement with the original MoE was 98.4%, against 98.7% for the rounded epilogue and the previous kernel (`tests/test_fused.py::test_rounds_like_the_model`).
- **Offsets.** Row offsets are int64, which is safe past 2^31 elements (e.g. 150k pairs × inter 16384).
- **Environment.** `SHARED_DELTA_MOE_AUTOTUNE=0` takes the first fitting config (fast tests). `SHARED_DELTA_MOE_TMA=0` forces pointer loads on Hopper.

H200, top-1 / top-2, ms per layer (hidden 2560, inter 9216, 5 deltas + zero expert; "graph" = CUDA graph replay):

| tokens | plain grouped | param (materialised) | **param Triton** | Triton, graph | DARE 0.2 block 64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.359 / 0.396 | 0.303 / 0.526 | 0.177 / 0.190 | **0.095 / 0.137** | 0.256 / 0.341 |
| 8 | 0.546 / 0.552 | 1.058 / 1.083 | 0.231 / 0.244 | **0.215 / 0.223** | 0.488 / 0.688 |
| 32 | 0.566 / 0.603 | 1.287 / 1.325 | 0.273 / 0.298 | **0.258 / 0.278** | 0.676 / 0.695 |
| 512 | 0.566 / 0.674 | 1.339 / 1.444 | 0.378 / 0.641 | **0.358 / 0.609** | 0.982 / 1.541 |
| 2048 | 0.972 / **1.332** | 1.641 / 2.220 | 0.970 / 1.822 | **0.929** / 1.749 | 2.298 / 4.220 |
| 8192 | **2.450 / 4.289** | 3.301 / 5.247 | 3.306 / 6.780 | 3.168 / 6.263 | 7.643 / 14.708 |

RTX 6000 Ada, top-1 / top-2:

| tokens | plain grouped | param (materialised) | **param Triton** | Triton, graph | DARE 0.2 block 64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.736 / 0.822 | 0.916 / 1.057 | 0.434 / 0.469 | **0.355 / 0.357** | 0.412 / 0.329 |
| 8 | 1.067 / 1.245 | 3.009 / 3.838 | 0.844 / 1.014 | 0.822 / 0.990 | **0.446 / 0.556** |
| 32 | 1.432 / 1.425 | 4.058 / 4.085 | 1.014 / 1.020 | 0.992 / 1.004 | **0.545 / 0.543** |
| 512 | 1.629 / 1.795 | 4.257 / 4.363 | 1.152 / 1.613 | **1.136** / 1.699 | 1.270 / 2.170 |
| 2048 | **2.786** / 5.504 | 5.572 / 8.045 | 2.866 / 5.817 | 3.107 / 5.636 | 3.545 / 7.043 |
| 8192 | 12.522 / 24.998 | 14.386 / 24.907 | 11.345 / 24.637 | **10.526 / 23.902** | 13.167 / 27.388 |

- Against the previous single-projection kernel (5522b23, the per-GPU tables below), H200 top-1: 1 token 0.336 → 0.177 ms eager, 0.095 in a graph; 32 tokens 0.359 → 0.273, 0.258 in a graph. Ada 1 token: 0.538 → 0.434. The graph numbers show how much of decode is launch overhead.
- DARE (bitmap, see [Seeded DARE deltas](#seeded-dare-deltas)) is faster than dense deltas on the Ada up to 32 tokens, because it reads 20% of the delta bytes; on the H200 the gather of kept values keeps it about 2x slower than dense.

### Training

`forward_param` is differentiable: with a gradient needed it runs the materialised path (cuBLAS on $W_b+\Delta_e$, any top-k, gradients to x, the routing weights and every weight). Two fused alternatives were built and measured (commit e0ce27a, then removed): a Triton backward that recomputes $h$ (one kernel computing $g$, $u$ and $dh'$; $dx$ through the down kernel; weight grads via `F.grouped_mm`), and the same Function on `F.grouped_mm` over stacked experts. Both were slower than the materialised path. Router-only training, top-2 / top-6, ms per forward+backward (peak GB above the weights):

| GPU, tokens, top-k | fused Triton | grouped GEMM | **materialised** |
|---|---:|---:|---:|
| H200, 4096, top-2 | 11.66 (0.87) | 8.53 (1.29) | **5.97** (1.19) |
| H200, 25000, top-2 | 64.41 (2.60) | 39.91 (3.85) | **27.93** (4.03) |
| H200, 25000, top-6 | 200.99 (6.52) | 112.32 (9.95) | **79.38** (10.84) |
| Ada, 4096, top-2 | 36.47 (0.87) | 39.43 (1.30) | **27.78** (1.20) |
| Ada, 25000, top-6 | 704.94 (6.52) | 610.53 (9.96) | **478.93** (10.84) |

With few experts (6), the materialisation costs one pass over the weights, and cuBLAS GEMMs on dense experts beat both the Triton GEMMs and `grouped_mm`. The fused backward only saves about 40% of activation memory.

`DareDeltaMoE` trains through the same path: `delta(e)` unpacks only the routed expert from the bitmap (`masked_scatter` of the kept values, differentiable in them), so the kept values can be made `nn.Parameter`s and trained. Unpacking with Philox on every access of `d_gate` / `d_up` / `d_down` (all experts each time) made a step 7.6 s. Whole Qwen3.5-4B text model, full-parameter training, batch 2 x 2048, bf16 weights, fused AdamW (bf16 states), random router (every delta routed: the worst case), H200, peak GiB and s/step:

| model | params / trainable | peak, checkpointing | peak, none | s/step, checkpointing |
|---|---|---:|---:|---:|
| dense Qwen3.5-4B | 4.21B / 4.21B | 35.9 | 65.5 | 0.4 |
| DARE 0.2/64, kept values frozen, top-1 / top-2 | 6.48B / 4.21B | 40.1 / 40.1 | 92.9 / 103.7 | 0.9 / 1.0 |
| DARE 0.2/64, kept values trained | 6.48B / 6.47B | 51.0 | 111.9 | 1.1 |
| dense deltas (`DeltaMoE`) trained | 15.53B / 15.53B | 118.7 | out of memory | 0.8 |

With gradient checkpointing, frozen DARE deltas cost only their own 4.2 GiB. Without it, every materialised $W_b+\Delta_e$ is held for the backward (142 MB per expert and layer).

## Speed (single MoE layer, bf16, eager, ms)

The per-implementation tables below were measured with the previous kernel (5522b23; one launch per projection). Its "param Triton" column is superseded by the tables above.

### H200 (sm90, 4.8 TB/s)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts). For comparison, transformers' stock `Qwen3_5MoeSparseMoeBlock` (`grouped_mm`) at the same size takes 0.36 / 0.47 / 0.49 / 2.72 ms at 1 / 32 / 512 / 8192 tokens (top-1).

top-1:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.251 | 0.351 | 0.235 | 0.227 | 0.336 | 0.232 | 0.323 |
| 8 | 0.578 | 0.488 | 0.947 | 0.758 | **0.358** | 0.620 | 0.574 |
| 32 | 0.818 | 0.592 | 1.282 | 1.050 | **0.359** | 0.779 | 0.606 |
| 512 | 0.880 | 0.566 | 1.345 | 1.146 | **0.557** | 0.982 | 0.743 |
| 2048 | 1.156 | 0.963 | 1.665 | 1.661 | 1.337 | 1.878 | 1.639 |
| 4096 | 1.688 | 1.326 | 2.213 | 2.196 | 1.794 | 3.131 | 2.720 |
| 8192 | 2.730 | 2.415 | 3.320 | 3.242 | 3.595 | 5.680 | 5.140 |

top-2:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.367 | 0.391 | 0.420 | 0.394 | **0.357** | 0.454 | 0.426 |
| 8 | 0.809 | 0.585 | 1.278 | 1.032 | **0.361** | 1.293 | 0.643 |
| 32 | 0.846 | 0.582 | 1.313 | 1.071 | **0.359** | 1.475 | 0.634 |
| 512 | 0.926 | 0.664 | 1.436 | 1.283 | 0.716 | 1.852 | 0.924 |
| 2048 | 1.650 | 1.311 | 2.177 | 2.160 | 1.773 | 3.475 | 2.443 |
| 4096 | 2.631 | 2.406 | 3.169 | 3.127 | 3.568 | 5.737 | 4.509 |
| 8192 | 4.628 | 4.392 | 5.261 | 5.252 | 7.067 | 10.236 | 8.546 |

- **Triton fused parameter aggregation** is the fastest of everything here from 8 to 512 tokens, the plain MoE and transformers included (top-1, 32 tokens: 0.36 vs 0.59 / 0.47). Decode is memory-bound. The kernel reads $W_b$ and $\Delta_e$ once each, writes no weights, and does a single GEMM's FLOPs; all pairs go through one launch per projection, and nothing synchronises the host.
- Three things mattered in the kernel:
  - **Masked delta load instead of a branch.** The zero expert's delta load is masked off rather than skipped with an `if`, so Triton can pipeline the loop. At 32 tokens: 0.57 → 0.48 ms.
  - **Summed weight as operand A.** The summed weight is the A operand, computing $y^T = (W_b+\Delta)\,x^T$; Hopper's wgmma can take A from registers. At 8192 tokens: 4.55 → 3.67 ms.
  - **No host sync for counts.** Group counts come from `scatter_add`, not `bincount`, and with few pairs the grid is sized from the pair count, not the largest group, so nothing synchronises the host. At 32 tokens: 0.48 → 0.36 ms.
- With many tokens the work is compute-bound, and the Triton GEMM loses to cuBLAS (8192 tokens: 3.66 vs 2.41 plain grouped); `forward_param` switches to the materialised version past the pair limit.
- **Per-matrix output aggregation** needs no custom kernel. It is about 1.3× the plain MoE at small token counts and about 2× at large ones, since it computes both $W_b$ and $\Delta_e$.
- **Grouped GEMM vs loop**: from 32 to 512 tokens, the grouped versions are 1.3–2.3× faster than the per-expert loops; at a single token the sort and count cost more than they save.

### RTX 6000 Ada (sm89, 960 GB/s)

top-1:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.363 | 0.663 | 0.808 | 0.509 | 0.538 | 0.530 | 0.815 |
| 8 | 1.650 | 1.357 | 3.935 | 2.402 | **1.065** | 1.600 | 1.379 |
| 32 | 1.657 | 1.376 | 4.024 | 2.454 | **1.078** | 1.580 | 1.413 |
| 512 | 1.859 | 1.591 | 4.225 | 2.823 | **1.244** | 2.368 | 2.095 |
| 2048 | 3.324 | 3.532 | 5.867 | 5.641 | **3.168** | 6.500 | 5.869 |
| 4096 | 6.162 | 6.219 | 8.153 | 8.009 | **5.935** | 12.223 | 11.810 |
| 8192 | 12.373 | 11.415 | 13.407 | 13.302 | 11.713 | 24.865 | 23.370 |

top-2:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.642 | 0.825 | 1.538 | 0.953 | **0.599** | 1.050 | 0.955 |
| 8 | 1.715 | 1.397 | 4.017 | 2.392 | **1.080** | 2.864 | 1.474 |
| 32 | 1.694 | 1.413 | 4.047 | 2.496 | **1.107** | 3.047 | 1.456 |
| 512 | 2.150 | 1.948 | 4.550 | 3.477 | **1.652** | 4.528 | 2.642 |
| 2048 | 5.998 | 6.270 | 8.158 | 7.856 | 6.903 | 11.486 | 9.812 |
| 4096 | 11.951 | 11.859 | 13.340 | 13.054 | **11.816** | 22.354 | 19.629 |
| 8192 | 22.403 | 22.105 | 23.916 | 23.836 | 27.769 | 43.258 | 38.772 |

- Lower bandwidth makes everything 2–4× slower than on the H200, but the ranking holds. From 8 to 512 tokens Triton is 1.2–1.3× faster than the plain MoE.
- Hopper's wgmma advantage is absent here: with many tokens Triton is on par with cuBLAS (top-1 at 4096–8192 tokens), and it beats materialised weights to higher pair counts, hence the higher limit for Ada and older.
- Two Ada-friendly configs (BK 32, 4–5 stages) were added to the autotune list. They took top-1 at 4096 tokens from 6.72 to 5.94 ms. On Hopper, autotuning still picks its own.

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
