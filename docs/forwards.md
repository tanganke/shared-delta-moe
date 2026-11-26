# Forward implementations

Notation ($W_b$, $\Delta_e$, slots, routing weights $r_i$, zero experts): [design.md](design.md).

| Public method | Computation | Implementations (private), picked automatically |
|---|---|---|
| `forward_param` | Parameter aggregation: each (token, slot) pair is multiplied by $W_b + \Delta_e$ of its expert, weighted by $r_i$ | `_forward_param_triton`: three Triton launches for all pairs (sort; gate/up with SiLU·up; down with the routing weight), adding $W_b$ and $\Delta_e$ tiles in registers (no weights written, 1× FLOPs); `_forward_param(x, min_rows)`: materialise $W_b + \Delta_e$ per expert and run cuBLAS (experts with fewer than `min_rows` rows compute $W_b x + \Delta_e x$ instead) |
| `forward_matrix` | Per-matrix output aggregation: gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `_forward_matrix_grouped`: flattens all (token, slot) pairs, sorts them by expert, one `F.grouped_mm` per projection for the deltas; `_forward_matrix`: loop over experts |
| `forward_dense(x, experts)` | Reference: plain MoE from stored full expert weights | `_forward_dense_grouped`: one `F.grouped_mm` per projection, like transformers' `grouped_mm` path; `_forward_dense`: loop over experts |

Selection rules (thresholds measured on an H200 and an RTX 6000 Ada at hidden 2560, inter 9216, 5 deltas + zero expert; retune for other GPUs or shapes):

- The module itself (`moe(x)`, `x` of any leading shape; what the Qwen3.5 blocks run): `forward_param`, except that `DareDeltaMoE` / `BitDeltaMoE` take `forward_matrix` on the CUDA kernels up to `MATRIX_FORWARD_PAIRS` (token, slot) pairs (DARE 256, BitDelta 64; H200 crossovers in the tables below) when those kernels apply. The two forwards round differently (about 1e-3 relative), so with packed deltas a batch's output depends slightly on its size.
- `forward_param`: Triton on CUDA bf16/fp16 without gradients, up to 8192 (token, slot) pairs on Hopper (sm90+) or 16384 on Ada and older. Beyond the limit the work is compute-bound, and cuBLAS on materialised weights wins; on Hopper that happens earlier, because cuBLAS's wgmma kernels are better than Triton's. With gradients (training) it takes the materialised path, or the fused Triton backward when `triton_train` is set (dense deltas); see [training.md](training.md). Triton (with or without gradients) also needs the routing sort to fit (`SORT_ONE_HOT_MAX`). The kernel itself: [fused-kernel.md](fused-kernel.md).
- `forward_matrix` of `DareDeltaMoE` / `BitDeltaMoE`: the CUDA kernels below when they apply. Otherwise, and for the other classes:
- `forward_matrix` / `forward_dense`: grouped GEMM when `F.grouped_mm` exists, the dtype is bf16/fp16/fp32, the device is CPU or a CUDA GPU with sm80+, and there is more than one pair; otherwise the loop.

## Output aggregation on the CUDA kernels (packed deltas)

`forward_matrix` of `DareDeltaMoE` / `BitDeltaMoE` runs on prebuilt CUDA kernels (`cuda_kernels/packed_moe.cu`,
`packed_ops.matrix_moe`; build as in [dare.md](dare.md#cuda-kernels-at-decode)) when the library is loaded, the GPU is
sm_90, no gradient is needed and there are at most `MATRIX_MAX_PAIRS` (1024) (token, slot) pairs; otherwise the
inherited forward runs on unpacked deltas. cuBLAS computes $x W_g^T$ and $x W_u^T$ for all tokens and $h W_d^T$ for
all pairs, so $W_b$ is read once however many groups are routed. Two kernels add only the deltas' products: gate/up
with SiLU·up per pair, then down with the routing weight; the slots are summed in torch. The parameter-aggregation
kernels instead rebuild $W_b + \Delta_g$ for every routed group.

- DARE: below about 5 pairs per group, a warp takes one row on CUDA cores. It writes the row's kept block indices to
  shared memory once, then streams the kept values with the activations they meet. From about 5 pairs per group, the
  tensor-core kernel below runs, with zeros for the dropped blocks and a running popcount to find the kept values.
- BitDelta: a CTA takes 16 rows and its warps split K, with the pairs as the N side of `mma.m16n8k16`. A byte of sign
  bits indexes a table of sign-flip masks XORed into the scale, as in the parameter-aggregation kernel.

Rounding is that of `_forward_matrix_grouped`: $x W_b^T$ and $x \Delta^T$ are each rounded to the model dtype before
their sum, and so are $h$ and $r_i y_i$. `tests/test_packed_matrix.py` checks these kernels against `_forward_matrix`
on the unpacked deltas. It covers DARE blocks 8 to 128, BitDelta matrix, row and block scales, zero experts, top-1 to
top-3, 16 deltas, ragged token counts up to 400 pairs, fp16, a CUDA graph and `torch.compile` (fullgraph, through the
ops `shared_delta_moe::matrix_moe_dare` / `_bit`). The models' default forward is unchanged.

H200, µs per layer, device time of every kernel after the router: hidden 2560, inter 9216, + zero expert, DARE 0.2 /
block 64, BitDelta matrix scales. "param" is `forward_param` (the CUDA kernels up to 16 pairs, then Triton); "matrix"
is `forward_matrix` with these kernels. Dense deltas use `forward_param` (Triton, the fastest dense path).
"bound" is the byte lower bound at 4.8 TB/s: $W_b$ once, each touched delta in its stored form, and the activations.

5 deltas, top-1:

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 80.4 | 66.4 | **55.7** | 61.5 | **54.2** | 29.5 / 29.5 |
| 4 | 122.7 | 329.0 | **105.4** | 158.0 | **110.1** | 47.4 / 33.2 |
| 8 | 194.8 | 464.4 | **159.9** | 199.5 | **138.5** | 53.3 / 35.1 |
| 16 | 191.1 | 585.0 | 210.7 | 320.1 | 192.4 | 59.3 / 38.9 |
| 64 | 234.1 | 688.7 | 312.0 | 363.2 | 311.8 | 59.8 / 39.3 |
| 512 | 344.9 | 931.4 | 1668.0 | 548.8 | 1632.2 | 64.2 / 43.7 |

(At 1 token, top-1, the token goes to the zero expert in this sample, so the param times are low; with a delta routed, DARE param is about 124 µs.)

5 deltas, top-2:

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 102.4 | 230.8 | **89.4** | 109.3 | **91.3** | 41.4 / 31.4 |
| 4 | 162.4 | 526.4 | 182.7 | 251.4 | 169.7 | 59.3 / 36.9 |
| 8 | 234.8 | 597.5 | **218.0** | 324.8 | **194.9** | 59.3 / 38.8 |
| 16 | 236.7 | 644.1 | 283.4 | 369.6 | **209.1** | 59.5 / 39.0 |
| 64 | 255.4 | 693.6 | 502.3 | 380.8 | 503.2 | 60.3 / 39.8 |
| 512 | 590.0 | 1484.9 | 3074.5 | 860.1 | 3100.2 | 68.1 / 47.7 |

15 deltas, top-3 (many slots per token, like a HierDeltaMoE with three groups):

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 156.1 | 332.4 | **113.9** | 179.7 | **138.2** | 47.3 / 35.0 |
| 4 | 437.9 | 943.9 | **268.2** | 448.7 | **275.3** | 77.1 / 44.3 |
| 8 | 566.3 | 1514.1 | **403.0** | 702.3 | **358.6** | 106.9 / 50.0 |
| 16 | 596.2 | 1632.6 | **538.8** | 877.6 | **461.9** | 119.0 / 57.5 |
| 64 | 642.9 | 1636.5 | 909.7 | 950.7 | 800.8 | 120.2 / 58.8 |

Bold marks the packed forms that beat dense deltas. Output aggregation is faster than parameter aggregation for
both formats from 1 to 64 tokens. It also beats dense deltas up to about 8–16 tokens, and up to 16 with many slots
per token. Of the time at 1 token, about 43 µs is the three cuBLAS GEMVs on $W_b$ (bound 29.5 µs). The kernels add
about 12–22 µs.

What limits it, and what did not work:

- From 16 tokens up, the delta kernels set the time. The CUDA-core DARE kernel is bound by instructions (one FMA per
  kept value per pair). The tensor-core kernels wait on load latency: each 32-wide chunk loads its signs (or bitmap,
  then values), and 8 pairs at most share them. At 512 tokens output aggregation is 2–5× slower than `forward_param`.
- Reusing the parameter-aggregation kernel with the $W_b$ ring removed (deltas only, epilogue adding $x W_b^T$) was
  slower than this at every size (DARE gate/up 36 µs at 1 token, 517 µs at 64 tokens). Its per-step overhead does not
  shrink with the bytes.
- BitDelta on CUDA cores (each lane adds or subtracts 32 activations per sign word) was instruction-bound already at
  4 tokens: 1.5 ms at 64 tokens, against 312 µs on the tensor cores.
