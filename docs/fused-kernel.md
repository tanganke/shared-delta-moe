# Fused Triton kernel

`triton_kernels/fused.py`, behind `forward_param` (inference):

- **Three launches, no host sync.** A one-program counting sort of the (token, slot) pairs by expert group (the routing itself is `DeltaMoE.topk` in torch, so every forward picks the same experts, ties included). Then gate/up with SiLU·up in the epilogue, writing $h$ in sorted order. Then down, scaled by the routing weight and written to the pair's slot. The kernels are persistent and derive the tile list from the group starts on the device, so a whole forward can be captured in a CUDA graph.
- **CUDA graph / `torch.compile`.** Eager calls launch directly. Under `torch.compile` the forward is one opaque custom op (`shared_delta_moe::fused_moe_dense` / `_dare`), so `fullgraph=True` and `mode="reduce-overhead"` work.
- **Tile order.** Grouped (GROUP_M) so programs running together share weight tiles in L2.
- **Split-K for decode.** When there are too few output tiles to occupy the SMs (one or two tokens), down splits K and adds float32 partials atomically. Dense deltas only.
- **Hopper.** From 256 pairs on sm90+ with dense deltas, weight tiles come in through TMA (host tensor descriptors). A 3-D descriptor over the deltas makes the zero expert's reads land out of bounds, so they read as zeros without a mask. Below 256 pairs, plain pointer loads are faster, because building descriptors costs more than TMA saves. DARE stays on pointer loads: with TMA for $W_b$ alone it was 2.3× slower (2048 tokens: 5.95 vs 2.63 ms). Two Triton 3.7.1 problems: `tl.range(warp_specialize=True)` fails to compile (NVGPUWarpSpecialization pass), and a 256-wide TMA box gives wrong results; configs with BN 256 are left out under TMA.
- **Rounding like the model.** The epilogues round $g$, $u$, SiLU($g$), $h$, $y$ and the routing weight to bf16 wherever the PyTorch MLP does (cuBLAS accumulation order aside, it matches `_forward_param` to about 3e-4 norm-relative per layer). An fp32 epilogue is closer to exact but drifts from the numerics the model was trained with: on a 32-layer Qwen3.5 its argmax agreement with the original MoE was 98.4%, against 98.7% for the rounded epilogue and the previous kernel (`tests/test_fused.py::test_rounds_like_the_model`).
- **Offsets.** Row offsets are int64, which is safe past 2^31 elements (e.g. 150k pairs × inter 16384).
- **Prefill.** From 640 pairs per group on (dense deltas, sm90), `fused_moe` runs materialised $W_b + \Delta$ and grouped GEMMs instead, which are faster there: [dense-prefill.md](dense-prefill.md).
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

- Against the previous single-projection kernel (5522b23, [speed-per-implementation.md](speed-per-implementation.md)), H200 top-1: 1 token 0.336 → 0.177 ms eager, 0.095 in a graph; 32 tokens 0.359 → 0.273, 0.258 in a graph. Ada 1 token: 0.538 → 0.434. The graph numbers show how much of decode is launch overhead.
- DARE (bitmap, see [dare.md](dare.md)) is faster than dense deltas on the Ada up to 32 tokens, because it reads 20% of the delta bytes; on the H200 the gather of kept values keeps it about 2x slower than dense.
