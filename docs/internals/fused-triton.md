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
- **Autotuning.** Keyed on the shapes and the pair count rounded up to a power of 4 (16, 64, 256, ...); results are
  kept in Triton's cache directory (`TRITON_CACHE_DIR`), so only the first run of a shape tunes.
- **Environment.** `SHARED_DELTA_MOE_AUTOTUNE=0` takes the first fitting config (fast tests). `SHARED_DELTA_MOE_TMA=0` forces pointer loads on Hopper.


## DARE (delta mode 1)

A row's bitmap of up to 8 words (blocks of 64: 2 to 5 words) is loaded once per tile into registers, so
the K loop only does bit arithmetic before gathering the kept values; longer rows read their word per tile.

## BitDelta (delta mode 2)

`forward_param` reads the sign bits directly (`triton_kernels/fused.py`, `Delta(2, bits, scales, block=32)`).

- **Loads.** Each K tile loads BK / 32 words per row and expands them to $\pm s$ in registers. One scale per row is
  loaded once per tile (kernel block 0); block scales are read per K tile, one per row when SB >= BK, else BK / SB per
  row, each spread over its block.
- **Rounding.** $\pm s$ is exact in the model dtype, so $W_b + \Delta$ rounds as in the materialised forward.
- **What carries over.** Like dense deltas, it supports split-K for decode, the persistent tile order, CUDA graphs
  and `torch.compile` (custom op `shared_delta_moe::fused_moe_bit`).
- **No TMA.** The sign words are not loaded through TMA.


## Group grids (delta mode 3)

`forward_param` of `GridDeltaMoE` / `GridHierDeltaMoE` reads the 2-bit codes directly (`Delta(3, codes, steps, lows,
block=g)`; [group grids](../guide/grid.md)): 2-bit codes (3 or 4 levels) and a power-of-2 group $g$ only, others use the
materialised forward.

- **Loads.** Each K tile loads BK / 16 words per row and expands entry $c$ to $q = (w \gg 2 (c \bmod 16))\,\&\,3$,
  then $\hat\delta = \ell + s q$ with the step $s$ and lowest level $\ell$ of group $\lfloor c / g \rfloor$: one per row
  when $g \ge$ BK, else BK / $g$ per row, each spread over its group. Symmetric grids pass
  $\ell = -s (L - 1) / 2$, built per call.
- **Rounding.** $\ell + s q$ is computed in float32 and rounded once to the model dtype, as the materialised forward's
  `addcmul`; then $W_b + \hat\delta$ in the model dtype.
- **What carries over.** Split-K for decode, the persistent tile order, CUDA graphs and `torch.compile` (custom op
  `shared_delta_moe::fused_moe_grid`); no TMA. Forward only: with a gradient the materialised path runs.
- **Routed linear layers.** `RoutedDeltaLinear` ([per-expert attention deltas](../guide/routed-deltas.md)) reuses the
  down kernel for its deltas alone, $\sum_i r_i \Delta_{e_i} x$, over a stride-0 zero base (every base load is the
  same cached zero), and adds $W_b x$ from cuBLAS.
- **Tests.** `tests/test_grid_kernel.py` (GPU): against the materialised forward for flat and two-level routing,
  $k_1$ wider than a group, symmetric grids and groups 32-128, every config and split-K, CUDA graphs and
  `torch.compile`.
