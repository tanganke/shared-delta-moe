# BitDelta deltas (1 bit per entry)

`BitDeltaMoE.from_delta_moe(m, scale="matrix")` stores each $\Delta_e$ as its sign, 1 bit per entry, times a scale
(BitDelta, Liu et al. 2024, "BitDelta: Your Fine-Tune May Only Be Worth One Bit"):

$$\Delta_e \approx \mathrm{diag}(s_e)\,\mathrm{sign}(\Delta_e)$$

- `scale="matrix"` (BitDelta): one scale per matrix, $s = \mathrm{mean}\,|\Delta_e|$. For a fixed sign matrix it is
  the scalar closest to $\Delta_e$ in Frobenius norm.
- `scale="row"`: one scale per output row, that row's mean $|\Delta_e|$. It costs R values per matrix.
- `scale=B` (an int, a power of 2 dividing the row): one scale per 1 x B block of a row, its mean $|\Delta_e|$, like
  DARE's blocks. It costs 16 / B extra bits per entry (B = 32: 1.5 bits per entry in all).

On a real model none of the finer scales helps (see Accuracy): the magnitude of a full fine-tune's delta does not
cluster by row or block, so use `"matrix"`.

A zero entry counts as +. The result is a different MoE from the original. Every forward of `BitDeltaMoE` equals the
plain MoE with experts $W_b + \mathrm{diag}(s_e)\,\mathrm{sign}(\Delta_e)$ (`tests/test_bitdelta.py`), not the
original experts.

```python
from shared_delta_moe import BitDeltaMoE, DeltaMoE
b = BitDeltaMoE.from_delta_moe(DeltaMoE.from_experts(base, experts, router, top_k=1), scale="row")
y = b.forward_param(x)
```

## Storage

The sign bits are `b_<proj>`, shape (N, R, C / 32) int32. Entry c of a row is bit c % 32 of word c // 32; a set bit
means +s. The scales are `s_<proj>`, shape (N, R, ceil(C / SB)), in the model dtype: entry c of a
row takes scale c // SB, SB = B for block scales, else the next power of 2 of C (one column). Both are in the state dict.

Rows must have a multiple of 32 entries, so hidden and inter must be multiples of 32.

One delta of a Qwen3.5-4B MoE layer (hidden 2560, inter 9216) takes:

| Format | Size |
|---|---:|
| dense bf16 | 141.6 MB |
| seeded DARE 0.2, block 64 | 28.3 MB of kept values, plus a 0.7 MB bitmap per layer |
| BitDelta | 8.85 MB of sign bits, plus 42 KB of row scales |

## Fused kernel: delta mode 2

`forward_param` reads the sign bits directly (`triton_kernels/fused.py`, `Delta(2, bits, scales, block=32)`).

- **Loads.** Each K tile loads BK / 32 words per row and expands them to $\pm s$ in registers. One scale per row is
  loaded once per tile (kernel block 0); block scales are read per K tile, one per row when SB >= BK, else BK / SB per
  row, each spread over its block.
- **Rounding.** $\pm s$ is exact in the model dtype, so $W_b + \Delta$ rounds as in the materialised forward.
- **What carries over.** Like dense deltas, it supports split-K for decode, the persistent tile order, CUDA graphs
  and `torch.compile` (custom op `shared_delta_moe::fused_moe_bit`).
- **No TMA.** The sign words are not loaded through TMA.

## Training the scales

The scales are `nn.Parameter`s, so BitDelta's scale distillation can be run through the materialised path.

- **What trains.** `delta(e)` unpacks only the routed expert, and $\mathrm{d}L/\mathrm{d}s_{e,r} = \sum_c \mathrm{d}L/\mathrm{d}\Delta_{e,r,c}\,\mathrm{sign}_{e,r,c}$ (checked in the tests). The signs are fixed.
- **No Triton backward.** With a gradient needed, `_use_triton` refuses the fused kernels, including `triton_train`,
  whose backward takes dense deltas only.

## Speed

H200, top-1 / top-2, ms per layer. The shape is hidden 2560, inter 9216, 5 deltas + zero expert; row scales;
"graph" = CUDA graph replay. Measured with `benchmarks/bench.py --only ...` on one
node, other GPUs running the tests.

| tokens | dense grouped (plain MoE) | param Triton, dense $\Delta$ | dense $\Delta$, graph | DARE 0.2 block 64 | **BitDelta** | BitDelta, graph |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.604 / 0.606 | 0.248 / 0.237 | 0.095 / 0.144 | 0.258 / 0.342 | 0.209 / 0.226 | 0.104 / 0.165 |
| 8 | 0.571 / 0.616 | 0.234 / 0.278 | 0.216 / 0.259 | 0.618 / 0.686 | 0.348 / 0.410 | 0.330 / 0.391 |
| 32 | 0.588 / 0.620 | 0.274 / 0.287 | 0.252 / 0.269 | 0.679 / 0.695 | 0.403 / 0.416 | 0.386 / 0.396 |
| 512 | 0.588 / 0.671 | 0.390 / 0.652 | 0.369 / 0.624 | 0.988 / 1.542 | 0.602 / 0.917 | 0.582 / 0.890 |
| 2048 | 0.998 / 1.335 | 0.962 / 1.836 | 0.933 / 1.767 | 2.300 / 4.227 | 1.416 / 2.526 | 1.393 / 2.494 |
| 8192 | 2.426 / 4.227 | 3.499 / 6.600 | 3.287 / 6.356 | 7.885 / 14.620 | 4.790 / 8.691 | 4.785 / 8.675 |

- **Against DARE.** BitDelta is 1.6–1.8x faster than DARE 0.2 / block 64 at every size, and its deltas are a third
  of DARE's bytes.
- **Against dense deltas.** It is about as fast at 1 token: 0.209 vs 0.248 ms eager, 0.104 vs 0.095 in a graph.
  From 8 tokens up it is 1.4–1.5x slower, although it reads 1/16 of the delta bytes. Expanding the bits (shift,
  mask, select per entry) costs more ALU time than the saved loads; the tile is no longer memory-bound.
- **Block scales** (same node, another run; row scales measured alongside, ms, top-1):

  | tokens | dense $\Delta$ | row scales | B = 32 | B = 128 |
  |---:|---:|---:|---:|---:|
  | 1 | 0.194 | 0.211 | 0.212 | 0.203 |
  | 8 | 0.252 | 0.343 | 0.385 | 0.382 |
  | 512 | 0.388 | 0.597 | 0.754 | 0.713 |
  | 8192 | 3.447 | 4.691 | 6.136 | 5.966 |

  12–30% slower than row scales from 8 tokens up.
- **CUDA kernels at decode.** Up to 16 (token, slot) pairs on sm_90 (H100 / H200 only), prebuilt CUDA kernels take over
  (`cuda_kernels/packed_moe.cu`; build and design in [dare.md](dare.md#cuda-kernels-at-decode)). A byte of sign
  bits indexes a 256-entry table of sign-flip masks, so $W_b \pm s$ costs one XOR and one bf16x2 add per two
  entries. They cover one scale per row or matrix, or blocks of 32 or more. Matrix scales, µs per layer, top-1 / top-2
  (% of the byte bound at 4.8 TB/s):

  | tokens | dense $\Delta$, Triton | BitDelta, Triton | **BitDelta, CUDA** |
  |---:|---:|---:|---:|
  | 1 | 80.5 / 117.0 | 89.4 / 144.6 (35% / 22%) | **78.0 / 108.2 (40% / 29%)** |
  | 8 | 191.4 / 234.6 | 276.4 / 312.8 (13% / 12%) | **198.3 / 275.6 (18% / 13%)** |
  | 16 | 228.4 / 236.2 | 363.6 / 369.0 (11% / 11%) | **310.2** / (Triton) |
  | 512 | 342.5 / 584.3 | 553.6 / 860.1 (8% / 6%) | (Triton) |

  It is 1.1–1.4x faster than Triton and about as fast as dense deltas at 1 token. From 8 tokens up, every routed
  group's rebuild of $W_b + \Delta_g$ keeps it latency-bound, so it does not reach the byte bound
  ([dare.md](dare.md#cuda-kernels-at-decode)).

## Accuracy

Qwen3.5-4B MoE (5 full-finetuned deltas + zero expert, top-1), accuracy in % (4 samples per question), against the
same model with dense deltas. Scale distillation: logit MSE to the dense-delta model, 200 steps, scales only
(qwen3-vl-moe `scripts/distill_bitdelta_scales.py`).

| deltas | engineering | history | law | NLI4CT |
|---|---:|---:|---:|---:|
| base model, no delta | 67.5 | 61.2 | 61.1 | 62.5 |
| dense bf16 | 73.5 | 75.4 | 68.4 | 79.0 |
| BitDelta, matrix scales | 73.5 | 74.2 | 67.5 | 75.8 |
| BitDelta, row scales | 73.4 | 74.2 | 67.2 | 75.8 |
| matrix scales, distilled | 73.6 | 74.2 | 68.3 | 76.8 |
| row scales, distilled | 73.6 | 74.1 | 68.3 | 76.7 |
| block 32 scales | 73.2 | 73.9 | 67.6 | 75.9 |
| block 128 scales | 73.5 | 74.0 | 67.3 | 75.9 |
| block 32, distilled | 73.8 | 74.5 | 68.6 | 77.0 |
| block 128, distilled | 73.9 | 74.2 | 68.2 | 77.1 |

- Row and block scales are no better than matrix scales (every difference within 0.5, inside the 95% intervals); the
  logit MSE to the dense-delta model is the same too. The error is each entry's own magnitude, which no shared scale
  recovers.
- Distillation recovers about 1 point on law and NLI4CT, nothing on history.
- The largest drop is on the set with the largest delta (NLI4CT, -2.3 after distillation).
