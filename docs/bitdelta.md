# BitDelta deltas (1 bit per entry)

`BitDeltaMoE.from_delta_moe(m, scale="matrix")` stores each $\Delta_e$ as its sign, 1 bit per entry, times a scale
(BitDelta, Liu et al. 2024, "BitDelta: Your Fine-Tune May Only Be Worth One Bit"):

$$\Delta_e \approx \mathrm{diag}(s_e)\,\mathrm{sign}(\Delta_e)$$

- `scale="matrix"` (BitDelta): one scale per matrix, $s = \mathrm{mean}\,|\Delta_e|$. For a fixed sign matrix it is
  the scalar closest to $\Delta_e$ in Frobenius norm.
- `scale="row"`: one scale per output row, that row's mean $|\Delta_e|$. It costs R values per matrix and is closer.

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
means +s. The scales are `s_<proj>`, shape (N, R), in the model dtype. Both are in the state dict.

Rows must have a multiple of 32 entries, so hidden and inter must be multiples of 32.

One delta of a Qwen3.5-4B MoE layer (hidden 2560, inter 9216) takes:

| Format | Size |
|---|---:|
| dense bf16 | 141.6 MB |
| seeded DARE 0.2, block 64 | 28.3 MB of kept values, plus a 0.7 MB bitmap per layer |
| BitDelta | 8.85 MB of sign bits, plus 42 KB of row scales |

## Fused kernel: delta mode 2

`forward_param` reads the sign bits directly (`triton_kernels/fused.py`, `Delta(2, bits, scales, block=32)`).

- **Loads.** Each K tile loads BK / 32 words per row and expands them to $\pm s$ in registers.
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
- **Room left.** Faster expansion (e.g. unpacking 32 entries with fewer integer ops, or dequantising once per tile
  into shared memory) is not done yet.

## Accuracy

Not yet measured on a real model.
