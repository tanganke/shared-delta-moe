# Group grids

`GridDeltaMoE.from_delta_moe(m, levels, group, symmetric, clip, x)` stores each $\Delta_e$ as low-bit codes on a grid
of $L$ evenly spaced values per group of $g$ entries along a row, the format that keeps the accuracy at 2 bits and
below in [delta-compression.md](delta-compression.md):

$$
\hat\delta = \ell + s\,q, \qquad q \in \{0, \dots, L - 1\},
$$

with per group a lowest level $\ell$ and a step $s$. A symmetric grid is centred on 0, $\ell = -s (L - 1) / 2$, and
stores only $s$. The codes come from `compress.quantize`: round to nearest without data (`x=None`), or GPTQ when `x`
gives each delta expert's MLP inputs (its domain's data). `clip=True` shrinks each group's span to the factor of least
rounding error.

`m` may be a `DeltaMoE` or a `HierDeltaMoE`; the result is a `GridDeltaMoE` or a `GridHierDeltaMoE`, with the same
base, router and routing. The steps and lowest levels are `nn.Parameter` objects, trainable through the materialised
forward; the codes are fixed.

| grid | `levels` | code bits | bits / entry, group 128 |
|---|---:|---:|---:|
| 2 bit | 4 | 2 | 2.25 (symmetric 2.125) |
| ternary | 3 | 2 | 2.125 (symmetric) |
| 1 bit | 2 | 1 | 1.25 (symmetric 1.125) |

Ternary codes take 2 bits here; a base-3 packing (5 codes per byte, 1.6 bits) is not implemented, so at the same
storage the 4-level grid is the better choice.

```python
from shared_delta_moe import GridDeltaMoE
g = GridDeltaMoE.from_delta_moe(m, levels=4, group=128)              # 2-bit RTN, no data
g = GridDeltaMoE.from_delta_moe(m, levels=2, group=128, x=inputs)    # 1-bit GPTQ, inputs[e]: (T_e, H)
y = g.forward_param(x)
```

## Storage

With $b$ code bits (1, 2 or 4) and projection (R, C) of delta $e$:

- `c_<proj>`: (N, R, C b / 32) int32 codes, low bits first: entry $c$ of a row starts at bit $b (c \bmod 32/b)$ of
  word $\lfloor c b / 32 \rfloor$.
- `s_<proj>`: (N, R, C / g) steps, and for asymmetric grids `z_<proj>`: (N, R, C / g) lowest levels, in the model
  dtype.

In bf16 the module's deltas differ from `compress`'s float32 simulation by up to one bf16 ulp of the group's largest
entry.

## Kernels and tensor parallelism

Without a gradient, 2-bit grids (3 or 4 levels) with a power-of-2 group run the fused Triton forward, which reads the
codes directly ([fused Triton kernel](../internals/fused-triton.md), mode 3); other grids, and every forward that
needs a gradient apart from router training (below), run the materialised one. No CUDA (`packed_ops`) kernel reads the codes, and the vLLM plugin takes
dense deltas only. For tensor parallelism over the intermediate dimension, the group must divide each rank's share of
it (Qwen3.5-27B, I = 17408 over 8 ranks: 2176 columns, so group 128 or 64, not 256).

## Router training

When a gradient is needed and every parameter except `router.weight` is frozen (2-bit codes, power-of-2 group, CUDA,
bf16/fp16), `forward_param` runs `_forward_decoded` in two steps:

- One Triton pass (`triton_kernels/decode.py`) decodes every expert into full weights, $W_b + (\ell + s q)$, with
  the base added as the last expert for the zero experts. Each entry is rounded as the materialised path rounds it.
- cuBLAS grouped GEMMs (`moe.grouped_moe`) then run over the sorted (token, slot) pairs.

The decoded weights are not kept for the backward. Outside activation checkpointing, `ops.recompute_saved` rebuilds
them there. Under checkpointing, the recompute decodes again and that layer keeps them until its backward. So a
training step decodes twice per layer, whatever the number of tokens.

Decoding takes about 5 ms for a 27B layer: 15 experts × 3 projections, 7.5 GiB of bf16 on H200. While a layer runs,
those weights add to peak memory. A step with trainable grids or base takes the materialised path. Measurements are
in [benchmarks](../internals/benchmarks.md#training).
