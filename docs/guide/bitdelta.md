# BitDelta

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
