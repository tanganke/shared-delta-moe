# DARE

`DareDeltaMoE.from_delta_moe(m, keep=0.2, block=64, seed=0)` keeps a random `keep` of each $\Delta_e$ (whole
$1 \times$ `block` blocks along the input dimension; `block=1` is element-wise DARE), rescaled by $1/\text{keep}$, and
stores only the kept values plus one int64 row start per weight row. The mask is not stored: it is Philox4x32-10 of
(unit, 3e + projection) under the seed, pure 32-bit integer arithmetic, so it can be regenerated anywhere
(`tests/test_dare.py` checks a Random123 known-answer vector and CPU == CUDA). The fused kernel reads it as a bitmap,
1 bit per block (`mask_bits`), built once from the seed and kept as a non-persistent buffer (`m_gate` / `m_up` /
`m_down`, rebuilt after loading): 0.7 MB per layer at block 64, 44 MB element-wise, against 142 MB of kept values at
keep 0.2. Storage per delta: `keep` of dense, e.g. 20%, plus the bitmap.

```python
from shared_delta_moe import DareDeltaMoE
d = DareDeltaMoE.from_delta_moe(m, keep=0.2, block=64, seed=0)  # m: a DeltaMoE
y = d(x)
```

How the kernels read it: [Fused Triton kernel](../internals/fused-triton.md#dare-delta-mode-1) and
[CUDA decode kernels](../internals/cuda-decode.md); speed: [Benchmarks](../internals/benchmarks.md).

Accuracy of block DARE on the real model:
ntu-projects `moe-indomain-acc` log [35] (blocks 4 to 256 at 20% match element-wise DARE).

