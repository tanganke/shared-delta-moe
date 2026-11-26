# Seeded DARE deltas

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
