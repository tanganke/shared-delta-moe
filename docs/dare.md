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

## CUDA kernels at decode

At up to 16 (token, slot) pairs, `forward_param` runs prebuilt CUDA kernels instead of the Triton ones
(`cuda_kernels/packed_moe.cu`, `packed_ops.py`). They cover DARE blocks of 8 or more and BitDelta, with hidden and inter
multiples of 64. A warp owns 16 weight rows over a K range and reads $W_b$ once per 64-wide K step. For each routed
group it adds the kept values in registers and multiplies with that group's pairs on the tensor cores. Popcounts over
the keep bitmap place the values, which stream through a per-warp `cp.async` pool that holds only what is kept.
Rounding is the same as in the Triton kernels; split-K sums run in a fixed order.

Build once per torch version on a node with nvcc (here a 2B compute node, `module load cuda/13.0`), writing
`sdm_packed-<torch>.so` into a directory the runtime also sees, e.g. on /data for the vLLM image:

```bash
SHARED_DELTA_MOE_CUDA_DIR=/data/projects/11620001/anketang/sdm-cuda python -m shared_delta_moe.cuda_kernels.packed_ops
```

The runtime loads the file by path, with the same variable and no nvcc. Without the library, or with
`SHARED_DELTA_MOE_CUDA=0`, everything runs on Triton. Under `torch.compile` the kernels run inside the same op
(`shared_delta_moe::fused_moe_dare`), so CUDA graphs work as before. `tests/test_packed_cuda.py` checks them against
the materialised forward: blocks 8 to 128, zero experts, top-2, split-K, fp16, a CUDA graph and `torch.compile`.

H200, µs per layer (torch.profiler device time of every kernel after the router), hidden 2560, inter 9216, 5 deltas
+ zero expert, DARE 0.2 / block 64, BitDelta matrix scales. "% BW" is the byte lower bound (W_b once, each touched
delta in its stored form, activations) over the time, at 4.8 TB/s:

| tokens | top-k | dense $\Delta$, Triton | DARE, Triton | **DARE, CUDA** | BitDelta, Triton | **BitDelta, CUDA** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 80.5 (73%) | 225.9 (16%) | **123.8 (29%)** | 89.4 (35%) | **78.0 (40%)** |
| 8 | 1 | 191.4 (77%) | 573.3 (9%) | **461.5 (12%)** | 276.4 (13%) | **198.3 (18%)** |
| 16 | 1 | 228.4 (78%) | 574.7 (9%) | **481.7 (11%)** | 363.6 (11%) | **310.2 (13%)** |
| 64 | 1 | 233.8 (76%) | 635.7 (9%) | (Triton) | 360.5 (11%) | (Triton) |
| 512 | 1 | 342.5 (53%) | 933.8 (7%) | (Triton) | 553.6 (8%) | (Triton) |
| 1 | 2 | 117.0 (76%) | 295.6 (12%) | **169.2 (21%)** | 144.6 (22%) | **108.2 (29%)** |
| 8 | 2 | 234.6 (75%) | 641.3 (9%) | **587.0 (10%)** | 312.8 (12%) | **275.6 (13%)** |
| 16 | 2 | 236.2 (75%) | 638.9 (9%) | (Triton) | 369.0 (11%) | (Triton) |
| 64 | 2 | 256.9 (69%) | 692.8 (9%) | (Triton) | 381.7 (10%) | (Triton) |
| 512 | 2 | 584.3 (32%) | 1487.6 (5%) | (Triton) | 860.1 (6%) | (Triton) |

- **Where it helps.** The CUDA path is 1.1–1.8x faster than Triton up to 16 pairs. It is still slower than dense
  deltas, except BitDelta at 1 token (78 vs 81 µs), and far from the byte bound.
- **Why it stops at 16 pairs.** At 24 pairs it is slower than Triton (DARE top-1, 24 tokens: 691 vs 633 µs), so
  `MAX_PAIRS = 16`.
- **What limits it.** Each routed group re-reads the $W_b$ chunk from shared memory and rebuilds $W_b + \Delta_g$.
  With 6 groups this step is latency-bound at the 8 warps per SM that the shared memory allows (W_b ring, delta pool,
  activations). Memory traffic is not the limit: per-warp clock counters showed under 10% of the time waiting for
  copies.
- **What did not help.** 32-wide K steps with 64-byte row segments: about 1 TB/s at 1 token. 128-wide K steps:
  shared memory cut the warps per SM, with no gain. A 2-slot ring: too little in flight. What helped: the table of
  sign masks (BitDelta), per-step hoisting of masks and scales, and two accumulator chains per tile (1.3–2x at 1
  token).
- **Next.** The step loop is latency-bound, so more warps per SM is what would help. Two warps could share one $W_b$
  ring and split the groups between them (named barriers per chunk). A 32-row tile would amortise the per-step
  control work.
