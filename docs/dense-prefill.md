# Dense prefill

From `DENSE_PREFILL_PAIRS_PER_GROUP` (640) × (`n_delta` + 1) (token, slot) pairs on, with dense deltas on sm90, the
fused forward (`triton_kernels.fused.fused_moe`, so `forward_param`, the `fused_moe_dense` op under `torch.compile`
and the vLLM plugin) leaves the Triton GEMMs for `cuda_kernels/dense_moe.py`:

1. The Triton routing sort, as before.
2. $W_b + \Delta_g$ for every group with rows into a scratch per projection, (G + 1, N, K), with the base in the
   last slot for the zero expert. One Triton add per projection; an empty group is skipped on the device, so there is
   no host sync. The scratch is reused across forwards and layers (one per shape: the 27B shard holds 1.07 GB).
3. `F.grouped_mm` (CUTLASS on sm90) for gate and up, SiLU · up in torch, `F.grouped_mm` for down, then back to pair
   order (a gather) times the routing weight, the slots summed. These are the materialised forward's own ops, so
   the rounding (g, u, SiLU(g), h, y, routing weight in bf16) is the model's; at 2048 tokens and more the output
   matches `_forward_param` bit for bit.

At large P the GEMMs are compute-bound and every weight tile is reused by many row tiles, so writing the sums once
(a few hundred MB, 0.2–0.5 ms) costs less than what CUTLASS's shared-memory-operand wgmma gains over a GEMM whose
weight operand is summed in registers (Triton: 34–39% of peak). Below the threshold, the extra pass over the
weights costs more than that, and Triton stays.

The routing sort's Triton kernel and the add need Triton; no compiled extension is needed, so the vLLM image (no
nvcc) runs it as is. Since this path has no upper limit, `forward_param` on Hopper no longer falls back to the
host-synchronising materialised loop past `TRITON_MAX_PAIRS_HOPPER`.

## Measurements

H200, bf16, ms per layer (torch.profiler, sum of kernel device time, routing sort included), % of 989 TFLOPS dense
bf16 in brackets (FLOPs = 6 · pairs · H · I). "Plain grouped" is the same `F.grouped_mm` forward on pre-materialised
experts (what a plain MoE costs; not a delta model). "CUDA" is the hand-written kernel below. **Bold**: what
`fused_moe` runs now.

Flat MoE, hidden 2560, inter 9216, 5 deltas + zero expert:

| top-k | tokens | pairs | Triton (before) | grouped | CUDA | plain grouped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 512 | **0.346** (21%) | 0.784 (9%) | 0.540 (14%) | 0.429 (17%) |
| 1 | 2048 | 2048 | **0.854** (34%) | 1.161 (25%) | 0.964 (30%) | 0.815 (36%) |
| 1 | 4096 | 4096 | 1.644 (36%) | **1.511** (39%) | 1.427 (41%) | 1.208 (49%) |
| 1 | 8192 | 8192 | 3.136 (37%) | **2.543** (46%) | 2.715 (43%) | 2.285 (51%) |
| 2 | 512 | 1024 | **0.524** (28%) | 0.889 (16%) | 0.517 (28%) | 0.531 (28%) |
| 2 | 2048 | 4096 | 1.559 (38%) | **1.543** (38%) | 1.453 (40%) | 1.192 (49%) |
| 2 | 4096 | 8192 | 2.978 (39%) | **2.571** (46%) | 2.823 (42%) | 2.279 (51%) |
| 2 | 8192 | 16384 | 6.296 (37%) | **4.594** (51%) | 5.288 (44%) | 4.040 (58%) |

Qwen3.8-27B upscale, one TP=2 shard: hidden 5120, inter 8704, 3 deltas in groups (2, 1), 2 slots per token (the
vLLM prefill case):

| tokens | pairs | Triton (before) | grouped | CUDA | plain grouped |
|---:|---:|---:|---:|---:|---:|
| 512 | 1024 | **0.782** (35%) | 1.233 (22%) | 0.858 (32%) | 0.783 (35%) |
| 1024 | 2048 | **1.400** (40%) | 1.595 (35%) | 1.371 (40%) | 1.122 (49%) |
| 2048 | 4096 | 2.667 (42%) | **2.473** (45%) | 2.539 (44%) | 2.028 (55%) |
| 8192 | 16384 | 13.222 (34%) | **7.931** (56%) | 10.123 (44%) | 7.271 (61%) |

The grouped path is 8–40% faster than Triton from the threshold on, and within 10–25% of the plain MoE; the rest is
the materialisation (27B, 8192 tokens: add 0.45 ms, base copies 0.13 ms) and the elementwise passes (SiLU · up,
gather, slot sum: ~0.5 ms). `python benchmarks/bench_dense_prefill.py [--kernels]` reproduces the tables.

## The hand-written CUDA kernel (opt-in)

`cuda_kernels/dense_kernel.cu`, two torch ops (`sdm_dense::gate_up`, `sdm_dense::down`), sm_90a, bf16. It fuses the
sum into the GEMM instead of materialising it: per K step one thread TMA-loads a 256 × 64 activation tile, a
128 × 64 $W_b$ tile and the matching $\Delta_g$ tile (128B swizzle) into a 3-stage mbarrier ring; two consumer
warpgroups add the $\Delta$ tile into the $W_b$ tile in shared memory (same swizzle on both, so the add is
elementwise; `__hadd2` rounds like the materialised bf16 sum), then issue `wgmma.m64n128k16` (both operands from
shared memory) on 128 rows each. For gate/up the weight tile is 64 gate rows over 64 up rows, so each thread holds
$g$ and $u$ of the same output element and SiLU · up happens in registers; down scales by the routing weight and
writes each pair's slot. The tile list comes from the group starts on the device (no host sync).

It beats Triton at large P (27B, 8192 tokens: 10.1 vs 13.2 ms) but loses to the grouped path everywhere it matters,
so it is off by default. Isolated (27B, 16384 pairs): gate/up 5.8 ms with the add, 4.8 ms with no delta (zero
group), against 3.65 ms for cuBLAS on the plain GEMM; the GEMM core reaches ~60% of peak where cuBLAS reaches ~80%,
and the in-shared-memory add costs another ~20%. What did not help: moving the add to three otherwise idle producer
warps (slower: the add is limited by shared-memory and L2 bandwidth, not issue slots), persistent CTAs with the
next tile's loads overlapping the epilogue (no change, within noise), GROUP_M 1–32 (4 is best by a few %). Closing
the gap would need what CUTLASS does for the plain GEMM (thread-block clusters with TMA multicast of the weight and
delta tiles, a TMA-store epilogue), and even then the fused add only saves the materialisation, ~7% of the grouped
path at 8192 tokens.

To try it, build once per torch version on a machine with nvcc (2B: a compute node, `module load cuda/13.0`):

```sh
SHARED_DELTA_MOE_CUDA_DIR=/data/projects/11620001/anketang/sdm-cuda python -m shared_delta_moe.cuda_kernels.dense_ext
```

which writes `sdm_dense.so` there; at runtime (nvcc not needed) set the same `SHARED_DELTA_MOE_CUDA_DIR` and
`SHARED_DELTA_MOE_DENSE_CUDA=1`. Without the library the grouped path runs. `tests/test_dense_prefill.py` checks both
against `_forward_param` (ragged, empty and zero-expert-only groups, top-2, sizes that are not tile multiples).
