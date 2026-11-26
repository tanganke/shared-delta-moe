# Dense prefill

From `DENSE_PREFILL_PAIRS_PER_GROUP` (640) × (`n_delta` + 1) (token, slot) pairs on, with dense deltas on sm90, the
fused forward (`triton_kernels.fused.fused_moe`, so `forward_param`, the `fused_moe_dense` op under `torch.compile`
and the vLLM plugin) leaves the Triton GEMMs for `cuda_kernels/dense_moe.py`:

1. The Triton routing sort, as before.
2. $W_b + \Delta_g$ for every group with rows into a scratch per projection, (G + 1, N, K), with the base in the
   last slot for the zero expert. One Triton add per projection; an empty group is skipped on the device, so there is
   no host sync. The scratch is reused across forwards and layers: 3 · (G + 1) · I · H elements per shape, dtype, device
   and stream (27B shard in bf16: 1.07 GB), never freed (`dense_moe._SCRATCH.clear()` releases it). It is never
   allocated during CUDA-graph capture: until it exists for the capturing stream, a captured forward stays on Triton.
3. `F.grouped_mm` (CUTLASS on sm90) for gate and up, SiLU · up in torch, `F.grouped_mm` for down, then back to pair
   order (a gather) times the routing weight, the slots summed. These are the materialised forward's own ops, so
   the rounding (g, u, SiLU(g), h, y, routing weight in bf16) is the model's; at 2048 tokens and more the output
   matches `_forward_param` bit for bit.

At large P the GEMMs are compute-bound and every weight tile is reused by many row tiles, so writing the sums once
(a few hundred MB, 0.2–0.5 ms) costs less than what CUTLASS's shared-memory-operand wgmma gains over a GEMM whose
weight operand is summed in registers (Triton: 34–39% of peak). Below the threshold, the extra pass over the
weights costs more than that, and Triton stays.

Inference only (bf16, no input that needs a gradient; fp16 would hit torch's host-synchronising grouped-GEMM
fallback), contiguous weights. The routing sort's Triton kernel and the add need Triton; no compiled extension is
needed, so the vLLM image (no nvcc) runs it as is. Since this path has no upper limit, `forward_param` on Hopper no longer falls back to the
host-synchronising materialised loop past `TRITON_MAX_PAIRS_HOPPER`.

## Measurements

H200, bf16, ms per layer (torch.profiler, sum of kernel device time, routing sort included), % of 989 TFLOPS dense
bf16 in brackets (FLOPs = 6 · pairs · H · I). "Plain grouped" is the same `F.grouped_mm` forward on pre-materialised
experts (what a plain MoE costs; not a delta model). **Bold**: what `fused_moe` runs now.

Flat MoE, hidden 2560, inter 9216, 5 deltas + zero expert:

| top-k | tokens | pairs | Triton (before) | grouped | plain grouped |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 512 | **0.346** (21%) | 0.784 (9%) | 0.429 (17%) |
| 1 | 2048 | 2048 | **0.854** (34%) | 1.161 (25%) | 0.815 (36%) |
| 1 | 4096 | 4096 | 1.644 (36%) | **1.511** (39%) | 1.208 (49%) |
| 1 | 8192 | 8192 | 3.136 (37%) | **2.543** (46%) | 2.285 (51%) |
| 2 | 512 | 1024 | **0.524** (28%) | 0.889 (16%) | 0.531 (28%) |
| 2 | 2048 | 4096 | 1.559 (38%) | **1.543** (38%) | 1.192 (49%) |
| 2 | 4096 | 8192 | 2.978 (39%) | **2.571** (46%) | 2.279 (51%) |
| 2 | 8192 | 16384 | 6.296 (37%) | **4.594** (51%) | 4.040 (58%) |

Qwen3.8-27B upscale, one TP=2 shard: hidden 5120, inter 8704, 3 deltas in groups (2, 1), 2 slots per token (the
vLLM prefill case):

| tokens | pairs | Triton (before) | grouped | plain grouped |
|---:|---:|---:|---:|---:|
| 512 | 1024 | **0.782** (35%) | 1.233 (22%) | 0.783 (35%) |
| 1024 | 2048 | **1.400** (40%) | 1.595 (35%) | 1.122 (49%) |
| 2048 | 4096 | 2.667 (42%) | **2.473** (45%) | 2.028 (55%) |
| 8192 | 16384 | 13.222 (34%) | **7.931** (56%) | 7.271 (61%) |

The grouped path is 8–40% faster than Triton from the threshold on, and within 10–25% of the plain MoE; the rest is
the materialisation (27B, 8192 tokens: add 0.45 ms, base copies 0.13 ms) and the elementwise passes (SiLU · up,
gather, slot sum: ~0.5 ms). `python benchmarks/bench_dense_prefill.py [--kernels]` reproduces the tables.

## The hand-written CUDA kernel (not kept)

A hand-written sm_90a kernel (TMA + mbarrier pipeline + `wgmma`, no CUTLASS) fused the sum into the GEMM instead: per
K step it TMA-loaded a 256 × 64 activation tile, the 128 × 64 $W_b$ tile and the matching $\Delta_g$ tile, the
consumer warpgroups added $\Delta$ into $W_b$ in shared memory (same 128B swizzle on both), then ran `wgmma` on it,
with SiLU · up (gate and up rows interleaved) and the routing weight in the epilogues. It was correct and beat Triton at
large sizes, but beat the grouped path only near 4096 pairs on the flat model (~6%) and lost clearly at the largest
sizes (ms, same cases as above): flat top-1 0.540 / 0.964 / 1.427 / 2.715 at
512 / 2048 / 4096 / 8192 tokens; flat top-2 0.517 / 1.453 / 2.823 / 5.288; 27B shard 0.858 / 1.371 / 2.539 / 10.12
at 512 / 1024 / 2048 / 8192 tokens. Its GEMM core reached ~60% of peak where cuBLAS reaches ~80% (27B gate/up with
no delta: 4.8 vs 3.65 ms), and the in-shared-memory add cost another ~20%. CUTLASS's mixed-input collective could
not be reused (its scale operand is MN-major, one entry per K block, not a full K-major delta tile), and even a
kernel at cuBLAS speed would only save the materialisation, ~7% of the grouped path at 8192 tokens. The code is in
commit 68c6ea5 (branch `cuda-dense-kernel`).
