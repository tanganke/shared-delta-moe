# Hand-written dense CUDA kernel (not kept)

A hand-written sm_90a kernel (TMA + mbarrier pipeline + `wgmma`, no CUTLASS) fused the sum into the GEMM instead: per
K step it TMA-loaded a 256 × 64 activation tile, the 128 × 64 $W_b$ tile and the matching $\Delta_g$ tile, the
consumer warpgroups added $\Delta$ into $W_b$ in shared memory (same 128B swizzle on both), then ran `wgmma` on it,
with SiLU · up (gate and up rows interleaved) and the routing weight in the epilogues. It was correct and beat Triton at
large sizes, but beat the grouped path only near 4096 pairs on the flat model (~6%) and lost clearly at the largest
sizes (ms, same cases as the [dense prefill benchmarks](benchmarks.md#dense-prefill)): flat top-1 0.540 / 0.964 / 1.427 / 2.715 at
512 / 2048 / 4096 / 8192 tokens; flat top-2 0.517 / 1.453 / 2.823 / 5.288; 27B shard 0.858 / 1.371 / 2.539 / 10.12
at 512 / 1024 / 2048 / 8192 tokens. Its GEMM core reached ~60% of peak where cuBLAS reaches ~80% (27B gate/up with
no delta: 4.8 vs 3.65 ms), and the in-shared-memory add cost another ~20%. CUTLASS's mixed-input collective could
not be reused (its scale operand is MN-major, one entry per K block, not a full K-major delta tile), and even a
kernel at cuBLAS speed would only save the materialisation, ~7% of the grouped path at 8192 tokens. The code is in
commit 68c6ea5 (branch `cuda-dense-kernel`).
