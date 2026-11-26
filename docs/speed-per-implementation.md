# Speed of each implementation (single MoE layer, bf16, eager, ms)

The per-implementation tables below were measured with the previous kernel (5522b23; one launch per projection). Its "param Triton" column is superseded by [fused-kernel.md](fused-kernel.md).

## H200 (sm90, 4.8 TB/s)

hidden 2560, inter 9216, 5 deltas + zero expert (the size of a Qwen3.5-4B MoE with 6 experts). For comparison, transformers' stock `Qwen3_5MoeSparseMoeBlock` (`grouped_mm`) at the same size takes 0.36 / 0.47 / 0.49 / 2.72 ms at 1 / 32 / 512 / 8192 tokens (top-1).

top-1:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.251 | 0.351 | 0.235 | 0.227 | 0.336 | 0.232 | 0.323 |
| 8 | 0.578 | 0.488 | 0.947 | 0.758 | **0.358** | 0.620 | 0.574 |
| 32 | 0.818 | 0.592 | 1.282 | 1.050 | **0.359** | 0.779 | 0.606 |
| 512 | 0.880 | 0.566 | 1.345 | 1.146 | **0.557** | 0.982 | 0.743 |
| 2048 | 1.156 | 0.963 | 1.665 | 1.661 | 1.337 | 1.878 | 1.639 |
| 4096 | 1.688 | 1.326 | 2.213 | 2.196 | 1.794 | 3.131 | 2.720 |
| 8192 | 2.730 | 2.415 | 3.320 | 3.242 | 3.595 | 5.680 | 5.140 |

top-2:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.367 | 0.391 | 0.420 | 0.394 | **0.357** | 0.454 | 0.426 |
| 8 | 0.809 | 0.585 | 1.278 | 1.032 | **0.361** | 1.293 | 0.643 |
| 32 | 0.846 | 0.582 | 1.313 | 1.071 | **0.359** | 1.475 | 0.634 |
| 512 | 0.926 | 0.664 | 1.436 | 1.283 | 0.716 | 1.852 | 0.924 |
| 2048 | 1.650 | 1.311 | 2.177 | 2.160 | 1.773 | 3.475 | 2.443 |
| 4096 | 2.631 | 2.406 | 3.169 | 3.127 | 3.568 | 5.737 | 4.509 |
| 8192 | 4.628 | 4.392 | 5.261 | 5.252 | 7.067 | 10.236 | 8.546 |

- **Triton fused parameter aggregation** is the fastest of everything here from 8 to 512 tokens, the plain MoE and transformers included (top-1, 32 tokens: 0.36 vs 0.59 / 0.47). Decode is memory-bound. The kernel reads $W_b$ and $\Delta_e$ once each, writes no weights, and does a single GEMM's FLOPs; all pairs go through one launch per projection, and nothing synchronises the host.
- Three things mattered in the kernel:
  - **Masked delta load instead of a branch.** The zero expert's delta load is masked off rather than skipped with an `if`, so Triton can pipeline the loop. At 32 tokens: 0.57 → 0.48 ms.
  - **Summed weight as operand A.** The summed weight is the A operand, computing $y^T = (W_b+\Delta)\,x^T$; Hopper's wgmma can take A from registers. At 8192 tokens: 4.55 → 3.67 ms.
  - **No host sync for counts.** Group counts come from `scatter_add`, not `bincount`, and with few pairs the grid is sized from the pair count, not the largest group, so nothing synchronises the host. At 32 tokens: 0.48 → 0.36 ms.
- With many tokens the work is compute-bound, and the Triton GEMM loses to cuBLAS (8192 tokens: 3.66 vs 2.41 plain grouped); `forward_param` switches to the materialised version past the pair limit.
- **Per-matrix output aggregation** needs no custom kernel. It is about 1.3× the plain MoE at small token counts and about 2× at large ones, since it computes both $W_b$ and $\Delta_e$.
- **Grouped GEMM vs loop**: from 32 to 512 tokens, the grouped versions are 1.3–2.3× faster than the per-expert loops; at a single token the sort and count cost more than they save.

## RTX 6000 Ada (sm89, 960 GB/s)

top-1:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.363 | 0.663 | 0.808 | 0.509 | 0.538 | 0.530 | 0.815 |
| 8 | 1.650 | 1.357 | 3.935 | 2.402 | **1.065** | 1.600 | 1.379 |
| 32 | 1.657 | 1.376 | 4.024 | 2.454 | **1.078** | 1.580 | 1.413 |
| 512 | 1.859 | 1.591 | 4.225 | 2.823 | **1.244** | 2.368 | 2.095 |
| 2048 | 3.324 | 3.532 | 5.867 | 5.641 | **3.168** | 6.500 | 5.869 |
| 4096 | 6.162 | 6.219 | 8.153 | 8.009 | **5.935** | 12.223 | 11.810 |
| 8192 | 12.373 | 11.415 | 13.407 | 13.302 | 11.713 | 24.865 | 23.370 |

top-2:

| tokens | plain loop | plain grouped | param (materialised) | param hybrid | **param Triton** | matrix loop | matrix grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.642 | 0.825 | 1.538 | 0.953 | **0.599** | 1.050 | 0.955 |
| 8 | 1.715 | 1.397 | 4.017 | 2.392 | **1.080** | 2.864 | 1.474 |
| 32 | 1.694 | 1.413 | 4.047 | 2.496 | **1.107** | 3.047 | 1.456 |
| 512 | 2.150 | 1.948 | 4.550 | 3.477 | **1.652** | 4.528 | 2.642 |
| 2048 | 5.998 | 6.270 | 8.158 | 7.856 | 6.903 | 11.486 | 9.812 |
| 4096 | 11.951 | 11.859 | 13.340 | 13.054 | **11.816** | 22.354 | 19.629 |
| 8192 | 22.403 | 22.105 | 23.916 | 23.836 | 27.769 | 43.258 | 38.772 |

- Lower bandwidth makes everything 2–4× slower than on the H200, but the ranking holds. From 8 to 512 tokens Triton is 1.2–1.3× faster than the plain MoE.
- Hopper's wgmma advantage is absent here: with many tokens Triton is on par with cuBLAS (top-1 at 4096–8192 tokens), and it beats materialised weights to higher pair counts, hence the higher limit for Ada and older.
- Two Ada-friendly configs (BK 32, 4–5 stages) were added to the autotune list. They took top-1 at 4096 tokens from 6.72 to 5.94 ms. On Hopper, autotuning still picks its own.

The delta form has uses beyond speed: $\Delta_e$ is a standalone tensor that can be compressed, sparsified or trained on its own; the zero expert has no parameters; and in per-matrix output aggregation the routing weight can stay in the computation graph.
