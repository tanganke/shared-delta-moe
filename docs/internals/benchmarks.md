# Benchmarks

Every speed table, by subject. Shapes are those of a Qwen3.5-4B MoE layer (hidden 2560, inter 9216, 5 deltas + zero
expert) unless a table says otherwise. How each path works: [Kernels](kernels.md).

## Fused Triton forward (parameter aggregation)

H200, top-1 / top-2, ms per layer (hidden 2560, inter 9216, 5 deltas + zero expert; "graph" = CUDA graph replay):

| tokens | plain grouped | param (materialised) | **param Triton** | Triton, graph | DARE 0.2 block 64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.359 / 0.396 | 0.303 / 0.526 | 0.177 / 0.190 | **0.095 / 0.137** | 0.256 / 0.341 |
| 8 | 0.546 / 0.552 | 1.058 / 1.083 | 0.231 / 0.244 | **0.215 / 0.223** | 0.488 / 0.688 |
| 32 | 0.566 / 0.603 | 1.287 / 1.325 | 0.273 / 0.298 | **0.258 / 0.278** | 0.676 / 0.695 |
| 512 | 0.566 / 0.674 | 1.339 / 1.444 | 0.378 / 0.641 | **0.358 / 0.609** | 0.982 / 1.541 |
| 2048 | 0.972 / **1.332** | 1.641 / 2.220 | 0.970 / 1.822 | **0.929** / 1.749 | 2.298 / 4.220 |
| 8192 | **2.450 / 4.289** | 3.301 / 5.247 | 3.306 / 6.780 | 3.168 / 6.263 | 7.643 / 14.708 |

RTX 6000 Ada, top-1 / top-2:

| tokens | plain grouped | param (materialised) | **param Triton** | Triton, graph | DARE 0.2 block 64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.736 / 0.822 | 0.916 / 1.057 | 0.434 / 0.469 | **0.355 / 0.357** | 0.412 / 0.329 |
| 8 | 1.067 / 1.245 | 3.009 / 3.838 | 0.844 / 1.014 | 0.822 / 0.990 | **0.446 / 0.556** |
| 32 | 1.432 / 1.425 | 4.058 / 4.085 | 1.014 / 1.020 | 0.992 / 1.004 | **0.545 / 0.543** |
| 512 | 1.629 / 1.795 | 4.257 / 4.363 | 1.152 / 1.613 | **1.136** / 1.699 | 1.270 / 2.170 |
| 2048 | **2.786** / 5.504 | 5.572 / 8.045 | 2.866 / 5.817 | 3.107 / 5.636 | 3.545 / 7.043 |
| 8192 | 12.522 / 24.998 | 14.386 / 24.907 | 11.345 / 24.637 | **10.526 / 23.902** | 13.167 / 27.388 |

- Against the previous single-projection kernel (5522b23, below), H200 top-1: 1 token 0.336 → 0.177 ms eager, 0.095 in a graph; 32 tokens 0.359 → 0.273, 0.258 in a graph. Ada 1 token: 0.538 → 0.434. The graph numbers show how much of decode is launch overhead.
- DARE (bitmap, see [DARE](../guide/dare.md)) is faster than dense deltas on the Ada up to 32 tokens, because it reads 20% of the delta bytes; on the H200 the gather of kept values keeps it about 2x slower than dense.

## DARE in the fused kernel: bitmap against Philox

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
what is left.

## BitDelta in the fused kernel

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
- **Block scales** (same node, another run; row scales measured alongside, ms, top-1):

  | tokens | dense $\Delta$ | row scales | B = 32 | B = 128 |
  |---:|---:|---:|---:|---:|
  | 1 | 0.194 | 0.211 | 0.212 | 0.203 |
  | 8 | 0.252 | 0.343 | 0.385 | 0.382 |
  | 512 | 0.388 | 0.597 | 0.754 | 0.713 |
  | 8192 | 3.447 | 4.691 | 6.136 | 5.966 |

  12–30% slower than row scales from 8 tokens up.

## CUDA decode kernels (parameter aggregation)

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

- **BitDelta.** 1.1–1.4x faster than Triton and about as fast as dense deltas at 1 token. From 8 tokens up, every
  routed group's rebuild of $W_b + \Delta_g$ keeps it latency-bound, so it does not reach the byte bound
  ([what limits it](cuda-decode.md#parameter-aggregation-forward-param)).

## Output aggregation on the CUDA kernels

H200, µs per layer, device time of every kernel after the router: hidden 2560, inter 9216, + zero expert, DARE 0.2 /
block 64, BitDelta matrix scales. "param" is `forward_param` (the CUDA kernels up to 16 pairs, then Triton); "matrix"
is `forward_matrix` with these kernels. Dense deltas use `forward_param` (Triton, the fastest dense path).
"bound" is the byte lower bound at 4.8 TB/s: $W_b$ once, each touched delta in its stored form, and the activations.

5 deltas, top-1:

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 80.4 | 66.4 | **55.7** | 61.5 | **54.2** | 29.5 / 29.5 |
| 4 | 122.7 | 329.0 | **105.4** | 158.0 | **110.1** | 47.4 / 33.2 |
| 8 | 194.8 | 464.4 | **159.9** | 199.5 | **138.5** | 53.3 / 35.1 |
| 16 | 191.1 | 585.0 | 210.7 | 320.1 | 192.4 | 59.3 / 38.9 |
| 64 | 234.1 | 688.7 | 312.0 | 363.2 | 311.8 | 59.8 / 39.3 |
| 512 | 344.9 | 931.4 | 1668.0 | 548.8 | 1632.2 | 64.2 / 43.7 |

(At 1 token, top-1, the token goes to the zero expert in this sample, so the param times are low; with a delta routed, DARE param is about 124 µs.)

5 deltas, top-2:

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 102.4 | 230.8 | **89.4** | 109.3 | **91.3** | 41.4 / 31.4 |
| 4 | 162.4 | 526.4 | 182.7 | 251.4 | 169.7 | 59.3 / 36.9 |
| 8 | 234.8 | 597.5 | **218.0** | 324.8 | **194.9** | 59.3 / 38.8 |
| 16 | 236.7 | 644.1 | 283.4 | 369.6 | **209.1** | 59.5 / 39.0 |
| 64 | 255.4 | 693.6 | 502.3 | 380.8 | 503.2 | 60.3 / 39.8 |
| 512 | 590.0 | 1484.9 | 3074.5 | 860.1 | 3100.2 | 68.1 / 47.7 |

15 deltas, top-3 (many slots per token, like a HierDeltaMoE with three groups):

| tokens | dense | DARE param | DARE matrix | BitDelta param | BitDelta matrix | bound DARE / Bit |
|---|---|---|---|---|---|---|
| 1 | 156.1 | 332.4 | **113.9** | 179.7 | **138.2** | 47.3 / 35.0 |
| 4 | 437.9 | 943.9 | **268.2** | 448.7 | **275.3** | 77.1 / 44.3 |
| 8 | 566.3 | 1514.1 | **403.0** | 702.3 | **358.6** | 106.9 / 50.0 |
| 16 | 596.2 | 1632.6 | **538.8** | 877.6 | **461.9** | 119.0 / 57.5 |
| 64 | 642.9 | 1636.5 | 909.7 | 950.7 | 800.8 | 120.2 / 58.8 |

Bold marks the packed forms that beat dense deltas. Output aggregation is faster than parameter aggregation for
both formats from 1 to 64 tokens. It also beats dense deltas up to about 8–16 tokens, and up to 16 with many slots
per token. Of the time at 1 token, about 43 µs is the three cuBLAS GEMVs on $W_b$ (bound 29.5 µs). The kernels add
about 12–22 µs.

## Dense prefill

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

Per-call buffers against a cache kept across calls (H200, 8192 tokens, grouped path, two alternating runs each,
device ms from the profiler / wall ms from CUDA events): 27B shard per-call 7.45–8.36 / 8.15–8.24, cached
8.43–8.50 / 8.23–8.33; flat top-2 per-call 4.46–4.64 / 4.77–4.78, cached 4.65–4.68 / 4.94–5.01. The caching
allocator hands the same blocks back, so the allocation costs nothing measurable; the peak during the call is the
same either way (+1.95 GiB and +1.71 GiB including activations).

## Training

Measured slower than the materialised path on H200 (as the first version, commit e0ce27a, was); it saves little memory, since weight gradients dominate when the weights train. hidden 2560, inter 9216, bf16, ms per forward + backward (peak GB above the weights), `benchmarks/bench_train.py`:

| H200, configuration, tokens | fused Triton | **materialised** |
|---|---:|---:|
| 5 deltas + zero expert, top-2, router only, 4096 | 10.96 (0.89) | **5.96** (1.21) |
| same, 25000 | 66.12 (2.72) | **27.94** (4.15) |
| 5 deltas, top-2, all weights, 4096 | 15.84 (1.99) | **11.55** (2.05) |
| same, 25000 | 92.99 (4.75) | **42.52** (5.22) |
| 5 deltas, top-6, all weights, 4096 | 44.57 (2.99) | **22.92** (3.17) |
| same, 25000 | 292.13 (12.07) | **117.42** (13.86) |
| Hier: 4 groups x (3 deltas + 1 zero), dense group routing, top-2 per group, all weights, 4096 | 67.64 (5.28) | **49.32** (6.47) |
| same, 25000 | 428.21 (18.35) | **186.00** (21.34) |


Whole Qwen3.5-4B text model, full-parameter training, batch 2 x 2048, bf16 weights, fused AdamW (bf16 states), random router (every delta routed: the worst case), H200, peak GiB and s/step:

| model | params / trainable | peak, checkpointing | peak, none | s/step, checkpointing |
|---|---|---:|---:|---:|
| dense Qwen3.5-4B | 4.21B / 4.21B | 35.9 | 65.5 | 0.4 |
| DARE 0.2/64, kept values frozen, top-1 / top-2 | 6.48B / 4.21B | 40.1 / 40.1 | 92.9 / 103.7 | 0.9 / 1.0 |
| DARE 0.2/64, kept values trained | 6.48B / 6.47B | 51.0 | 111.9 | 1.1 |
| dense deltas (`DeltaMoE`) trained | 15.53B / 15.53B | 118.7 | out of memory | 0.8 |

The "none" column predates `DeltaMoE._mlp_remat`. With it (the materialised weights recomputed in the backward; same H200, 2026-11-27), without checkpointing: DARE frozen top-1 92.9 → 71.7 GiB (0.6 → 0.8 s/step), DARE trained 111.8 → 90.7 GiB (0.8 → 0.9 s/step); dense deltas trained still out of memory; with checkpointing unchanged (118.7 GiB).

## Per implementation, previous kernel (single MoE layer, bf16, eager, ms)

The per-implementation tables below were measured with the previous kernel (5522b23; one launch per projection). Its "param Triton" column is superseded by the section above.

### H200 (sm90, 4.8 TB/s)

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

### RTX 6000 Ada (sm89, 960 GB/s)

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
