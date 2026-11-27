# CUDA decode kernels

Prebuilt CUDA kernels for DARE, BitDelta and (output aggregation) 2-bit grid deltas at decode sizes, on sm_90 (H100 / H200). Two families: parameter aggregation inside `forward_param`, and output aggregation behind `forward_matrix`. Speed tables: [Benchmarks](benchmarks.md).

## Parameter aggregation (`forward_param`)

At up to 16 (token, slot) pairs, `forward_param` runs prebuilt CUDA kernels instead of the Triton ones
(`cuda_kernels/packed_moe.cu`, `packed_ops.py`). They cover DARE blocks of 8 or more and BitDelta, with hidden and inter
multiples of 64, on sm_90 (H100 / H200) only: the build targets sm_90a and the layout needs Hopper's 227 KB of
shared memory per CTA, so on any other GPU, and for shapes or tensors they do not cover, the forward runs on Triton.
A warp owns 16 weight rows over a K range and reads $W_b$ once per 64-wide K step. For each routed
group it adds the kept values in registers and multiplies with that group's pairs on the tensor cores. Popcounts over
the keep bitmap place the values, which stream through a per-warp `cp.async` pool that holds only what is kept.
Rounding is the same as in the Triton kernels; split-K sums run in a fixed order. For BitDelta, a byte of sign bits
indexes a 256-entry table of sign-flip masks, so $W_b \pm s$ costs one XOR and one bf16x2 add per two entries; the
kernels cover one scale per row or matrix, or blocks of 32 or more.

Build once per torch version on a node with nvcc (here a 2B compute node, `module load cuda/13.0`), writing
`sdm_packed-<torch>-<source hash>.so` (an edited `packed_moe.cu` is never matched with an old build) into a directory the runtime also sees, e.g. on /data for the vLLM image:

```bash
SHARED_DELTA_MOE_CUDA_DIR=/data/projects/11620001/anketang/sdm-cuda python -m shared_delta_moe.cuda_kernels.packed_ops
```

The runtime loads the file by path, with the same variable and no nvcc. Without the library, if it fails to load, or with
`SHARED_DELTA_MOE_CUDA=0`, everything runs on Triton. Under `torch.compile` the kernels run inside the same op
(`shared_delta_moe::fused_moe_dare`), so CUDA graphs work as before. `tests/test_packed_cuda.py` checks them against
the materialised forward: blocks 8 to 128, zero experts, top-2, split-K, fp16, a CUDA graph and `torch.compile`.

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

## Output aggregation (`forward_matrix`)

`forward_matrix` of `DareDeltaMoE` / `BitDeltaMoE` runs on prebuilt CUDA kernels (`cuda_kernels/packed_moe.cu`,
`packed_ops.matrix_moe`; build as in [the section above](#parameter-aggregation-forward-param)) when the library is loaded, the GPU is
sm_90, no gradient is needed and there are at most `MATRIX_MAX_PAIRS` (1024) (token, slot) pairs; otherwise the
inherited forward runs on unpacked deltas. cuBLAS computes $x W_g^T$ and $x W_u^T$ for all tokens and $h W_d^T$ for
all pairs, so $W_b$ is read once however many groups are routed. Two kernels add only the deltas' products: gate/up
with SiLU·up per pair, then down with the routing weight; the slots are summed in torch. The parameter-aggregation
kernels instead rebuild $W_b + \Delta_g$ for every routed group.

- DARE: below about 5 pairs per group, a warp takes one row on CUDA cores. It writes the row's kept block indices to
  shared memory once, then streams the kept values with the activations they meet. From about 5 pairs per group, the
  tensor-core kernel below runs, with zeros for the dropped blocks and a running popcount to find the kept values.
- BitDelta: a CTA takes 16 rows and its warps split K, with the pairs as the N side of `mma.m16n8k16`. A byte of sign
  bits indexes a table of sign-flip masks XORed into the scale, as in the parameter-aggregation kernel.
- 2-bit group grids (mode 3, `GridDeltaMoE`): the same tensor-core kernel. Each lane's 8 entries of a 32-wide chunk
  are half a code word, decoded in registers to $\ell + s\,q$ (the group's lowest level and step, $q \in \{0, \dots, 3\}$)
  and packed into bf16 pairs. Groups of 32 or more (a power of 2), rows a multiple of 16. `GridDeltaMoE` takes this
  path up to `MATRIX_MAX_PAIRS` pairs, and so does the vLLM plugin's grid MLP. With groups a multiple of 128 and an
  even number of them per row, the grid kernels below run instead.
- Routed deltas ([per-expert attention and lm_head deltas](../guide/routed-deltas.md)), 2-bit grids: the down kernel
  alone, over a zero base product (`packed_ops.routed_matrix`), up to 64 pairs; beyond that the Triton down kernel.

### Grid kernels (`grid_gate_up`, `grid_down`)

Over a 128-entry block $b$ inside one group, every entry is $\delta_j = \ell + s\,q_j$, so for an activation $x$

$$
\sum_{j \in b} \delta_j x_j = s \sum_{j \in b} q_j x_j + \ell X_b, \qquad X_b = \sum_{j \in b} x_j .
$$

The tensor cores then only need $q_j \in \{0, \dots, 3\}$, which needs no arithmetic to decode. The kernel ORs the
two code bits into the mantissa of 1.0: $v_j = 1 + q_j / 4$ (bf16 bits 5–6 of `0x3F80`, fp16 bits 8–9 of `0x3C00`),
one shift and one `lop3` for two entries. Since $\sum_j v_j x_j = X_b + \tfrac14 \sum_j q_j x_j$,

$$
\sum_{j \in b} \delta_j x_j = 4s \, P_b + (\ell - 4s) X_b, \qquad P_b = \sum_{j \in b} v_j x_j ,
$$

with $P_b$ from `mma.m16n8k16` in fp32 and the fold in fp32. Unlike the kernel above, the deltas are not rounded to
bf16, and the error against fp32 is the same (one bf16 ulp of the output). The block sums $X_b$ (fp32) come from
torch. So does a reordering of the activations, `packed_ops._prep`: lane $q$ of a row takes code word $q$ of the
block (entries $32q \dots 32q + 31$), and register $i$ of its half word $w$ holds entries $16w + i$ and $16w + i + 8$
(code bits $2i$ and $2i + 16$). The activations are interleaved to match, `x'[16w + 2i + j] = x[16w + i + 8j]`, so
that a lane's B fragment is two plain 32-bit loads.

There are two kernels, chosen by the pairs of a busy group, about $(2P + G) / (G + 1)$:

- **Up to 8 (`grid_small_kernel`)**, batch 1 to about 16: a CTA owns 16 rows and its 4 warps split K. The next
  block's codes, scales, $X_b$ and activations are loaded into registers while the current one computes.
- **More (`grid_kernel`)**: a CTA owns 64 rows (a warp each) and runs 16 or 32 pairs per pass. A `cp.async`
  pipeline of 3–4 shared-memory stages holds each block's codes, scales, activations and $X_b$. Each thread works out
  its copy addresses once per pass, and the block loop only adds a stride.

Qwen3.8-27B MLP shapes at TP=2 on H200 (G = 14, H = 5120, I = 8704 per rank, codes 37.6 MB per expert and layer), one
layer's MoE deltas, CUDA graph timing. The bandwidth is that of the codes read:

| tokens, top-k | experts read | kernel above | grid kernels |
|---|---:|---:|---:|
| 1, top-2 | 2 | 0.124 ms (0.61 TB/s) | 0.059 ms (1.27 TB/s) |
| 16, top-2 | 13 | 0.711 ms (0.69 TB/s) | 0.328 ms (1.49 TB/s) |
| 64, top-2 | 14 | 1.303 ms (0.40 TB/s) | 0.60 ms (0.87 TB/s) |
| 16, top-10 | 14 | 1.514 ms (0.35 TB/s) | 0.61 ms (0.87 TB/s) |
| 64, top-10 | 14 | 4.589 ms (0.11 TB/s) | 1.49 ms (0.35 TB/s) |

Nsight Compute shows `grid_kernel` bound by instruction issue, not memory: about 300 warp instructions per 128-entry
block and pass, and DRAM at 10–17% of its bandwidth. On 2B it is at
`/data/apps/nvhpc/26.3/Linux_x86_64/26.3/profilers/13.1/Nsight_Compute/ncu` and also runs in the Glutton container.
Moving the address arithmetic out of the block loop gave the last 20%. With many pairs, the per-pair `mma` and fold
instructions set the time, not DRAM. Past 32 pairs per group, each further pass also decodes the codes again.

Qwen3.8-27B shapes at TP=2 on H200 (moe27-14-B: 14 experts top-2 plus one zero expert; routed deltas in 4 sets), per
decode step, summed over the 64 layers:

| batch | grid MLP MoE, Triton | grid MLP MoE, CUDA matrix | dense MLP | routed deltas, Triton | routed deltas, CUDA matrix (to 64 pairs) | base projections |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.3 ms | 14.2 ms | 5.7 ms | 22.1 ms | 11.5 ms | 2.9 ms |
| 16 | 103.5 ms | 48.4 ms | 5.9 ms | 37.0 ms | 15.1 ms | 2.9 ms |
| 64 | 122.0 ms | 72.4 ms | 5.7 ms | 39.6 ms | 33.7 ms | 3.0 ms |

The Triton MoE kernel rebuilds $W_b + \Delta_e$ for each routed expert, so at 16 tokens it reads the 267 MB base once
for each of the 11 experts the batch touches; the matrix path reads it once. The remaining gap is in the delta kernels
(about 0.7 TB/s from the codes) and a fixed 0.06 ms per routed call (routing sort and launches).

Rounding is that of `_forward_matrix_grouped`: $x W_b^T$ and $x \Delta^T$ are each rounded to the model dtype before
their sum, and so are $h$ and $r_i y_i$. `tests/test_packed_matrix.py` checks these kernels against `_forward_matrix`
on the unpacked deltas. It covers DARE blocks 8 to 128, BitDelta matrix, row and block scales, zero experts, top-1 to
top-3, 16 deltas, ragged token counts up to 400 pairs, fp16, a CUDA graph and `torch.compile` (fullgraph, through the
ops `shared_delta_moe::matrix_moe_dare` / `_bit`). The models' default forward is unchanged.

What limits it, and what did not work:

- From 16 tokens up, the delta kernels set the time. The CUDA-core DARE kernel is bound by instructions (one FMA per
  kept value per pair). The tensor-core kernels wait on load latency: each 32-wide chunk loads its signs (or bitmap,
  then values), and 8 pairs at most share them. At 512 tokens output aggregation is 2–5× slower than `forward_param`.
- Reusing the parameter-aggregation kernel with the $W_b$ ring removed (deltas only, epilogue adding $x W_b^T$) was
  slower than this at every size (DARE gate/up 36 µs at 1 token, 517 µs at 64 tokens). Its per-step overhead does not
  shrink with the bytes.
- BitDelta on CUDA cores (each lane adds or subtracts 32 activations per sign word) was instruction-bound already at
  4 tokens: 1.5 ms at 64 tokens, against 312 µs on the tensor cores.
