# Serving with vLLM

`vllm_plugin.py` is a vLLM general plugin (entry point in `pyproject.toml`). With the package installed in vLLM's
environment, vLLM serves a converted HierDeltaMoE checkpoint as it is, with no materialisation:

```bash
vllm serve DST --tensor-parallel-size 2
```

Without installing, put `src` on `PYTHONPATH` together with a directory holding a `*.dist-info` with the entry
point (`[vllm.general_plugins]` `shared_delta_moe = shared_delta_moe.vllm_plugin:register` in `entry_points.txt`):
vLLM finds plugins only through package metadata, in every process, workers included.

## What runs

- **Decoder MLPs:** every one is a `HierDeltaMoE` on the fused Triton kernel, with dense deltas or, with
  `hier_delta_moe.levels`, 2-bit [group grids](grid.md) (mode 3 of the kernel, decoded tile by tile).
- **[Routed attention and `lm_head` deltas](routed-deltas.md)** (`routed_deltas`, 2-bit grids only): they ride on
  vLLM's own projections, `qkv_proj` and `o_proj` in attention, `in_proj_qkvz`, `in_proj_ba` and `out_proj` in
  linear attention, and `lm_head`. Each projection's `quant_method.apply` adds, after the base product $W_b x$,

  $$
  \sum_i r_i \, \Delta_{e_i} x
  $$

  for this rank's shard, with one more fused launch. The block's `delta_router` routes the block input in its
  first projection, and the block's other projections reuse that routing.

## Tensor parallelism

- **MLPs:** split along the intermediate dimension I, as vLLM's own MLP. $W_g$, $W_u$ and their deltas are split
  by rows, $W_d$ and its deltas by columns. Grid codes, steps and lowest levels are split along the same dimension,
  so the grid group must divide I / tp (Qwen3.8-27B: 17408 / 8 = 2176, which 128 divides). An all-reduce sums the down
  projections.
- **Routed deltas:** `codes`, `steps` and `lows` get the expert dimension in front and are loaded by the
  projection's own weight loader. They are therefore sharded exactly as the weight: the merged shards of
  `qkv_proj`, `in_proj_qkvz` and `in_proj_ba`, replicated KV heads, and `lm_head`'s padded vocabulary (padding rows
  stay zero). Row-parallel projections need their per-rank input width divisible by the group.
- **Routers:** replicated, so every rank routes alike.

torch.compile and CUDA graphs work: the MLP and the routed deltas are registered custom ops.

`hier_delta_moe.force: [g, e]` in the text config serves the model with every token forced to one expert
([forced routing](hier.md)). For example, a directory of symlinks to the checkpoint with an edited `config.json`.

## Checks

A tiny Qwen3.5 (4 layers, 3 fine-tunes in 2 groups, random routers), converted as dense, grid MLP, routed, and grid
MLP + routed with sets. vLLM's prompt log-probabilities against the HF model:

| variant | TP | mode | mean abs. diff | max abs. diff |
|---|---|---|---:|---:|
| the base, no plugin | 1 | eager | 0.0039 | 0.013 |
| dense | 1 | eager | 0.0031 | 0.009 |
| grid MLP | 1 | eager | 0.0036 | 0.010 |
| routed | 1 | eager | 0.0047 | 0.017 |
| grid MLP + routed, sets | 1 | eager | 0.0041 | 0.016 |
| grid MLP + routed, sets | 2 | eager | 0.0043 | 0.023 |
| grid MLP + routed, sets | 2 | compiled, CUDA graphs | 0.0050 | 0.017 |

All are at the base model's own bf16 difference. Checked with vLLM 0.32.0 on H200.

Qwen3.8-27B moe27-14-B (14 experts, top-2, 2-bit grid MLP deltas and 2-bit routed deltas in 4 sets), TP=2 on two
H200, greedy decode of 256 tokens, answers coherent:

| batch | eager, Triton only | compiled, CUDA graphs, CUDA matrix kernels | + grid kernels (128-entry groups) | + `routed_small` |
|---:|---:|---:|---:|---:|
| 1 | 7.1 tok/s | 36.5 tok/s | 39.6 tok/s | 44.8 tok/s |
| 16 | 50.3 tok/s | 340 tok/s | 385 tok/s | 418 tok/s |
| 64 | 215 tok/s | 527 tok/s | 681 tok/s | 688 tok/s |

Run with `SHARED_DELTA_MOE_CUDA_DIR` set (the prebuilt kernels, [CUDA decode kernels](../internals/cuda-decode.md))
and, at TP > 1, `VLLM_DISABLE_COMPILE_CACHE=1`: a cached compiled artifact had device 0 baked in and failed on rank 1.
The cache key also misses changes to this package's ops: after an upgrade, a cached artifact failed with "too many
values to unpack", so disable the cache or clear `VLLM_CACHE_ROOT` then too.

The fused Triton kernels autotune on the first batch of each new size class (pair count rounded up to a power of
4), which takes minutes for a large model; results then stay in `TRITON_CACHE_DIR` for later runs. vLLM's warmup
covers the decode sizes, but a first long prefill can still tune past `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (300 s),
which kills the engine: set it to 3600 for a first run, or set `SHARED_DELTA_MOE_AUTOTUNE=0` (first fitting config,
no tuning).
