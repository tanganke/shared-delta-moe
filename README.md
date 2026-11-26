# shared-delta-moe

A small Python library: a top-k SwiGLU MoE written as **a shared base expert plus per-expert deltas**, with several forward implementations that are exactly equivalent to the original MoE, including a Triton kernel that fuses $W_b+\Delta_e$ into the GEMM.

```
src/shared_delta_moe/models/delta_moe.py  DeltaMoE: from_experts, routing, the forward implementations
src/shared_delta_moe/ops.py               stateless helpers: mlp, grouping routed rows, one grouped GEMM
src/shared_delta_moe/triton_kernels/      fused.py: the fused parameter-aggregation forward (sort, gate_up + SiLU,
                                          down + routing weight; dense or seeded-DARE deltas; Hopper TMA); train.py:
                                          its backward for training (dense deltas, triton_train)
src/shared_delta_moe/models/hier.py       HierDeltaMoE: groups of deltas + zero experts, two-level routing
src/shared_delta_moe/models/dare.py       seeded DARE: Philox mask, its bitmap, pack, DareDeltaMoE
src/shared_delta_moe/models/bitdelta.py   BitDelta: 1-bit sign deltas with per-matrix or per-row scales, BitDeltaMoE
src/shared_delta_moe/models/qwen3_5/qwen3_5_delta_moe/
                                          transformers' Qwen3.5 MoE with DeltaMoE blocks (forward_param): modeling.py
                                          (patches the decoder layer for configs with delta_moe_zero_experts), convert.py
                                          (plain MoE checkpoint -> delta layout)
src/shared_delta_moe/models/qwen3_5/qwen3_5_hier_delta_moe/
                                          dense Qwen3.5 / Qwen3.8 with HierDeltaMoE MLPs: modeling.py, convert.py
                                          (base + domain fine-tunes -> one group per domain, other tensors merged)
src/shared_delta_moe/merging.py           merge methods for the shared tensors (simple_average, task_arithmetic, ties)
src/shared_delta_moe/constants.py         HAS_TRITON and the thresholds that pick an implementation
src/shared_delta_moe/type_hints.py        shape-annotated tensor aliases (T tokens, H hidden, ...)
src/shared_delta_moe/testing.py           random_experts for tests and benchmarks
tests/test_delta_moe.py                   every forward == plain MoE, token by token (Triton part needs CUDA)
tests/test_dare.py                        Philox known-answer, CPU == CUDA masks, bitmap == mask, DARE forwards and gradients
tests/test_hier.py                        HierDeltaMoE forwards and gradients == the two-level ensemble, token by token
tests/test_qwen3_5_hier_delta_moe.py     tiny dense Qwen3.5 upscaled == ensemble of its fine-tunes' MLPs
tests/test_train.py                       the Triton backward == the materialised path (x, router, every weight; Hier)
tests/test_bitdelta.py                    sign packing, BitDelta scales, forwards == plain MoE, scale gradients, Triton mode 2
tests/test_qwen3_5_delta_moe.py           tiny Qwen3.5 MoE: converted == stock MoE at top-1, top-2, soft routing
tests/test_fused.py                       the fused kernels: sort, every autotune config, split-K, int64 offsets,
                                          CUDA graph, torch.compile
benchmarks/bench.py                       per-layer time of each implementation (CUDA)
benchmarks/bench_train.py                 training step: Triton backward vs materialised, time and memory
```

```bash
uv sync                                  # or: pip install -e .   (Triton ships with torch on Linux + CUDA)
uv run pytest                            # or: python tests/test_delta_moe.py
uv run python benchmarks/bench.py --top-k 2 --tokens 1,32,512,8192
```

```python
from shared_delta_moe import DeltaMoE
m = DeltaMoE.from_experts(base, experts, router, top_k=1, num_zero_experts=1)
y = m.forward_param(x)   # or forward_matrix(x); forward_dense(x, m.dense_experts()) is the plain-MoE baseline
```

## Docs

Details live in `docs/`, one topic per file; this README keeps only the layout, setup and this index.

- [design.md](docs/design.md): the delta form, zero experts, routing, what is and is not equivalent, `from_experts`, dtypes
- [forwards.md](docs/forwards.md): `forward_param` / `forward_matrix` / `forward_dense` and how an implementation is picked
- [fused-kernel.md](docs/fused-kernel.md): the fused Triton forward (sort, gate_up, down; CUDA graph, TMA, split-K, rounding) and its speed
- [dare.md](docs/dare.md): seeded DARE deltas, the mask bitmap, speed
- [bitdelta.md](docs/bitdelta.md): 1-bit BitDelta deltas (sign bits plus scales), the kernel's mode 2, speed
- [training.md](docs/training.md): gradients through `forward_param`, the fused Triton backward (`triton_train`), DARE training memory
- [hier.md](docs/hier.md): the two-level shared delta MoE (`HierDeltaMoE`): groups, dense or top-k group routing, zero experts; upscaling a dense Qwen3.5 / Qwen3.8 and its domain fine-tunes (`qwen3_5_hier_delta_moe`, merge methods)
- [qwen3_5.md](docs/qwen3_5.md): Qwen3.5 MoE in transformers with DeltaMoE blocks, checkpoint converter
- [speed-per-implementation.md](docs/speed-per-implementation.md): every implementation per layer, H200 and RTX 6000 Ada (previous kernel 5522b23)
- [transformers-decode.md](docs/transformers-decode.md): why the plain MoE looked slow in transformers' `generate()` (`batched_mm` switch) and the fix
