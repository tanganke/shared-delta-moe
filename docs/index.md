# shared-delta-moe

A top-k SwiGLU MoE stored as one shared base expert plus a delta per expert, $W_e = W_b + \Delta_e$, with
dense, seeded-DARE and 1-bit BitDelta deltas, a two-level variant (`HierDeltaMoE`), fused Triton and CUDA kernels,
Qwen3.5 integration in transformers and a vLLM plugin. Source:
[github.com/tanganke/shared-delta-moe](https://github.com/tanganke/shared-delta-moe).

```{toctree}
:maxdepth: 1

getting-started
```

```{toctree}
:caption: User guide
:maxdepth: 2

guide/deltamoe
guide/hier
guide/moe
guide/delta-formats
guide/qwen3_5
guide/vllm
guide/training
guide/router-training
guide/similarity
```

```{toctree}
:caption: Internals
:maxdepth: 2

internals/dispatch
internals/kernels
internals/benchmarks
```

```{toctree}
:caption: Notes
:maxdepth: 1

internals/transformers-decode
internals/cuda-dense-kernel
```

```{toctree}
:caption: Reference
:maxdepth: 2

api
```
