# shared-delta-moe

A top-k SwiGLU MoE stored as one shared base expert plus a delta per expert, $W_e = W_b + \Delta_e$, with
dense, seeded-DARE, 1-bit BitDelta and two-level (`HierDeltaMoE`) deltas, a fused Triton forward and backward, and
Qwen3.5 integration in transformers. Setup and the repo layout are in the
[README](https://github.com/tanganke/shared-delta-moe#readme).

```{toctree}
:caption: Model
:maxdepth: 2

design
forwards
hier
```

```{toctree}
:caption: Delta formats
:maxdepth: 2

dare
bitdelta
```

```{toctree}
:caption: Kernels and training
:maxdepth: 2

fused-kernel
training
```

```{toctree}
:caption: transformers
:maxdepth: 2

qwen3_5
transformers-decode
```

```{toctree}
:caption: Measurements
:maxdepth: 2

speed-per-implementation
```

```{toctree}
:caption: Reference
:maxdepth: 2

api
```
