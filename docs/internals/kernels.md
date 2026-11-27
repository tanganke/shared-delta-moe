# Kernels

Which forward runs when is in [Forward dispatch](dispatch.md); every timing is in [Benchmarks](benchmarks.md).

- [Fused Triton kernel](fused-triton.md): parameter aggregation, $W_b + \Delta$ added in registers inside the GEMM; dense,
  DARE, BitDelta and 2-bit group-grid deltas; inference and (dense) training.
- [CUDA decode kernels](cuda-decode.md): DARE and BitDelta at a few (token, slot) pairs on Hopper, both aggregations.
- [Dense prefill](dense-prefill.md): from a few hundred pairs per group, materialised $W_b + \Delta$ and grouped GEMMs.

```{toctree}
:maxdepth: 1

fused-triton
cuda-decode
dense-prefill
```
