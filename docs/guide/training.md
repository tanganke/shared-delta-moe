# Training

`forward_param` is differentiable. By default, with a gradient needed it runs the materialised path (cuBLAS on $W_b+\Delta_e$, any top-k, gradients to x, the routing weights and every weight). Autograd does not keep the materialised $W_b+\Delta_e$ for the backward: `DeltaMoE._mlp_remat` swaps each saved copy for a recipe and recomputes the sum (one add per weight, plus the unpacking for DARE / BitDelta) when the backward needs it. Without gradient checkpointing this cut a whole Qwen3.5-4B DARE step from 92.9 to 71.7 GiB (trained kept values: 111.8 to 90.7 GiB) for 0.1-0.2 s more per step; under checkpointing (or any other saved-tensor hooks), under autocast, or with `m.remat = False` it stays off, as the checkpoint already keeps only the layer input. `tests/test_remat.py` checks values and every gradient unchanged.

**Fused Triton backward (`m.triton_train = True`, dense deltas, `triton_kernels/train.py`).** One autograd.Function around the fused forward that saves only its inputs (x, the routing, the weights): no $W_b+\Delta_e$ and nothing per (token, slot) pair is held for the backward. The backward recomputes $g$, $u$ in the kernel that forms $dh' = dy\,(W_d+D_d)$ (all three GEMMs share the output tile), rounding $g$, $u$, SiLU($g$) and $h$ as the forward does, so $dL/dr$ is the gradient of that forward; $dx$ runs through the down kernel on $[dL/dg \mid dL/du]$; weight gradients use `F.grouped_mm` over the sorted pairs (the base gets the fp32 sum over every group). It is taken with `triton_train` set, Triton on CUDA, weights and x in bf16/fp16, hidden and inter multiples of 8 (`grouped_mm` needs 16-byte rows), and any pair count; otherwise the materialised path. `HierDeltaMoE` ([hier.md](hier.md)) trains through it unchanged; DARE deltas do not (materialised path). `tests/test_train.py` checks output and every gradient against the materialised path.

Measured on the H200 it is slower than the materialised path (as the first version, commit e0ce27a, was), and it
saves little memory, since weight gradients dominate when the weights train (tables:
[Benchmarks](../internals/benchmarks.md#training)). With few experts the materialisation is one pass over the weights, and cuBLAS GEMMs on dense experts beat the Triton GEMMs; the Triton path is for when a lower peak matters more than speed (about 35% lower when only the router trains, 3-18% with every weight). (On an RTX 6000 Ada at hidden 512 / inter 1024 it was 4-7x faster; not measured there at full size.) The earlier grouped-GEMM variant of the same Function (also e0ce27a) stays removed.

`DareDeltaMoE` trains through the same path: `delta(e)` unpacks only the routed expert from the bitmap (`masked_scatter` of the kept values, differentiable in them), so the kept values can be made `nn.Parameter`s and trained. Unpacking with Philox on every access of `d_gate` / `d_up` / `d_down` (all experts each time) made a step 7.6 s. Peak memory and step time of a whole Qwen3.5-4B with DARE deltas are in
[Benchmarks](../internals/benchmarks.md#training). With gradient checkpointing, frozen DARE deltas cost only their own 4.2 GiB. Without it, the materialised $W_b+\Delta_e$ (142 MB per expert and layer) used to be held for the backward; it is now recomputed (above).

## Training BitDelta scales

The scales are `nn.Parameter`s, so BitDelta's scale distillation can be run through the materialised path.

- **What trains.** `delta(e)` unpacks only the routed expert, and $\mathrm{d}L/\mathrm{d}s_{e,r} = \sum_c \mathrm{d}L/\mathrm{d}\Delta_{e,r,c}\,\mathrm{sign}_{e,r,c}$ (checked in the tests). The signs are fixed.
- **No Triton backward.** With a gradient needed, `_use_triton` refuses the fused kernels, including `triton_train`,
  whose backward takes dense deltas only.

