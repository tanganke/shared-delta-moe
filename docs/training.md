# Training

`forward_param` is differentiable. By default, with a gradient needed it runs the materialised path (cuBLAS on $W_b+\Delta_e$, any top-k, gradients to x, the routing weights and every weight).

**Fused Triton backward (`m.triton_train = True`, dense deltas, `triton_kernels/train.py`).** One autograd.Function around the fused forward that saves only its inputs (x, the routing, the weights): no $W_b+\Delta_e$ and nothing per (token, slot) pair is held for the backward. The backward recomputes $g$, $u$ in the kernel that forms $dh' = dy\,(W_d+D_d)$ (all three GEMMs share the output tile), rounding $g$, $u$, SiLU($g$) and $h$ as the forward does, so $dL/dr$ is the gradient of that forward; $dx$ runs through the down kernel on $[dL/dg \mid dL/du]$; weight gradients use `F.grouped_mm` over the sorted pairs (the base gets the fp32 sum over every group). It is taken with `triton_train` set, Triton on CUDA, weights and x in bf16/fp16, hidden and inter multiples of 8 (`grouped_mm` needs 16-byte rows), and any pair count; otherwise the materialised path. `HierDeltaMoE` ([hier.md](hier.md)) trains through it unchanged; DARE deltas do not (materialised path). `tests/test_train.py` checks output and every gradient against the materialised path.

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

With few experts the materialisation is one pass over the weights, and cuBLAS GEMMs on dense experts beat the Triton GEMMs; the Triton path is for when a lower peak matters more than speed (about 35% lower when only the router trains, 3-18% with every weight). (On an RTX 6000 Ada at hidden 512 / inter 1024 it was 4-7x faster; not measured there at full size.) The earlier grouped-GEMM variant of the same Function (also e0ce27a) stays removed.

`DareDeltaMoE` trains through the same path: `delta(e)` unpacks only the routed expert from the bitmap (`masked_scatter` of the kept values, differentiable in them), so the kept values can be made `nn.Parameter`s and trained. Unpacking with Philox on every access of `d_gate` / `d_up` / `d_down` (all experts each time) made a step 7.6 s. Whole Qwen3.5-4B text model, full-parameter training, batch 2 x 2048, bf16 weights, fused AdamW (bf16 states), random router (every delta routed: the worst case), H200, peak GiB and s/step:

| model | params / trainable | peak, checkpointing | peak, none | s/step, checkpointing |
|---|---|---:|---:|---:|
| dense Qwen3.5-4B | 4.21B / 4.21B | 35.9 | 65.5 | 0.4 |
| DARE 0.2/64, kept values frozen, top-1 / top-2 | 6.48B / 4.21B | 40.1 / 40.1 | 92.9 / 103.7 | 0.9 / 1.0 |
| DARE 0.2/64, kept values trained | 6.48B / 6.47B | 51.0 | 111.9 | 1.1 |
| dense deltas (`DeltaMoE`) trained | 15.53B / 15.53B | 118.7 | out of memory | 0.8 |

With gradient checkpointing, frozen DARE deltas cost only their own 4.2 GiB. Without it, every materialised $W_b+\Delta_e$ is held for the backward (142 MB per expert and layer).
