# Training

`forward_param` is differentiable: with a gradient needed it runs the materialised path (cuBLAS on $W_b+\Delta_e$, any top-k, gradients to x, the routing weights and every weight). Two fused alternatives were built and measured (commit e0ce27a, then removed): a Triton backward that recomputes $h$ (one kernel computing $g$, $u$ and $dh'$; $dx$ through the down kernel; weight grads via `F.grouped_mm`), and the same Function on `F.grouped_mm` over stacked experts. Both were slower than the materialised path. Router-only training, top-2 / top-6, ms per forward+backward (peak GB above the weights):

| GPU, tokens, top-k | fused Triton | grouped GEMM | **materialised** |
|---|---:|---:|---:|
| H200, 4096, top-2 | 11.66 (0.87) | 8.53 (1.29) | **5.97** (1.19) |
| H200, 25000, top-2 | 64.41 (2.60) | 39.91 (3.85) | **27.93** (4.03) |
| H200, 25000, top-6 | 200.99 (6.52) | 112.32 (9.95) | **79.38** (10.84) |
| Ada, 4096, top-2 | 36.47 (0.87) | 39.43 (1.30) | **27.78** (1.20) |
| Ada, 25000, top-6 | 704.94 (6.52) | 610.53 (9.96) | **478.93** (10.84) |

With few experts (6), the materialisation costs one pass over the weights, and cuBLAS GEMMs on dense experts beat both the Triton GEMMs and `grouped_mm`. The fused backward only saves about 40% of activation memory.

`DareDeltaMoE` trains through the same path: `delta(e)` unpacks only the routed expert from the bitmap (`masked_scatter` of the kept values, differentiable in them), so the kept values can be made `nn.Parameter`s and trained. Unpacking with Philox on every access of `d_gate` / `d_up` / `d_down` (all experts each time) made a step 7.6 s. Whole Qwen3.5-4B text model, full-parameter training, batch 2 x 2048, bf16 weights, fused AdamW (bf16 states), random router (every delta routed: the worst case), H200, peak GiB and s/step:

| model | params / trainable | peak, checkpointing | peak, none | s/step, checkpointing |
|---|---|---:|---:|---:|
| dense Qwen3.5-4B | 4.21B / 4.21B | 35.9 | 65.5 | 0.4 |
| DARE 0.2/64, kept values frozen, top-1 / top-2 | 6.48B / 4.21B | 40.1 / 40.1 | 92.9 / 103.7 | 0.9 / 1.0 |
| DARE 0.2/64, kept values trained | 6.48B / 6.47B | 51.0 | 111.9 | 1.1 |
| dense deltas (`DeltaMoE`) trained | 15.53B / 15.53B | 118.7 | out of memory | 0.8 |

With gradient checkpointing, frozen DARE deltas cost only their own 4.2 GiB. Without it, every materialised $W_b+\Delta_e$ is held for the backward (142 MB per expert and layer).
