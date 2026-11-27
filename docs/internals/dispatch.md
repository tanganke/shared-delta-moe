# Forward dispatch

Notation ($W_b$, $\Delta_e$, slots, routing weights $r_i$, zero experts): [design.md](../guide/deltamoe.md).

| Public method | Computation | Implementations (private), picked automatically |
|---|---|---|
| `forward_param` | Parameter aggregation: each (token, slot) pair is multiplied by $W_b + \Delta_e$ of its expert, weighted by $r_i$ | `_forward_param_triton`: three Triton launches for all pairs (sort; gate/up with SiLU·up; down with the routing weight), adding $W_b$ and $\Delta_e$ tiles in registers (no weights written, 1× FLOPs); `_forward_param(x, min_rows)`: materialise $W_b + \Delta_e$ per expert and run cuBLAS (experts with fewer than `min_rows` rows compute $W_b x + \Delta_e x$ instead) |
| `forward_matrix` | Per-matrix output aggregation: gate/up compute $W_b x$ once for all tokens, plus $\Delta_e x$ per slot. Each slot keeps its own activation $h_i$; down computes $W_b h_i + \Delta_e h_i$; the slots are summed with weights $r_i$ | `_forward_matrix_grouped`: flattens all (token, slot) pairs, sorts them by expert, one `F.grouped_mm` per projection for the deltas; `_forward_matrix`: loop over experts |
| `MoE.forward` (`to_moe()`) | Reference: the plain MoE, full expert weights stored | `_forward_grouped`: one `F.grouped_mm` per projection, like transformers' `grouped_mm` path; `_forward_loop`: loop over experts |

Selection rules (thresholds measured on an H200 and an RTX 6000 Ada at hidden 2560, inter 9216, 5 deltas + zero expert; retune for other GPUs or shapes):

- The module itself (`moe(x)`, `x` of any leading shape; what the Qwen3.5 blocks run): `forward_param`, except that `DareDeltaMoE` / `BitDeltaMoE` take `forward_matrix` on the CUDA kernels up to `MATRIX_FORWARD_PAIRS` (token, slot) pairs (DARE 256, BitDelta 64; H200 crossovers in the tables below) when those kernels apply. The two forwards round differently (about 1e-3 relative), so with packed deltas a batch's output depends slightly on its size.
- `forward_param`: Triton on CUDA bf16/fp16 without gradients, up to 8192 (token, slot) pairs on Hopper (sm90+) or 16384 on Ada and older. Beyond the limit the work is compute-bound, and cuBLAS on materialised weights wins; on Hopper that happens earlier, because cuBLAS's wgmma kernels are better than Triton's. With gradients (training) it takes the materialised path, or the fused Triton backward when `triton_train` is set (dense deltas); see [training.md](../guide/training.md). Triton (with or without gradients) also needs the routing sort to fit (`SORT_ONE_HOT_MAX`). The kernel itself: [fused-kernel.md](fused-triton.md).
- `forward_matrix` of `DareDeltaMoE` / `BitDeltaMoE`: the CUDA kernels below when they apply. Otherwise, and for the other classes:
- `forward_matrix` / `MoE.forward`: grouped GEMM when `F.grouped_mm` exists, the dtype is bf16/fp16/fp32, the device is CPU or a CUDA GPU with sm80+, and there is more than one pair; otherwise the loop.

