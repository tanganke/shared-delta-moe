# Dense prefill

From `DENSE_PREFILL_PAIRS_PER_GROUP` (640) × (`n_delta` + 1) (token, slot) pairs on, with dense deltas on sm90, the
fused forward (`triton_kernels.fused.fused_moe`, so `forward_param`, the `fused_moe_dense` op under `torch.compile`
and the vLLM plugin) leaves the Triton GEMMs for `cuda_kernels/dense_moe.py`:

1. The Triton routing sort, as before.
2. $W_b + \Delta_g$ for every group with rows into a buffer per projection, (G + 1, N, K), with the base in the
   last slot for the zero expert. One Triton add per projection; an empty group is skipped on the device, so there is
   no host sync. The buffers are allocated per call and freed when it returns (the caching allocator, or a CUDA
   graph's private pool, owns them): a transient peak of 3 · (G + 1) · I · H elements during the call (27B shard in
   bf16: 1.07 GB). Reusing them across calls saves nothing measurable (below).
3. `F.grouped_mm` (CUTLASS on sm90) for gate and up, SiLU · up in torch, `F.grouped_mm` for down, then back to pair
   order (a gather) times the routing weight, the slots summed. These are the materialised forward's own ops, so
   the rounding (g, u, SiLU(g), h, y, routing weight in bf16) is the model's; at 2048 tokens and more the output
   matches `_forward_param` bit for bit.

At large P the GEMMs are compute-bound and every weight tile is reused by many row tiles, so writing the sums once
(a few hundred MB, 0.2–0.5 ms) costs less than what CUTLASS's shared-memory-operand wgmma gains over a GEMM whose
weight operand is summed in registers (Triton: 34–39% of peak). Below the threshold, the extra pass over the
weights costs more than that, and Triton stays.

Inference only (bf16, no input that needs a gradient; fp16 would hit torch's host-synchronising grouped-GEMM
fallback), contiguous weights. The routing sort's Triton kernel and the add need Triton; no compiled extension is
needed, so the vLLM image (no nvcc) runs it as is. With dense deltas on Hopper, `forward_param` has no upper pair
limit: past `TRITON_MAX_PAIRS_HOPPER` it stays on the fused forward (grouped path, or Triton when `supports` says
no) instead of the host-synchronising materialised loop, so a prefill of any size can be captured in a CUDA graph.

