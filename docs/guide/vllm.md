# Serving with vLLM

`vllm_plugin.py` is a vLLM general plugin (entry point in `pyproject.toml`): with the package installed in vLLM's
environment, vLLM serves a converted checkpoint as it is, with no materialisation, every decoder MLP a
`HierDeltaMoE` on the fused Triton kernel:

```bash
vllm serve DST --tensor-parallel-size 2
```

Tensor parallelism splits the intermediate dimension ($W_g$, $W_u$ and their deltas by rows, $W_d$ and its deltas by
columns), the router is replicated, and an all-reduce sums the down projections. torch.compile and CUDA graphs work
(the kernel is one custom op). Dense MLP deltas only: checkpoints with grid MLP deltas (`hier_delta_moe.levels`)
or [per-expert attention and lm_head deltas](routed-deltas.md) (`routed_deltas`) are refused at load. Checked with vLLM 0.29.0 on the synthetic Qwen3.8-27B upscale (3
fine-tunes in 2 groups, random routers), TP=2 on H200: coherent greedy output; per rank 74.6 GiB of weights.
