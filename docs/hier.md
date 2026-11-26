# Two-level shared delta MoE (`HierDeltaMoE`)

An optional shared base expert plus delta experts arranged in groups, routed first over groups and then within the chosen groups
(`models/hier.py`). Group $i$ holds $E_i$ delta experts with weights $W_b+\Delta_{ij}$ and $Z_i$ zero experts (no
parameters, output 0); $E_i$ and $Z_i$ may differ between groups, and a group may hold only zero experts.

- Group router: $h = \mathrm{softmax}(R_0 x)$ over the $G$ groups; dense / soft (`group_top_k=None`: every group,
  weight $h_i$) or top-$k_0$ renormalised over the chosen groups.
- Within group $i$: softmax over its $E_i+Z_i$ logits, top-$k_1$ (`top_k`), renormalised within the group; zero
  experts compete like the others.

$$
y = [\mathrm{MLP}(x; W_b)] + \sum_{i \in S(x)} h_i(x) \sum_{j \in \text{top-}k_1,\ j \text{ a delta}} \tilde p_{ij}(x)\, \mathrm{MLP}(x; W_b+\Delta_{ij})
$$

The bracketed base term is there with `shared_base=True` (the default) and absent with `shared_base=False`, the
form for upscaling a dense model from its fine-tunes (below): with experts close to the base, the base term would make
each MLP about $2f(x)$. Either way this equals the plain two-level ensemble of the full experts $\theta_{ij}=W_b+\Delta_{ij}$ (with the base as a shared
expert of weight 1 when `shared_base`) (`from_experts` takes those full experts and stores $\Delta_{ij}=\theta_{ij}-W_b$). One linear layer
holds both routers: rows $0..G-1$ are $R_0$, then group by group its $E_i$ delta rows and $Z_i$ zero-expert rows.

**Usage.**

```python
import torch
from shared_delta_moe import HierDeltaMoE, random_experts

H, I = 1024, 4096
full = random_experts(H, I, 6, torch.bfloat16, "cuda")    # (gate, up, down) each
base, groups = full[0], [full[1:3], full[3:4], full[4:6]]  # E = 2, 1, 2 experts per group
zeros = [1, 0, 0]                                         # Z: one zero expert in group 0
router = torch.randn(3 + sum(len(g) + z for g, z in zip(groups, zeros)), H, dtype=torch.bfloat16, device="cuda") * H**-0.5
moe = HierDeltaMoE.from_experts(base, groups, router, zeros, top_k=1, group_top_k=None, shared_base=False)

x = torch.randn(32, H, dtype=torch.bfloat16, device="cuda")
y = moe.forward_param(x)  # the fused Triton kernel on CUDA; differentiable (Triton backward with triton_train)
```

`from_experts` takes the full experts $\theta_{ij}$ and stores $\Delta_{ij}=\theta_{ij}-W_b$. To start from shapes
instead, use `HierDeltaMoE(hidden, inter, group_sizes, group_zeros, top_k, group_top_k, shared_base=...)` and load
`gate`/`up`/`down`, `d_gate`/`d_up`/`d_down` and `router.weight`.

**Implementation.** `topk` flattens the routing into ordinary `DeltaMoE` slots: per token $S\cdot k_1$ slots
($S=G$ dense, else $k_0$) weighted $h_i\tilde p_{ij}$, plus, with `shared_base`, one base slot of weight 1; a picked zero expert becomes a base
slot of weight 0. Every `DeltaMoE` forward (materialised, split, grouped, fused Triton) and the Triton training path
([training.md](training.md)) then compute $y$ unchanged, gradients through the two softmaxes included.

**Limits.** Zero-weight base slots are still computed (one base MLP each in the parameter-aggregation forwards):
waste that grows with $S\cdot k_1$ minus the deltas picked. Dense deltas only: `DareDeltaMoE.from_delta_moe` refuses a
`HierDeltaMoE`. The routing tables are built per device on first use (not buffers), so modules made on the meta device,
as transformers' `from_pretrained` does, work. `tests/test_hier.py` checks every forward and the gradients against the ensemble
written out token by token.

## Upscaling a dense Qwen3.5 / Qwen3.8 model from domain fine-tunes

```bash
python -m shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert BASE DST \
    --group medical=FT1,FT2 --group code=FT3 [--zeros 0] [--top-k 1] [--group-top-k K] [--shared-base] \
    [--merge simple_average | task_arithmetic | ties_merging] [--merge-arg scaling_factor=0.3 --merge-arg threshold=20]
```

```python
import shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe  # patches Qwen3_5DecoderLayer
model = Qwen3_5ForConditionalGeneration.from_pretrained(DST)  # or Qwen3_5ForCausalLM for a text-only checkpoint
```

One group per domain; each fine-tune's decoder MLP becomes an expert of its group ($\Delta_{ij}=\theta_{ij}-W_b$,
subtracted in fp32), `BASE`'s MLP is $W_b$. Every other tensor (attention, linear attention, norms, embeddings,
`lm_head`, vision tower, MTP) is merged across the fine-tunes that have it with a method of `merging.py` (a
tensor no fine-tune has, like Qwen3.8's `mtp.*` that transformers drops on load, is kept from the base), named and parametrised
after fusion_bench (`simple_average` default; TIES trims per tensor, not over the whole task vector). Routers start at
zero (uniform), to be trained. The text config gets `hier_delta_moe` (groups, sizes, zeros, top-k, `shared_base`,
merge). Tensors are streamed from the safetensors shards and written in shards (`--shard-gb`), so memory stays at a
few tensors. `tests/test_qwen3_5_hier_delta_moe.py` converts a tiny model with perturbed fine-tunes and compares its
logits with the stock dense model running the ensemble of the fine-tunes' full MLPs.

## Serving with vLLM

`vllm_plugin.py` is a vLLM general plugin (entry point in `pyproject.toml`): with the package installed in vLLM's
environment, vLLM serves a converted checkpoint as it is, with no materialisation, every decoder MLP a
`HierDeltaMoE` on the fused Triton kernel:

```bash
vllm serve DST --tensor-parallel-size 2
```

Tensor parallelism splits the intermediate dimension ($W_g$, $W_u$ and their deltas by rows, $W_d$ and its deltas by
columns), the router is replicated, and an all-reduce sums the down projections. torch.compile and CUDA graphs work
(the kernel is one custom op). Dense deltas only. Checked with vLLM 0.29.0 on the synthetic Qwen3.8-27B upscale (3
fine-tunes in 2 groups, random routers), TP=2 on H200: coherent greedy output; per rank 74.6 GiB of weights.
