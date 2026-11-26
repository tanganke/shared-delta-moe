# Getting started

## Install

```bash
pip install -e .                         # torch only; add extras as needed:
pip install -e ".[triton,transformers,merging]"
```

- `triton`: the fused kernels (Triton ships with torch on Linux + CUDA).
- `transformers`: the Qwen3.5 blocks and checkpoint converters.
- `merging`: fusion_bench, for the HierDeltaMoE converter's merge methods.

For development, with uv:

```bash
git clone https://github.com/tanganke/fusion_bench ../fusion_bench   # uv installs it editable from there
uv sync
uv run pytest
```

The CUDA decode kernels for DARE / BitDelta deltas on Hopper are optional and built once with nvcc; see
[CUDA decode kernels](internals/cuda-decode.md). Without them everything runs on Triton or PyTorch.

## One MoE layer

```python
import torch
from shared_delta_moe import DeltaMoE, random_experts

H, I, n = 256, 512, 4                                    # hidden, intermediate, delta experts
full = random_experts(H, I, n + 1, torch.float32, "cpu")  # n + 1 experts, each (gate, up, down)
base, experts = full[0], full[1:]                         # expert 0 becomes W_b (and the zero expert)
router = torch.randn(n + 1, H) * H**-0.5                  # rows 0..n-1: the deltas, row n: the base
moe = DeltaMoE.from_experts(base, experts, router, top_k=2, num_zero_experts=1)

x = torch.randn(8, H)
y = moe(x)                                                # = the plain top-2 MoE over full
```

`moe(x)` takes any leading shape. `DeltaMoE` stores $W_b$ once and one delta per expert
([DeltaMoE](guide/deltamoe.md)); the deltas can be compressed ([Delta formats](guide/delta-formats.md)), and
`HierDeltaMoE` routes over groups of them ([HierDeltaMoE](guide/hier.md)).

## A whole model

```bash
# a plain Qwen3.5 MoE -> DeltaMoE in every layer
python -m shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe.convert PLAIN_MOE OUT
# a dense base + domain fine-tunes -> HierDeltaMoE in every layer
python -m shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert BASE DST --group medical=FT1,FT2 --group code=FT3
```

```python
import shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe  # patches transformers' decoder layer
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(DST)  # Qwen3_5ForCausalLM for text-only
```

Details: [Converting Qwen3.5 models](guide/qwen3_5.md); serving the result: [Serving with vLLM](guide/vllm.md).
