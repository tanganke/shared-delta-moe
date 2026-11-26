# Plain MoE decode in transformers (`batched_mm`)

Running the full model with transformers (Qwen3.5-4B MoE, 6 experts, H200), batch-32 decode measured 338 tok/s for the original MoE, while both forms here reached 711 / 860 tok/s. That is **not** because these forms are faster. It's because the original MoE was given a slow path by transformers:

1. Qwen3.5 MoE experts default to `grouped_mm` (`torch._grouped_mm`, one cutlass grouped GEMM for all experts). A single layer takes 0.47 ms at 32 tokens.
2. But `generate()` uses `_optimize_model_for_decode` to automatically **switch every `grouped_mm` to `batched_mm` during decode** (transformers 5.16, `generation/utils.py`).
3. `batched_mm` gathers **one full copy of the expert weights per token** (`self.gate_up_proj[expert_ids]`), then runs `bmm`. At batch 1 that's about the same as reading the weights directly. At batch 32, every layer copies 32 expert-sized matrices (about 4.5 GB). The profile shows 90% of the time in `aten::index` (gather) and `aten::bmm`.
4. The forms here don't go through the experts module, so they're never switched.

Disable the switch and the original MoE becomes the fastest:

| Batch-32 decode (full model, ctx 512) | tok/s | ms/step |
|---|---:|---:|
| Original MoE (with transformers' switch to `batched_mm`) | 338 | 94.7 |
| Original MoE (stays on `grouped_mm`) | **867** | 36.9 |
| Parameter aggregation (materialised, pure PyTorch) | 711 | 45.0 |
| Per-matrix output aggregation (loop, pure PyTorch) | 860 | 37.2 |

How to disable it (batch 1 gets slightly slower, 31 → 28 tok/s):

```python
import contextlib
from transformers.generation.utils import GenerationMixin
GenerationMixin._optimize_model_for_decode = contextlib.contextmanager(lambda self: (yield))
```

A better fix than disabling it outright is to use `batched_mm` only when there are few tokens. Full-model decode (H200, context 512, ms per step):

| batch | transformers default (always `batched_mm` in decode) | always `grouped_mm` | auto, threshold 8 |
|---:|---:|---:|---:|
| 1 | 31.7 | 38.2 | 33.3 |
| 2 | 32.2 | 39.4 | 34.0 |
| 4 | 31.8 | 39.6 | 34.0 |
| 8 | 33.1 | 38.9 | 34.4 |
| 16 | 52.9 | 39.8 | 36.7 |
| 32 | 94.3 | 39.1 | 37.5 |

`grouped_mm` is flat at about 39 ms (bound by framework overhead, not memory traffic); `batched_mm` is faster up to batch 8, then grows with the number of weight copies. Registering an `auto` implementation (`batched_mm` when tokens × top_k ≤ 8, otherwise `grouped_mm`; prefill naturally takes `grouped_mm`) stays within about 2 ms of the best at every batch size. Between-run noise is about ±2 ms, so 4 and 8 are indistinguishable as thresholds.

```python
import contextlib
import transformers.integrations.moe as moe
from transformers.generation.utils import GenerationMixin

def auto_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    f = moe.batched_mm_experts_forward if top_k_index.numel() <= 8 else moe.grouped_mm_experts_forward
    return f(self, hidden_states, top_k_index, top_k_weights)

moe.ExpertsInterface.register("auto", auto_experts_forward)
GenerationMixin._optimize_model_for_decode = contextlib.contextmanager(lambda self: (yield))
model.set_experts_implementation("auto")
```

The takeaway: with large experts (9216 × 2560) and batch > 1, transformers' "switch to `batched_mm` during decode" heuristic makes things slower.
