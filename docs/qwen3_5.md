# Qwen3.5 in transformers

```bash
python -m shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe.convert PLAIN_MOE OUT [--base INDEX] [--top-k K]
```

```python
import shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe  # patches Qwen3_5MoeDecoderLayer
model = AutoModelForImageTextToText.from_pretrained(OUT, num_experts_per_tok=2)  # or AutoModelForCausalLM
```

The plain MoE's expert BASE becomes $W_b$ and the zero expert; the others become $\Delta_e$. Its shared expert must
have no output. `num_experts_per_tok` is the top-k at load time (= `num_experts` for soft routing over all experts).
