"""Qwen3.5 DeltaMoE in transformers: convert a tiny plain MoE checkpoint (base = one of its experts), load it with
the patched decoder layer, and compare logits to the stock MoE at top-1, top-2 and soft routing (fp32, CPU).
"""

import tempfile
from pathlib import Path

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe import Qwen3_5DeltaMoeBlock
from shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe.convert import (
    main as convert_main,
)


def test_converted_matches_stock_moe():
    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_experts=6, num_experts_per_tok=1, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
    )  # fmt: skip
    plain = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for layer in plain.model.layers:
            for p in (
                layer.mlp.experts.gate_up_proj,
                layer.mlp.experts.down_proj,
                layer.mlp.gate.weight,
            ):
                p.normal_(std=0.3)
            layer.mlp.shared_expert.down_proj.weight.zero_()  # the plain MoE has no shared-expert output
    x = torch.randint(0, 128, (2, 24))
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "plain", Path(d) / "delta"
        plain.save_pretrained(src)
        convert_main([str(src), str(dst), "--base", "2"])
        for k in (1, 2, 6):  # 6 = every expert: soft routing
            stock = Qwen3_5MoeForCausalLM.from_pretrained(
                src, num_experts_per_tok=k
            ).eval()
            delta = Qwen3_5MoeForCausalLM.from_pretrained(
                dst, num_experts_per_tok=k
            ).eval()
            blocks = [layer.mlp for layer in delta.model.layers]
            assert all(
                isinstance(b, Qwen3_5DeltaMoeBlock) and b.top_k == k for b in blocks
            )
            assert blocks[0].n_delta == 5 and blocks[0].num_zero_experts == 1
            with torch.no_grad():
                want, got = stock(x).logits, delta(x).logits
            err = ((got - want).abs().max() / want.abs().max()).item()
            assert err < 1e-5, (k, err)
            routed = stock.model.layers[0].mlp.gate(
                stock.model.embed_tokens(x).view(-1, 32)
            )[2]
            assert (routed == 2).any() and (
                routed != 2
            ).any(), "hits the base expert and others"


if __name__ == "__main__":
    test_converted_matches_stock_moe()
    print("ok")
