"""transformers' Qwen3.5 MoE (Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration) with every sparse MoE block a
DeltaMoE run by forward_param.

A text config with `delta_moe_zero_experts: Z` (an int, 0 allowed) turns it on: num_experts = n deltas + Z router rows,
rows n .. n+Z-1 being zero experts (the base MLP itself); num_experts_per_tok is the top-k (num_experts: soft routing
over every expert). There is no shared expert. Checkpoint keys per layer: mlp.{gate,up,down} (W_b),
mlp.d_{gate,up,down} (Delta_e) and mlp.router.weight; convert.py writes them from a plain MoE. Importing this module
patches Qwen3_5MoeDecoderLayer.__init__; configs without the field build the stock block. to_bitdelta turns a loaded
model's blocks into BitDelta (1-bit deltas, models/bitdelta.py).
"""

from typing import Any

from torch import Tensor
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

from ...bitdelta import BitDeltaMoE
from ...delta_moe import DeltaMoE


class Qwen3_5DeltaMoeBlock(DeltaMoE):
    def __init__(self, config: Any) -> None:
        z = config.delta_moe_zero_experts
        n = config.num_experts - z
        super().__init__(
            config.hidden_size,
            config.moe_intermediate_size,
            n,
            config.num_experts_per_tok,
            z,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:  # type: ignore[override]
        shape = hidden_states.shape
        return self.forward_param(hidden_states.reshape(-1, shape[-1])).view(shape)


class Qwen3_5BitDeltaMoeBlock(BitDeltaMoE):
    forward = Qwen3_5DeltaMoeBlock.forward


def to_bitdelta(model: Any, scale: str = "matrix") -> Any:
    """Replace every Qwen3_5DeltaMoeBlock of a loaded model with its BitDelta form (BitDeltaMoE.from_delta_moe), in
    place; the dense deltas are dropped."""
    for name, mod in list(model.named_modules()):
        if type(mod) is Qwen3_5DeltaMoeBlock:
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, Qwen3_5BitDeltaMoeBlock.from_delta_moe(mod, scale))
    return model


_layer_init = M.Qwen3_5MoeDecoderLayer.__init__


def _init(self: Any, config: Any, layer_idx: int) -> None:
    _layer_init(self, config, layer_idx)
    if getattr(config, "delta_moe_zero_experts", None) is not None:
        self.mlp = Qwen3_5DeltaMoeBlock(config)


M.Qwen3_5MoeDecoderLayer.__init__ = _init  # type: ignore[method-assign]
