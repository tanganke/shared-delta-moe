"""transformers' dense Qwen3.5 (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration; Qwen3.8 uses it) with every
decoder MLP a HierDeltaMoE run by forward_param.

A text config with `hier_delta_moe: {group_sizes, group_zeros, top_k, group_top_k, shared_base}` turns it on (see
models/hier.py; group_top_k null = dense group routing). Checkpoint keys per layer: mlp.{gate,up,down} (W_b),
mlp.d_{gate,up,down} (Delta, groups in order) and mlp.router.weight (G group rows, then each group's delta and zero
rows); convert.py writes them. Importing this module patches Qwen3_5DecoderLayer.__init__; configs without the field
build the stock MLP.
"""

from typing import Any

from torch import Tensor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from ...hier import HierDeltaMoE


class Qwen3_5HierDeltaMoeBlock(HierDeltaMoE):
    """The decoder MLP replacement: a HierDeltaMoE built from a Qwen3.5 text config, run by forward_param.

    Args:
        config: a text config with hidden_size, intermediate_size and the hier_delta_moe dict (group_sizes,
            group_zeros, top_k; optional group_top_k, default None = dense group routing, and shared_base, default
            False).
    """

    def __init__(self, config: Any) -> None:
        c = config.hier_delta_moe
        super().__init__(config.hidden_size, config.intermediate_size, c["group_sizes"], c["group_zeros"], c["top_k"],
                         c.get("group_top_k"), shared_base=c.get("shared_base", False))  # fmt: skip

    def forward(self, hidden_states: Tensor) -> Tensor:  # type: ignore[override]
        """Apply the MoE to every token through forward_param.

        Args:
            hidden_states: (..., H) hidden states, any leading shape.

        Returns:
            Tensor of the same shape as hidden_states.
        """
        shape = hidden_states.shape
        return self.forward_param(hidden_states.reshape(-1, shape[-1])).view(shape)


_layer_init = M.Qwen3_5DecoderLayer.__init__


def _init(self: Any, config: Any, layer_idx: int) -> None:
    _layer_init(self, config, layer_idx)
    if getattr(config, "hier_delta_moe", None) is not None:
        self.mlp = Qwen3_5HierDeltaMoeBlock(config)


M.Qwen3_5DecoderLayer.__init__ = _init  # type: ignore[method-assign]
