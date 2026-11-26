r"""transformers' Qwen3.5 MoE (``Qwen3_5MoeForCausalLM``, ``Qwen3_5MoeForConditionalGeneration``) with every sparse
MoE block a DeltaMoE run by ``forward_param``.

A text config with ``delta_moe_zero_experts: Z`` (an int, 0 allowed) turns it on: ``num_experts`` = n deltas + Z
router rows, rows n .. n+Z-1 being zero experts (the base MLP itself); ``num_experts_per_tok`` is the top-k
(``num_experts``: soft routing over every expert). There is no shared expert. Checkpoint keys per layer:
``mlp.{gate,up,down}`` (:math:`W_b`), ``mlp.d_{gate,up,down}`` (:math:`\Delta_e`) and ``mlp.router.weight``;
``convert.py`` writes them from a plain MoE. Importing this module patches ``Qwen3_5MoeDecoderLayer.__init__``;
configs without the field build the stock block. ``to_bitdelta`` turns a loaded model's blocks into BitDelta (1-bit
deltas, ``models/bitdelta.py``).
"""

from typing import Any

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

from ...bitdelta import BitDeltaMoE
from ...delta_moe import DeltaMoE
from ..common import block_forward, patch_decoder_layer


class Qwen3_5DeltaMoeBlock(DeltaMoE):
    """A Qwen3.5 sparse MoE block as a DeltaMoE, run by ``forward_param``.

    Args:
        config: The Qwen3.5 MoE text config: ``hidden_size``, ``moe_intermediate_size``, ``num_experts`` (N deltas
            + Z), ``num_experts_per_tok`` (top-k) and ``delta_moe_zero_experts`` (Z).
    """

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

    forward = block_forward


class Qwen3_5BitDeltaMoeBlock(BitDeltaMoE):
    """A Qwen3.5 sparse MoE block as a BitDeltaMoE (1-bit deltas), with ``Qwen3_5DeltaMoeBlock``'s forward.

    Built by ``to_bitdelta`` from a ``Qwen3_5DeltaMoeBlock``; constructor args as ``BitDeltaMoE``.
    """

    forward = block_forward


def to_bitdelta(model: Any, scale: "str | int" = "matrix") -> Any:
    """Replace every ``Qwen3_5DeltaMoeBlock`` of a loaded model with its BitDelta form, in place.

    Uses ``BitDeltaMoE.from_delta_moe``; the dense deltas are dropped.

    Args:
        model: A loaded transformers Qwen3.5 MoE model with ``Qwen3_5DeltaMoeBlock`` blocks.
        scale: Delta scales, passed to ``BitDeltaMoE.from_delta_moe``: ``"matrix"`` (BitDelta), ``"row"`` or an int
            block size.

    Returns:
        The same model, modified.
    """
    for name, mod in list(model.named_modules()):
        if type(mod) is Qwen3_5DeltaMoeBlock:
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, Qwen3_5BitDeltaMoeBlock.from_delta_moe(mod, scale))
    return model


patch_decoder_layer(M.Qwen3_5MoeDecoderLayer, "delta_moe_zero_experts", Qwen3_5DeltaMoeBlock)
