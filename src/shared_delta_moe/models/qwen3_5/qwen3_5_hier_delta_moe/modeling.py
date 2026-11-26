r"""transformers' dense Qwen3.5 (``Qwen3_5ForCausalLM``, ``Qwen3_5ForConditionalGeneration``; Qwen3.8 uses it) with
every decoder MLP a HierDeltaMoE run by ``forward_param``.

A text config with ``hier_delta_moe: {group_sizes, group_zeros, top_k, group_top_k}`` turns it on (see
``models/hier.py``; ``group_top_k`` null = dense group routing). Checkpoint keys per layer:

- ``mlp.{gate,up,down}``: :math:`W_b`.
- ``mlp.d_{gate,up,down}``: :math:`\Delta`, groups in order.
- ``mlp.router.weight``: :math:`G` group rows, then each group's delta and zero rows.

``convert.py`` writes them. Importing this module patches ``Qwen3_5DecoderLayer.__init__``; configs without the field
build the stock MLP.
"""

from typing import Any

from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from ...hier import HierDeltaMoE
from ..common import block_forward, patch_decoder_layer


class Qwen3_5HierDeltaMoeBlock(HierDeltaMoE):
    """The decoder MLP replacement: a HierDeltaMoE built from a Qwen3.5 text config, run by ``forward_param``.

    Args:
        config: a text config with ``hidden_size``, ``intermediate_size`` and the ``hier_delta_moe`` dict
            (``group_sizes``, ``group_zeros``, ``top_k``; optional ``group_top_k``, default None = dense group
            routing).
    """

    def __init__(self, config: Any) -> None:
        c = config.hier_delta_moe
        super().__init__(config.hidden_size, config.intermediate_size, c["group_sizes"], c["group_zeros"], c["top_k"],
                         c.get("group_top_k"))  # fmt: skip

    forward = block_forward


patch_decoder_layer(M.Qwen3_5DecoderLayer, "hier_delta_moe", Qwen3_5HierDeltaMoeBlock)
