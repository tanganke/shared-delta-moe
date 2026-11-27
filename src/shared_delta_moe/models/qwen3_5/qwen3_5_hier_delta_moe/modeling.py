r"""transformers' dense Qwen3.5 (``Qwen3_5ForCausalLM``, ``Qwen3_5ForConditionalGeneration``; Qwen3.8 uses it) with
every decoder MLP a HierDeltaMoE run by ``DeltaMoE.forward``.

A text config with ``hier_delta_moe: {group_sizes, group_zeros, top_k, group_top_k}`` turns it on (see
``models/hier.py``; ``group_top_k`` null = dense group routing; ``levels`` set, with ``group`` and ``symmetric``:
group-grid deltas, ``Qwen3_5GridHierDeltaMoeBlock``, keys ``mlp.{c,s,z}_{gate,up,down}`` as in ``models/grid.py``).
Checkpoint keys per layer:

- ``mlp.{gate,up,down}``: :math:`W_b`.
- ``mlp.d_{gate,up,down}``: :math:`\Delta`, groups in order.
- ``mlp.router.weight``: :math:`G` group rows, then each group's delta and zero rows.

``convert.py`` writes them. Importing this module patches ``Qwen3_5DecoderLayer.__init__``; configs without the field
build the stock MLP.
"""

from typing import Any

from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from ...grid import GridHierDeltaMoE
from ...hier import HierDeltaMoE
from ..common import patch_decoder_layer


class Qwen3_5HierDeltaMoeBlock(HierDeltaMoE):
    """The decoder MLP replacement: a HierDeltaMoE built from a Qwen3.5 text config, run by ``DeltaMoE.forward``.

    Args:
        config: a text config with ``hidden_size``, ``intermediate_size`` and the ``hier_delta_moe`` dict
            (``group_sizes``, ``group_zeros``, ``top_k``; optional ``group_top_k``, default None = dense group
            routing; optional ``force`` [g, e], every token to expert e of group g, ``DeltaMoE.force``).
    """

    def __init__(self, config: Any) -> None:
        c = config.hier_delta_moe
        super().__init__(config.hidden_size, config.intermediate_size, c["group_sizes"], c["group_zeros"], c["top_k"],
                         c.get("group_top_k"))  # fmt: skip
        self.force = c.get("force")  # [g, e]: every token to expert e of group g (DeltaMoE.force)


class Qwen3_5GridHierDeltaMoeBlock(GridHierDeltaMoE):
    """``Qwen3_5HierDeltaMoeBlock`` with group-grid deltas (``hier_delta_moe`` has ``levels``; optional ``group``,
    default 128, and ``symmetric``, default False).

    Args:
        config: The text config.
    """

    def __init__(self, config: Any) -> None:
        c = config.hier_delta_moe
        super().__init__(config.hidden_size, config.intermediate_size, c["group_sizes"], c["group_zeros"], c["top_k"],
                         c.get("group_top_k"), levels=c["levels"], group=c.get("group", 128),
                         symmetric=c.get("symmetric", False))  # fmt: skip
        self.force = c.get("force")


def _block(config: Any) -> HierDeltaMoE:
    grid = config.hier_delta_moe.get("levels") is not None
    return (Qwen3_5GridHierDeltaMoeBlock if grid else Qwen3_5HierDeltaMoeBlock)(config)


patch_decoder_layer(M.Qwen3_5DecoderLayer, "hier_delta_moe", _block)
