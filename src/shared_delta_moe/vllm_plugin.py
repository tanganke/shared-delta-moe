r"""vLLM plugin: dense Qwen3.5 / Qwen3.8 checkpoints with ``hier_delta_moe`` in the text config (the layout of
``models/qwen3_5/qwen3_5_hier_delta_moe``, written by its ``convert.py``) run every decoder MLP as a HierDeltaMoE on
the fused Triton kernel, with tensor parallelism.

It is registered as a vLLM general plugin (entry point ``vllm.general_plugins`` in ``pyproject.toml``), so vLLM loads
it in every process, workers included. ``register`` patches vLLM's ``Qwen3_5DecoderLayer`` as ``modeling.py`` patches
transformers'.

Tensor parallelism splits the intermediate dimension, as vLLM's own MLP does:

- :math:`W_g`, :math:`W_u` and their deltas by rows, :math:`W_d` and its deltas by columns.
- The router is replicated, so every rank routes alike.
- An all-reduce sums the partial down projections.

The forward is always the fused Triton kernel, a registered custom op, so torch.compile and CUDA graphs see one node.
Dense deltas only.
"""

from functools import partial
from typing import Any

import torch
from torch import Tensor

from .models.hier import HierDeltaMoE

# name -> dimension sharded over the tensor-parallel ranks (the intermediate one); others are replicated
SHARD_DIM = {"gate": 0, "up": 0, "down": 1, "d_gate": 1, "d_up": 1, "d_down": 2}


def _load_shard(param: Tensor, loaded: Tensor, dim: int, rank: int, tp: int) -> None:
    n = param.shape[dim]
    assert loaded.shape[dim] == n * tp, f"checkpoint {tuple(loaded.shape)} vs {tp} shards of {tuple(param.shape)}"
    param.data.copy_(loaded.narrow(dim, rank * n, n))


class VllmHierDeltaMoeBlock(HierDeltaMoE):
    """The decoder MLP in vLLM: this rank's slice of a HierDeltaMoE.

    Args:
        config: a text config with ``hidden_size``, ``intermediate_size`` (divisible by the tensor-parallel size)
            and the ``hier_delta_moe`` dict (see ``Qwen3_5HierDeltaMoeBlock``).
    """

    def __init__(self, config: Any) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        c, inter = config.hier_delta_moe, config.intermediate_size
        assert c.get("levels") is None, "vllm_plugin: grid MLP deltas (hier_delta_moe.levels) not supported yet"
        assert getattr(config, "routed_deltas", None) is None, "vllm_plugin: routed_deltas not supported yet"
        assert inter % tp == 0, f"intermediate_size {inter} does not split over {tp} ranks"
        super().__init__(config.hidden_size, inter // tp, c["group_sizes"], c["group_zeros"], c["top_k"],
                         c.get("group_top_k"))  # fmt: skip
        self.tp = tp
        for name, dim in SHARD_DIM.items():
            getattr(self, name).weight_loader = partial(_load_shard, dim=dim, rank=rank, tp=tp)
        self._tables(self.gate.device)  # built now, not while vLLM traces the forward

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        from vllm.distributed import tensor_model_parallel_all_reduce

        from .triton_kernels import fused_moe

        shape = x.shape
        x = x.reshape(-1, shape[-1])
        idx, r = self.topk(x)
        y = fused_moe(x, idx, r, self.n_delta, (self.gate, self.up, self.down), self._kernel_deltas())
        if self.tp > 1:
            y = tensor_model_parallel_all_reduce(y)
        return y.view(shape)


def register() -> None:
    """Patch vLLM's ``Qwen3_5DecoderLayer.__init__`` (idempotent): decoder layers whose text config has
    ``hier_delta_moe`` get a ``VllmHierDeltaMoeBlock`` as ``mlp``; others keep vLLM's MLP, and so do the MTP layers
    (prefix ``mtp.layers.N``, built from the same class), whose MLP ``convert.py`` keeps dense."""
    import vllm.model_executor.models.qwen3_5 as M

    if getattr(M.Qwen3_5DecoderLayer, "_hier_delta_moe", False):
        return
    init = M.Qwen3_5DecoderLayer.__init__

    def _init(self: Any, vllm_config: Any, layer_type: str, prefix: str = "") -> None:
        init(self, vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        if getattr(config, "hier_delta_moe", None) is not None and "mtp" not in prefix.split("."):
            self.mlp = VllmHierDeltaMoeBlock(config)

    M.Qwen3_5DecoderLayer.__init__ = _init  # type: ignore[method-assign]
    M.Qwen3_5DecoderLayer._hier_delta_moe = True  # type: ignore[attr-defined]
