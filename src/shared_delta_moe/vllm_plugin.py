r"""vLLM plugin: dense Qwen3.5 / Qwen3.8 checkpoints with ``hier_delta_moe`` in the text config (the layout of
``models/qwen3_5/qwen3_5_hier_delta_moe``, written by its ``convert.py``) run every decoder MLP as a HierDeltaMoE on
the fused Triton kernel, with tensor parallelism. With ``routed_deltas`` (2-bit grids) the attention, linear-attention
and ``lm_head`` projections also get their routed per-expert (or per-set) deltas (``models/routed_linear.py``).

It is registered as a vLLM general plugin (entry point ``vllm.general_plugins`` in ``pyproject.toml``), so vLLM loads
it in every process, workers included. ``register`` patches vLLM's ``Qwen3_5DecoderLayer`` as ``modeling.py`` patches
transformers', and ``Qwen3_5ForCausalLMBase`` for ``lm_head``.

Tensor parallelism splits the intermediate dimension, as vLLM's own MLP does:

- :math:`W_g`, :math:`W_u` and their deltas by rows, :math:`W_d` and its deltas by columns (grid MLP deltas: codes,
  steps and lowest levels along the same dimension; the group must divide each rank's share of I).
- The router is replicated, so every rank routes alike.
- An all-reduce sums the partial down projections.

Routed deltas ride on vLLM's own projections (``qkv_proj``, ``o_proj``, ``in_proj_qkvz``, ``in_proj_ba``,
``out_proj``, ``lm_head``). Each gets ``codes`` / ``steps`` / ``lows`` with the expert dimension in front, loaded by
the projection's own weight loader, so they are sharded exactly as its weight (merged shards, vocabulary padding
included). Its ``quant_method.apply`` adds :math:`\sum_i r_i \Delta_{e_i} x` for this rank's shard, before any
all-reduce or gather. The block's replicated ``delta_router`` routes the block input in its first projection, which
copies the routing into the block's ``route_idx`` / ``route_r`` buffers for the others.

The MLP and the routed deltas are fused Triton kernels, registered custom ops under torch.compile.
"""

from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from .models.grid import GridHierDeltaMoE, code_bits
from .models.hier import HierDeltaMoE
from .models.routed_linear import set_table, to_sets

# name -> dimension sharded over the tensor-parallel ranks (the intermediate one); others are replicated
SHARD_DIM = {"gate": 0, "up": 0, "down": 1, "d_gate": 1, "d_up": 1, "d_down": 2}
GRID_SHARD_DIM = {"gate": 0, "up": 0, "down": 1} | {f"{t}_{p}": 2 if p == "down" else 1 for t in "csz"
                                                     for p in ("gate", "up", "down")}  # fmt: skip
# attention block -> (projections that read the block input and route it, the output projection)
ROUTED = {"self_attn": (("qkv_proj",), "o_proj"), "linear_attn": (("in_proj_qkvz", "in_proj_ba"), "out_proj")}


def _load_shard(param: Tensor, loaded: Tensor, dim: int, rank: int, tp: int) -> None:
    n = param.shape[dim]
    assert loaded.shape[dim] == n * tp, f"checkpoint {tuple(loaded.shape)} vs {tp} shards of {tuple(param.shape)}"
    param.data.copy_(loaded.narrow(dim, rank * n, n))


def _tp() -> tuple[int, int]:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


class _VllmBlock:
    """The forward of both blocks: route, the fused kernel on this rank's slice (grids at decode sizes: the CUDA
    matrix kernel), all-reduce."""

    grid = False

    def forward(self, x: Tensor) -> Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        from .triton_kernels import fused_moe
        from .triton_kernels.fused import grid_moe_any

        shape = x.shape
        x = x.reshape(-1, shape[-1])
        idx, r = self.topk(x)  # type: ignore[attr-defined]
        base, deltas = (self.gate, self.up, self.down), self._kernel_deltas()  # type: ignore[attr-defined]
        y = (grid_moe_any if self.grid else fused_moe)(x, idx, r, self.n_delta, base, deltas)  # type: ignore[attr-defined]
        if self.tp > 1:  # type: ignore[attr-defined]
            y = tensor_model_parallel_all_reduce(y)
        return y.view(shape)


class VllmHierDeltaMoeBlock(_VllmBlock, HierDeltaMoE):
    """The decoder MLP in vLLM: this rank's slice of a HierDeltaMoE (dense deltas).

    Args:
        config: a text config with ``hidden_size``, ``intermediate_size`` (divisible by the tensor-parallel size)
            and the ``hier_delta_moe`` dict (see ``Qwen3_5HierDeltaMoeBlock``).
    """

    def __init__(self, config: Any) -> None:
        tp, rank = _tp()
        c, inter = config.hier_delta_moe, config.intermediate_size
        assert inter % tp == 0, f"intermediate_size {inter} does not split over {tp} ranks"
        HierDeltaMoE.__init__(self, config.hidden_size, inter // tp, c["group_sizes"], c["group_zeros"], c["top_k"],
                              c.get("group_top_k"))  # fmt: skip
        self.tp = tp
        for name, dim in SHARD_DIM.items():
            getattr(self, name).weight_loader = partial(_load_shard, dim=dim, rank=rank, tp=tp)
        self._tables(self.gate.device)  # built now, not while vLLM traces the forward


class VllmGridHierDeltaMoeBlock(_VllmBlock, GridHierDeltaMoE):
    """The decoder MLP in vLLM with grid deltas (``hier_delta_moe.levels``, 2-bit): this rank's slice of a
    GridHierDeltaMoE, on the fused kernel's mode 3.

    Args:
        config: as ``VllmHierDeltaMoeBlock``; ``intermediate_size / tp`` divisible by the grid group.
    """

    def __init__(self, config: Any) -> None:
        tp, rank = _tp()
        c, inter = config.hier_delta_moe, config.intermediate_size
        g = c.get("group", 128)
        assert inter % tp == 0 and inter // tp % g == 0, f"intermediate_size {inter} over {tp} ranks, group {g}"
        GridHierDeltaMoE.__init__(self, config.hidden_size, inter // tp, c["group_sizes"], c["group_zeros"],
                                  c["top_k"], c.get("group_top_k"), levels=c["levels"], group=g,
                                  symmetric=c.get("symmetric", False))  # fmt: skip
        assert self.kernel_ok(), "vllm_plugin: grid MLP deltas need 2-bit codes and a power-of-2 group"
        self.tp, self.grid = tp, True
        for name, dim in GRID_SHARD_DIM.items():
            if hasattr(self, name):  # symmetric grids have no z_
                getattr(self, name).weight_loader = partial(_load_shard, dim=dim, rank=rank, tp=tp)
        self._tables(self.gate.device)


def _grids(lin: nn.Module, n: int, c: dict[str, Any], dim: str, loader: Any) -> None:
    """Add zero ``codes`` / ``steps`` / ``lows`` for n deltas of this rank's (R, C) shard of ``lin.weight``, marked
    for ``loader`` (``output_dim`` / ``input_dim`` shifted by the expert dimension in front)."""
    w = lin.weight
    R, C = w.shape
    g = c.get("group", 128)
    assert code_bits(c["levels"]) == 2 and g & (g - 1) == 0, "vllm_plugin: routed deltas need 2-bit grids"
    assert C % g == 0, f"{type(lin).__name__}: {C} input columns per rank, group {g}"
    shapes = {"codes": ((n, R, C // 16), torch.int32), "steps": ((n, R, C // g), w.dtype)}
    if not c.get("symmetric", False):
        shapes["lows"] = shapes["steps"]
    for name, (shape, dtype) in shapes.items():
        p = nn.Parameter(torch.zeros(shape, dtype=dtype, device=w.device), requires_grad=False)
        setattr(p, dim, 2 if dim == "input_dim" else 1)
        p.weight_loader = loader
        lin.register_parameter(name, p)


def _route_into(lin: nn.Module, owner: nn.Module | None, route: Any, c: dict[str, Any], n: int) -> None:
    """Wrap ``lin.quant_method.apply``: the base product plus this shard's routed deltas. ``route`` routes x (the
    block's first projection: also into ``owner``'s ``route_idx`` / ``route_r`` buffers, by in-place copies, which
    torch.compile and CUDA graphs keep, unlike Python state); None reads the first T rows of those buffers."""
    from .triton_kernels.fused import Delta, routed_deltas_any

    inner, L, g, sym = lin.quant_method.apply, c["levels"], c.get("group", 128), c.get("symmetric", False)

    def apply(layer: nn.Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
        y = inner(layer, x, bias)
        if layer is not lin:  # a quant method shared with another layer
            return y
        xs = x.reshape(-1, x.shape[-1])
        T = xs.shape[0]
        if route is not None:
            idx, r = route(xs)
            if owner is not None:
                owner.route_idx[:T].copy_(idx)
                owner.route_r[:T].copy_(r)
        else:
            idx, r = owner.route_idx[:T], owner.route_r[:T]  # type: ignore[union-attr,index]
        z = -lin.steps * ((L - 1) / 2) if sym else lin.lows
        return y + routed_deltas_any(xs, idx, r, n, Delta(3, lin.codes, lin.steps, z, g)).view(y.shape)

    lin.quant_method.apply = apply


def _router(layer: Any, owner: nn.Module, c: dict[str, Any]) -> tuple[Any, int]:
    """Add ``owner.delta_router`` (replicated, shaped like the layer's MoE router); return the routing (the MoE's
    ``topk`` with that router, slots mapped to sets when ``c`` has them) and the delta count."""
    w, n_delta, sets = layer.mlp.router.weight, layer.mlp.n_delta, c.get("sets")
    owner.delta_router = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    table = None if sets is None else set_table(sets, w.device)
    n = n_delta if sets is None else len(sets)

    def route(x: Tensor) -> tuple[Tensor, Tensor]:
        idx, r = layer.mlp.topk(x, owner.delta_router)
        return (idx, r) if table is None else (to_sets(idx, n_delta, table, n), r)

    return route, n


def _routed_attention(layer: Any, c: dict[str, Any], max_tokens: int) -> None:
    for block, (entries, out) in ROUTED.items():
        attn = getattr(layer, block, None)
        if attn is None:
            continue
        route, n = _router(layer, attn, c)
        dev, k = attn.delta_router.weight.device, layer.mlp.top_k
        attn.register_buffer("route_idx", torch.zeros(max_tokens, k, dtype=torch.long, device=dev), persistent=False)
        attn.register_buffer("route_r", torch.zeros(max_tokens, k, dtype=torch.float32, device=dev), persistent=False)
        for name in (*entries, out):
            lin = getattr(attn, name)
            _grids(lin, n, c, "input_dim" if name == out else "output_dim", lin.weight_loader)  # v1: plain params
            _route_into(lin, attn, route if name == entries[0] else None, c, n)


def _routed_lm_head(model: Any, c: dict[str, Any]) -> None:
    lm = model.lm_head
    route, n = _router(model.model.layers[-1], lm, c)

    def load(param: Tensor, loaded: Tensor) -> None:  # this rank's vocabulary rows; padding rows stay zero
        si = lm.shard_indices
        size = si.org_vocab_end_index - si.org_vocab_start_index
        param.data[:, :size].copy_(loaded.narrow(1, si.org_vocab_start_index, size))

    _grids(lm, n, c, "output_dim", load)
    _route_into(lm, None, route, c, n)


def register() -> None:
    """Patch vLLM's ``Qwen3_5DecoderLayer.__init__`` (idempotent): decoder layers whose text config has
    ``hier_delta_moe`` get a ``VllmHierDeltaMoeBlock`` (``VllmGridHierDeltaMoeBlock`` with ``levels``) as ``mlp``,
    and with ``routed_deltas`` routed deltas on their attention projections. Others keep vLLM's MLP, and so do the
    MTP layers (prefix ``mtp.layers.N``, built from the same class), whose MLP ``convert.py`` keeps dense.
    ``Qwen3_5ForCausalLMBase.__init__`` adds ``lm_head``'s routed deltas."""
    import vllm.model_executor.models.qwen3_5 as M

    if getattr(M.Qwen3_5DecoderLayer, "_hier_delta_moe", False):
        return
    init, lm_init = M.Qwen3_5DecoderLayer.__init__, M.Qwen3_5ForCausalLMBase.__init__

    def _init(self: Any, vllm_config: Any, layer_type: str, prefix: str = "") -> None:
        init(self, vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        c, rd = getattr(config, "hier_delta_moe", None), getattr(config, "routed_deltas", None)
        if c is None or "mtp" in prefix.split("."):
            return
        self.mlp = (VllmHierDeltaMoeBlock if c.get("levels") is None else VllmGridHierDeltaMoeBlock)(config)
        if rd is not None:
            assert rd.get("levels") is not None, "vllm_plugin: dense routed deltas not supported (--routed-levels)"
            _routed_attention(self, rd, vllm_config.scheduler_config.max_num_batched_tokens)

    def _lm_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        lm_init(self, vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        rd = getattr(config, "routed_deltas", None)
        if rd is not None and getattr(config, "hier_delta_moe", None) is not None and hasattr(self.lm_head, "weight"):
            assert not config.tie_word_embeddings, "vllm_plugin: routed lm_head deltas need an untied lm_head"
            hd = getattr(vllm_config.model_config, "head_dtype", None)
            # another head dtype bypasses lm_head.quant_method.apply (LogitsProcessor._apply_head), and the deltas
            assert hd is None or hd == vllm_config.model_config.dtype, "vllm_plugin: routed lm_head needs head_dtype unset"
            _routed_lm_head(self, rd)

    M.Qwen3_5DecoderLayer.__init__ = _init  # type: ignore[method-assign]
    M.Qwen3_5ForCausalLMBase.__init__ = _lm_init  # type: ignore[method-assign]
    M.Qwen3_5DecoderLayer._hier_delta_moe = True  # type: ignore[attr-defined]
