"""Shared by the two Qwen3.5 integrations (``qwen3_5_delta_moe``, ``qwen3_5_hier_delta_moe``): the
transformers decoder-layer patch, and safetensors checkpoint I/O for the converters. (vLLM's decoder layer has another
constructor; ``vllm_plugin.py`` patches it on its own.)

A text config with ``routed_deltas: {levels, group, symmetric}`` (levels null = dense deltas) also gives every
attention and linear-attention projection and ``lm_head`` per-expert deltas (``routed_linear.RoutedDeltaLinear``):
each attention block gets a ``delta_router`` (shaped like its layer's MoE router, routed the same way), ``lm_head``
one routed like the last layer's MoE. Embeddings, norms and the linear attention's conv1d / ``A_log`` / ``dt_bias``
stay shared."""

import functools
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from ..routed_linear import RoutedDeltaLinear

ROUTED = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
          "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}  # fmt: skip


def patch_decoder_layer(layer: type, field: str, block: Callable[[Any], Any]) -> None:
    """Patch ``layer.__init__``: a layer whose text config has ``field`` (not None) gets ``block(config)`` as its
    ``mlp``; other configs build the stock MLP.

    Args:
        layer: transformers' decoder layer class, constructed as ``layer(config, layer_idx)``.
        field: the text config attribute that turns the block on.
        block: the MLP replacement, built from the text config.
    """
    init = layer.__init__

    def _init(self: Any, config: Any, layer_idx: int) -> None:
        init(self, config, layer_idx)
        if getattr(config, field, None) is not None:
            self.mlp = block(config)
            if getattr(config, "routed_deltas", None) is not None:
                for name, projs in ROUTED.items():
                    if hasattr(self, name):
                        route_through(getattr(self, name), projs, self, config.routed_deltas)

    layer.__init__ = _init  # type: ignore[method-assign]


def _routed(old: nn.Linear, n_delta: int, c: dict[str, Any]) -> RoutedDeltaLinear:
    assert old.bias is None, "routed deltas on a biased projection"
    w = old.weight
    return RoutedDeltaLinear(old.in_features, old.out_features, n_delta, c.get("levels"), c.get("group", 128),
                             c.get("symmetric", False), w.device, w.dtype)  # fmt: skip


def _set_route(layer: Any, lins: list[RoutedDeltaLinear], module: nn.Module, args: tuple, kwargs: dict) -> None:
    x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    x = x.reshape(-1, x.shape[-1]).to(module.delta_router.weight.device)  # (accelerate moves inputs after the hook)
    route = layer.mlp.topk(x, module.delta_router)  # the layer's MoE at call time
    for lin in lins:
        lin.route = route


def route_through(module: nn.Module, projs: tuple[str, ...], layer: Any, c: dict[str, Any]) -> None:
    """Replace ``module``'s projections ``projs`` with RoutedDeltaLinears, add ``module.delta_router`` and a forward
    pre-hook that routes ``module``'s input with it (``layer.mlp.topk``) for all of them.

    Args:
        module: The attention block (input ``hidden_states``), or ``lm_head`` itself.
        projs: Attribute names of its linear layers; ``()`` = ``module`` is the linear layer (already replaced).
        layer: The decoder layer whose ``mlp`` (a DeltaMoE) routes.
        c: The ``routed_deltas`` config entry.
    """
    moe = layer.mlp
    lins = []
    for p in projs:
        setattr(module, p, _routed(getattr(module, p), moe.n_delta, c))
        lins.append(getattr(module, p))
    w = moe.router.weight
    module.delta_router = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    module.register_forward_pre_hook(functools.partial(_set_route, layer, lins or [module]), with_kwargs=True)


def _patch_lm_head(model: type) -> None:
    init = model.__init__

    def _init(self: Any, config: Any) -> None:
        init(self, config)
        text = getattr(config, "text_config", config)
        c = getattr(text, "routed_deltas", None)
        if c is None:
            return
        assert not getattr(config, "tie_word_embeddings", False), "routed lm_head deltas need an untied lm_head"
        last = [m for m in self.modules() if isinstance(m, M.Qwen3_5DecoderLayer)][-1]
        self.lm_head = _routed(self.lm_head, last.mlp.n_delta, c)
        route_through(self.lm_head, (), last, c)

    model.__init__ = _init  # type: ignore[method-assign]


for _m in (M.Qwen3_5ForCausalLM, M.Qwen3_5ForConditionalGeneration):
    _patch_lm_head(_m)


class Checkpoint:
    """A safetensors checkpoint directory (sharded with model.safetensors.index.json, or one model.safetensors), read
    one tensor at a time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.files = {k: self.path / f for k, f in json.loads(index.read_text())["weight_map"].items()}
        else:
            f = self.path / "model.safetensors"
            with safe_open(f, "pt") as h:
                self.files = dict.fromkeys(h.keys(), f)
        self._open: dict[Path, Any] = {}

    def keys(self) -> list[str]:
        return list(self.files)

    def get(self, key: str) -> torch.Tensor:
        f = self.files[key]
        if f not in self._open:
            self._open[f] = safe_open(f, "pt")
        return self._open[f].get_tensor(key)


class ShardWriter:
    """Write tensors to ``dst`` in shards of about ``shard_bytes``, then the index (``close``)."""

    def __init__(self, dst: Path, shard_bytes: int) -> None:
        self.dst, self.limit, self.buf, self.size, self.n, self.map, self.total = dst, shard_bytes, {}, 0, 0, {}, 0

    def add(self, key: str, t: torch.Tensor) -> None:
        self.buf[key] = t.contiguous()
        self.size += t.numel() * t.element_size()
        if self.size >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.n += 1
        name = f"model-{self.n:05d}.safetensors"
        save_file(self.buf, self.dst / name, metadata={"format": "pt"})
        self.map.update(dict.fromkeys(self.buf, name))
        self.total += self.size
        self.buf, self.size = {}, 0

    def close(self) -> None:
        self.flush()
        index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.map.items()))}
        (self.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def copy_config(src: Path, dst: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Copy ``src``'s non-weight files (config, tokenizer, ...) to ``dst``, the config with ``edit`` applied to its
    text config (``text_config``, or the whole config of a text-only model)."""
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy(f, dst / f.name)
    config = json.loads((src / "config.json").read_text())
    edit(config.get("text_config", config))
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
