"""Shared by the two Qwen3.5 integrations (``qwen3_5_delta_moe``, ``qwen3_5_hier_delta_moe``): the block forward, the
transformers decoder-layer patch, and safetensors checkpoint I/O for the converters. (vLLM's decoder layer has another
constructor; ``vllm_plugin.py`` patches it on its own.)"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor


def block_forward(self: Any, hidden_states: Tensor) -> Tensor:
    """The decoder MLP forward of a DeltaMoE block: ``forward_param`` over the flattened tokens.

    Args:
        hidden_states: (..., H) hidden states, any leading shape.

    Returns:
        Tensor of the same shape as ``hidden_states``.
    """
    shape = hidden_states.shape
    return self.forward_param(hidden_states.reshape(-1, shape[-1])).view(shape)


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

    layer.__init__ = _init  # type: ignore[method-assign]


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
