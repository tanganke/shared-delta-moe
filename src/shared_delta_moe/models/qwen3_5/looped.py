"""transformers' Qwen3.5 / Qwen3.8 text model with decoder layers ``[start, start + size)`` run as a loop
(``models/looped.py``), whatever their MLPs (stock, DeltaMoE, HierDeltaMoE, with or without ``lora``).

``loop_layers(model, start, size, repeats)`` changes a loaded model in place:

- Round 0 uses the layers' own cache slots; round r > 0 of layer ``start + j`` uses slot
  ``num_hidden_layers + (r - 1) * size + j``, through an alias of the layer that shares every parameter and differs
  only in its attention's ``layer_idx``. The text config's ``layer_types`` gets those slots' types appended, so the
  ``DynamicCache`` that ``generate`` builds from the config has them.
- The text config gets ``loop`` (start, size, repeats, routing, feedback_rank, update_init); the model gets
  ``loop_update`` (a ``LoopUpdate``, repeats > 1) and, in training mode, ``loop_aux``: the balance losses of the
  loop's flat DeltaMoE blocks from the last forward (``looped.balance_loss``).
- Under gradient checkpointing every layer call is a checkpoint, the loop's too (``looped.keep_routing``).

The adapters (``loop_update``, routers, LoRA) are saved as a state dict; no ``from_pretrained`` path builds the loop.
"""

import copy
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from ..delta_moe import DeltaMoE
from ..hier import HierDeltaMoE
from ..looped import ROUTINGS, LoopUpdate, keep_routing, loop, loopable


def _alias(layer: nn.Module, slot: int) -> nn.Module:
    """``layer`` with its attention's cache slot ``slot``; every parameter, buffer and hook shared."""
    new = copy.copy(layer)
    new._modules = dict(layer._modules)
    name = "linear_attn" if "linear_attn" in layer._modules else "self_attn"
    attn = copy.copy(layer._modules[name])
    attn.layer_idx = slot
    new._modules[name] = attn
    return new


def text_model(model: nn.Module) -> M.Qwen3_5TextModel:
    """The ``Qwen3_5TextModel`` inside ``model`` (or model itself)."""
    if isinstance(model, M.Qwen3_5TextModel):
        return model
    return next(m for m in model.modules() if isinstance(m, M.Qwen3_5TextModel))


def loop_layers(model: nn.Module, start: int, size: int, repeats: int, routing: str = "fixed",
                feedback_rank: int = 64, update_init: float = 0.1) -> nn.Module:  # fmt: skip
    """Make layers ``[start, start + size)`` of a loaded Qwen3.5 model a loop (in place; see the module docstring).

    Args:
        model: ``Qwen3_5ForCausalLM``, ``Qwen3_5ForConditionalGeneration`` or the text model.
        start: First looped layer.
        size: Layers in the loop.
        repeats: Rounds, >= 1 (1: the plain model, with fixed or collected routing).
        routing: ``"fixed"`` or ``"dynamic"``.
        feedback_rank: Rank of the feedback.
        update_init: The update gate at init.

    Returns:
        model.
    """
    text = text_model(model)
    cfg = text.config
    n = cfg.num_hidden_layers
    assert getattr(cfg, "loop", None) is None, "already looped"
    assert 0 <= start and size >= 1 and start + size <= n and repeats >= 1 and routing in ROUTINGS
    stack = list(text.layers[start:start + size])
    text.__dict__["_loop_aliases"] = [[_alias(layer, n + (r - 1) * size + j) for j, layer in enumerate(stack)]
                                      for r in range(1, repeats)]  # fmt: skip  # not submodules: no duplicate keys
    types = list(cfg.layer_types[:n]) + [cfg.layer_types[start + j] for _ in range(1, repeats) for j in range(size)]
    loop_cfg = dict(start=start, size=size, repeats=repeats, routing=routing, feedback_rank=feedback_rank,
                    update_init=update_init)  # fmt: skip
    for c in {id(cfg): cfg, id(model.config.get_text_config()): model.config.get_text_config()}.values():
        c.layer_types, c.loop = types, loop_cfg
    ref = stack[0].input_layernorm.weight
    text.loop_update = (LoopUpdate(cfg.hidden_size, feedback_rank, cfg.rms_norm_eps, update_init, ref.device)
                        if repeats > 1 else None)  # fmt: skip
    text.__dict__["_loop_moes"] = [loopable(layer.mlp) for layer in stack if isinstance(layer.mlp, DeltaMoE)]
    text.loop_aux = None
    text.__class__ = LoopedQwen3_5TextModel
    return model


class LoopedQwen3_5TextModel(M.Qwen3_5TextModel):
    """``Qwen3_5TextModel`` whose forward runs the loop of ``config.loop`` (made by ``loop_layers``)."""

    @M.merge_with_config_defaults
    @M.capture_outputs
    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None, past_key_values: Any = None,
                inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None,
                **kwargs: Any) -> M.Qwen3_5ModelOutputWithPast:  # fmt: skip
        # Qwen3_5TextModel.forward (transformers 5.16), its layer loop replaced
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = M.DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        else:
            text_position_ids = None
        if not isinstance(masks := attention_mask, dict):
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                           "past_key_values": past_key_values, "position_ids": text_position_ids}  # fmt: skip
            masks = {"full_attention": M.create_causal_mask(**mask_kwargs),
                     "linear_attention": M.create_recurrent_attention_mask(**mask_kwargs)}  # fmt: skip
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        c, n = self.config.loop, self.config.num_hidden_layers
        a, b = c["start"], c["start"] + c["size"]

        def call(layer: nn.Module, h: torch.Tensor, i: int, run: Any) -> torch.Tensor:
            return run(h, position_embeddings=position_embeddings, attention_mask=masks[self.config.layer_types[i]],
                       position_ids=text_position_ids, past_key_values=past_key_values, use_cache=use_cache,
                       **kwargs)  # fmt: skip

        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        def stack(h: torch.Tensor, r: int) -> torch.Tensor:
            layers = self.layers[a:b] if r == 0 else self._loop_aliases[r - 1]
            for j, layer in enumerate(layers):  # nn.Module.__call__: our checkpoint, not the layer's own

                def one(x: torch.Tensor, layer: nn.Module = layer, i: int = a + j) -> torch.Tensor:
                    return call(layer, x, i, lambda *y, **k: nn.Module.__call__(layer, *y, **k))

                h = checkpoint(keep_routing(one, self._loop_moes), h, use_reentrant=False) if ckpt else one(h)
            return h

        self.loop_aux = [] if self.training else None
        aux = self.loop_aux if any(not isinstance(m, HierDeltaMoE) for m in self._loop_moes) else None
        h = inputs_embeds
        for i in range(a):
            h = call(self.layers[i], h, i, self.layers[i])
        h = loop(stack, h, self.loop_update, c["repeats"], self._loop_moes, c["routing"], aux)
        for i in range(b, n):
            h = call(self.layers[i], h, i, self.layers[i])
        return M.Qwen3_5ModelOutputWithPast(last_hidden_state=self.norm(h), past_key_values=past_key_values)
