r"""Per-expert LoRA on a DeltaMoE: expert e's MLP gets :math:`\tfrac{\alpha}{r} B_{e,p} A_{e,p}\,x` added to each
projection p, in activation space (unmerged, as peft's LoRA; a merge into bf16 weights would round small updates
away). A zero expert (the base alone) gets none: give the base its own LoRA as an all-zero delta instead.

The block then always runs the materialised per-expert loop (``weights(e)``); the fused Triton, grouped and matrix
forwards read the deltas directly and would skip the LoRA.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..type_hints import Tokens
from .delta_moe import DeltaMoE
from .packed import PROJ


class _ExpertLoRA:
    """Mixin (``add_expert_lora``)."""

    lora_scale: float

    def lora(self, p: str, e: int, x: Tensor) -> Tensor:
        """Expert e's LoRA term of projection p for inputs x, in x's dtype."""
        a, b = getattr(self, f"lora_a_{p}")[e], getattr(self, f"lora_b_{p}")[e]
        return F.linear(F.linear(x.to(a.dtype), a), b).to(x.dtype) * self.lora_scale

    def expert_mlp(self, x: Tokens, e: int) -> Tokens:
        """SwiGLU MLP of expert e (< 0: a zero expert, no LoRA) on x (T, H)."""
        g, u, d = self.weights(e)  # type: ignore[attr-defined]
        if e < 0:
            return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)
        h = F.silu(F.linear(x, g) + self.lora("gate", e, x)) * (F.linear(x, u) + self.lora("up", e, x))
        return F.linear(h, d) + self.lora("down", e, h)

    def _forward_auto(self, x: Tokens) -> Tokens:
        idx, r = self.route(x)  # type: ignore[attr-defined]
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            out = out.index_add(0, rows, (r[rows, slots, None] * self.expert_mlp(x[rows], e)).to(out.dtype))
        return out


_LORA: dict[type, type] = {}


def add_expert_lora(moe: DeltaMoE, rank: int, alpha: float, dtype: torch.dtype = torch.float32) -> DeltaMoE:
    r"""Give every delta expert of ``moe`` a LoRA on gate, up and down (in place): parameters ``lora_a_<p>``
    (N, r, in) kaiming uniform as ``nn.Linear``, ``lora_b_<p>`` (N, out, r) zero, so the output is unchanged at init.

    Args:
        moe: Any DeltaMoE (HierDeltaMoE, packed formats).
        rank: r.
        alpha: :math:`\alpha`; the scale is :math:`\alpha / r`.
        dtype: Dtype of the LoRA parameters (float32: trained in full precision).

    Returns:
        moe, its class now a subclass with ``_ExpertLoRA``.
    """
    assert not isinstance(moe, _ExpertLoRA) and rank > 0
    cls = type(moe)
    if cls not in _LORA:
        _LORA[cls] = type(f"LoRA{cls.__name__}", (_ExpertLoRA, cls), {})
    dev, (inter, hidden), n = moe.gate.device, moe.gate.shape, moe.n_delta
    for p, (o, i) in zip(PROJ, ((inter, hidden), (inter, hidden), (hidden, inter))):
        a = torch.empty(n, rank, i, device=dev, dtype=dtype).uniform_(-1 / math.sqrt(i), 1 / math.sqrt(i))
        setattr(moe, f"lora_a_{p}", nn.Parameter(a))
        setattr(moe, f"lora_b_{p}", nn.Parameter(torch.zeros(n, o, rank, device=dev, dtype=dtype)))
    moe.__class__ = _LORA[cls]
    moe.lora_scale = alpha / rank
    return moe
