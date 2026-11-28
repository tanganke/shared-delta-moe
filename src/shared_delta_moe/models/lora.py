r"""Per-expert LoRA on a DeltaMoE: expert e's MLP gets :math:`\tfrac{\alpha}{r} B_{e,p} A_{e,p}\,x` added to each
projection p, in activation space (unmerged, as peft's LoRA; a merge into bf16 weights would round small updates
away). A zero expert (the base alone) gets none: give the base its own LoRA as an all-zero delta instead.

The block runs the materialised per-expert loop (``weights(e)``): the fused Triton, grouped and matrix forwards read
the deltas directly and would skip the LoRA. The exception is training a 2-bit ``GridDeltaMoE`` with frozen base and
deltas, where only the router and the LoRA train, with or without grad. That runs ``grouped_lora_moe`` on the decoded experts
(``GridDeltaMoE._decoded``, rebuilt for the backward rather than kept). The decoded weights take no
gradient; the LoRA is a separate low-rank branch per projection.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..ops import gmm, recompute_saved, sort_rows
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

    def _router_only(self) -> bool:
        """Only the router and the LoRA train (``GridDeltaMoE._use_decoded``)."""
        return all(not p.requires_grad for n, p in self.named_parameters()  # type: ignore[attr-defined]
                   if n != "router.weight" and not n.startswith("lora_"))  # fmt: skip

    def _forward_decoded(self, x: Tokens) -> Tokens:
        """``grouped_lora_moe`` on the decoded experts (rebuilt for the backward)."""
        idx, r = self.topk(x)  # type: ignore[attr-defined]
        ws = self._decoded()  # type: ignore[attr-defined]
        loras = [(getattr(self, f"lora_a_{p}"), getattr(self, f"lora_b_{p}")) for p in PROJ]
        with recompute_saved(ws, self._decoded):  # type: ignore[attr-defined]  # every zero expert -> the base
            y = grouped_lora_moe(x.to(ws[0].dtype), idx.clamp(max=self.n_delta), r, ws, loras,  # type: ignore[attr-defined]
                                 self.lora_scale)  # fmt: skip
        return y.to(x.dtype)

    def _forward_auto(self, x: Tokens) -> Tokens:
        # with or without grad: a reentrant checkpoint runs the forward under no_grad, its recompute with grad, and
        # both must take the same path
        use = getattr(self, "_decodable", None)
        if use is not None and use(x):
            return self._forward_decoded(x)
        idx, r = self.route(x)  # type: ignore[attr-defined]
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            out = out.index_add(0, rows, (r[rows, slots, None] * self.expert_mlp(x[rows], e)).to(out.dtype))
        return out


def grouped_lora_moe(x: Tokens, idx: Tensor, r: Tensor, ws: "list[Tensor]", loras: "list[tuple[Tensor, Tensor]]",
                     scale: float) -> Tokens:  # fmt: skip
    r"""``moe.grouped_moe`` with expert e's LoRA :math:`s\,B_{e,p}A_{e,p}` on each projection p, as
    ``_ExpertLoRA.expert_mlp``: grouped GEMMs over (token, slot) pairs sorted by expert, the LoRA in its own dtype.

    Args:
        x: (T, H) tokens, in the weights' dtype.
        idx: (T, k) expert per slot, <= N; N is the base (a zero expert), which has no LoRA.
        r: (T, k) routing weights.
        ws: Full weights gate (N + 1, I, H), up (N + 1, I, H), down (N + 1, H, I), the base last.
        loras: (A (N, r, in), B (N, out, r)) per projection, gate, up, down.
        scale: s, :math:`\alpha / r`.

    Returns:
        (T, H) in x's dtype.
    """
    k = idx.shape[1]
    tok = torch.arange(idx.numel(), device=x.device) // k  # pair -> token
    rows, offs = sort_rows(idx.flatten(), ws[0].shape[0])
    xs = x[tok[rows]]

    def proj(v: Tensor, w: Tensor, a: Tensor, b: Tensor) -> Tensor:
        a, b = (torch.cat([t, t.new_zeros(1, *t.shape[1:])]) for t in (a, b))  # the base: no LoRA
        return gmm(v, w, offs) + gmm(gmm(v.to(a.dtype), a, offs), b, offs).to(v.dtype) * scale

    h = F.silu(proj(xs, ws[0], *loras[0])) * proj(xs, ws[1], *loras[1])
    o = torch.empty(idx.numel(), x.shape[-1], dtype=x.dtype, device=x.device)
    o = o.index_copy(0, rows, proj(h, ws[2], *loras[2]))
    return (o.view(-1, k, x.shape[-1]) * r.to(x.dtype)[..., None]).sum(1)


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
