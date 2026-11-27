r"""The plain MoE that the delta forms are equivalent to: every expert stored as full weights, no shared base.

``MoE`` is a flat top-k SwiGLU MoE over :math:`E` experts; ``HierMoE`` the two-level (group, then expert) one. They
route exactly as ``DeltaMoE`` / ``HierDeltaMoE`` (same softmax, top-k and renormalisation, in float32), so
``DeltaMoE.to_moe()`` / ``HierDeltaMoE.to_moe()`` give the same function with :math:`W_b + \Delta_e` written out and
each zero expert stored as a copy of :math:`W_b`. They are the baseline the delta forms are measured against (tests,
benchmarks), and a plain module to start training or converting from.

Forwards: one ``F.grouped_mm`` per projection over all (token, slot) pairs (like transformers' ``grouped_mm`` path)
when ``use_grouped``, else a loop over the routed experts; both differentiable.
"""

import functools
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..ops import gmm, mlp, softmax_topk, sort_rows, use_grouped
from ..type_hints import MLPWeights, Tokens


class MoE(nn.Module):
    r"""Top-k SwiGLU MoE with :math:`E` full experts: :math:`y = \sum_{i=1}^{k} r_i \, \mathrm{MLP}(x; W_{e_i})`.

    The parameters are uninitialised (``torch.empty``); fill them with ``from_experts`` or ``load_state_dict``.

    Args:
        hidden: H, the model width.
        inter: I, the MLP intermediate width.
        n_experts: E, the number of experts.
        top_k: K, experts picked per token, :math:`1 \le K \le E`.
        device: Device of the parameters.
        dtype: Dtype of the parameters.

    Attributes:
        router: ``nn.Linear(H, E, bias=False)``.
        gate: (E, I, H) gate weights.
        up: (E, I, H) up weights.
        down: (E, H, I) down weights.
    """

    def __init__(
        self,
        hidden: int,
        inter: int,
        n_experts: int,
        top_k: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= n_experts
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        self.top_k, self.n_experts = top_k, n_experts
        self.router = nn.Linear(hidden, n_experts, bias=False, **factory)
        self.gate = nn.Parameter(torch.empty(n_experts, inter, hidden, **factory))
        self.up = nn.Parameter(torch.empty(n_experts, inter, hidden, **factory))
        self.down = nn.Parameter(torch.empty(n_experts, hidden, inter, **factory))

    @torch.no_grad()
    def _fill(self, experts: Sequence[MLPWeights], router: Tensor) -> "MoE":
        assert router.shape == self.router.weight.shape and len(experts) == self.n_experts
        self.router.weight.copy_(router)
        for i, dst in enumerate((self.gate, self.up, self.down)):
            dst.copy_(torch.stack([w[i] for w in experts]))
        return self

    @classmethod
    def from_experts(cls, experts: Sequence[MLPWeights], router: Tensor, top_k: int = 1) -> "MoE":
        """Build from full expert weights; device and dtype follow ``experts[0]``.

        Args:
            experts: E experts, each (gate (I, H), up (I, H), down (H, I)).
            router: (E, H) router weight, row e routes to ``experts[e]``.
            top_k: K, experts picked per token.

        Returns:
            The MoE holding copies of the weights.
        """
        gate = experts[0][0]
        m = cls(gate.shape[1], gate.shape[0], len(experts), top_k, gate.device, gate.dtype)
        return m._fill(experts, router)

    def topk(self, x: Tokens) -> tuple[Tensor, Tensor]:
        """Route tokens as ``DeltaMoE.topk``: float32 softmax, top-k, renormalised.

        Args:
            x: (T, H) tokens.

        Returns:
            (idx, r): idx (T, K) int64 expert index per slot; r (T, K) float32 weights, rows summing to 1.
        """
        return softmax_topk(self.router(x), self.top_k)

    def _forward_loop(self, x: Tokens) -> Tokens:
        """Loop over the routed experts, one MLP on the rows routed to each."""
        idx, r = self.topk(x)
        r = r.to(x.dtype)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)  # (top-k picks distinct experts: a row at most once)
            y = mlp(x[rows], self.gate[e], self.up[e], self.down[e])
            out = out.index_add(0, rows, (r[rows, slots, None] * y).to(out.dtype))
        return out

    def _forward_grouped(self, x: Tokens) -> Tokens:
        """All (token, slot) pairs sorted by expert, one grouped GEMM per projection."""
        idx, r = self.topk(x)
        tok = torch.arange(idx.numel(), device=x.device) // self.top_k  # pair -> token
        rows, offs = sort_rows(idx.flatten(), self.n_experts)
        xs = x[tok[rows]]
        h = F.silu(gmm(xs, self.gate, offs)) * gmm(xs, self.up, offs)
        o = torch.empty(idx.numel(), x.shape[-1], dtype=x.dtype, device=x.device).index_copy(0, rows, gmm(h, self.down, offs))
        return (o.view(-1, self.top_k, x.shape[-1]) * r.to(x.dtype)[..., None]).sum(1)

    def forward(self, x: Tensor) -> Tensor:
        """The MoE over tokens of any leading shape: grouped GEMM when ``use_grouped``, else the expert loop.

        Args:
            x: (..., H) tokens.

        Returns:
            MoE output, same shape as ``x``.
        """
        t = x.reshape(-1, x.shape[-1])
        y = self._forward_grouped(t) if use_grouped(t, self.top_k) else self._forward_loop(t)
        return y.view(x.shape)


@functools.cache
def hier_tables(group_sizes: tuple[int, ...], group_zeros: tuple[int, ...], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """The lookup tables of a two-level router, built once per layout and device (cached; not buffers, since
    transformers builds models on the meta device).

    Args:
        group_sizes: :math:`E_i`, experts with parameters per group.
        group_zeros: :math:`Z_i`, zero experts per group (all 0 for ``HierMoE``).
        device: Device of the tables.

    Returns:
        (col, delta, pad), each (G, M) with M the widest group: group-local expert j -> its column among the expert
        logits (= its index among all experts, group by group), its delta index (-1: zero expert or padding), and
        whether j is padding.
    """
    widths = [e + z for e, z in zip(group_sizes, group_zeros)]
    M, col, delta, pad = max(widths), [], [], []
    c = d = 0
    for e, w in zip(group_sizes, widths):
        col.append([c + j if j < w else 0 for j in range(M)])
        delta.append([d + j if j < e else -1 for j in range(M)])
        pad.append([j >= w for j in range(M)])
        c, d = c + w, d + e
    with torch.inference_mode(False):  # shared by every module of this layout: never inference tensors (no backward)
        return tuple(torch.tensor(v, device=device) for v in (col, delta, pad))  # type: ignore[return-value]


def hier_topk(logits: Tensor, col: Tensor, pad: Tensor, k1: int, k0: int | None) -> tuple[Tensor, Tensor, Tensor]:
    r"""Two-level routing: softmax over the G groups (dense, or top-:math:`k_0` renormalised), then within each kept
    group softmax over its experts, top-:math:`k_1` renormalised; slot weight :math:`h_i \tilde p_{ij}`.

    Args:
        logits: (T, G + sum of group widths) router output; columns G.. are the experts, group by group.
        col: (G, M) column table from ``hier_tables``.
        pad: (G, M) padding table from ``hier_tables``.
        k1: :math:`k_1`, experts picked per group.
        k0: :math:`k_0`, groups picked per token; None = every group.

    Returns:
        (gi, pj, w): gi (T, S) kept groups, pj (T, S, k1) group-local picks, w (T, S * k1) float32 slot weights.
    """
    G = col.shape[0]
    logits = logits.float()
    h = logits[:, :G].softmax(-1)
    p = logits[:, G:][:, col].masked_fill(pad, float("-inf")).softmax(-1)  # T G M
    pv, pj = p.topk(k1, dim=-1)
    pv = pv / pv.sum(-1, keepdim=True)
    gi = torch.arange(G, device=logits.device).expand(len(logits), G)
    if k0 is not None:
        h, gi = h.topk(k0, dim=-1)
        h = h / h.sum(-1, keepdim=True)
        pick = gi[..., None].expand(-1, -1, k1)
        pv, pj = pv.gather(1, pick), pj.gather(1, pick)
    return gi, pj, (h[..., None] * pv).flatten(1)


class HierMoE(MoE):
    r"""Two-level top-k MoE of full experts: :math:`y = \sum_i h_i \sum_{j \in \text{top-}k_1} \tilde p_{ij}\,
    \mathrm{MLP}(x; \theta_{ij})`, routed as ``HierDeltaMoE`` (see ``hier.py``) with every expert stored in full.

    Group :math:`i` holds :math:`E_i` experts, stored in order (group 0's first). The router has :math:`G + \sum_i E_i`
    rows: :math:`R_0` (G rows), then each group's expert rows. Flattened into :math:`S \cdot k_1` MoE slots per token
    (:math:`S = G` dense, else :math:`k_0`), so the MoE forwards run unchanged.

    Args:
        hidden: H, the model width.
        inter: I, the MLP intermediate width.
        group_sizes: :math:`E_i > 0` experts per group.
        top_k: :math:`k_1`, experts picked per group, :math:`1 \le k_1 \le \min_i E_i`.
        group_top_k: :math:`k_0`, groups picked per token; None = dense routing over all groups.
        device: Device of the parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        hidden: int,
        inter: int,
        group_sizes: Sequence[int],
        top_k: int = 1,
        group_top_k: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        G = len(group_sizes)
        assert G > 0 and 1 <= top_k <= max(group_sizes)  # narrower groups pad (weight 0), as HierDeltaMoE
        assert group_top_k is None or 1 <= group_top_k <= G
        super().__init__(hidden, inter, sum(group_sizes), top_k, device, dtype)  # router replaced below
        self.group_sizes, self.k1, self.group_top_k = list(group_sizes), top_k, group_top_k
        self.top_k = (G if group_top_k is None else group_top_k) * top_k  # MoE slots per token
        self.router = nn.Linear(hidden, G + sum(group_sizes), bias=False, device=device, dtype=dtype)

    def _tables(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        return hier_tables(tuple(self.group_sizes), (0,) * len(self.group_sizes), device)

    @classmethod
    def from_experts(  # type: ignore[override]
        cls,
        groups: Sequence[Sequence[MLPWeights]],
        router: Tensor,
        top_k: int = 1,
        group_top_k: int | None = None,
    ) -> "HierMoE":
        r"""Build from full experts in groups; device and dtype follow the first expert.

        Args:
            groups: ``groups[i]`` = group :math:`i`'s experts, each (gate (I, H), up (I, H), down (H, I)).
            router: :math:`(G + \sum_i E_i, H)` router weight, laid out as in the class docstring.
            top_k: :math:`k_1`, experts picked per group.
            group_top_k: :math:`k_0`, groups picked per token; None = dense.

        Returns:
            The HierMoE holding copies of the weights.
        """
        gate = groups[0][0][0]
        m = cls(gate.shape[1], gate.shape[0], [len(g) for g in groups], top_k, group_top_k, gate.device, gate.dtype)
        return m._fill([w for g in groups for w in g], router)

    def topk(self, x: Tokens) -> tuple[Tensor, Tensor]:
        """Route tokens to the flattened slots.

        Args:
            x: (T, H) tokens.

        Returns:
            (idx, w), both (T, ``top_k``): idx the int64 expert index per slot, w the float32 slot weight.
        """
        col, _, pad = self._tables(x.device)
        gi, pj, w = hier_topk(self.router(x), col, pad, self.k1, self.group_top_k)
        return col[gi[..., None], pj].flatten(1), w
