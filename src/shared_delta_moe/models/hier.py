"""Two-level shared delta MoE: an optional shared base expert (weight 1) plus grouped delta experts, routed group then
expert.

Group i holds E_i delta experts (weights W_b + Delta_ij) and Z_i zero experts (no parameters, output 0). Routing:

  h = softmax(R_0 x) over the G groups; dense (every group, weight h_i) or top-k0 renormalised
  p_i = softmax over group i's E_i + Z_i experts; top-k1 renormalised within the group (zero experts compete too)
  y = [MLP(x; W_b)] + sum_i h_i sum_{j in top-k1, j a delta} p_ij MLP(x; W_b + Delta_ij)

The bracketed term only with shared_base (the default); without it (upscaling a dense model from its fine-tunes) y is
the routed ensemble alone. Either way y equals the plain two-level ensemble of full experts theta_ij = W_b + Delta_ij. One linear layer holds both
routers: rows 0..G-1 are R_0, then group by group E_i delta rows and Z_i zero-expert rows.

The routing is flattened into DeltaMoE slots: per token S * k1 slots (S = G dense, else k0) weighted h_i p_ij, plus
one base slot weighted 1 with shared_base; a picked zero expert becomes a base slot with weight 0. Every DeltaMoE forward (materialised,
split, grouped, fused Triton) then computes y unchanged.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..type_hints import MLPWeights, RouterWeight, Tokens
from .delta_moe import DeltaMoE


class HierDeltaMoE(DeltaMoE):
    """Two-level (group, then expert) shared delta MoE, flattened into DeltaMoE routing slots.

    Group i holds E_i delta experts and Z_i zero experts; the module stores the shared base W_b and the deltas
    Delta_ij of all groups in order (n_delta = sum_i E_i). The router is one linear layer of G + sum_i (E_i + Z_i)
    rows, laid out as in the module docstring. Construction leaves the weights uninitialised; fill them with
    from_experts or load_state_dict.

    Args:
        hidden: H, the model width.
        inter: I, the MLP intermediate width.
        group_sizes: E_i, the number of delta experts in each group (G = len(group_sizes) > 0).
        group_zeros: Z_i >= 0, the number of zero experts in each group (same length as group_sizes).
        top_k: k1, experts picked per group, 1 <= k1 <= min_i (E_i + Z_i).
        group_top_k: k0, groups picked per token, 1 <= k0 <= G; None = dense (soft) routing over all groups.
        device: device of the parameters.
        dtype: dtype of the parameters.
        shared_base: add MLP(x; W_b) with weight 1 (one extra base slot per token).
    """

    def __init__(
        self,
        hidden: int,
        inter: int,
        group_sizes: Sequence[int],
        group_zeros: Sequence[int],
        top_k: int = 1,
        group_top_k: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        shared_base: bool = True,
    ) -> None:
        G = len(group_sizes)
        assert G == len(group_zeros) and G > 0 and min(group_zeros) >= 0
        assert 1 <= top_k <= min(e + z for e, z in zip(group_sizes, group_zeros))
        assert group_top_k is None or 1 <= group_top_k <= G
        super().__init__(hidden, inter, sum(group_sizes), 1, 1, device, dtype)  # router replaced below
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        self.group_sizes, self.group_zeros = list(group_sizes), list(group_zeros)
        self.k1, self.group_top_k, self.num_zero_experts = top_k, group_top_k, sum(group_zeros)
        self.shared_base = shared_base
        self.top_k = (G if group_top_k is None else group_top_k) * top_k + shared_base  # DeltaMoE slots per token
        widths = [e + z for e, z in zip(group_sizes, group_zeros)]
        self.router = nn.Linear(hidden, G + sum(widths), bias=False, **factory)
        self._table_cache: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = {}

    def _tables(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """(G, M) group-local expert -> column of the expert logits, delta index (-1: zero expert or padding), padding.
        Built per device on first use, not buffers: transformers builds models on the meta device."""
        if device not in self._table_cache:
            widths = [e + z for e, z in zip(self.group_sizes, self.group_zeros)]
            M, col, delta, pad = max(widths), [], [], []
            c = d = 0
            for e, w in zip(self.group_sizes, widths):
                col.append([c + j if j < w else 0 for j in range(M)])
                delta.append([d + j if j < e else -1 for j in range(M)])
                pad.append([j >= w for j in range(M)])
                c, d = c + w, d + e
            self._table_cache[device] = tuple(torch.tensor(v, device=device) for v in (col, delta, pad))  # type: ignore[assignment]
        return self._table_cache[device]

    @classmethod
    @torch.no_grad()
    def from_experts(  # type: ignore[override]
        cls,
        base: MLPWeights,
        groups: Sequence[Sequence[MLPWeights]],
        router: RouterWeight,
        group_zeros: Sequence[int],
        top_k: int = 1,
        group_top_k: int | None = None,
        shared_base: bool = True,
    ) -> "HierDeltaMoE":
        """Build from the plain two-level ensemble of full experts.

        Stores W_b = base and Delta_ij = theta_ij - W_b; device and dtype follow base.

        Args:
            base: (gate (I, H), up (I, H), down (H, I)), the shared expert W_b.
            groups: groups[i] = group i's full experts theta_ij, each as (gate, up, down) shaped like base;
                E_i = len(groups[i]).
            router: (G + sum_i (E_i + Z_i), H) router weight, laid out as in the module docstring.
            group_zeros: Z_i, the number of zero experts in each group.
            top_k: k1, experts picked per group.
            group_top_k: k0, groups picked per token; None = dense routing over all groups.
            shared_base: add MLP(x; W_b) with weight 1.

        Returns:
            The HierDeltaMoE holding W_b, the deltas and the router.
        """
        gate, up, down = base
        m = cls(gate.shape[1], gate.shape[0], [len(g) for g in groups], group_zeros, top_k, group_top_k,
                gate.device, gate.dtype, shared_base)  # fmt: skip
        assert router.shape == m.router.weight.shape
        m.router.weight.copy_(router)
        experts = [w for g in groups for w in g]
        for dst, b in ((m.gate, gate), (m.up, up), (m.down, down)):
            dst.copy_(b)
        for dst, b, i in ((m.d_gate, gate, 0), (m.d_up, up, 1), (m.d_down, down, 2)):
            dst.copy_(torch.stack([w[i] for w in experts]) - b if experts else dst)
        return m

    def topk(self, x: Tokens) -> tuple[Tensor, Tensor]:
        """Route tokens to the flattened DeltaMoE slots.

        Per token S * k1 slots (S = G if group_top_k is None, else k0) weighted h_i p_ij; a picked zero expert becomes
        a base slot with weight 0. With shared_base the last slot is the shared base with weight 1.

        Args:
            x: (T, H) tokens.

        Returns:
            (idx, weight), both (T, top_k): idx is the int64 delta index per slot (n_delta: the base), weight the
            float32 routing weight per slot.
        """
        G = len(self.group_sizes)
        col, delta, pad = self._tables(x.device)
        logits = self.router(x).float()
        h = logits[:, :G].softmax(-1)
        p = logits[:, G:][:, col].masked_fill(pad, float("-inf")).softmax(-1)  # T G M
        pv, pj = p.topk(self.k1, dim=-1)
        pv = pv / pv.sum(-1, keepdim=True)
        gi = torch.arange(G, device=x.device).expand(len(x), G)
        if self.group_top_k is not None:
            h, gi = h.topk(self.group_top_k, dim=-1)
            h = h / h.sum(-1, keepdim=True)
            pick = gi[..., None].expand(-1, -1, self.k1)
            pv, pj = pv.gather(1, pick), pj.gather(1, pick)
        e = delta[gi[..., None], pj].flatten(1)  # T S*k1
        w = (h[..., None] * pv).flatten(1).masked_fill(e < 0, 0.0)
        e = torch.where(e < 0, self.n_delta, e)
        if not self.shared_base:
            return e, w
        one = torch.ones_like(w[:, :1])
        return torch.cat((e, torch.full_like(e[:, :1], self.n_delta)), 1), torch.cat((w, one), 1)
