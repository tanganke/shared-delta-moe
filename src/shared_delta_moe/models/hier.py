"""Two-level shared delta MoE: a shared base expert (weight 1) plus grouped delta experts, routed group then expert.

Group i holds E_i delta experts (weights W_b + Delta_ij) and Z_i zero experts (no parameters, output 0). Routing:

  h = softmax(R_0 x) over the G groups; dense (every group, weight h_i) or top-k0 renormalised
  p_i = softmax over group i's E_i + Z_i experts; top-k1 renormalised within the group (zero experts compete too)
  y = MLP(x; W_b) + sum_i h_i sum_{j in top-k1, j a delta} p_ij MLP(x; W_b + Delta_ij)

which equals the plain two-level ensemble of full experts theta_ij = W_b + Delta_ij. One linear layer holds both
routers: rows 0..G-1 are R_0, then group by group E_i delta rows and Z_i zero-expert rows.

The routing is flattened into DeltaMoE slots: per token S * k1 slots (S = G dense, else k0) weighted h_i p_ij, plus
one base slot weighted 1; a picked zero expert becomes a base slot with weight 0. Every DeltaMoE forward (materialised,
split, grouped, fused Triton) then computes y unchanged.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..type_hints import MLPWeights, RouterWeight, Tokens
from .delta_moe import DeltaMoE


class HierDeltaMoE(DeltaMoE):
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
    ) -> None:
        """group_sizes[i] = E_i deltas, group_zeros[i] = Z_i zero experts; top_k = k1 per group; group_top_k = k0,
        None = dense (soft) over all groups. Uninitialised; fill with from_experts or load_state_dict."""
        G = len(group_sizes)
        assert G == len(group_zeros) and G > 0 and min(group_zeros) >= 0
        assert 1 <= top_k <= min(e + z for e, z in zip(group_sizes, group_zeros))
        assert group_top_k is None or 1 <= group_top_k <= G
        super().__init__(hidden, inter, sum(group_sizes), 1, 1, device, dtype)  # router replaced below
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        self.group_sizes, self.group_zeros = list(group_sizes), list(group_zeros)
        self.k1, self.group_top_k, self.num_zero_experts = top_k, group_top_k, sum(group_zeros)
        self.top_k = (G if group_top_k is None else group_top_k) * top_k + 1  # DeltaMoE slots per token
        widths = [e + z for e, z in zip(group_sizes, group_zeros)]
        self.router = nn.Linear(hidden, G + sum(widths), bias=False, **factory)
        # (G, M) group-local expert -> column of the expert logits / delta index (-1: zero expert or padding)
        M, col, delta, pad = max(widths), [], [], []
        c = d = 0
        for e, w in zip(group_sizes, widths):
            col.append([c + j if j < w else 0 for j in range(M)])
            delta.append([d + j if j < e else -1 for j in range(M)])
            pad.append([j >= w for j in range(M)])
            c, d = c + w, d + e
        for n, v in (("_col", col), ("_delta", delta), ("_pad", pad)):
            self.register_buffer(n, torch.tensor(v, device=device), persistent=False)

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
    ) -> "HierDeltaMoE":
        """From the plain two-level ensemble: base = the shared expert, groups[i] = group i's full experts, router =
        (G + sum_i (E_i + Z_i), H) laid out as in the module docstring. Stores W_b and Delta_ij = theta_ij - W_b."""
        gate, up, down = base
        m = cls(gate.shape[1], gate.shape[0], [len(g) for g in groups], group_zeros, top_k, group_top_k,
                gate.device, gate.dtype)  # fmt: skip
        assert router.shape == m.router.weight.shape
        m.router.weight.copy_(router)
        experts = [w for g in groups for w in g]
        for dst, b in ((m.gate, gate), (m.up, up), (m.down, down)):
            dst.copy_(b)
        for dst, b, i in ((m.d_gate, gate, 0), (m.d_up, up, 1), (m.d_down, down, 2)):
            dst.copy_(torch.stack([w[i] for w in experts]) - b if experts else dst)
        return m

    def topk(self, x: Tokens) -> tuple[Tensor, Tensor]:
        """(T, top_k) delta index (n_delta: base) and float32 weight per slot; the last slot is the shared base."""
        G = len(self.group_sizes)
        logits = self.router(x).float()
        h = logits[:, :G].softmax(-1)
        p = logits[:, G:][:, self._col].masked_fill(self._pad, float("-inf")).softmax(-1)  # T G M
        pv, pj = p.topk(self.k1, dim=-1)
        pv = pv / pv.sum(-1, keepdim=True)
        gi = torch.arange(G, device=x.device).expand(len(x), G)
        if self.group_top_k is not None:
            h, gi = h.topk(self.group_top_k, dim=-1)
            h = h / h.sum(-1, keepdim=True)
            pick = gi[..., None].expand(-1, -1, self.k1)
            pv, pj = pv.gather(1, pick), pj.gather(1, pick)
        e = self._delta[gi[..., None], pj].flatten(1)  # T S*k1
        w = (h[..., None] * pv).flatten(1).masked_fill(e < 0, 0.0)
        e = torch.where(e < 0, self.n_delta, e)
        one = torch.ones_like(w[:, :1])
        return torch.cat((e, torch.full_like(e[:, :1], self.n_delta)), 1), torch.cat((w, one), 1)
