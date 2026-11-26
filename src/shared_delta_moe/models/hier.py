r"""Two-level shared delta MoE: grouped delta experts over one shared base :math:`W_b`, routed group then expert.

Group :math:`i` holds :math:`E_i` delta experts (weights :math:`W_b + \Delta_{ij}`) and :math:`Z_i` zero experts (no
parameters, :math:`\Delta = 0`: the base MLP itself, as in ``DeltaMoE``). "Shared" is the storage: :math:`W_b` once,
one :math:`\Delta_{ij}` per delta expert. Routing:

- :math:`h = \mathrm{softmax}(R_0 x)` over the :math:`G` groups; dense (every group, weight :math:`h_i`) or
  top-:math:`k_0` renormalised.
- :math:`p_i` = softmax over group :math:`i`'s :math:`E_i + Z_i` experts; top-:math:`k_1` renormalised within the group
  (zero experts compete too).
- The output:

  .. math::

     y = \sum_i h_i \sum_{j \in \text{top-}k_1} p_{ij}\, \mathrm{MLP}(x; W_b + \Delta_{ij}),
     \qquad \Delta_{ij} = 0 \text{ for a zero expert}

The weights sum to 1, so :math:`y` is the plain two-level ensemble of full experts :math:`\theta_{ij} = W_b +
\Delta_{ij}` (:math:`\theta = W_b` for a zero expert). One linear layer holds both routers: rows :math:`0 \ldots G-1`
are :math:`R_0`, then group by group :math:`E_i` delta rows and :math:`Z_i` zero-expert rows.

The routing is flattened into DeltaMoE slots: per token :math:`S \cdot k_1` slots (:math:`S = G` dense, else
:math:`k_0`) weighted :math:`h_i p_{ij}`; a picked zero expert is a base slot. Every DeltaMoE forward (materialised,
split, grouped, fused Triton) then computes :math:`y` unchanged.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..type_hints import MLPWeights, RouterWeight, Tokens
from .delta_moe import DeltaMoE


class HierDeltaMoE(DeltaMoE):
    r"""Two-level (group, then expert) shared delta MoE, flattened into DeltaMoE routing slots.

    Group :math:`i` holds :math:`E_i` delta experts and :math:`Z_i` zero experts; the module stores the shared base
    :math:`W_b` and the deltas :math:`\Delta_{ij}` of all groups in order (:math:`n_\text{delta} = \sum_i E_i`). The
    router is one linear layer of :math:`G + \sum_i (E_i + Z_i)` rows, laid out as in the module docstring.
    Construction leaves the weights uninitialised; fill them with ``from_experts`` or ``load_state_dict``.

    Args:
        hidden: :math:`H`, the model width.
        inter: :math:`I`, the MLP intermediate width.
        group_sizes: :math:`E_i`, the number of delta experts in each group (:math:`G` = ``len(group_sizes)`` > 0).
        group_zeros: :math:`Z_i \ge 0`, the number of zero experts in each group (same length as ``group_sizes``).
        top_k: :math:`k_1`, experts picked per group, :math:`1 \le k_1 \le \min_i (E_i + Z_i)`.
        group_top_k: :math:`k_0`, groups picked per token, :math:`1 \le k_0 \le G`; None = dense (soft) routing over
            all groups.
        device: device of the parameters.
        dtype: dtype of the parameters.
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
    ) -> None:
        G = len(group_sizes)
        assert G == len(group_zeros) and G > 0 and min(group_zeros) >= 0
        assert 1 <= top_k <= min(e + z for e, z in zip(group_sizes, group_zeros))
        assert group_top_k is None or 1 <= group_top_k <= G
        super().__init__(hidden, inter, sum(group_sizes), 1, 1, device, dtype)  # router replaced below
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        self.group_sizes, self.group_zeros = list(group_sizes), list(group_zeros)
        self.k1, self.group_top_k, self.num_zero_experts = top_k, group_top_k, sum(group_zeros)
        self.top_k = (G if group_top_k is None else group_top_k) * top_k  # DeltaMoE slots per token
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
    ) -> "HierDeltaMoE":
        r"""Build from the plain two-level ensemble of full experts.

        Stores :math:`W_b` = ``base`` and :math:`\Delta_{ij} = \theta_{ij} - W_b`; device and dtype follow ``base``.

        Args:
            base: (gate (I, H), up (I, H), down (H, I)), the shared expert :math:`W_b`.
            groups: ``groups[i]`` = group :math:`i`'s full experts :math:`\theta_{ij}`, each as (gate, up, down) shaped
                like ``base``; :math:`E_i` = ``len(groups[i])``.
            router: :math:`(G + \sum_i (E_i + Z_i), H)` router weight, laid out as in the module docstring.
            group_zeros: :math:`Z_i`, the number of zero experts in each group.
            top_k: :math:`k_1`, experts picked per group.
            group_top_k: :math:`k_0`, groups picked per token; None = dense routing over all groups.

        Returns:
            The HierDeltaMoE holding :math:`W_b`, the deltas and the router.
        """
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
        r"""Route tokens to the flattened DeltaMoE slots.

        Per token :math:`S \cdot k_1` slots (:math:`S = G` if ``group_top_k`` is None, else :math:`k_0`) weighted
        :math:`h_i p_{ij}`; a picked zero expert is a base slot.

        Args:
            x: (T, H) tokens.

        Returns:
            (idx, weight), both (T, ``top_k``): ``idx`` is the int64 delta index per slot (``n_delta``: the base),
            ``weight`` the float32 routing weight per slot.
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
        return torch.where(e < 0, self.n_delta, e), (h[..., None] * pv).flatten(1)
