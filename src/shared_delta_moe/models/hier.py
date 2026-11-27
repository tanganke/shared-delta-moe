r"""Two-level shared delta MoE: grouped delta experts over one shared base :math:`W_b`, routed group then expert.

Group :math:`i` holds :math:`E_i` delta experts (weights :math:`W_b + \Delta_{ij}`) and :math:`Z_i` zero experts (no
parameters, :math:`\Delta = 0`: the base MLP itself, as in ``DeltaMoE``). "Shared" is the storage: :math:`W_b` once,
one :math:`\Delta_{ij}` per delta expert. Routing:

- :math:`h = \mathrm{softmax}(R_0 x)` over the :math:`G` groups; dense (every group, weight :math:`h_i`) or
  top-:math:`k_0` renormalised.
- :math:`p_i` = softmax over group :math:`i`'s :math:`E_i + Z_i` experts; top-:math:`k_1` renormalised within the group
  to :math:`\tilde p_{ij}` (zero experts compete too).
- The output:

  .. math::

     y = \sum_i h_i \sum_{j \in \text{top-}k_1} \tilde p_{ij}\, \mathrm{MLP}(x; W_b + \Delta_{ij}),
     \qquad \Delta_{ij} = 0 \text{ for a zero expert}

The weights sum to 1, so :math:`y` is the plain two-level ensemble of full experts :math:`\theta_{ij} = W_b +
\Delta_{ij}` (:math:`\theta = W_b` for a zero expert). One linear layer holds both routers: rows :math:`0 \ldots G-1`
are :math:`R_0`, then group by group :math:`E_i` delta rows and :math:`Z_i` zero-expert rows.

The routing is flattened into DeltaMoE slots: per token :math:`S \cdot k_1` slots (:math:`S = G` dense, else
:math:`k_0`) weighted :math:`h_i \tilde p_{ij}`; a picked zero expert is a base slot. Every DeltaMoE forward (materialised,
split, grouped, fused Triton) then computes :math:`y` unchanged.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..type_hints import MLPWeights, RouterWeight, Tokens
from .delta_moe import DeltaMoE, forced_slots
from ..ops import replay
from .moe import HierMoE, hier_tables, hier_topk


class HierDeltaMoE(DeltaMoE):
    r"""Two-level (group, then expert) shared delta MoE, flattened into DeltaMoE routing slots.

    Group :math:`i` holds :math:`E_i` delta experts and :math:`Z_i` zero experts; the module stores the shared base
    :math:`W_b` and the deltas :math:`\Delta_{ij}` of all groups in order (:math:`n_\text{delta} = \sum_i E_i`). The
    router is one linear layer of :math:`G + \sum_i (E_i + Z_i)` rows, laid out as in the module docstring.
    Construction leaves the weights uninitialised; fill them with ``from_experts`` or ``load_state_dict``.

    Args:
        hidden: :math:`H`, the model width.
        inter: :math:`I`, the MLP intermediate width.
        group_sizes: :math:`E_i`, the number of delta experts in each group (:math:`G` = ``len(group_sizes)``, :math:`G > 0`).
        group_zeros: :math:`Z_i \ge 0`, the number of zero experts in each group (same length as ``group_sizes``).
        top_k: :math:`k_1`, experts picked per group, :math:`1 \le k_1 \le \max_i (E_i + Z_i)`; in a group narrower
            than :math:`k_1` the extra slots are padding, a base slot of weight 0 (:math:`k_1 = \max_i (E_i + Z_i)`:
            dense routing within every group).
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
        assert 1 <= top_k <= max(e + z for e, z in zip(group_sizes, group_zeros))
        assert group_top_k is None or 1 <= group_top_k <= G
        super().__init__(hidden, inter, sum(group_sizes), 1, 1, device, dtype)  # router replaced below
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        self.group_sizes, self.group_zeros = list(group_sizes), list(group_zeros)
        self.k1, self.group_top_k, self.num_zero_experts = top_k, group_top_k, sum(group_zeros)
        self.top_k = (G if group_top_k is None else group_top_k) * top_k  # DeltaMoE slots per token
        widths = [e + z for e, z in zip(group_sizes, group_zeros)]
        self.router = nn.Linear(hidden, G + sum(widths), bias=False, **factory)

    def _tables(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """(G, M) group-local expert -> column of the expert logits, delta index (-1: zero expert or padding), padding.
        Built once per layout and device (``moe.hier_tables``), not buffers: transformers builds models on the meta device."""
        return hier_tables(tuple(self.group_sizes), tuple(self.group_zeros), device)

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

    def _forced_index(self) -> int:
        """``topk``'s delta index for ``force`` = (g, e): group g's e-th expert, or ``n_delta`` (the base) for one of
        its zero experts (group weight and in-group weight both 1)."""
        g, e = (int(v) for v in self.force)
        assert 0 <= g < len(self.group_sizes) and 0 <= e < self.group_sizes[g] + self.group_zeros[g], f"force {self.force}"
        return sum(self.group_sizes[:g]) + e if e < self.group_sizes[g] else self.n_delta

    def to_moe(self) -> HierMoE:  # type: ignore[override]
        r"""The equivalent plain two-level MoE: group :math:`i` holds its :math:`E_i` experts :math:`W_b + \Delta_{ij}`
        and then :math:`Z_i` stored copies of :math:`W_b`; the router (same layout) and top-k are unchanged.

        Returns:
            A new ``moe.HierMoE`` with group sizes :math:`E_i + Z_i`, on this module's device and dtype.
        """
        groups, d = [], 0
        with torch.no_grad():
            for e, z in zip(self.group_sizes, self.group_zeros):
                groups.append([self.weights(j) for j in range(d, d + e)] + [self.weights(-1)] * z)
                d += e
            return HierMoE.from_experts(groups, self.router.weight, self.k1, self.group_top_k)

    def topk(self, x: Tokens, router: nn.Module | None = None) -> tuple[Tensor, Tensor]:
        r"""Route tokens to the flattened DeltaMoE slots.

        Per token :math:`S \cdot k_1` slots (:math:`S = G` if ``group_top_k`` is None, else :math:`k_0`) weighted
        :math:`h_i \tilde p_{ij}`; a picked zero expert is a base slot.

        Args:
            x: (T, H) tokens.
            router: Another router shaped like ``self.router`` (``DeltaMoE.topk``); None = ``self.router``.

        Returns:
            (idx, weight), both (T, ``top_k``): ``idx`` is the int64 delta index per slot (``n_delta``: the base),
            ``weight`` the float32 routing weight per slot.
        """
        if self.force is not None:
            return forced_slots(x, self._forced_index(), self.top_k)
        col, delta, pad = self._tables(x.device)
        router = self.router if router is None else router
        logits = router(x)
        pick = replay(router, logits, lambda: hier_topk(logits.detach(), col, pad, self.k1, self.group_top_k)[:2])
        gi, pj, w = hier_topk(logits, col, pad, self.k1, self.group_top_k, pick)
        e = delta[gi[..., None], pj].flatten(1)  # T S*k1
        return torch.where(e < 0, self.n_delta, e), w
