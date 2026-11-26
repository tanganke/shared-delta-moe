r"""Top-k SwiGLU MoE written as a shared base expert plus per-expert deltas, with two exact forward forms.

Routed expert :math:`e` has weights :math:`W_e = W_b + \Delta_e` for each of gate, up and down. Router rows
0 .. n-1 are the deltas; rows n .. n+Z-1 are zero experts (``num_zero_experts`` = Z, default 1): no parameters, each
stands for the base MLP itself. Routing is softmax, top-k, renormalised over the k picks (weights :math:`r`). Both
forms below equal the plain MoE

.. math::

    \sum_{i=1}^{k} r_i \, \mathrm{MLP}(x; W_{e_i})

with experts :math:`[W_b + \Delta_1, \dots, W_b + \Delta_n] + [W_b] \times Z`; they differ only in where the addition
happens:

``forward_param``
    Parameter aggregation: per expert, materialise :math:`W_b + \Delta_e` once and run one MLP on the (token, slot)
    pairs routed to it, weighted by :math:`r` (small groups skip the materialisation).
``forward_matrix``
    Per-matrix output aggregation: gate/up are :math:`W_b x` (once for all tokens) :math:`+ \Delta_e x` per slot;
    each slot has its own activation :math:`h_i`, down is :math:`W_b h_i + \Delta_e h_i`; slots summed with weights
    :math:`r`.
``forward_dense``
    The plain MoE baseline, from stored full experts.

Each public forward picks an implementation (``_forward_*``) by hardware and input, with thresholds in
``constants.py``: ``forward_param`` uses the fused Triton kernel on CUDA bf16/fp16 for a moderate number of (token,
slot) pairs, else materialised weights; ``forward_matrix`` / ``forward_dense`` use grouped GEMM (``F.grouped_mm``)
when available for the dtype and device and there is more than one pair, else a loop.

Adding :math:`r \Delta_e x` into one shared activation instead (merging weights per token) is not the same function
when :math:`k > 1`, because SwiGLU is nonlinear; each slot keeps its own activation here. With :math:`k = 1`,
:math:`r = 1` and both reduce to one MLP.

Plain PyTorch (``F.linear``, ``F.grouped_mm``, ``index_add``), plus an optional Triton kernel for parameter
aggregation that adds :math:`W_b + \Delta_e` tile by tile in registers instead of writing the sum out
(``triton_kernels/``).
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..constants import (
    GROUPED_DTYPES,
    GROUPED_MIN_PAIRS,
    HAS_TRITON,
    PARAM_MIN_ROWS,
    SORT_ONE_HOT_MAX,
    TRITON_DTYPES,
    TRITON_MAX_PAIRS_HOPPER,
    TRITON_MAX_PAIRS_OLDER,
    TRITON_MIN_PAIRS,
)
from ..ops import gmm, groups, mlp, pow2, sort_rows
from ..type_hints import (
    DenseExperts,
    Experts,
    Group,
    Hidden,
    MLPWeights,
    PairHidden,
    RouteWeights,
    RouterWeight,
    Tokens,
)


class DeltaMoE(nn.Module):
    r"""Top-k SwiGLU MoE stored as a shared base expert :math:`W_b` plus per-expert deltas :math:`\Delta_e`.

    The parameters are uninitialised (``torch.empty``); fill them with ``from_experts`` or ``load_state_dict``.

    Args:
        hidden: H, the model width.
        inter: I, the MLP intermediate width.
        n_delta: :math:`N`, the number of routed experts with a delta.
        top_k: :math:`K`, experts picked per token, :math:`1 \le K \le N + Z`.
        num_zero_experts: :math:`Z \ge 0`, router rows :math:`N, \ldots, N+Z-1` that route to the base MLP itself (no parameters).
        device: Device of the parameters.
        dtype: Dtype of the parameters.

    Attributes:
        triton_train: Train through the fused Triton forward + backward (``triton_kernels/train.py``) instead of
            materialised cuBLAS: holds no :math:`W_b + \Delta_e` and no per-pair activations for the backward, but
            measured slower (README, Training). Default False.
        router: ``nn.Linear(H, N+Z, bias=False)``; rows :math:`N, \ldots, N+Z-1` are the zero experts.
        gate: (I, H) base gate weight :math:`W_b`.
        up: (I, H) base up weight :math:`W_b`.
        down: (H, I) base down weight :math:`W_b`.
        d_gate: (N, I, H) gate deltas :math:`\Delta_e`.
        d_up: (N, I, H) up deltas :math:`\Delta_e`.
        d_down: (N, H, I) down deltas :math:`\Delta_e`.

    Raises:
        AssertionError: If ``num_zero_experts < 0`` or ``top_k`` is outside 1 .. ``n_delta + num_zero_experts``.
    """

    # train through the fused Triton forward + backward (triton_kernels/train.py) instead of materialised cuBLAS:
    # holds no W_b + Delta_e and no per-pair activations for the backward, but measured slower (README, Training)
    triton_train: bool = False
    gate: nn.Parameter  # I H
    up: nn.Parameter  # I H
    down: nn.Parameter  # H I
    d_gate: nn.Parameter  # N I H
    d_up: nn.Parameter  # N I H
    d_down: nn.Parameter  # N H I

    def __init__(
        self,
        hidden: int,
        inter: int,
        n_delta: int,
        top_k: int = 1,
        num_zero_experts: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        assert num_zero_experts >= 0 and 1 <= top_k <= n_delta + num_zero_experts

        def p(*shape: int) -> nn.Parameter:
            return nn.Parameter(torch.empty(*shape, **factory))

        self.top_k, self.num_zero_experts, self.n_delta = (
            top_k,
            num_zero_experts,
            n_delta,
        )
        self.router = nn.Linear(
            hidden, n_delta + num_zero_experts, bias=False, **factory
        )
        self.gate, self.up, self.down = (
            p(inter, hidden),
            p(inter, hidden),
            p(hidden, inter),
        )
        # (experts, rows, cols) of each projection's deltas, for the packed formats (models/packed.py)
        self.shapes = {"gate": (n_delta, inter, hidden), "up": (n_delta, inter, hidden), "down": (n_delta, hidden, inter)}
        self._init_deltas(n_delta, inter, hidden, factory)

    def _init_deltas(
        self, n: int, inter: int, hidden: int, factory: dict[str, Any]
    ) -> None:
        """Delta storage; a packed format (``packed.PackedDeltaMoE``) overrides this and ``d_gate``/``d_up``/``d_down``."""
        self.d_gate, self.d_up, self.d_down = (
            nn.Parameter(torch.empty(n, inter, hidden, **factory)),
            nn.Parameter(torch.empty(n, inter, hidden, **factory)),
            nn.Parameter(torch.empty(n, hidden, inter, **factory)),
        )

    @classmethod
    @torch.no_grad()
    def from_experts(
        cls,
        base: MLPWeights,
        experts: Sequence[MLPWeights],
        router: RouterWeight,
        top_k: int = 1,
        num_zero_experts: int = 1,
    ) -> "DeltaMoE":
        r"""Build a DeltaMoE from a plain MoE.

        Stores :math:`W_b` and :math:`\Delta_e = W_e - W_b` in the inputs' dtype, on the inputs' device.

        Args:
            base: The base expert's (gate (I, H), up (I, H), down (H, I)), stored as :math:`W_b`.
            experts: The other N experts' full weights :math:`W_e`, each (gate, up, down) shaped like base.
            router: (N+Z, H) the MoE's router weight; rows :math:`N, \ldots, N+Z-1` route to the base.
            top_k: :math:`K`, experts picked per token.
            num_zero_experts: :math:`Z`; 0: the base is only shared, never an expert on its own.

        Returns:
            The DeltaMoE with :math:`H`, :math:`I` from ``base`` and :math:`N` = ``len(experts)``.

        Raises:
            AssertionError: If ``router.shape[0] != len(experts) + num_zero_experts``.
        """
        gate, up, down = base
        assert router.shape[0] == len(experts) + num_zero_experts
        m = cls(
            gate.shape[1],
            gate.shape[0],
            len(experts),
            top_k,
            num_zero_experts,
            dtype=gate.dtype,
            device=gate.device,
        )
        m.router.weight.copy_(router)
        for dst, b in ((m.gate, gate), (m.up, up), (m.down, down)):
            dst.copy_(b)
        for dst, b, i in ((m.d_gate, gate, 0), (m.d_up, up, 1), (m.d_down, down, 2)):
            dst.copy_(torch.stack([w[i] for w in experts]) - b)
        return m

    def topk(self, x: Tokens) -> tuple[Tensor, Tensor]:
        """Route tokens: top-k router indices and their renormalised weights.

        Float32 softmax and renormalisation, as transformers' MoE routers do (bf16 can flip near-ties). Every forward
        routes through this one call, so they all pick the same experts, ties included.

        Args:
            x: (T, H) tokens.

        Returns:
            A tuple (idx, r): idx (T, K) int64 router index per slot (>= ``n_delta``: a zero expert); r (T, K) float32
            routing weights, each row renormalised to sum to 1.
        """
        r, i = self.router(x).softmax(-1, dtype=torch.float).topk(self.top_k, dim=-1)
        return i, r / r.sum(-1, keepdim=True)

    def route(self, x: Tokens) -> tuple[Experts, RouteWeights]:
        """``topk`` with every zero expert as -1 and the weights in the model dtype (used by the PyTorch forwards).

        Args:
            x: (T, H) tokens.

        Returns:
            A tuple (idx, r): idx (T, K) int64 delta index per slot, -1 = any zero expert; r (T, K) routing weights
            in x's dtype, rows summing to 1.
        """
        i, r = self.topk(x)
        return torch.where(i < self.n_delta, i, -1), r.to(x.dtype)

    def delta(self, e: int) -> MLPWeights:
        r""":math:`\Delta_e` for gate, up, down; ``dare.DareDeltaMoE`` unpacks only expert e here.

        Args:
            e: Delta index, :math:`0 \le e <` ``n_delta``.

        Returns:
            (gate delta (I, H), up delta (I, H), down delta (H, I)).
        """
        return self.d_gate[e], self.d_up[e], self.d_down[e]

    def weights(self, e: int) -> MLPWeights:
        r"""Full weights of expert e: :math:`W_b + \Delta_e`, or :math:`W_b` itself for a zero expert.

        Args:
            e: Delta index, or < 0 (-1) for a zero expert.

        Returns:
            (gate (I, H), up (I, H), down (H, I)); materialised sums for :math:`e \ge 0`, the base parameters for :math:`e < 0`.
        """
        if e < 0:
            return self.gate, self.up, self.down
        dg, du, dd = self.delta(e)
        return self.gate + dg, self.up + du, self.down + dd

    def mlp_split(self, x: Tokens, e: int) -> Tokens:
        r""":math:`\mathrm{MLP}(x; W_b + \Delta_e)` without forming the sum: every projection is
        :math:`W_b x + \Delta_e x`.

        Args:
            x: (T, H) tokens.
            e: Delta index, :math:`0 \le e <` ``n_delta``.

        Returns:
            (T, H) SwiGLU MLP output of expert e.
        """
        dg, du, dd = self.delta(e)
        g = F.linear(x, self.gate) + F.linear(x, dg)
        u = F.linear(x, self.up) + F.linear(x, du)
        h = F.silu(g) * u
        return F.linear(h, self.down) + F.linear(h, dd)

    def _forward_param(self, x: Tokens, min_rows: int = 0) -> Tokens:
        r"""``min_rows > 0``: an expert with fewer routed rows skips materialising :math:`W_b + \Delta_e` and uses
        ``mlp_split`` (reads :math:`W_b` and :math:`\Delta_e` once, writes no weights: cheaper while memory-bound, twice the FLOPs).
        """
        idx, r = self.route(x)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            # under top-k two zero experts can share a row; index_add sums them
            rows, slots = (idx == e).nonzero(as_tuple=True)
            if 0 <= e and len(rows) < min_rows:
                y = self.mlp_split(x[rows], e)
            else:
                y = mlp(x[rows], *self.weights(e))
            out = out.index_add(0, rows, (r[rows, slots, None] * y).to(out.dtype))  # (autocast: y in another dtype)
        return out

    def _forward_matrix(self, x: Tokens) -> Tokens:
        idx, r = self.route(x)
        gb: Hidden = F.linear(x, self.gate)
        ub: Hidden = F.linear(x, self.up)
        out = torch.zeros_like(x)
        for j in range(self.top_k):
            gs: list[Group] = [grp for grp in groups(idx[:, j], r[:, j]) if grp[0] >= 0]
            ds = {e: self.delta(e) for e, _, _ in gs}
            g, u = gb, ub
            for e, rows, _ in gs:
                g = g.index_add(0, rows, F.linear(x[rows], ds[e][0]))
                u = u.index_add(0, rows, F.linear(x[rows], ds[e][1]))
            h: Hidden = F.silu(g) * u
            o: Tokens = F.linear(h, self.down)
            for e, rows, _ in gs:
                o = o.index_add(0, rows, F.linear(h[rows], ds[e][2]))
            out = out + r[:, j, None] * o
        return out

    def dense_experts(self) -> DenseExperts:
        r"""The equivalent plain MoE's full expert weights, for ``forward_dense``.

        Returns:
            [gate (N+1, I, H), up (N+1, I, H), down (N+1, H, I)], index 0 = base, index e+1 = :math:`W_b + \Delta_e`.
        """
        return [
            torch.cat((b[None], b + d))
            for b, d in (
                (self.gate, self.d_gate),
                (self.up, self.d_up),
                (self.down, self.d_down),
            )
        ]

    def _forward_dense(self, x: Tokens, experts: DenseExperts) -> Tokens:
        """Plain top-k MoE with stored full experts, grouped the same way (the baseline)."""
        idx, r = self.route(x)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            out = out.index_add(
                0,
                rows,
                r[rows, slots, None] * mlp(x[rows], *(w[e + 1] for w in experts)),
            )
        return out

    def _forward_matrix_grouped(self, x: Tokens) -> Tokens:
        """``forward_matrix`` with the deltas of all (token, slot) pairs done as one grouped GEMM per projection."""
        idx, r = self.route(x)
        tok = torch.arange(idx.numel(), device=x.device) // self.top_k  # pair -> token
        rows, offs = sort_rows(idx.flatten(), self.n_delta)  # pairs with a delta
        xs = x[tok[rows]]
        g = F.linear(x, self.gate)[tok].index_add(0, rows, gmm(xs, self.d_gate, offs))
        u = F.linear(x, self.up)[tok].index_add(0, rows, gmm(xs, self.d_up, offs))
        h: PairHidden = F.silu(g) * u  # each slot keeps its own activation
        o = F.linear(h, self.down).index_add(0, rows, gmm(h[rows], self.d_down, offs))
        return (o.view(-1, self.top_k, x.shape[-1]) * r[..., None]).sum(1)

    def _forward_dense_grouped(self, x: Tokens, experts: DenseExperts) -> Tokens:
        """Plain top-k MoE, all (token, slot) pairs as one grouped GEMM per projection (like transformers' ``grouped_mm``)."""
        idx, r = self.route(x)
        tok = torch.arange(idx.numel(), device=x.device) // self.top_k
        rows, offs = sort_rows(
            idx.flatten() + 1, len(experts[0])
        )  # every pair; zero experts -> 0 = base
        xs = x[tok[rows]]
        h = F.silu(gmm(xs, experts[0], offs)) * gmm(xs, experts[1], offs)
        o = torch.empty(idx.numel(), x.shape[-1], dtype=x.dtype, device=x.device)
        o[rows] = gmm(h, experts[2], offs)
        return (o.view(-1, self.top_k, x.shape[-1]) * r[..., None]).sum(1)

    def _forward_param_triton(self, x: Tokens) -> Tokens:
        r"""Parameter aggregation with :math:`W_b + \Delta_e` fused into the GEMMs (``triton_kernels/fused.py``): the
        routing (``topk``) sorted in one kernel, gate/up with SiLU * up in one, down with the routing weights in one; no host
        sync, so it can be captured in a CUDA graph. CUDA, bf16/fp16."""
        from ..triton_kernels import fused_moe

        idx, r = self.topk(x)
        return fused_moe(x, idx, r, self.n_delta, (self.gate, self.up, self.down), self._kernel_deltas())

    def _kernel_deltas(self) -> "tuple[Delta, Delta, Delta]":
        """The deltas in the form the fused kernels take; ``dare.DareDeltaMoE`` overrides it with its packed storage."""
        from ..triton_kernels import Delta

        return Delta(0, self.d_gate), Delta(0, self.d_up), Delta(0, self.d_down)

    def _grouped(self, x: Tokens) -> bool:
        """Use grouped GEMM? Needs ``F.grouped_mm`` with this dtype on this device, and more than one (token, slot) pair
        (with one pair the sort and count cost more than the loop saves)."""
        if not hasattr(F, "grouped_mm") or x.dtype not in GROUPED_DTYPES:
            return False
        if x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) < (
            8,
            0,
        ):
            return False
        return (
            x.device.type in ("cuda", "cpu")
            and x.shape[0] * self.top_k >= GROUPED_MIN_PAIRS
        )

    def _needs_grad(self, x: Tokens) -> bool:
        return torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in self.parameters()))

    def _use_triton(self, x: Tokens) -> bool:
        """Fused Triton kernels? Triton, CUDA, a ``TRITON_DTYPES`` dtype, a routing sort that fits; with a gradient
        needed only when ``triton_train`` is set (any pair count), else ``TRITON_MIN_PAIRS`` ..
        ``TRITON_MAX_PAIRS_*`` (token, slot) pairs (on Hopper with dense deltas no upper limit)."""
        if not (HAS_TRITON and x.is_cuda and x.dtype in TRITON_DTYPES and self.gate.dtype == x.dtype):
            return False  # (weights in another dtype: e.g. an fp32 model under bf16 autocast)
        if pow2(self.top_k) * pow2(self.n_delta + 2) > SORT_ONE_HOT_MAX:
            return False  # the routing sort's one-hot, even one token at a time
        if self._needs_grad(x):
            # weight gradients use F.grouped_mm: 16-byte aligned rows (hidden and inter multiples of 8 in bf16/fp16)
            return self.triton_train and hasattr(F, "grouped_mm") and self.gate.shape[0] % 8 == self.gate.shape[1] % 8 == 0
        hopper = torch.cuda.get_device_capability(x.device) >= (9, 0)
        P = x.shape[0] * self.top_k
        if hopper and self._kernel_deltas()[0].mode == 0:
            # dense deltas: no upper limit, the fused forward goes to its grouped-GEMM path past
            # DENSE_PREFILL_PAIRS_PER_GROUP * (n_delta + 1) pairs (cuda_kernels/dense_moe.py) when it can, and stays
            # capturable either way (the materialised loop syncs with the host)
            return P >= TRITON_MIN_PAIRS
        limit = TRITON_MAX_PAIRS_HOPPER if hopper else TRITON_MAX_PAIRS_OLDER
        return TRITON_MIN_PAIRS <= P <= limit

    def forward_param(self, x: Tokens) -> Tokens:
        r"""MoE forward by parameter aggregation.

        Uses the fused Triton kernels when ``_use_triton`` (with the Triton backward when a gradient is needed), else
        materialised :math:`W_b + \Delta_e` (experts with fewer than ``PARAM_MIN_ROWS`` routed rows skip the
        materialisation; differentiable).

        Args:
            x: (T, H) tokens.

        Returns:
            (T, H) MoE output, :math:`\sum_{i=1}^{k} r_i \, \mathrm{MLP}(x; W_{e_i})`.
        """
        if self._use_triton(x):
            return self._forward_param_autograd(x) if self._needs_grad(x) else self._forward_param_triton(x)
        return self._forward_param(x, min_rows=PARAM_MIN_ROWS)

    def _forward_param_autograd(self, x: Tokens) -> Tokens:
        """Differentiable fused parameter aggregation: the routing in torch autograd (the router gets its gradient
        through the routing weights), the MoE in one ``autograd.Function`` that saves only its inputs; dense deltas."""
        from ..triton_kernels.train import fused_moe_autograd

        idx, r = self.topk(x)
        return fused_moe_autograd(x, idx, r, (self.gate, self.up, self.down),
                                  (self.d_gate, self.d_up, self.d_down), self.n_delta)  # fmt: skip

    def forward_matrix(self, x: Tokens) -> Tokens:
        r"""MoE forward by per-matrix output aggregation.

        Uses grouped GEMM when available and worth it, else the per-expert loop.

        Args:
            x: (T, H) tokens.

        Returns:
            (T, H) MoE output, :math:`\sum_{i=1}^{k} r_i \, \mathrm{MLP}(x; W_{e_i})`.
        """
        return (
            self._forward_matrix_grouped(x)
            if self._grouped(x)
            else self._forward_matrix(x)
        )

    def forward_dense(self, x: Tokens, experts: DenseExperts) -> Tokens:
        """Plain MoE baseline forward from stored full experts.

        Uses grouped GEMM when available and worth it, else the per-expert loop.

        Args:
            x: (T, H) tokens.
            experts: Full expert weights as returned by ``dense_experts``, index 0 = base.

        Returns:
            (T, H) MoE output.
        """
        if self._grouped(x):
            return self._forward_dense_grouped(x, experts)
        return self._forward_dense(x, experts)
