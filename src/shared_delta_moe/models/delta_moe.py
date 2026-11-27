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

The plain MoE itself, with :math:`W_b + \Delta_e` written out, is ``to_moe()`` (``moe.MoE``).

Each public forward picks an implementation (``_forward_*``) by hardware and input, with thresholds in
``constants.py``: ``forward_param`` uses the fused Triton kernel on CUDA bf16/fp16 for a moderate number of (token,
slot) pairs, else materialised weights; ``forward_matrix`` uses grouped GEMM (``F.grouped_mm``)
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
    HAS_TRITON,
    PARAM_MIN_ROWS,
    SORT_ONE_HOT_MAX,
    TRITON_DTYPES,
    TRITON_MAX_PAIRS_HOPPER,
    TRITON_MAX_PAIRS_OLDER,
    TRITON_MIN_PAIRS,
)
from ..ops import gmm, groups, mlp, pow2, replay, softmax_topk, sort_rows, use_grouped
from ..type_hints import (
    Experts,
    Group,
    Hidden,
    MLPWeights,
    PairHidden,
    RouteWeights,
    RouterWeight,
    Tokens,
)
from .moe import MoE


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
            measured slower (docs/internals/benchmarks.md, Training). Default False.
        remat: The materialised training path recomputes :math:`W_b + \Delta_e` in the backward instead of keeping it
            (``_mlp_remat``): ~21 GiB less on a whole Qwen3.5-4B without checkpointing, 15-30% slower step. Set
            False when the step fits anyway. Default True.
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
    # holds no W_b + Delta_e and no per-pair activations for the backward, but measured slower (docs/internals/benchmarks.md, Training)
    triton_train: bool = False
    remat: bool = True  # recompute W_b + Delta_e in the materialised backward (_mlp_remat)
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

    def to_moe(self) -> "MoE":
        r"""The equivalent plain MoE: experts :math:`[W_b + \Delta_1, \dots, W_b + \Delta_N] + [W_b] \times Z`, the
        same router and top-k (a zero expert becomes a stored copy of :math:`W_b`). Packed formats unpack their deltas.

        Returns:
            A new ``moe.MoE`` with :math:`N + Z` experts, on this module's device and dtype.
        """
        with torch.no_grad():
            experts = [self.weights(e) for e in range(self.n_delta)] + [self.weights(-1)] * self.num_zero_experts
            return MoE.from_experts(experts, self.router.weight, self.top_k)

    def topk(self, x: Tokens, router: nn.Module | None = None) -> tuple[Tensor, Tensor]:
        """Route tokens: top-k router indices and their renormalised weights.

        Float32 softmax and renormalisation, as transformers' MoE routers do (bf16 can flip near-ties). Every forward
        routes through this one call, so they all pick the same experts, ties included; an activation-checkpoint
        recompute reuses the forward's picks (``ops.replay``).

        Args:
            x: (T, H) tokens.
            router: Another router of the same shape as ``self.router``, routed the same way (``routed_linear``: the
                attention and lm_head deltas); None = ``self.router``.

        Returns:
            A tuple (idx, r): idx (T, K) int64 router index per slot (>= ``n_delta``: a zero expert); r (T, K) float32
            routing weights, each row renormalised to sum to 1.
        """
        router = self.router if router is None else router
        logits = router(x)
        pick = replay(router, logits, lambda: softmax_topk(logits.detach(), self.top_k)[0])
        return softmax_topk(logits, self.top_k, pick)

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

    def _mlp_remat(self, x: Tokens, e: int) -> Tokens:
        r"""``mlp(x, *self.weights(e))``, but autograd does not keep :math:`W_b + \Delta_e` for the backward: the
        saved copies are swapped for a recipe and recomputed (one add per weight) when the backward needs them, so a
        training step holds the activations only, not one materialised expert per routed expert and layer. Not done
        under other saved-tensor hooks (activation checkpointing, offloading), which already decide what is kept, nor
        with ``remat`` False or without torch's (private) hook query. Saves nothing under autocast: F.linear then
        saves its own cast copy, which is kept. Values and gradients are unchanged: the recomputed sum is the same sum."""
        w = self.weights(e)
        top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
        if not self.remat or e < 0 or not torch.is_grad_enabled() or top is None or top(False) is not None:
            return mlp(x, *w)
        # metadata only: autograd keeps the hooks with every saved tensor, so a closure over w would keep it alive
        ptr = {t.untyped_storage().data_ptr(): (i, t.dtype, t.untyped_storage().nbytes()) for i, t in enumerate(w)}
        cache: dict[int, Tensor] = {}

        def pack(t: Tensor) -> "Tensor | tuple":
            # the weight itself or a view of it (F.linear may save W.t()); w[i] is contiguous, so offset 0 = same start
            i, dtype, nbytes = ptr.get(t.untyped_storage().data_ptr(), (-1, None, 0))
            if i < 0 or t.dtype != dtype or t.untyped_storage().nbytes() != nbytes:
                return t
            return i, t.shape, t.stride(), t.storage_offset()

        def unpack(p: "Tensor | tuple") -> Tensor:
            if isinstance(p, Tensor):
                return p
            i, shape, stride, offset = p
            if i not in cache:
                with torch.no_grad():
                    cache.update(enumerate(self.weights(e)))
            return cache.pop(i).as_strided(shape, stride, offset)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return mlp(x, *w)  # w (the only reference left) is freed on return

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
                y = self._mlp_remat(x[rows], e)
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
        return use_grouped(x, self.top_k)

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

    def forward(self, x: Tensor) -> Tensor:
        """The module's forward: the MoE over tokens of any leading shape.

        Args:
            x: (..., H) tokens.

        Returns:
            MoE output, same shape as ``x``: ``forward_param``, except where a subclass picks another forward by batch
            size (``PackedDeltaMoE``: ``forward_matrix`` at small batches).
        """
        shape = x.shape
        return self._forward_auto(x.reshape(-1, shape[-1])).view(shape)

    def _forward_auto(self, x: Tokens) -> Tokens:
        return self.forward_param(x)

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
