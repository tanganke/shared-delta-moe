"""Top-k SwiGLU MoE written as a shared base expert plus per-expert deltas, with two exact forward forms.

Routed expert e has weights W_e = W_b + Delta_e for each of gate, up and down. Router rows 0 .. n-1 are the
deltas; rows n .. n+Z-1 are zero experts (num_zero_experts = Z, default 1): no parameters, each stands for the base
MLP itself. Routing is softmax, top-k, renormalised over the k picks (weights r). Both forms below equal the plain MoE
sum_i r_i MLP(x; W_{e_i})  with experts [W_b + Delta_1, ..., W_b + Delta_n] + [W_b] * Z; they differ only in where
the addition happens:

  forward_param   parameter aggregation: per expert, materialise W_b + Delta_e once and run one MLP on the
                  (token, slot) pairs routed to it, weighted by r (small groups skip the materialisation)
  forward_matrix  per-matrix output aggregation: gate/up are W_b x (once for all tokens) + Delta_e x per slot; each
                  slot has its own activation h_i, down is W_b h_i + Delta_e h_i; slots summed with weights r
  forward_dense   the plain MoE baseline, from stored full experts

Each public forward picks an implementation (_forward_*) by hardware and input, with thresholds in constants.py:
forward_param uses the fused Triton kernel on CUDA bf16/fp16 for a moderate number of (token, slot) pairs, else
materialised weights; forward_matrix / forward_dense use grouped GEMM (F.grouped_mm) when available for the dtype and
device and there is more than one pair, else a loop.

Adding r Delta_e x into one shared activation instead (merging weights per token) is not the same function when k > 1,
because SwiGLU is nonlinear; each slot keeps its own activation here. With k = 1, r = 1 and both reduce to one MLP.

Plain PyTorch (F.linear, F.grouped_mm, index_add), plus an optional Triton kernel for parameter aggregation that
adds W_b + Delta_e tile by tile in registers instead of writing the sum out (triton_kernels.py).
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .constants import (
    GROUPED_DTYPES,
    GROUPED_MIN_PAIRS,
    HAS_TRITON,
    PARAM_MIN_ROWS,
    TRITON_DTYPES,
    TRITON_MAX_PAIRS_HOPPER,
    TRITON_MAX_PAIRS_OLDER,
    TRITON_MIN_PAIRS,
)
from .ops import gmm, groups, mlp, sort_rows
from .type_hints import (
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
        """Uninitialised (torch.empty); fill with from_experts or load_state_dict."""
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
        self._init_deltas(n_delta, inter, hidden, factory)

    def _init_deltas(
        self, n: int, inter: int, hidden: int, factory: dict[str, Any]
    ) -> None:
        """Delta storage; a subclass with another format (dare.DareDeltaMoE) overrides this and d_gate/d_up/d_down."""
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
        """From a plain MoE: base = the base expert's (gate, up, down), experts = the other experts' full weights W_e,
        router = the MoE's router weight, rows n..n+Z-1 routing to the base (Z = num_zero_experts; 0: the base is only
        shared, never an expert on its own). Stores W_b and Delta_e = W_e - W_b in the inputs' dtype.
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

    def route(self, x: Tokens) -> tuple[Experts, RouteWeights]:
        r, i = self.router(x).softmax(-1).topk(self.top_k, dim=-1)
        return torch.where(i < self.n_delta, i, -1), r / r.sum(-1, keepdim=True)

    def weights(self, e: int) -> MLPWeights:
        if e < 0:
            return self.gate, self.up, self.down
        return (
            self.gate + self.d_gate[e],
            self.up + self.d_up[e],
            self.down + self.d_down[e],
        )

    def mlp_split(self, x: Tokens, e: int) -> Tokens:
        """MLP(x; W_b + Delta_e) without forming the sum: every projection is W_b x + Delta_e x."""
        g = F.linear(x, self.gate) + F.linear(x, self.d_gate[e])
        u = F.linear(x, self.up) + F.linear(x, self.d_up[e])
        h = F.silu(g) * u
        return F.linear(h, self.down) + F.linear(h, self.d_down[e])

    def _forward_param(self, x: Tokens, min_rows: int = 0) -> Tokens:
        """min_rows > 0: an expert with fewer routed rows skips materialising W_b + Delta_e and uses mlp_split
        (reads W_b and Delta_e once, writes no weights: cheaper while memory-bound, twice the FLOPs).
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
            out = out.index_add(0, rows, r[rows, slots, None] * y)
        return out

    def _forward_matrix(self, x: Tokens) -> Tokens:
        idx, r = self.route(x)
        gb: Hidden = F.linear(x, self.gate)
        ub: Hidden = F.linear(x, self.up)
        out = torch.zeros_like(x)
        for j in range(self.top_k):
            gs: list[Group] = [grp for grp in groups(idx[:, j], r[:, j]) if grp[0] >= 0]
            g, u = gb, ub
            for e, rows, _ in gs:
                g = g.index_add(0, rows, F.linear(x[rows], self.d_gate[e]))
                u = u.index_add(0, rows, F.linear(x[rows], self.d_up[e]))
            h: Hidden = F.silu(g) * u
            o: Tokens = F.linear(h, self.down)
            for e, rows, _ in gs:
                o = o.index_add(0, rows, F.linear(h[rows], self.d_down[e]))
            out = out + r[:, j, None] * o
        return out

    def dense_experts(self) -> DenseExperts:
        """The equivalent plain MoE: [gate (N+1, I, H), up (N+1, I, H), down (N+1, H, I)], index 0 = base."""
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
        """forward_matrix with the deltas of all (token, slot) pairs done as one grouped GEMM per projection."""
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
        """Plain top-k MoE, all (token, slot) pairs as one grouped GEMM per projection (like transformers' grouped_mm)."""
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
        """Parameter aggregation with W_b + Delta_e fused into the GEMM (triton_kernels.delta_gmm), all (token, slot)
        pairs in one launch per projection. CUDA, bf16/fp16."""
        idx, r = self.route(x)
        n, k = self.n_delta, self.top_k
        e = idx.flatten()
        e = torch.where(e < 0, n, e)  # every zero expert -> group n (W_b alone)
        order = e.argsort()
        counts = torch.zeros(n + 1, dtype=torch.int32, device=x.device)
        counts.scatter_add_(
            0, e, torch.ones_like(e, dtype=torch.int32)
        )  # no host sync, unlike bincount
        ends = counts.cumsum(0, dtype=torch.int32)
        starts = ends - counts
        # largest group bounds the grid; for few pairs use the pair count instead of syncing to read it
        max_rows = e.numel() if e.numel() <= 1024 else int(counts.max())
        xs = x[order // k]
        groups = (starts, ends, max_rows)
        g = self._fused_gmm(xs, 0, *groups)
        u = self._fused_gmm(xs, 1, *groups)
        o = self._fused_gmm(F.silu(g) * u, 2, *groups)
        out = torch.empty_like(o)
        out[order] = o
        return (out.view(-1, k, x.shape[-1]) * r[..., None]).sum(1)

    def _fused_gmm(
        self, x: Tensor, j: int, starts: Tensor, ends: Tensor, max_rows: int
    ) -> Tensor:
        """Projection j (0 gate, 1 up, 2 down) of rows grouped by expert, W_b + Delta_g fused in the GEMM."""
        from .triton_kernels import delta_gmm

        wb = (self.gate, self.up, self.down)[j]
        return delta_gmm(
            x, wb, (self.d_gate, self.d_up, self.d_down)[j], starts, ends, max_rows
        )

    def _grouped(self, x: Tokens) -> bool:
        """Use grouped GEMM? Needs F.grouped_mm with this dtype on this device, and more than one (token, slot) pair
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

    def _use_triton(self, x: Tokens) -> bool:
        """Fused Triton parameter aggregation? Triton, CUDA, a TRITON_DTYPES dtype, and a (token, slot) pair count
        within TRITON_MIN_PAIRS .. TRITON_MAX_PAIRS_{HOPPER,OLDER} (see constants)."""
        if not (HAS_TRITON and x.is_cuda and x.dtype in TRITON_DTYPES):
            return False
        hopper = torch.cuda.get_device_capability(x.device) >= (9, 0)
        limit = TRITON_MAX_PAIRS_HOPPER if hopper else TRITON_MAX_PAIRS_OLDER
        return TRITON_MIN_PAIRS <= x.shape[0] * self.top_k <= limit

    def forward_param(self, x: Tokens) -> Tokens:
        """Parameter aggregation: the fused Triton kernel when _use_triton, else materialised W_b + Delta_e (experts
        with fewer than PARAM_MIN_ROWS routed rows skip the materialisation)."""
        if self._use_triton(x):
            return self._forward_param_triton(x)
        return self._forward_param(x, min_rows=PARAM_MIN_ROWS)

    def forward_matrix(self, x: Tokens) -> Tokens:
        """Per-matrix output aggregation: grouped GEMM when available and worth it, else the per-expert loop."""
        return (
            self._forward_matrix_grouped(x)
            if self._grouped(x)
            else self._forward_matrix(x)
        )

    def forward_dense(self, x: Tokens, experts: DenseExperts) -> Tokens:
        """Plain MoE baseline: grouped GEMM when available and worth it, else the per-expert loop."""
        if self._grouped(x):
            return self._forward_dense_grouped(x, experts)
        return self._forward_dense(x, experts)
