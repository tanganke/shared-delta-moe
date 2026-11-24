"""Top-k SwiGLU MoE written as a shared base expert plus per-expert deltas, with two exact forward forms.

Routed expert e has weights W_e = W_b + Delta_e for each of gate, up and down. Router index 0 is the zero expert:
it has no parameters and stands for the base MLP itself. Routing is softmax, top-k, renormalised over the k picks
(weights r). Both forms below equal the plain MoE  sum_i r_i MLP(x; W_{e_i})  with experts
[W_b, W_b + Delta_1, ..., W_b + Delta_n]; they differ only in where the addition happens:

  forward_param   parameter aggregation: per expert, materialise W_b + Delta_e once and run one MLP on the
                  (token, slot) pairs routed to it, weighted by r
  forward_matrix  per-matrix output aggregation: gate/up are W_b x (once for all tokens) + Delta_e x per slot; each
                  slot has its own activation h_i, down is W_b h_i + Delta_e h_i; slots summed with weights r

Adding r Delta_e x into one shared activation instead (merging weights per token) is not the same function when k > 1,
because SwiGLU is nonlinear; each slot keeps its own activation here. With k = 1, r = 1 and both reduce to one MLP.

Plain PyTorch only (F.linear, index_add); no custom kernels.

    python delta_moe.py [--top-k K] [--no-bench]   # self-check (float64, k = 1 and 2), then a benchmark if CUDA
"""
import sys
import time
from collections.abc import Callable
from typing import Annotated, Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Shapes as annotations (T tokens, H hidden, I intermediate, N deltas, K top-k, R rows of one group).
Tokens: TypeAlias = Annotated[Tensor, "T H"]
Hidden: TypeAlias = Annotated[Tensor, "T I"]
Experts: TypeAlias = Annotated[Tensor, "T K int64, delta index, -1 = zero expert"]
RouteWeights: TypeAlias = Annotated[Tensor, "T K, rows sum to 1"]
Rows: TypeAlias = Annotated[Tensor, "R int64 token indices"]
Coef: TypeAlias = Annotated[Tensor, "R 1 router weight of those rows"]
InWeight: TypeAlias = Annotated[Tensor, "I H"]
OutWeight: TypeAlias = Annotated[Tensor, "H I"]
MLPWeights: TypeAlias = tuple[InWeight, InWeight, OutWeight]
Group: TypeAlias = tuple[int, Rows, Coef]
DenseExperts: TypeAlias = list[Annotated[Tensor, "N+1 out in"]]


def mlp(x: Tokens, gate: InWeight, up: InWeight, down: OutWeight) -> Tokens:
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def groups(e: Annotated[Tensor, "T int64"], r: Annotated[Tensor, "T"]) -> list[Group]:
    """One routing slot -> [(delta index or -1, token rows, their weights)] for the experts it uses."""
    out: list[Group] = []
    for k in e.unique().tolist():
        rows = (e == k).nonzero().squeeze(-1)
        out.append((k, rows, r[rows, None]))
    return out


class DeltaMoE(nn.Module):
    gate: nn.Parameter  # I H
    up: nn.Parameter  # I H
    down: nn.Parameter  # H I
    d_gate: nn.Parameter  # N I H
    d_up: nn.Parameter  # N I H
    d_down: nn.Parameter  # N H I

    def __init__(self, hidden: int, inter: int, n_delta: int, top_k: int = 1, **factory: Any) -> None:
        """factory: dtype / device for every parameter."""
        super().__init__()
        assert 1 <= top_k <= n_delta + 1

        def p(*shape: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(*shape, **factory) * shape[-1] ** -0.5)

        self.top_k = top_k
        self.router = nn.Linear(hidden, n_delta + 1, bias=False, **factory)
        self.gate, self.up, self.down = p(inter, hidden), p(inter, hidden), p(hidden, inter)
        self.d_gate, self.d_up, self.d_down = p(n_delta, inter, hidden), p(n_delta, inter, hidden), p(n_delta, hidden, inter)

    def route(self, x: Tokens) -> tuple[Experts, RouteWeights]:
        r, i = self.router(x).softmax(-1).topk(self.top_k, dim=-1)
        return i - 1, r / r.sum(-1, keepdim=True)

    def weights(self, e: int) -> MLPWeights:
        if e < 0:
            return self.gate, self.up, self.down
        return self.gate + self.d_gate[e], self.up + self.d_up[e], self.down + self.d_down[e]

    def forward_param(self, x: Tokens) -> Tokens:
        idx, r = self.route(x)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)  # an expert appears at most once per token
            out = out.index_add(0, rows, r[rows, slots, None] * mlp(x[rows], *self.weights(e)))
        return out

    def forward_matrix(self, x: Tokens) -> Tokens:
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
        return [torch.cat((b[None], b + d)) for b, d in ((self.gate, self.d_gate), (self.up, self.d_up), (self.down, self.d_down))]

    def forward_dense(self, x: Tokens, experts: DenseExperts) -> Tokens:
        """Plain top-k MoE with stored full experts, grouped the same way (the baseline)."""
        idx, r = self.route(x)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            out = out.index_add(0, rows, r[rows, slots, None] * mlp(x[rows], *(w[e + 1] for w in experts)))
        return out


def check() -> None:
    for k in (1, 2):
        torch.manual_seed(0)
        m = DeltaMoE(16, 24, 3, top_k=k, dtype=torch.float64)
        x = torch.randn(200, 16, dtype=torch.float64)
        idx, r = m.route(x)
        assert len(idx.unique()) == 4, "check should hit the zero expert and every delta"
        assert k == 1 or (idx == -1).any(-1).logical_and((idx >= 0).any(-1)).any(), "zero expert mixed with a delta"
        experts = m.dense_experts()
        ref = torch.stack([sum(ri * mlp(xi, *(w[ei + 1] for w in experts)) for ei, ri in zip(es.tolist(), rs))
                           for xi, es, rs in zip(x, idx, r)])  # token by token
        for f in (m.forward_param, m.forward_matrix, lambda x: m.forward_dense(x, experts)):
            assert torch.allclose(f(x), ref, atol=1e-10), (k, (f(x) - ref).abs().max())
        n = sum(p.numel() for name, p in m.named_parameters() if name != "router.weight")
        assert n == 4 * 3 * 16 * 24, "base + 3 deltas = 4 MLPs, the zero expert adds nothing"
    print("check ok (top-1, top-2)")


@torch.no_grad()
def bench(hidden: int = 2560, inter: int = 9216, n_delta: int = 5, top_k: int = 1) -> None:
    torch.manual_seed(0)
    m = DeltaMoE(hidden, inter, n_delta, top_k=top_k, dtype=torch.bfloat16, device="cuda")
    experts = m.dense_experts()
    forms: dict[str, Callable[[Tokens], Tokens]] = {"dense MoE": lambda x: m.forward_dense(x, experts), "param": m.forward_param, "matrix": m.forward_matrix}
    print(f"{torch.cuda.get_device_name()}, hidden {hidden}, inter {inter}, {n_delta} deltas + zero expert, top-{top_k}, bf16, eager")
    print(f"{'tokens':>8} " + " ".join(f"{k:>14}" for k in forms) + "   (ms per layer)")
    for t in (1, 32, 512, 8192):
        x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
        row: list[float] = []
        for f in forms.values():
            for _ in range(3):
                f(x)
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(20):
                f(x)
            torch.cuda.synchronize()
            row.append((time.perf_counter() - s) / 20 * 1e3)
        print(f"{t:>8} " + " ".join(f"{v:>14.3f}" for v in row))


if __name__ == "__main__":
    check()
    if torch.cuda.is_available() and "--no-bench" not in sys.argv:
        bench(top_k=int(sys.argv[sys.argv.index("--top-k") + 1]) if "--top-k" in sys.argv else 1)
