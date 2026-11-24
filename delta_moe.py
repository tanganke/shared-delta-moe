"""Top-k SwiGLU MoE written as a shared base expert plus per-expert deltas, with two exact forward forms.

Routed expert e has weights W_e = W_b + Delta_e for each of gate, up and down. Router rows 0 .. Z-1 are zero
experts (num_zero_experts = Z, default 1): no parameters, each stands for the base MLP itself; rows Z .. Z+n-1 are the
deltas. Routing is softmax, top-k, renormalised over the k picks (weights r). Both forms below equal the plain MoE
sum_i r_i MLP(x; W_{e_i})  with experts [W_b] * Z + [W_b + Delta_1, ..., W_b + Delta_n]; they differ only in where
the addition happens:

  forward_param   parameter aggregation: per expert, materialise W_b + Delta_e once and run one MLP on the
                  (token, slot) pairs routed to it, weighted by r
  forward_matrix  per-matrix output aggregation: gate/up are W_b x (once for all tokens) + Delta_e x per slot; each
                  slot has its own activation h_i, down is W_b h_i + Delta_e h_i; slots summed with weights r

Adding r Delta_e x into one shared activation instead (merging weights per token) is not the same function when k > 1,
because SwiGLU is nonlinear; each slot keeps its own activation here. With k = 1, r = 1 and both reduce to one MLP.

Plain PyTorch only (F.linear, index_add); no custom kernels.

    python delta_moe.py [--top-k K] [--no-bench]   # self-check (float64, k = 1 and 2), then a benchmark if CUDA
"""

import itertools
import sys
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Shapes as annotations (T tokens, H hidden, I intermediate, N deltas, K top-k, R rows of one group).
Tokens: TypeAlias = Annotated[Tensor, "T H"]
Hidden: TypeAlias = Annotated[Tensor, "T I"]
Experts: TypeAlias = Annotated[Tensor, "T K int64, delta index, -1 = any zero expert"]
RouteWeights: TypeAlias = Annotated[Tensor, "T K, rows sum to 1"]
Rows: TypeAlias = Annotated[Tensor, "R int64 token indices"]
Coef: TypeAlias = Annotated[Tensor, "R 1 router weight of those rows"]
InWeight: TypeAlias = Annotated[Tensor, "I H"]
OutWeight: TypeAlias = Annotated[Tensor, "H I"]
MLPWeights: TypeAlias = tuple[InWeight, InWeight, OutWeight]
Group: TypeAlias = tuple[int, Rows, Coef]
RouterWeight: TypeAlias = Annotated[Tensor, "Z+N H, rows 0..Z-1 = zero experts (base)"]
DenseExperts: TypeAlias = list[Annotated[Tensor, "N+1 out in, index 0 = base"]]


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

        self.top_k, self.num_zero_experts = top_k, num_zero_experts
        self.router = nn.Linear(
            hidden, num_zero_experts + n_delta, bias=False, **factory
        )
        self.gate, self.up, self.down = (
            p(inter, hidden),
            p(inter, hidden),
            p(hidden, inter),
        )
        self.d_gate, self.d_up, self.d_down = (
            p(n_delta, inter, hidden),
            p(n_delta, inter, hidden),
            p(n_delta, hidden, inter),
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
        router = the MoE's router weight, rows 0..Z-1 routing to the base (Z = num_zero_experts; 0: the base is only
        shared, never an expert on its own). Stores W_b and Delta_e = W_e - W_b in the inputs' dtype."""
        gate, up, down = base
        assert router.shape[0] == num_zero_experts + len(experts)
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
        return (i - self.num_zero_experts).clamp(min=-1), r / r.sum(-1, keepdim=True)

    def weights(self, e: int) -> MLPWeights:
        if e < 0:
            return self.gate, self.up, self.down
        return (
            self.gate + self.d_gate[e],
            self.up + self.d_up[e],
            self.down + self.d_down[e],
        )

    def forward_param(self, x: Tokens) -> Tokens:
        idx, r = self.route(x)
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slots = (idx == e).nonzero(
                as_tuple=True
            )  # under top-k two zero experts can share a row; index_add sums them
            out = out.index_add(
                0, rows, r[rows, slots, None] * mlp(x[rows], *self.weights(e))
            )
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
        return [
            torch.cat((b[None], b + d))
            for b, d in (
                (self.gate, self.d_gate),
                (self.up, self.d_up),
                (self.down, self.d_down),
            )
        ]

    def forward_dense(self, x: Tokens, experts: DenseExperts) -> Tokens:
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


def random_experts(
    hidden: int, inter: int, n: int, dtype: torch.dtype, device: str
) -> list[MLPWeights]:
    """n random (gate, up, down), scaled 1/sqrt(fan_in) so activations stay O(1) (for the check and benchmark)."""
    r = lambda *shape: (
        torch.randn(*shape, dtype=dtype, device=device) * shape[-1] ** -0.5
    )
    return [(r(inter, hidden), r(inter, hidden), r(hidden, inter)) for _ in range(n)]


def check() -> None:
    for z, k in itertools.product((0, 1, 2), (1, 2)):
        torch.manual_seed(0)
        full = random_experts(
            16, 24, 4, torch.float64, "cpu"
        )  # random weights only so that the experts differ
        m = DeltaMoE.from_experts(
            full[0],
            full[1:],
            torch.randn(z + 3, 16, dtype=torch.float64),
            top_k=k,
            num_zero_experts=z,
        )
        x = torch.randn(200, 16, dtype=torch.float64)
        idx, r = m.route(x)
        assert idx[idx >= 0].unique().tolist() == [0, 1, 2] and (idx == -1).any() == (
            z > 0
        ), "hit every delta, and the zero experts iff any"
        assert (
            z == 0
            or k == 1
            or (idx == -1).any(-1).logical_and((idx >= 0).any(-1)).any()
        ), "zero expert mixed with a delta"
        assert z < 2 or k == 1 or (idx == -1).all(-1).any(), (
            "two zero experts on one token"
        )
        experts = m.dense_experts()
        assert all(
            torch.allclose(w[e], f[j])
            for e, f in enumerate(full)
            for j, w in enumerate(experts)
        )
        ref = torch.stack(
            [
                sum(
                    ri * mlp(xi, *(w[ei + 1] for w in experts))
                    for ei, ri in zip(es.tolist(), rs)
                )
                for xi, es, rs in zip(x, idx, r)
            ]
        )  # token by token
        for f in (
            m.forward_param,
            m.forward_matrix,
            lambda x: m.forward_dense(x, experts),
        ):
            assert torch.allclose(f(x), ref, atol=1e-10), (
                z,
                k,
                (f(x) - ref).abs().max(),
            )
        n = sum(
            p.numel() for name, p in m.named_parameters() if name != "router.weight"
        )
        assert n == 4 * 3 * 16 * 24, (
            "base + 3 deltas = 4 MLPs, zero experts add nothing"
        )
    print("check ok (0/1/2 zero experts x top-1/top-2)")


@torch.no_grad()
def bench(
    hidden: int = 2560, inter: int = 9216, n_delta: int = 5, top_k: int = 1
) -> None:
    torch.manual_seed(0)
    full = random_experts(hidden, inter, n_delta + 1, torch.bfloat16, "cuda")
    m = DeltaMoE.from_experts(
        full[0],
        full[1:],
        torch.randn(n_delta + 1, hidden, dtype=torch.bfloat16, device="cuda"),
        top_k,
    )
    del full
    experts = m.dense_experts()
    forms: dict[str, Callable[[Tokens], Tokens]] = {
        "dense MoE": lambda x: m.forward_dense(x, experts),
        "param": m.forward_param,
        "matrix": m.forward_matrix,
    }
    print(
        f"{torch.cuda.get_device_name()}, hidden {hidden}, inter {inter}, {n_delta} deltas + zero expert, top-{top_k}, bf16, eager"
    )
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
        bench(
            top_k=int(sys.argv[sys.argv.index("--top-k") + 1])
            if "--top-k" in sys.argv
            else 1
        )
