"""Top-1 SwiGLU MoE written as a shared base expert plus per-expert deltas, with two exact forward forms.

Routed expert k has weights W_k = W_b + Delta_k for each of gate, up and down. Router index 0 is the zero expert:
it has no parameters and its tokens get the plain base MLP. Both forms below equal the dense MoE with experts
[W_b, W_b + Delta_1, ..., W_b + Delta_n]; they differ only in where the addition happens:

  forward_param   parameter aggregation: per routed group, materialise W_b + Delta_k, then run one MLP
  forward_matrix  per-matrix output aggregation: every projection is W_b x (all tokens) + Delta_k x (its group)

Plain PyTorch only (F.linear, index_add); no custom kernels. Top-1 routing with weight 1 is assumed (a router
weight r would scale the Delta_k terms in forward_matrix).

    python delta_moe.py          # self-check (float64), then a benchmark if CUDA is available
"""
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn


def mlp(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


class DeltaMoE(nn.Module):
    def __init__(self, hidden, inter, n_delta, **factory):
        super().__init__()
        p = lambda *shape: nn.Parameter(torch.randn(*shape, **factory) * shape[-1] ** -0.5)
        self.router = nn.Linear(hidden, n_delta + 1, bias=False, **factory)
        self.gate, self.up, self.down = p(inter, hidden), p(inter, hidden), p(hidden, inter)
        self.d_gate, self.d_up, self.d_down = p(n_delta, inter, hidden), p(n_delta, inter, hidden), p(n_delta, hidden, inter)

    def groups(self, x):
        """[(delta index or -1 for the zero expert, token rows)] for the experts this batch uses."""
        k = self.router(x).argmax(-1) - 1
        return [(e, (k == e).nonzero().squeeze(-1)) for e in k.unique().tolist()]

    def forward_param(self, x):
        out = torch.empty_like(x)
        for e, rows in self.groups(x):
            w = (self.gate, self.up, self.down)
            if e >= 0:
                w = (self.gate + self.d_gate[e], self.up + self.d_up[e], self.down + self.d_down[e])
            out[rows] = mlp(x[rows], *w)
        return out

    def forward_matrix(self, x):
        groups = [(e, rows) for e, rows in self.groups(x) if e >= 0]
        g, u = F.linear(x, self.gate), F.linear(x, self.up)
        for e, rows in groups:
            g = g.index_add(0, rows, F.linear(x[rows], self.d_gate[e]))
            u = u.index_add(0, rows, F.linear(x[rows], self.d_up[e]))
        h = F.silu(g) * u
        out = F.linear(h, self.down)
        for e, rows in groups:
            out = out.index_add(0, rows, F.linear(h[rows], self.d_down[e]))
        return out

    def dense_experts(self):
        """The equivalent plain MoE: stacked full expert weights, index 0 = base."""
        return [torch.cat((b[None], b + d)) for b, d in ((self.gate, self.d_gate), (self.up, self.d_up), (self.down, self.d_down))]

    def forward_dense(self, x, experts):
        """Plain top-1 MoE with stored full experts, grouped the same way (the baseline)."""
        out = torch.empty_like(x)
        for e, rows in self.groups(x):
            out[rows] = mlp(x[rows], *(w[e + 1] for w in experts))
        return out


def check():
    torch.manual_seed(0)
    m = DeltaMoE(16, 24, 3, dtype=torch.float64)
    x = torch.randn(200, 16, dtype=torch.float64)
    k = m.router(x).argmax(-1)
    assert len(k.unique()) == 4, "check should hit the zero expert and every delta"
    experts = m.dense_experts()
    ref = torch.stack([mlp(xi, *(w[ki] for w in experts)) for xi, ki in zip(x, k)])  # token by token
    for f in (m.forward_param, m.forward_matrix, lambda x: m.forward_dense(x, experts)):
        assert torch.allclose(f(x), ref, atol=1e-10), (f(x) - ref).abs().max()
    n = sum(p.numel() for name, p in m.named_parameters() if name != "router.weight")
    assert n == 4 * 3 * 16 * 24, "base + 3 deltas = 4 MLPs, the zero expert adds nothing"
    print("check ok")


@torch.no_grad()
def bench(hidden=2560, inter=9216, n_delta=5):
    torch.manual_seed(0)
    m = DeltaMoE(hidden, inter, n_delta, dtype=torch.bfloat16, device="cuda")
    experts = m.dense_experts()
    forms = {"dense MoE": lambda x: m.forward_dense(x, experts), "param": m.forward_param, "matrix": m.forward_matrix}
    print(f"{torch.cuda.get_device_name()}, hidden {hidden}, inter {inter}, {n_delta} deltas + zero expert, bf16, eager")
    print(f"{'tokens':>8} " + " ".join(f"{k:>14}" for k in forms) + "   (ms per layer)")
    for t in (1, 32, 512, 8192):
        x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
        row = []
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
        bench()
