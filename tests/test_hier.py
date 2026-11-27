"""Every forward of HierDeltaMoE equals the plain two-level ensemble of full experts, token by token, written out
from the formula. python tests/test_hier.py also runs it."""

import itertools

import torch

from shared_delta_moe import HierDeltaMoE, mlp, random_experts
from shared_delta_moe.constants import HAS_TRITON

SIZES, ZEROS = [2, 0, 3], [1, 2, 0]  # a group of only zero experts, a group without any
CASES = list(itertools.product((1, 2), (None, 1, 2)))  # (k1, k0); k0 None = dense


def make(k1, k0, dtype=torch.float32, device="cpu", hidden=16, inter=24, sizes=SIZES, zeros=ZEROS):
    torch.manual_seed(0)
    full = random_experts(hidden, inter, 1 + sum(sizes), dtype, device)
    groups, c = [], 1
    for e in sizes:
        groups.append(full[c : c + e])
        c += e
    router = torch.randn(len(sizes) + sum(sizes) + sum(zeros), hidden, dtype=dtype, device=device)
    m = HierDeltaMoE.from_experts(full[0], groups, router, zeros, k1, k0)
    return m, (full[0], groups, router), torch.randn(100, hidden, dtype=dtype, device=device)


def reference(src, k1, k0, x, zeros=ZEROS):
    """y = sum_{i in S} h_i sum_{j in top-k1} p_ij MLP(x; theta_ij), theta = W_b for a zero expert, per token."""
    base, groups, router = src
    G, out = len(groups), []
    for xt in x:
        lg = (router @ xt).float()
        h = lg[:G].softmax(-1)
        S = range(G) if k0 is None else h.topk(k0).indices.tolist()
        hs = h[list(S)].sum()
        y, c = 0, G
        for i in range(G):
            w = len(groups[i]) + zeros[i]
            if i in S:
                pv, pj = lg[c : c + w].softmax(-1).topk(k1)
                for v, j in zip(pv / pv.sum(), pj.tolist()):
                    theta = groups[i][j] if j < len(groups[i]) else base
                    y = y + (h[i] / hs * v).to(x.dtype) * mlp(xt[None], *theta)[0]
            c += w
        out.append(y)
    return torch.stack(out)


def forwards(m):
    return {
        "param": m.forward_param,
        "param split": lambda x: m._forward_param(x, min_rows=10**9),
        "matrix": m.forward_matrix,
        "matrix loop": m._forward_matrix,
        "matrix grouped": m._forward_matrix_grouped,
        "moe": m.to_moe(),
    }


def test_forwards_match_ensemble():
    for k1, k0 in CASES:
        m, src, x = make(k1, k0)
        idx, r = m.route(x)
        assert (idx >= 0).any()
        torch.testing.assert_close(r.sum(-1), torch.ones(len(x)), msg="weights sum to 1")
        if k0 is None or k0 > 1:
            assert ((idx < 0) & (r > 0)).any(), "a zero expert picked, weighted as the base"
        ref = reference(src, k1, k0, x)
        for name, f in forwards(m).items():
            if name == "matrix grouped" and not hasattr(torch.nn.functional, "grouped_mm"):
                continue
            torch.testing.assert_close(f(x), ref, atol=1e-5, rtol=1e-5, msg=f"{name} k1={k1} k0={k0}")


def test_gradients_match_ensemble():
    """dL/dx and dL/d every parameter (router, base, deltas) through the trainable forwards equal the ensemble's,
    the ensemble built from the module's own parameters (theta_ij = W_b + Delta_ij)."""
    for k1, k0 in CASES:
        m, _, x = make(k1, k0)
        groups, c = [], 0
        for e in SIZES:
            groups.append([(m.gate + m.d_gate[j], m.up + m.d_up[j], m.down + m.d_down[j]) for j in range(c, c + e)])
            c += e
        src = ((m.gate, m.up, m.down), groups, m.router.weight)
        fs = {"ensemble": lambda x: reference(src, k1, k0, x)} | {n: forwards(m)[n] for n in ("param", "param split", "matrix loop")}
        grads = {}
        for name, f in fs.items():
            xg = x.clone().requires_grad_()
            m.zero_grad()
            (f(xg) ** 2).sum().backward()
            grads[name] = [xg.grad] + [p.grad.clone() for p in m.parameters()]
        for name, g in grads.items():
            for a, b in zip(g, grads["ensemble"]):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4, msg=f"{name} k1={k1} k0={k0}")


def test_state_dict_roundtrip():
    m, _, x = make(2, None)
    n = HierDeltaMoE(16, 24, SIZES, ZEROS, 2, None)
    n.load_state_dict(m.state_dict())
    torch.testing.assert_close(n.forward_param(x), m.forward_param(x))
    # built on the meta device (as transformers' from_pretrained does), then materialised and loaded
    n = HierDeltaMoE(16, 24, SIZES, ZEROS, 2, None, device="meta").to_empty(device="cpu")
    n.load_state_dict(m.state_dict())
    torch.testing.assert_close(n.forward_param(x), m.forward_param(x))


def test_triton_matches_materialised():
    """Includes many groups x slots (16 slots, 48 deltas), where the routing sort once ran out of shared memory."""
    if not (torch.cuda.is_available() and HAS_TRITON):
        print("skipped: no CUDA")
        return
    big = ([3] * 16, [1] * 16)
    for k1, k0, (sizes, zeros) in [(*c, (SIZES, ZEROS)) for c in CASES] + [(2, 8, big), (2, None, big)]:
        m, src, x = make(k1, k0, torch.bfloat16, "cuda", 128, 256, sizes, zeros)
        with torch.no_grad():
            assert m._use_triton(x)
            torch.testing.assert_close(m.forward_param(x), m._forward_param(x), atol=5e-2, rtol=5e-2)  # matches the ensemble in fp32


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
