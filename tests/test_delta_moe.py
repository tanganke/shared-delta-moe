"""Every forward equals the plain MoE, token by token. Runs under pytest, or directly: python tests/test_delta_moe.py
(the Triton test needs CUDA and is skipped without it)."""

import itertools

import torch

from shared_delta_moe import DeltaMoE, mlp, random_experts
from shared_delta_moe.constants import HAS_TRITON

CASES = list(itertools.product((0, 1, 2), (1, 2)))  # (zero experts, top-k)


def make(
    z: int, k: int, hidden: int = 16, inter: int = 24, dtype=torch.float32, device="cpu"
):
    torch.manual_seed(0)
    full = random_experts(
        hidden, inter, 4, dtype, device
    )  # random only so that the experts differ
    router = torch.randn(3 + z, hidden, dtype=dtype, device=device)
    m = DeltaMoE.from_experts(full[0], full[1:], router, top_k=k, num_zero_experts=z)
    return m, full, torch.randn(200, hidden, dtype=dtype, device=device)


def reference(m: DeltaMoE, x: torch.Tensor) -> torch.Tensor:
    idx, r = m.route(x)
    experts = m.dense_experts()
    return torch.stack(
        [
            sum(
                ri * mlp(xi, *(w[ei + 1] for w in experts))
                for ei, ri in zip(es.tolist(), rs)
            )
            for xi, es, rs in zip(x, idx, r)
        ]
    )


def test_routing_covers_cases():
    for z, k in CASES:
        m, _, x = make(z, k)
        idx, _ = m.route(x)
        assert idx[idx >= 0].unique().tolist() == [0, 1, 2], "every delta is hit"
        assert (idx == -1).any() == (z > 0), "zero experts hit iff there are any"
        if z and k > 1:
            assert (
                (idx == -1).any(-1) & (idx >= 0).any(-1)
            ).any(), "a zero expert mixed with a delta"
        if z > 1 and k > 1:
            assert (idx == -1).all(-1).any(), "two zero experts on one token"


def test_from_experts_and_parameter_count():
    for z, k in CASES:
        m, full, _ = make(z, k)
        experts = m.dense_experts()
        assert all(
            torch.allclose(w[e], f[j])
            for e, f in enumerate(full)
            for j, w in enumerate(experts)
        )
        n = sum(
            p.numel() for name, p in m.named_parameters() if name != "router.weight"
        )
        assert (
            n == 4 * 3 * 16 * 24
        ), "base + 3 deltas = 4 MLPs, zero experts add nothing"


def test_forwards_match_plain_moe():
    for z, k in CASES:
        m, _, x = make(z, k)
        experts = m.dense_experts()
        ref = reference(m, x)
        forms = {
            "param": m._forward_param,
            "param split": lambda x: m._forward_param(x, min_rows=10**9),
            "param mixed": lambda x: m._forward_param(x, min_rows=100),
            "matrix loop": m._forward_matrix,
            "matrix grouped": m._forward_matrix_grouped,
            "dense loop": lambda x: m._forward_dense(x, experts),
            "dense grouped": lambda x: m._forward_dense_grouped(x, experts),
            "forward_param": m.forward_param,
            "forward_matrix": m.forward_matrix,
            "forward_dense": lambda x: m.forward_dense(x, experts),
        }
        for name, f in forms.items():
            for t in (1, 200):  # 1 token takes the loop branch of the public forwards
                err = (f(x[:t]) - ref[:t]).abs().max().item()
                assert err < 1e-5, (z, k, name, t, err)


def test_gradients_match_plain_moe():
    """Training through forward_param (the materialised path): dL/dx and dL/d router match the plain MoE, for top-k
    up to every expert, zero experts included (their weight is in the renormalised top-k)."""
    for z, k in CASES + [(1, 4), (2, 5)]:  # k = n + Z: soft routing over every expert
        m, _, x = make(z, k)
        w = torch.randn(200, 16)
        grads = {}
        for name, f in (
            ("reference", lambda x: reference(m, x)),
            ("forward_param", m.forward_param),
            ("param split", lambda x: m._forward_param(x, min_rows=10**9)),
        ):
            xg = x.clone().requires_grad_(True)
            m.zero_grad()
            (f(xg) * w).sum().backward()
            grads[name] = (xg.grad, m.router.weight.grad.clone())
        for name, (gx, gr) in grads.items():
            ref_x, ref_r = grads["reference"]
            assert (gr.abs().max() > 1e-3) == (k > 1), (z, k, name)  # top-1 renormalises to 1: no router gradient (rounding only)
            assert torch.allclose(gx, ref_x, atol=1e-5) and torch.allclose(gr, ref_r, atol=1e-4), (z, k, name)


def test_triton_matches_materialised_param():
    if not torch.cuda.is_available():
        print("skipped: no CUDA")
        return
    for (z, k), (hidden, inter) in itertools.product(
        CASES, ((256, 384), (200, 300))
    ):  # 200/300: ragged tiles
        m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
        for t in (1, 37, 200):
            want = m._forward_param(x[:t])
            for name, f in (
                ("triton", m._forward_param_triton),
                ("forward_param", m.forward_param),
            ):
                got = f(x[:t])
                rel = ((got - want).abs().max() / want.abs().max()).item()
                assert rel < 2e-2, (name, z, k, hidden, t, rel)


def test_training_skips_forward_only_kernel():
    if not torch.cuda.is_available():
        print("skipped: no CUDA")
        return
    m, _, x = make(1, 2, 256, 384, torch.bfloat16, "cuda")
    m.forward_param(x[:37]).float().sum().backward()  # grad enabled: the materialised path, not the Triton kernel
    assert m.d_gate.grad is not None and m.d_gate.grad.abs().sum() > 0
    with torch.no_grad():
        assert m._use_triton(x[:37]) == HAS_TRITON


def test_module_forward_is_forward_param():
    m, _, x = make(1, 2)
    with torch.no_grad():
        torch.testing.assert_close(m(x.view(4, 50, -1)), m.forward_param(x).view(4, 50, -1))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
