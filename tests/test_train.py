"""Training through forward_param (triton_kernels/train.py): output and gradients (x, router, every weight) of the
fused autograd path == the materialised PyTorch path (same dtype, so that routing is identical), for top-1 .. top-all, zero experts mixed into the
top-k, router-only training (frozen weights) and DARE deltas. Needs CUDA and Triton (skipped otherwise).
python tests/test_train.py also runs it."""

import copy
import itertools

import torch
from test_delta_moe import make

from shared_delta_moe import DareDeltaMoE
from shared_delta_moe.constants import HAS_TRITON

GPU = torch.cuda.is_available() and HAS_TRITON


def rel(got, want):
    return ((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12)).item()


def grads(m, x, f, dy):
    m.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = f(m, x)
    y.backward(dy.to(y.dtype))
    named = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
    return y.detach(), x.grad, named


def check(m, t, router_only=False, tol=5e-2, fwd=lambda mod, xx: mod.forward_param(xx)):
    torch.manual_seed(1)
    x = torch.randn(t, m.gate.shape[1], dtype=torch.bfloat16, device="cuda")
    dy = torch.randn_like(x)
    if router_only:
        for n, p in m.named_parameters():
            p.requires_grad_(n == "router.weight")
    # the reference runs in the same dtype: a float32 router can pick another expert for a token near a tie
    ref_m = copy.deepcopy(m)
    y, gx, gp = grads(m, x, fwd, dy)
    y0, gx0, gp0 = grads(ref_m, x, lambda mod, xx: mod._forward_param(xx), dy)
    assert rel(y, y0) < tol, ("y", rel(y, y0))
    assert rel(gx, gx0) < tol, ("dx", rel(gx, gx0))
    assert gp.keys() == gp0.keys()
    for n in gp:
        assert gp[n] is not None, n
        if n == "router.weight" and m.top_k == 1:  # r = p / p = 1: the gradient is 0, both sides rounding noise
            assert gp[n].abs().max() < 1e-3 * gx0.abs().max() * t, (n, gp[n].abs().max())
            continue
        assert rel(gp[n], gp0[n]) < tol, (n, rel(gp[n], gp0[n]))
    return gp.keys()


def test_gradients_match_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    # (zero experts, top-k): one zero expert; zero experts mixed into top-2; every expert (3 deltas + 3 zero = top-6)
    for z, k in ((1, 1), (1, 2), (2, 2), (3, 6)):
        for t in (1, 37, 300):
            m, _, _ = make(z, k, 192, 320, torch.bfloat16, "cuda")
            names = check(m, t)
            assert "router.weight" in names and "d_down" in names
            print(f"  ok z={z} top-{k} tokens {t}", flush=True)


def test_non_contiguous_weights():
    """Weights in another memory layout (column-major copies of gate, up and their deltas): each tensor's own strides
    are used (gate and up once shared one set of strides and gave wrong results)."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for name in ("gate", "up", "d_gate", "d_up", "down", "d_down"):
        m, _, _ = make(1, 2, 192, 320, torch.bfloat16, "cuda")
        p = getattr(m, name)
        col = p.detach().transpose(-1, -2).contiguous().transpose(-1, -2)  # same values, column-major
        setattr(m, name, torch.nn.Parameter(col))
        check(m, 37)
        with torch.no_grad():
            x = torch.randn(37, 192, dtype=torch.bfloat16, device="cuda")
            assert rel(m._forward_param_triton(x), m._forward_param(x)) < 2e-2, name


def test_grouped_variant():
    """The grouped-GEMM autograd variant (large pair counts) against the materialised path, every case above."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe.triton_kernels.train import grouped_moe_autograd

    def fwd(mod, xx):
        idx, r = mod.topk(xx)
        return grouped_moe_autograd(xx, idx, r, (mod.gate, mod.up, mod.down), (mod.d_gate, mod.d_up, mod.d_down),
                                    mod.n_delta)  # fmt: skip

    for (z, k), t in itertools.product(((1, 1), (1, 2), (2, 2), (3, 6)), (1, 37, 300)):
        m, _, _ = make(z, k, 192, 320, torch.bfloat16, "cuda")
        check(m, t, fwd=fwd)
        check(m, t, router_only=True, fwd=fwd)


def test_router_only():
    """Frozen weights, trainable router: only dL/dx and dL/dr are formed (no weight-gradient GEMMs)."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for z, k in ((1, 2), (3, 6)):
        m, _, _ = make(z, k, 192, 320, torch.bfloat16, "cuda")
        assert list(check(m, 300, router_only=True)) == ["router.weight"]


def test_dare_trains_router():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, _ = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    d = DareDeltaMoE.from_delta_moe(m, 0.5, 64, seed=3)
    # reference: a plain DeltaMoE holding the DARE module's unpacked deltas
    dense = copy.deepcopy(m)
    with torch.no_grad():
        for n in ("gate", "up", "down"):
            getattr(dense, f"d_{n}").copy_(getattr(d, f"d_{n}"))
    torch.manual_seed(1)
    x = torch.randn(64, 192, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn_like(x)
    _, gx, gp = grads(d, x, lambda mod, xx: mod.forward_param(xx), dy)
    _, gx0, gp0 = grads(dense, x, lambda mod, xx: mod.forward_param(xx), dy)
    assert rel(gx, gx0) < 1e-2 and rel(gp["router.weight"], gp0["router.weight"]) < 1e-2


def test_checkpoint_recompute():
    """torch.utils.checkpoint re-runs the forward in backward: same gradients as without it."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from torch.utils.checkpoint import checkpoint

    m, _, _ = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    torch.manual_seed(1)
    x = torch.randn(300, 192, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn_like(x)
    _, gx, gp = grads(m, x, lambda mod, xx: mod.forward_param(xx), dy)
    _, gx1, gp1 = grads(m, x, lambda mod, xx: checkpoint(mod.forward_param, xx, use_reentrant=False), dy)
    assert rel(gx, gx1) < 1e-3, "same recompute (dL/dr sums by atomics, so x's router part may differ in rounding)"
    for n in gp:
        assert rel(gp[n], gp1[n]) < 1e-2, n


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
