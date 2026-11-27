"""Training through forward_param with triton_train (triton_kernels/train.py): output and gradients (x, router,
every weight) of the fused autograd path == the materialised PyTorch path (same dtype, so that routing is identical),
for top-1 .. top-all, zero experts mixed into the top-k, router-only training (frozen weights), HierDeltaMoE; DARE
keeps the materialised path. Needs CUDA and Triton (skipped otherwise). python tests/test_train.py also runs it."""

import copy
import itertools

import torch
import test_hier as hier
from test_delta_moe import make

from shared_delta_moe import DareDeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.grid import GridDeltaMoE

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


def check(m, t, router_only=False, tol=5e-2, fwd=lambda mod, xx: mod.forward_param(xx), triton=True):
    """triton False: a grid module's decoded router-training path (GridDeltaMoE._forward_decoded) instead."""
    m.triton_train = True
    torch.manual_seed(1)
    x = torch.randn(t, m.gate.shape[1], dtype=torch.bfloat16, device="cuda")
    dy = torch.randn_like(x)
    if router_only:
        for n, p in m.named_parameters():
            p.requires_grad_(n == "router.weight")
    # the reference runs in the same dtype: a float32 router can pick another expert for a token near a tie
    ref_m = copy.deepcopy(m)
    with torch.enable_grad():
        assert m._use_triton(x.requires_grad_()) if triton else m._use_decoded(x.requires_grad_()), "takes the path"
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


def test_router_only():
    """Frozen weights, trainable router: only dL/dx and dL/dr are formed (no weight-gradient GEMMs)."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for z, k in ((1, 2), (3, 6)):
        m, _, _ = make(z, k, 192, 320, torch.bfloat16, "cuda")
        assert list(check(m, 300, router_only=True)) == ["router.weight"]


def test_dare_trains_materialised():
    """DARE deltas: training stays on the materialised path even with triton_train."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, _ = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    d = DareDeltaMoE.from_delta_moe(m, 0.5, 64, seed=3)
    d.triton_train = True
    x = torch.randn(64, 192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    assert not d._use_triton(x)
    with torch.no_grad():
        assert d._use_triton(x)


def test_grid_router_only():
    """2-bit grid deltas (flat and Hier, asymmetric / symmetric, groups 32 .. 128) with frozen weights train through
    the decoded experts and grouped GEMMs (GridDeltaMoE._forward_decoded, also under checkpointing): output, dL/dx and
    the router's gradient match the materialised path; trainable grids or base keep the materialised path."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return

    def soft(m):  # make's randn router saturates the softmax (router gradients ~1e-24, all rounding)
        with torch.no_grad():
            m.router.weight.mul_(0.05)
        return m

    from torch.utils.checkpoint import checkpoint

    ckpt = lambda mod, xx: checkpoint(mod.forward_param, xx, use_reentrant=False)
    for z, k in ((1, 2), (3, 6)):
        m, _, _ = make(z, k, 256, 384, torch.bfloat16, "cuda")
        soft(m)
        for levels, group, sym in ((4, 128, False), (4, 32, False), (3, 64, True)):
            g = GridDeltaMoE.from_delta_moe(m, levels, group, sym)
            for t in (1, 300):
                assert list(check(g, t, router_only=True, triton=False)) == ["router.weight"]
        check(g, 300, router_only=True, triton=False, fwd=ckpt)
    for k1, k0 in ((2, None), (5, 2)):
        h, _, _ = hier.make(k1, k0, torch.bfloat16, "cuda", 256, 384, [4, 4, 4, 2], [1, 1, 1, 1])
        check(GridDeltaMoE.from_delta_moe(soft(h), 4, 128), 300, router_only=True, triton=False)
    g = GridDeltaMoE.from_delta_moe(m, 4, 128)
    for n, p in g.named_parameters():
        p.requires_grad_(n == "router.weight")
    x = torch.randn(64, 256, device="cuda", requires_grad=True)  # an fp32 residual stream under bf16 autocast
    with torch.autocast("cuda", torch.bfloat16):
        assert g._use_decoded(x)
        y = g(x)
        with torch.no_grad():
            y0 = g._forward_param(x)
    assert y.dtype == torch.float32 and rel(y, y0) < 2e-2
    y.square().sum().backward()
    assert x.grad is not None and g.router.weight.grad is not None
    assert not g._use_decoded(x.detach().requires_grad_())  # fp32 without autocast: the materialised path
    g.gate.requires_grad_(True)  # trainable base (or grids): the materialised path
    assert not g._use_decoded(torch.randn(8, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True))


def test_hier():
    """HierDeltaMoE (two-level routing flattened into slots, dense and top-k group routing, zero experts) trains
    through the same Function, including many groups x slots."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    big = ([3] * 16, [1] * 16)
    for k1, k0, (sizes, zeros) in [(2, None, (hier.SIZES, hier.ZEROS)), (1, 2, (hier.SIZES, hier.ZEROS)), (2, 8, big)]:
        m, _, _ = hier.make(k1, k0, torch.bfloat16, "cuda", 192, 320, sizes, zeros)
        for t in (1, 300):
            check(m, t)
        check(m, 300, router_only=True)


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
    m.triton_train = True
    _, gx, gp = grads(m, x, lambda mod, xx: mod.forward_param(xx), dy)
    _, gx1, gp1 = grads(m, x, lambda mod, xx: checkpoint(mod.forward_param, xx, use_reentrant=False), dy)
    assert rel(gx, gx1) < 1e-3, "same recompute (dL/dr sums by atomics, so x's router part may differ in rounding)"
    for n in gp:
        assert rel(gp[n], gp1[n]) < 1e-2, n


def test_autocast_and_unaligned():
    """A bf16 model under autocast with top-2 (the slot sum would come out fp32), and hidden / inter not multiples of
    8 (grouped_mm needs 16-byte rows: the materialised path instead)."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, _ = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    m.triton_train = True
    x = torch.randn(64, 192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", torch.bfloat16):
        assert m._use_triton(x)
        y = m.forward_param(x)
    assert y.dtype == torch.bfloat16
    y.float().square().sum().backward()
    assert x.grad is not None and m.d_down.grad is not None
    m, _, _ = make(1, 2, 200, 324, torch.bfloat16, "cuda")
    m.triton_train = True
    x = torch.randn(64, 200, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    assert not m._use_triton(x)
    m.forward_param(x).sum().backward()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
