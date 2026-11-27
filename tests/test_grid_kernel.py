"""Fused Triton mode 3 (2-bit group grids): GridDeltaMoE / GridHierDeltaMoE's fused forward against the materialised
one (bf16, CUDA), every config and split-K, CUDA graphs and torch.compile; RoutedDeltaLinear's fused delta path
against its loop. GPU only (skipped otherwise); python tests/test_grid_kernel.py also runs it."""

import torch

from shared_delta_moe import DeltaMoE, HierDeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models import grid as Gd
from shared_delta_moe.models.routed_linear import RoutedDeltaLinear
from shared_delta_moe import compress as C

GPU = torch.cuda.is_available() and HAS_TRITON


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def _module(hier, levels, group, symmetric, hidden=256, inter=384):
    torch.manual_seed(0)
    if hier:  # widths 3 (2 + 1 zero) and 4: k1 = 4 pads the first group, dense routing in both
        m = HierDeltaMoE(hidden, inter, [2, 4], [1, 0], top_k=4, group_top_k=None, device="cuda", dtype=torch.bfloat16)
    else:
        m = DeltaMoE(hidden, inter, 5, top_k=2, num_zero_experts=1, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.normal_(std=0.02 if n.startswith("d_") else (1.0 if "router" in n else 0.05))
    g = Gd.GridDeltaMoE.from_delta_moe(m, levels, group, symmetric)
    return g, torch.randn(256, hidden, device="cuda", dtype=torch.bfloat16)


CASES = [(False, 4, 128, False), (True, 4, 128, False), (True, 3, 64, True), (False, 4, 32, False)]


def test_fused_matches_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for case in CASES:
        g, x = _module(*case)
        assert g.kernel_ok()
        for t in (1, 3, 37, 200):
            with torch.no_grad():
                got, want = g._forward_param_triton(x[:t]), g._forward_param(x[:t])
            assert rel(got, want) < 2e-2, (case, t, rel(got, want))
        with torch.no_grad():  # the module's own forward takes the fused path at these sizes
            assert g._use_triton(x[:8]) and rel(g(x[:8]), g._forward_param(x[:8])) < 2e-2


def test_every_config_and_split_k():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe.triton_kernels import fused as F

    for case in CASES[1:3]:
        g, x = _module(*case)
        dg, du, dd = g._kernel_deltas()
        with torch.no_grad():
            for t in (1, 37):
                xs = x[:t]
                want = g._forward_param(xs)
                r = F.route(*g.topk(xs), g.n_delta)
                h_ref = F.gate_up(xs, r, g.gate, g.up, dg, du, F.GATE_UP_CONFIGS[0])
                for c in F.GATE_UP_CONFIGS:
                    try:
                        h = F.gate_up(xs, r, g.gate, g.up, dg, du, c)
                    except ValueError:
                        continue
                    assert rel(h, h_ref) < 2e-2, ("gate_up", case, c, t)
                for c in F.DOWN_CONFIGS:
                    for s in (1, 2, 4):
                        try:
                            y = F.down(h_ref, r, g.down, dd, c, split_k=s)
                        except ValueError:
                            continue
                        assert rel(y, want) < 2e-2, ("down", case, c, s, t)


def test_cuda_graph_and_compile():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    g, x = _module(True, 4, 128, False)
    xs = x[:8].clone()
    with torch.no_grad():
        g._forward_param_triton(xs)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g._forward_param_triton(xs)
        torch.cuda.current_stream().wait_stream(s)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            out = g._forward_param_triton(xs)
        for _ in range(2):
            xs.copy_(torch.randn_like(xs))
            cg.replay()
            assert rel(out, g._forward_param(xs)) < 2e-2
        torch._dynamo.reset()
        f = torch.compile(g._forward_param_triton, fullgraph=True)
        assert rel(f(x[:37]), g._forward_param(x[:37])) < 2e-2


def test_routed_linear_decoded_loop():
    """Frozen grids under a gradient: the loop decodes each expert in Triton (and rebuilds it for the backward) and
    matches the PyTorch-unpacked loop in output and in the x and routing-weight gradients."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    for levels, group, sym in ((4, 128, False), (3, 64, True)):
        Cn, R, N, K, T = 512, 384, 4, 3, 300
        lin = RoutedDeltaLinear(Cn, R, N, levels, group, sym, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.normal_(std=0.05)
            for e in range(N):
                q, lo, st = C.quantize(torch.randn(R, Cn, device="cuda") * 0.01, None, levels, group, sym)
                lin.codes[e], lin.steps[e] = Gd.pack_codes(q, 2), st.bfloat16()
                if not sym:
                    lin.lows[e] = lo.bfloat16()
        lin.requires_grad_(False)
        idx = torch.randint(0, N + 2, (T, K), device="cuda")
        r0 = torch.rand(T, K, device="cuda").softmax(-1)
        x0 = torch.randn(T, Cn, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(T, R, device="cuda", dtype=torch.bfloat16)
        outs = []
        for dec, ck in ((False, False), (True, False), (True, True)):
            lin._decodable = (lambda x, d=dec: d) if not dec else RoutedDeltaLinear._decodable.__get__(lin)
            x, r = x0.clone().requires_grad_(True), r0.clone().requires_grad_(True)

            def f(x, r):
                lin.route = (idx, r)
                return lin(x)

            y = checkpoint(f, x, r, use_reentrant=False) if ck else f(x, r)
            assert lin._decodable(x) == dec
            y.backward(dy)
            outs.append((y.detach(), x.grad, r.grad))
        for got in outs[1:]:
            for a, b in zip(got, outs[0]):
                assert rel(a.float(), b.float()) < 1e-2, (levels, rel(a.float(), b.float()))


def test_routed_linear_fused():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    torch.manual_seed(0)
    for levels, group, sym in ((4, 128, False), (3, 64, True)):
        Cn, R, N, K = 512, 384, 4, 3
        lin = RoutedDeltaLinear(Cn, R, N, levels, group, sym, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.normal_(std=0.05)
            for e in range(N):
                q, lo, st = C.quantize(torch.randn(R, Cn, device="cuda") * 0.01, None, levels, group, sym)
                lin.codes[e], lin.steps[e] = Gd.pack_codes(q, 2), st.bfloat16()
                if not sym:
                    lin.lows[e] = lo.bfloat16()
        for T in (1, 5, 300):
            idx = torch.randint(0, N + 2, (T, K), device="cuda")  # >= N: no delta
            r = torch.rand(T, K, device="cuda").softmax(-1)
            lin.route = (idx, r)
            x = torch.randn(T, Cn, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                assert lin._fused(x, r, idx.numel())
                got = lin(x)
            assert not lin._fused(x, r.requires_grad_(True), idx.numel()), "router gradient: the loop"
            lin.route = (idx, r)
            want = lin(x)  # the loop (a gradient is needed)
            want.float().sum().backward()
            assert r.grad is not None and r.grad.abs().sum() > 0
            assert rel(got, want.detach()) < 2e-2, (levels, T, rel(got, want.detach()))


def test_cuda_matrix_matches_triton():
    """The CUDA matrix kernels (mode 3, decode sizes: ``packed_ops.matrix_moe_any``, ``routed_matrix``) against the
    fused Triton kernels, where the library is built (sm_90)."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe.cuda_kernels import packed_ops
    from shared_delta_moe.triton_kernels import fused as F

    for case in CASES:
        g, x = _module(*case)
        base, deltas = (g.gate, g.up, g.down), g._kernel_deltas()
        if not packed_ops.supports(x[:4], 4 * g.top_k, base, deltas, matrix=True):
            print("skipped: CUDA library not built / not sm_90, or group < 32")
            continue
        for t in (1, 3, 37, 200):
            with torch.no_grad():
                idx, r = g.topk(x[:t])
                got = packed_ops.matrix_moe_any(x[:t], idx, r, g.n_delta, base, deltas)
                want = F.fused_moe(x[:t], idx, r, g.n_delta, base, deltas)
            assert rel(got, want) < 2e-2, (case, t, rel(got, want))
        with torch.no_grad():
            assert g._matrix_cuda(x[:8]) and rel(g(x[:8]), g._forward_param(x[:8])) < 2e-2
        d = deltas[0]
        for t in (1, 5, 64):
            idx = torch.randint(0, g.n_delta + 1, (t, 2), device="cuda")
            r = torch.rand(t, 2, device="cuda")
            got = packed_ops.routed_matrix(x[:t], idx, r, g.n_delta, d)
            want = F.routed_deltas(x[:t], idx, r, g.n_delta, d)
            assert rel(got, want) < 2e-2, (case, t, rel(got, want))



def test_grid_ops_match_fp32():
    """grid_gate_up / grid_down (packed_ops.matrix_moe, routed_matrix: 2-bit grids factored per 128-entry block)
    against fp32 on the decoded deltas: grid_small_kernel (few pairs) and grid_kernel (2 and 4 pair tiles), bf16
    and fp16, groups 128 and 256, and routed rows not a multiple of 64, also by routed_small (no routing sort)."""
    import torch.nn.functional as F

    from shared_delta_moe.cuda_kernels import packed_ops
    from shared_delta_moe.models.grid import grid_delta
    from shared_delta_moe.triton_kernels import Delta
    from shared_delta_moe.triton_kernels.fused import route

    if not (torch.cuda.is_available() and packed_ops.available() and torch.cuda.get_device_capability() == (9, 0)):
        print("skipped: CUDA library not built / not sm_90")
        return
    torch.manual_seed(0)
    G, H, I, dev = 3, 512, 1024, "cuda"
    rel = lambda a, b: ((a.float() - b).abs().max() / b.abs().max()).item()
    for dt in (torch.bfloat16, torch.float16):
        for group in (128, 256):
            def grids(n, k):
                return (torch.randint(-2**31, 2**31 - 1, (G, n, k // 16), device=dev, dtype=torch.int32),
                        (torch.rand(G, n, k // group, device=dev) * 0.02).to(dt),
                        (torch.rand(G, n, k // group, device=dev) * -0.03).to(dt))
            cs = (grids(I, H), grids(I, H), grids(H, I))
            deltas = tuple(Delta(3, c, s_, z, group) for c, s_, z in cs)
            dec = [[grid_delta(c[e], s_[e], z[e], 4, 2).float() for e in range(G)] for c, s_, z in cs]
            base = tuple((torch.randn(*sh, device=dev) * 0.05).to(dt) for sh in ((I, H), (I, H), (H, I)))
            for T, k in ((1, 2), (6, 2), (20, 2), (40, 3)):  # small kernel; 2 and 4 pair tiles
                idx = torch.stack([torch.randperm(G + 1, device=dev)[:k] for _ in range(T)])  # G: a zero expert
                rw = torch.rand(T, k, device=dev).softmax(-1)
                x = torch.randn(T, H, device=dev, dtype=dt)
                got = packed_ops.matrix_moe(x, route(idx, rw, G), base, deltas)
                want = torch.zeros(T, H, device=dev)
                for t in range(T):
                    for j in range(k):
                        e, xf = int(idx[t, j]), x[t].float()
                        w = [b.float() + (dec[m][e] if e < G else 0) for m, b in enumerate(base)]
                        h = F.silu(w[0] @ xf) * (w[1] @ xf)
                        want[t] += rw[t, j] * (w[2] @ h)
                assert rel(got, want) < 2e-2, (dt, group, T, k, rel(got, want))
            R = 48  # routed rows: not a multiple of 64
            c, s_, z = grids(R, H)
            for T in (1, 9, 30):
                idx = torch.randint(0, G + 1, (T, 2), device=dev)
                idx[:, 0] %= G  # every token some delta (want != 0)
                rw = torch.rand(T, 2, device=dev)
                x = torch.randn(T, H, device=dev, dtype=dt)
                d = Delta(3, c, s_, z, group)
                got = packed_ops.routed_matrix(x, idx, rw, G, d)
                assert packed_ops.routed_small_ok(x, G, d)
                small = packed_ops.routed_small(x, idx, rw, d)  # no-sort kernel, same sums
                want = torch.zeros(T, R, device=dev)
                for t in range(T):
                    for j in range(2):
                        e = int(idx[t, j])
                        if e < G:
                            want[t] += rw[t, j] * (grid_delta(c[e], s_[e], z[e], 4, 2).float() @ x[t].float())
                assert rel(got, want) < 2e-2, (dt, group, "routed", T, rel(got, want))
                assert rel(small, want) < 2e-2, (dt, group, "routed_small", T, rel(small, want))

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
