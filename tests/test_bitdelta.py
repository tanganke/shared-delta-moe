"""BitDelta: sign bits pack and unpack exactly, the scales are BitDelta's, every forward of BitDeltaMoE equals the plain
MoE with W_b + diag(s) sign(Delta), and the fused Triton mode 2 matches the materialised forward. python
tests/test_bitdelta.py also runs it."""

import itertools

import torch
from test_delta_moe import CASES, make, reference

from shared_delta_moe import BitDeltaMoE, DeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.bitdelta import PROJ, pack_signs, unpack_signs

GPU = torch.cuda.is_available() and HAS_TRITON
H, I = 32, 64  # multiples of 32: one sign word per 32 entries


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_pack_signs_round_trip():
    """Several words per row, bit 31 (negative int32 words) and exact zeros (counted as +) included."""
    d = torch.randn(3, 5, 96)
    d[0, 0, :7] = 0
    w = pack_signs(d)
    assert w.shape == (3, 5, 3) and w.dtype == torch.int32 and (w < 0).any()
    assert torch.equal(unpack_signs(w, torch.float32), torch.where(d >= 0, 1.0, -1.0))
    assert pack_signs(torch.randn(0, 4, 64)).shape == (0, 4, 2)


def test_scales_and_deltas():
    """Block 16: 2 scales per gate / up row (C = H = 32), 4 per down row (C = I = 64)."""
    m, _, _ = make(1, 1, H, I)
    for scale in ("matrix", "row", 16):
        b = BitDeltaMoE.from_delta_moe(m, scale)
        for n in PROJ:
            d, s = getattr(m, f"d_{n}"), getattr(b, f"s_{n}")
            N, R, C = d.shape
            a = d.abs()
            want = {"row": lambda: a.mean(-1, keepdim=True), "matrix": lambda: a.mean((1, 2))[:, None, None].expand(N, R, 1),
                    16: lambda: a.view(N, R, C // 16, 16).mean(-1)}[scale]()  # fmt: skip
            assert s.shape == want.shape and torch.allclose(s, want), (scale, n)
            per_entry = s.repeat_interleave(C // s.shape[-1], -1)  # every row's scales spread over its entries
            sign = torch.where(d >= 0, 1.0, -1.0)
            assert torch.equal(getattr(b, f"d_{n}"), per_entry * sign), (scale, n)
    assert (BitDeltaMoE.from_delta_moe(m, 16).d_down - m.d_down).norm() < (BitDeltaMoE.from_delta_moe(m, "row").d_down - m.d_down).norm()


def test_bitdelta_forwards_match_plain_moe():
    for (z, k), scale in itertools.product(CASES, ("matrix", "row", 16)):
        m, _, x = make(z, k, H, I)
        b = BitDeltaMoE.from_delta_moe(m, scale)
        ref = reference(b, x)
        for name, f in {
            "param": b._forward_param,
            "param split": lambda x: b._forward_param(x, min_rows=10**9),
            "matrix": b._forward_matrix,
            "matrix grouped": b._forward_matrix_grouped,
            "moe grouped": b.to_moe()._forward_grouped,
        }.items():
            assert rel(f(x), ref) < 1e-5, (z, k, scale, name)


def test_state_dict_round_trip():
    """Saved and loaded (also into a model made on meta): sign bits as int32 buffers, scales as parameters."""
    m, _, x = make(2, 2, H, I)
    b = BitDeltaMoE.from_delta_moe(m, 16)
    assert {f"b_{n}" for n in PROJ} | {f"s_{n}" for n in PROJ} <= set(b.state_dict())
    for device in ("meta", "cpu"):
        b2 = BitDeltaMoE(H, I, m.n_delta, 2, 2, device=device, scale=16).to_empty(device="cpu")
        b2.load_state_dict(b.state_dict())
        assert torch.equal(b2._forward_param(x), b._forward_param(x)), device
    empty = BitDeltaMoE(H, I, 0, 1, 1)  # only the zero expert
    assert empty.d_gate.shape == (0, I, H)


def test_scale_gradients():
    """Scale distillation goes through the materialised forward: dL/ds[e, r] = sum_c dL/dDelta[e, r, c] sign[e, r, c],
    with the dense DeltaMoE holding the same (unpacked) deltas; x's gradient matches too."""
    for (z, k), scale in (((1, 1), "row"), ((2, 2), 16)):
        m, _, x = make(z, k, H, I)
        b = BitDeltaMoE.from_delta_moe(m, scale)
        dense = DeltaMoE(H, I, m.n_delta, k, z)
        with torch.no_grad():
            dense.router.weight.copy_(b.router.weight)
            for n in PROJ:
                getattr(dense, n).copy_(getattr(b, n))
                getattr(dense, f"d_{n}").copy_(getattr(b, f"d_{n}"))
        w = torch.randn(200, H)
        gx = []
        for mod in (b, dense):
            xg = x.clone().requires_grad_(True)
            (mod._forward_param(xg) * w).sum().backward()
            gx.append(xg.grad)
        assert torch.allclose(gx[0], gx[1], atol=1e-5), (z, k)
        for n in PROJ:
            sign = unpack_signs(getattr(b, f"b_{n}"), torch.float32)
            s = getattr(b, f"s_{n}")
            gd = getattr(dense, f"d_{n}").grad * sign
            want = gd.view(*s.shape, -1).sum(-1)  # summed over each scale's entries
            assert torch.allclose(getattr(b, f"s_{n}").grad, want, atol=1e-5), (z, k, n)


def test_packed_refuses_hier():
    """A HierDeltaMoE has a two-level router, which the flat packed forms (BitDelta, DARE) cannot hold."""
    from shared_delta_moe import DareDeltaMoE, HierDeltaMoE

    h = HierDeltaMoE(H, I, [2, 1], [1, 1], 1)
    for convert in (lambda: BitDeltaMoE.from_delta_moe(h), lambda: DareDeltaMoE.from_delta_moe(h, 0.5)):
        try:
            convert()
        except TypeError:
            continue
        raise AssertionError("converted a HierDeltaMoE")


def _gpu(z, k, hidden=192, inter=320, scale="row"):
    m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
    return BitDeltaMoE.from_delta_moe(m, scale), x


def test_triton_matches_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    # 192 / 320 leave ragged tiles; matrix scales once; block 32 (< every BK) and 128 (>= some BK) per row block
    for (z, k), (hidden, inter), scale in (((0, 1), (192, 320), "row"), ((2, 2), (192, 320), "row"),
                                           ((1, 2), (256, 384), "matrix"), ((2, 2), (192, 320), 32),
                                           ((1, 2), (256, 384), 128)):  # fmt: skip
        b, x = _gpu(z, k, hidden, inter, scale)
        for t in (1, 37, 200):
            with torch.no_grad():
                got, want = b._forward_param_triton(x[:t]), b._forward_param(x[:t])
            assert rel(got, want) < 2e-2, (z, k, hidden, scale, t, rel(got, want))


def test_every_config_and_split_k():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for scale in ("row", 32, 128):
        _every_config(*_gpu(2, 2, 256, 384, scale))


def _every_config(b, x):
    from shared_delta_moe.triton_kernels import fused as F

    dg, du, dd = b._kernel_deltas()
    with torch.no_grad():
        for t in (1, 37):
            xs = x[:t]
            want = b._forward_param(xs)
            r = F.route(*b.topk(xs), b.n_delta)
            h_ref = F.gate_up(xs, r, b.gate, b.up, dg, du, F.GATE_UP_CONFIGS[0])
            for c in F.GATE_UP_CONFIGS:
                try:
                    h = F.gate_up(xs, r, b.gate, b.up, dg, du, c)
                except ValueError:  # too much shared memory for this GPU
                    continue
                assert rel(h, h_ref) < 2e-2, ("gate_up", c, t)
            for c in F.DOWN_CONFIGS:
                for s in (1, 2, 4):
                    try:
                        y = F.down(h_ref, r, b.down, dd, c, split_k=s)
                    except ValueError:
                        continue
                    assert rel(y, want) < 2e-2, ("down", c, s, t)


def test_cuda_graph_and_compile():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    b, x = _gpu(1, 2, 256, 384, 64)
    xs = x[:8].clone()
    with torch.no_grad():
        b._forward_param_triton(xs)  # autotune outside the capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            b._forward_param_triton(xs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = b._forward_param_triton(xs)
        for _ in range(2):
            xs.copy_(torch.randn_like(xs))
            g.replay()
            assert rel(out, b._forward_param(xs)) < 2e-2
        torch._dynamo.reset()
        f = torch.compile(b._forward_param_triton, fullgraph=True)
        assert rel(f(x[:37]), b._forward_param(x[:37])) < 2e-2


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
